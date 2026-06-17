import random
import os

import torch

from cg.api import (
    Observation,
    State,
    PlayerState,
    Pokemon,
    Option,
    SelectData,
    Card,
    OptionType,
    SelectType,
    SelectContext,
    EnergyType,
    CardType,
    SpecialConditionType,
    CardData,
    AreaType,
)
from cg.api import all_card_data, all_attack
from agent.evaluate import get_card_data, get_attack_data, evaluate_state, _can_use_attack
from agent.features import encode_observation

_card_db_cache: dict[int, CardData] | None = None
_policy_model = None
_policy_device = "cpu"
_policy_enabled = False
_policy_blend = 0.7


def load_policy_model(path: str | None = None, device: str = "cpu"):
    global _policy_model, _policy_device, _policy_enabled
    if path is None:
        for candidate in [
            "model.pt",
            "model.pth",
            os.path.join(os.path.dirname(__file__), "model.pt"),
            os.path.join(os.path.dirname(__file__), "..", "model.pt"),
            "/kaggle_simulations/agent/model.pt",
        ]:
            if os.path.exists(candidate):
                path = candidate
                break
    if path is None or not os.path.exists(path):
        _policy_enabled = False
        return

    from agent.network import load_model
    try:
        _policy_model = load_model(path, device=device)
        _policy_device = device
        _policy_enabled = True
    except Exception:
        _policy_enabled = False


def _card_db() -> dict[int, CardData]:
    return get_card_data()


def choose_action(obs: Observation, use_model: bool = True) -> list[int]:
    select = obs.select
    if select is None:
        return []

    state = obs.current
    my_index = state.yourIndex
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    select_type = select.type
    context = select.context
    options = select.option
    min_count = select.minCount
    max_count = select.maxCount

    if max_count == 0:
        return []

    if len(options) == 1:
        return [0]

    heuristic_action = _choose_action_heuristic(obs, select, state, my_index)

    if use_model and _policy_enabled and _policy_model is not None and select_type in (SelectType.MAIN, SelectType.ATTACK, SelectType.CARD):
        model_action = _choose_action_model(obs, select, heuristic_action)
        if model_action is not None:
            return model_action

    return heuristic_action


def _choose_action_model(obs: Observation, select: SelectData, heuristic_action: list[int]) -> list[int] | None:
    try:
        state_features, option_features = encode_observation(obs)
        if not option_features or not state_features:
            return None

        from agent.network import score_actions
        scores, value = score_actions(
            _policy_model,
            state_features,
            option_features,
            device=_policy_device,
        )

        num_options = len(option_features)
        if num_options == 0:
            return None

        best_idx = max(range(num_options), key=lambda i: scores[i] if i < len(scores) else -1e9)

        if 0 <= best_idx < num_options:
            blend = random.random()
            if blend < _policy_blend:
                return [best_idx]
            else:
                return heuristic_action
        return None
    except Exception:
        return None


def _choose_action_heuristic(obs: Observation, select: SelectData, state: State, my_index: int) -> list[int]:
    select_type = select.type
    context = select.context
    options = select.option
    min_count = select.minCount
    max_count = select.maxCount

    if max_count == 0:
        return []

    if len(options) == 1:
        return [0]

    if select_type == SelectType.YES_NO:
        return _handle_yes_no(obs, select, state, my_index)

    if select_type == SelectType.MAIN:
        return _handle_main(obs, select, state, my_index)

    if select_type == SelectType.ATTACK:
        return _handle_attack(obs, select, state, my_index)

    if select_type == SelectType.ENERGY:
        return _handle_energy(obs, select, state, my_index)

    if select_type == SelectType.CARD:
        return _handle_card_select(obs, select, state, my_index)

    if select_type == SelectType.COUNT:
        return _handle_count(obs, select, state, my_index)

    if select_type == SelectType.SPECIAL_CONDITION:
        return _handle_special_condition(obs, select, state, my_index)

    if min_count == max_count and min_count > 0:
        indices = list(range(min_count))
        if len(indices) <= len(options):
            return indices[:max_count]

    return random.sample(range(len(options)), min(max_count, len(options)))


def _handle_yes_no(obs: Observation, select: SelectData, state: State, my_index: int) -> list[int]:
    context = select.context
    options = select.option

    if context == SelectContext.IS_FIRST:
        return [_option_index_for_type(options, OptionType.YES)]

    if context == SelectContext.MULLIGAN:
        me = state.players[my_index]
        if me.hand is not None:
            has_basic = any(
                _card_db().get(c.id) is not None
                and _card_db()[c.id].cardType == CardType.POKEMON
                and _card_db()[c.id].basic
                for c in me.hand
            )
            if has_basic:
                return [_option_index_for_type(options, OptionType.NO)]
        return [_option_index_for_type(options, OptionType.YES)]

    if context == SelectContext.ACTIVATE:
        return [_option_index_for_type(options, OptionType.YES)]

    return [_option_index_for_type(options, OptionType.YES)]


def _option_index_for_type(options: list[Option], target_type: OptionType) -> int:
    for i, opt in enumerate(options):
        if opt.type == target_type:
            return i
    return 0


def _handle_main(obs: Observation, select: SelectData, state: State, my_index: int) -> list[int]:
    options = select.option
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    scored = []
    for i, opt in enumerate(options):
        score = _score_main_option(opt, state, my_index)
        scored.append((score, i, opt))

    scored.sort(key=lambda x: -x[0])

    return [scored[0][1]]


def _score_main_option(opt: Option, state: State, my_index: int) -> float:
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    card_db = _card_db()
    attack_db = get_attack_data()

    if opt.type == OptionType.ATTACK:
        return _score_attack_option(opt, state, my_index)

    if opt.type == OptionType.END:
        return 10.0

    if opt.type == OptionType.PLAY:
        cd = card_db.get(opt.cardId, None) if opt.cardId else None
        if cd is None:
            return 0.0
        return _score_play_card(cd, opt, state, my_index)

    if opt.type == OptionType.ATTACH:
        return _score_attach(opt, state, my_index)

    if opt.type == OptionType.EVOLVE:
        return _score_evolve(opt, state, my_index)

    if opt.type == OptionType.ABILITY:
        return 30.0

    if opt.type == OptionType.RETREAT:
        return _score_retreat(opt, state, my_index)

    if opt.type == OptionType.DISCARD:
        return 0.0

    return 0.0


def _score_attack_option(opt: Option, state: State, my_index: int) -> float:
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    if my_active is None or opp_active is None:
        return 0.0

    attack_db = get_attack_data()
    atk = attack_db.get(opt.attackId) if opt.attackId else None
    if atk is None:
        return 50.0

    score = 50.0 + atk.damage * 0.3

    if atk.damage >= opp_active.hp:
        score += 300.0
        opp_prizes_taken = 6 - len([p for p in opp.prize if p is not None])
        card_db = _card_db()
        opp_cd = card_db.get(opp_active.id)
        if opp_cd and opp_cd.ex:
            score += 200.0
        if opp_cd and opp_cd.megaEx:
            score += 300.0

    damage_ratio = atk.damage / opp_active.maxHp if opp_active.maxHp > 0 else 0
    if damage_ratio >= 1.0:
        score += 200.0
    elif damage_ratio >= 0.5:
        score += 50.0

    return score


def _score_play_card(cd: CardData, opt: Option, state: State, my_index: int) -> float:
    me = state.players[my_index]

    if cd.cardType == CardType.SUPPORTER:
        if not state.supporterPlayed:
            score = 25.0
            card_db = _card_db()
            if me.hand is not None:
                hand_size = len(me.hand)
                score += min(hand_size * 0.5, 10.0)
            return score
        return -10.0

    if cd.cardType == CardType.ITEM:
        return 20.0

    if cd.cardType == CardType.TOOL:
        return 15.0

    if cd.cardType == CardType.STADIUM:
        return 12.0

    if cd.cardType == CardType.POKEMON:
        return _score_play_pokemon(cd, opt, state, my_index)

    if cd.cardType == CardType.BASIC_ENERGY or cd.cardType == CardType.SPECIAL_ENERGY:
        if not state.energyAttached:
            return 18.0
        return -5.0

    return 5.0


def _score_play_pokemon(cd: CardData, opt: Option, state: State, my_index: int) -> float:
    me = state.players[my_index]
    my_active = me.active[0] if me.active else None
    score = 15.0

    if my_active is None and cd.basic:
        score += 50.0

    if cd.ex:
        score += 10.0
    if cd.megaEx:
        score -= 5.0
    if cd.stage2:
        score += 5.0
    elif cd.stage1:
        score += 3.0

    if len(me.bench) < me.benchMax:
        score += 5.0
    else:
        score -= 10.0

    return score


def _score_attach(opt: Option, state: State, my_index: int) -> float:
    me = state.players[my_index]
    if state.energyAttached:
        return -10.0

    my_active = me.active[0] if me.active else None
    if my_active is None:
        return 0.0

    score = 20.0

    card_db = _card_db()
    if opt.inPlayArea == AreaType.ACTIVE:
        score += 10.0
    elif opt.inPlayArea == AreaType.BENCH:
        score -= 5.0

    return score


def _score_evolve(opt: Option, state: State, my_index: int) -> float:
    score = 25.0
    card_db = _card_db()
    cd = card_db.get(opt.cardId) if opt.cardId else None
    if cd:
        if cd.ex:
            score += 10.0
        if cd.stage2:
            score += 15.0
        elif cd.stage1:
            score += 5.0
    return score


def _score_retreat(opt: Option, state: State, my_index: int) -> float:
    me = state.players[my_index]
    my_active = me.active[0] if me.active else None

    if my_active is None:
        return -20.0
    if state.retreated:
        return -20.0

    score = 0.0
    if my_active.maxHp > 0:
        hp_ratio = my_active.hp / my_active.maxHp
        if hp_ratio < 0.3:
            score += 40.0
        elif hp_ratio < 0.5:
            score += 20.0
        else:
            score -= 10.0

    if me.paralyzed or me.asleep or me.confused:
        score += 30.0

    if my_active.hp > 0 and my_active.hp <= 30:
        score += 50.0

    return score


def _handle_attack(obs: Observation, select: SelectData, state: State, my_index: int) -> list[int]:
    options = select.option
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    attack_db = get_attack_data()

    best_score = -1.0
    best_idx = 0
    for i, opt in enumerate(options):
        score = 0.0
        atk = attack_db.get(opt.attackId) if opt.attackId else None
        if atk:
            score += atk.damage * 0.5
            if opp_active:
                if atk.damage >= opp_active.hp:
                    score += 1000.0
                elif atk.damage >= opp_active.hp * 0.5:
                    score += 100.0
        else:
            score += 10.0

        if score > best_score:
            best_score = score
            best_idx = i

    return [best_idx]


def _handle_energy(obs: Observation, select: SelectData, state: State, my_index: int) -> list[int]:
    options = select.option
    context = select.context

    my_active = state.players[my_index].active[0] if state.players[my_index].active else None

    if context == SelectContext.DISCARD_ENERGY:
        if my_active and my_active.energies:
            return list(range(min(select.minCount, len(options))))

    energy_scores = []
    for i, opt in enumerate(options):
        score = 5.0
        if my_active:
            card_db = _card_db()
            active_cd = card_db.get(my_active.id)
            if active_cd:
                target_energy = EnergyType(opt.energyIndex) if opt.energyIndex is not None else None
        if opt.inPlayArea == AreaType.ACTIVE:
            score += 10.0
        energy_scores.append((score, i))

    if select.minCount > 0:
        energy_scores.sort(key=lambda x: -x[0])
        count = min(select.minCount, len(energy_scores))
        return [e[1] for e in energy_scores[:count]]

    return [energy_scores[0][1]]


def _handle_card_select(obs: Observation, select: SelectData, state: State, my_index: int) -> list[int]:
    context = select.context
    options = select.option
    min_count = select.minCount
    max_count = select.maxCount
    me = state.players[my_index]

    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        return _select_setup_pokemon(options, state, my_index, prefer_active=True)

    if context == SelectContext.SETUP_BENCH_POKEMON:
        return _select_setup_pokemon(options, state, my_index, prefer_active=False)

    if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        return _select_switch(options, state, my_index)

    if context == SelectContext.TO_BENCH:
        return _select_bench_pokemon(options, state, my_index)

    if context == SelectContext.DISCARD:
        return _select_discard(options, state, my_index)

    if context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE:
        return _select_damage_target(options, state, my_index)

    if context == SelectContext.HEAL or context == SelectContext.REMOVE_DAMAGE_COUNTER:
        return _select_heal_target(options, state, my_index)

    if context == SelectContext.EVOLVES_TO or context == SelectContext.EVOLVES_FROM:
        return _select_evolution(options, state, my_index)

    if context == SelectContext.ATTACH_TO or context == SelectContext.ATTACH_FROM:
        return _select_attach_target(options, state, my_index)

    count = max(min_count, 1)
    if count <= len(options):
        return list(range(count))
    return random.sample(range(len(options)), min(max_count, len(options)))


def _select_setup_pokemon(options: list[Option], state: State, my_index: int, prefer_active: bool) -> list[int]:
    card_db = _card_db()
    scored = []
    for i, opt in enumerate(options):
        cd = card_db.get(opt.cardId) if opt.cardId else None
        score = 0.0
        if cd:
            if prefer_active:
                if cd.basic and cd.hp and cd.hp > 50:
                    score += cd.hp * 0.5
                if cd.ex:
                    score += 20.0
                if cd.stage2 and not cd.basic:
                    score -= 30.0
            else:
                if cd.basic:
                    score += 10.0
                if cd.hp:
                    score += cd.hp * 0.3
                if cd.ex:
                    score += 15.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _select_switch(options: list[Option], state: State, my_index: int) -> list[int]:
    me = state.players[my_index]
    card_db = _card_db()

    scored = []
    for i, opt in enumerate(options):
        score = 0.0
        if opt.area == AreaType.BENCH:
            for mon in me.bench:
                if mon.id == opt.cardId:
                    score += mon.hp * 0.3
                    cd = card_db.get(mon.id)
                    if cd:
                        if cd.ex:
                            score += 10.0
                    if mon.hp == mon.maxHp:
                        score += 20.0

                    has_usable_attack = False
                    if cd and cd.attacks:
                        attack_db = get_attack_data()
                        for atk_id in cd.attacks:
                            atk = attack_db.get(atk_id)
                            if atk and _can_use_attack(mon, atk):
                                has_usable_attack = True
                                score += atk.damage * 0.2
                                break
                    if has_usable_attack:
                        score += 30.0

                    break
        elif opt.type == OptionType.YES:
            score -= 50.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _select_bench_pokemon(options: list[Option], state: State, my_index: int) -> list[int]:
    card_db = _card_db()
    scored = []
    for i, opt in enumerate(options):
        cd = card_db.get(opt.cardId) if opt.cardId else None
        score = 0.0
        if cd:
            if cd.basic:
                score += 10.0
            if cd.hp:
                score += cd.hp * 0.2
            if cd.ex:
                score += 5.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _select_discard(options: list[Option], state: State, my_index: int) -> list[int]:
    card_db = _card_db()
    scored = []
    for i, opt in enumerate(options):
        cd = card_db.get(opt.cardId) if opt.cardId else None
        score = 0.0
        if cd:
            if cd.cardType == CardType.BASIC_ENERGY:
                score += 5.0
            elif cd.cardType == CardType.POKEMON and cd.basic:
                score -= 20.0
            elif cd.cardType == CardType.SUPPORTER:
                score += 10.0
            elif cd.cardType == CardType.ITEM:
                score += 8.0
            elif cd.cardType == CardType.TOOL:
                score += 8.0
        scored.append((score, i))

    scored.sort(key=lambda x: x[0])
    min_count = 1
    return [scored[0][1]][:min_count]


def _select_damage_target(options: list[Option], state: State, my_index: int) -> list[int]:
    opp = state.players[1 - my_index]
    scored = []
    for i, opt in enumerate(options):
        score = 0.0
        if opt.playerIndex == 1 - my_index:
            if opt.area == AreaType.ACTIVE:
                opp_active = opp.active[0] if opp.active else None
                if opp_active:
                    score += (1.0 - opp_active.hp / opp_active.maxHp) * 50.0 if opp_active.maxHp > 0 else 0
                    card_db = _card_db()
                    cd = card_db.get(opp_active.id)
                    if cd and cd.ex:
                        score += 20.0
                    if cd and cd.megaEx:
                        score += 30.0
            elif opt.area == AreaType.BENCH:
                score += 5.0
        else:
            score -= 100.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _select_heal_target(options: list[Option], state: State, my_index: int) -> list[int]:
    me = state.players[my_index]
    scored = []
    for i, opt in enumerate(options):
        score = 0.0
        if opt.playerIndex == my_index:
            if opt.area == AreaType.ACTIVE and me.active and me.active[0]:
                active = me.active[0]
                if active.maxHp > 0:
                    score += (1.0 - active.hp / active.maxHp) * 50.0
            elif opt.area == AreaType.BENCH:
                for mon in me.bench:
                    if mon.id == opt.cardId and mon.maxHp > 0:
                        score += (1.0 - mon.hp / mon.maxHp) * 30.0
                        break
        else:
            score -= 100.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _select_evolution(options: list[Option], state: State, my_index: int) -> list[int]:
    card_db = _card_db()
    scored = []
    for i, opt in enumerate(options):
        cd = card_db.get(opt.cardId) if opt.cardId else None
        score = 0.0
        if cd:
            if cd.stage2:
                score += 30.0
            elif cd.stage1:
                score += 15.0
            if cd.hp:
                score += cd.hp * 0.3
            if cd.ex:
                score += 10.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _select_attach_target(options: list[Option], state: State, my_index: int) -> list[int]:
    me = state.players[my_index]
    my_active = me.active[0] if me.active else None

    scored = []
    for i, opt in enumerate(options):
        score = 0.0
        if opt.inPlayArea == AreaType.ACTIVE or (opt.area == AreaType.ACTIVE):
            score += 20.0
            if my_active:
                card_db = _card_db()
                cd = card_db.get(my_active.id)
                if cd and cd.attacks:
                    score += 10.0
        elif opt.inPlayArea == AreaType.BENCH or (opt.area == AreaType.BENCH):
            score += 5.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _handle_count(obs: Observation, select: SelectData, state: State, my_index: int) -> list[int]:
    context = select.context
    options = select.option

    if context == SelectContext.DRAW_COUNT:
        max_draw = max(opt.number for opt in options if opt.number is not None) if options else 0
        for i, opt in enumerate(options):
            if opt.number == max_draw:
                return [i]
        return [0]

    if context == SelectContext.DAMAGE_COUNTER_COUNT:
        opp = state.players[1 - my_index]
        opp_active = opp.active[0] if opp.active else None
        if opp_active and opp_active.maxHp > 0:
            remaining_hp = opp_active.hp
            best_damage = 0
            best_idx = 0
            for i, opt in enumerate(options):
                if opt.number is not None:
                    damage = opt.number * 10
                    if damage >= remaining_hp:
                        return [i]
                    if damage > best_damage:
                        best_damage = damage
                        best_idx = i
            return [best_idx]
        return [0]

    return [0]


def _handle_special_condition(obs: Observation, select: SelectData, state: State, my_index: int) -> list[int]:
    context = select.context
    options = select.option

    me = state.players[my_index]
    opp = state.players[1 - my_index]

    if context == SelectContext.AFFECT_SPECIAL_CONDITION:
        for i, opt in enumerate(options):
            if opt.specialConditionType == SpecialConditionType.PARALYZE:
                return [i]
            if opt.specialConditionType == SpecialConditionType.SLEEP:
                return [i]

    if context == SelectContext.RECOVER_SPECIAL_CONDITION:
        for i, opt in enumerate(options):
            if opt.specialConditionType == SpecialConditionType.PARALYZE and me.paralyzed:
                return [i]
            if opt.specialConditionType == SpecialConditionType.POISON and me.poisoned:
                return [i]
            if opt.specialConditionType == SpecialConditionType.BURN and me.burned:
                return [i]
            if opt.specialConditionType == SpecialConditionType.CONFUSE and me.confused:
                return [i]
            if opt.specialConditionType == SpecialConditionType.SLEEP and me.asleep:
                return [i]

    return [0]