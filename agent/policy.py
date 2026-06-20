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
from agent.evaluate import get_card_data, get_attack_data, evaluate_state, _can_use_attack, _has_ex_immunity, _effective_damage
from agent.features import encode_observation
from agent.strategies import StrategyProfile, strategy_action_bias

_card_db_cache: dict[int, CardData] | None = None
# Per-player policy net (MULTI_DECK_ENSEMBLE_PLAN.md Phase B: two different
# specialist checkpoints play each other in the same process, so the model
# must be indexed by player like _my_strategy already is).
_policy_model: list = [None, None]
_policy_device: list = ["cpu", "cpu"]
_policy_enabled: list = [False, False]
_policy_blend = 0.7
_mcts_enabled = True
_mcts_max_time_ms = 1500.0
_mcts_min_options = 4

# Strategy-guided nudging (MEGA_LUCARIO_STRATEGY_PLAN.md, generalized for
# multiple decks in MULTI_DECK_ENSEMBLE_PLAN.md). Indexed per-player so two
# different strategy-bearing decks can play each other with each side getting
# its own bias — gated per-slot so a deck with no profile (index is None)
# behaves exactly as before.
_my_strategy: list[StrategyProfile | None] = [None, None]
STRATEGY_BIAS_WEIGHT = 1.0


def set_my_strategy(profile: StrategyProfile | None, index: int | None = None) -> None:
    global _my_strategy
    if index is None:
        _my_strategy = [profile, profile]
    else:
        _my_strategy[index] = profile


def get_my_strategy(my_index: int) -> StrategyProfile | None:
    return _my_strategy[my_index]


def _strategy_bias(opt: Option, state: State, my_index: int) -> float:
    profile = _my_strategy[my_index]
    if profile is None:
        return 0.0
    return STRATEGY_BIAS_WEIGHT * strategy_action_bias(opt, state, my_index, profile)


def load_policy_model(path: str | None = None, device: str = "cpu", index: int | None = None):
    if index is None:
        load_policy_model(path, device, index=0)
        load_policy_model(path, device, index=1)
        return

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
        _policy_enabled[index] = False
        return

    from agent.network import load_model
    try:
        _policy_model[index] = load_model(path, device=device)
        _policy_device[index] = device
        _policy_enabled[index] = True
    except Exception:
        _policy_enabled[index] = False


def _card_db() -> dict[int, CardData]:
    return get_card_data()


def _resolve_card_id(state: State, opt: Option, my_index: int) -> int | None:
    """For SelectType.CARD options the engine never populates Option.cardId —
    verified empirically across SETUP_ACTIVE_POKEMON, SETUP_BENCH_POKEMON,
    SWITCH, TO_ACTIVE, and DISCARD contexts. Per the OptionType.CARD field
    spec (area/index/playerIndex), the real card lives at
    state.players[opt.playerIndex].<area>[opt.index] instead."""
    if opt.index is None:
        return None
    owner = opt.playerIndex if opt.playerIndex is not None else my_index
    if owner < 0 or owner >= len(state.players):
        return None
    player = state.players[owner]
    # OptionType.PLAY at the MAIN select level has no area field at all (the
    # spec just says "index within the hand") — area=None means hand there.
    if opt.area is None or opt.area == AreaType.HAND:
        hand = player.hand
        if hand and opt.index < len(hand):
            return hand[opt.index].id
    elif opt.area == AreaType.ACTIVE:
        mon = player.active[0] if player.active else None
        return mon.id if mon else None
    elif opt.area == AreaType.BENCH:
        if player.bench and opt.index < len(player.bench):
            return player.bench[opt.index].id
    elif opt.area == AreaType.DISCARD:
        if player.discard and opt.index < len(player.discard):
            return player.discard[opt.index].id
    elif opt.area == AreaType.PRIZE:
        if opt.index < len(player.prize):
            card = player.prize[opt.index]
            return card.id if card else None
    return None


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

    if _mcts_enabled and _should_use_mcts(obs, select_type, options, state, my_index):
        mcts_action = _try_mcts(obs)
        if mcts_action is not None:
            return mcts_action

    heuristic_action = _choose_action_heuristic(obs, select, state, my_index)

    action = heuristic_action
    if use_model and _policy_enabled[my_index] and _policy_model[my_index] is not None and select_type in (SelectType.MAIN, SelectType.ATTACK, SelectType.CARD):
        model_action = _choose_action_model(obs, select, heuristic_action, my_index)
        if model_action is not None:
            action = model_action

    if min_count > 0 and len(action) < min_count:
        action = list(range(min(min_count, len(options))))
    if max_count > 0 and len(action) > max_count:
        action = action[:max_count]

    return action


def _choose_action_model(obs: Observation, select: SelectData, heuristic_action: list[int], my_index: int) -> list[int] | None:
    try:
        state_features, option_features = encode_observation(obs)
        if not option_features or not state_features:
            return None

        from agent.network import score_actions
        scores, value = score_actions(
            _policy_model[my_index],
            state_features,
            option_features,
            device=_policy_device[my_index],
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


def _should_use_mcts(obs: Observation, select_type: int, options, state: State, my_index: int) -> bool:
    if obs.search_begin_input is None:
        return False
    if len(options) < _mcts_min_options:
        return False
    if select_type == SelectType.ATTACK:
        return True
    if select_type == SelectType.MAIN:
        me = state.players[my_index]
        my_active = me.active[0] if me.active else None
        opp_active = state.players[1 - my_index].active[0] if state.players[1 - my_index].active else None
        has_attack = any(o.type == OptionType.ATTACK for o in options)
        has_playable = any(o.type == OptionType.PLAY for o in options)
        turn_late = state.turn >= 2
        if has_attack and my_active and opp_active:
            return True
        if has_playable and turn_late:
            return True
    return False


def _try_mcts(obs: Observation) -> list[int] | None:
    try:
        from agent.search import mcts_search, build_search_inputs
        inputs = build_search_inputs(obs)
        if not inputs:
            return None
        return mcts_search(obs, max_simulations=40, max_time_ms=_mcts_max_time_ms, **inputs)
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

    if not scored:
        return [0]

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
        cid = _resolve_card_id(state, opt, my_index)
        cd = card_db.get(cid) if cid else None
        base = 0.0 if cd is None else _score_play_card(cd, opt, state, my_index)
        return base + _strategy_bias(opt, state, my_index)

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

    raw_damage = atk.damage
    effective_damage = _effective_damage(my_active, opp_active, raw_damage)

    if effective_damage <= 0 and raw_damage > 0:
        return -200.0

    score = 50.0 + effective_damage * 0.3

    if effective_damage >= opp_active.hp:
        score += 300.0
        card_db = _card_db()
        opp_cd = card_db.get(opp_active.id)
        if opp_cd and opp_cd.ex:
            score += 200.0
        if opp_cd and opp_cd.megaEx:
            score += 300.0

    damage_ratio = effective_damage / opp_active.maxHp if opp_active.maxHp > 0 else 0
    if damage_ratio >= 1.0:
        score += 200.0
    elif damage_ratio >= 0.5:
        score += 50.0

    score += _strategy_bias(opt, state, my_index)

    return score


def _score_play_card(cd: CardData, opt: Option, state: State, my_index: int) -> float:
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    if cd.cardType == CardType.SUPPORTER:
        if not state.supporterPlayed:
            score = 25.0
            card_db = _card_db()
            if me.hand is not None:
                hand_size = len(me.hand)
                score += min(hand_size * 0.5, 10.0)

            if cd.name and ("boss" in cd.name.lower() or "orders" in cd.name.lower()):
                if opp_active and _has_ex_immunity(opp_active):
                    if my_active:
                        my_cd = card_db.get(my_active.id)
                        if my_cd and (my_cd.ex or my_cd.megaEx):
                            if len(opp.bench) > 0:
                                score += 200.0

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

    score += _strategy_bias(opt, state, my_index)

    return score


def _score_evolve(opt: Option, state: State, my_index: int) -> float:
    score = 25.0
    card_db = _card_db()
    cid = _resolve_card_id(state, opt, my_index)
    cd = card_db.get(cid) if cid else None
    if cd:
        if cd.ex:
            score += 10.0
        if cd.stage2:
            score += 15.0
        elif cd.stage1:
            score += 5.0

    score += _strategy_bias(opt, state, my_index)

    return score


def _score_retreat(opt: Option, state: State, my_index: int) -> float:
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    if my_active is None:
        return -20.0
    if state.retreated:
        return -20.0

    score = 0.0
    card_db = _card_db()

    if opp_active and _has_ex_immunity(opp_active):
        my_cd = card_db.get(my_active.id)
        if my_cd and (my_cd.ex or my_cd.megaEx):
            score += 150.0

            has_non_ex_bench = False
            for mon in me.bench:
                bcd = card_db.get(mon.id)
                if bcd and not bcd.ex and not bcd.megaEx and bcd.attacks:
                    has_non_ex_bench = True
                    break
            if not has_non_ex_bench:
                score -= 50.0

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

    score += _strategy_bias(opt, state, my_index)

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
            raw_damage = atk.damage
            if my_active and opp_active:
                effective_damage = _effective_damage(my_active, opp_active, raw_damage)
                if effective_damage <= 0 and raw_damage > 0:
                    score -= 200.0
                else:
                    score += effective_damage * 0.5
                    if effective_damage >= opp_active.hp:
                        score += 1000.0
                    elif effective_damage >= opp_active.hp * 0.5:
                        score += 100.0
            else:
                score += raw_damage * 0.5
        else:
            score += 10.0

        score += _strategy_bias(opt, state, my_index)

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
        return _select_discard(options, state, my_index, min_count=min_count)

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
        cid = _resolve_card_id(state, opt, my_index)
        cd = card_db.get(cid) if cid else None
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
    opp = state.players[1 - my_index]
    opp_active = opp.active[0] if opp.active else None
    card_db = _card_db()

    facing_wall = opp_active is not None and _has_ex_immunity(opp_active)
    opp_active_cd = card_db.get(opp_active.id) if opp_active else None
    opp_active_is_ex = opp_active_cd is not None and (opp_active_cd.ex or opp_active_cd.megaEx)
    profile = _my_strategy[my_index]

    scored = []
    for i, opt in enumerate(options):
        score = 0.0
        if opt.area == AreaType.BENCH and opt.index is not None and opt.index < len(me.bench):
            mon = me.bench[opt.index]
            bcd = card_db.get(mon.id)
            if bcd:
                if facing_wall and not bcd.ex and not bcd.megaEx and bcd.attacks:
                    score += 100.0
                if bcd.ex or bcd.megaEx:
                    score += 10.0
                if mon.hp == mon.maxHp:
                    score += 20.0
                if profile is not None and mon.id in profile.prefer_passive_wall_ids and opp_active_is_ex:
                    score += 130.0

            has_usable_attack = False
            if bcd and bcd.attacks:
                attack_db = get_attack_data()
                for atk_id in bcd.attacks:
                    atk = attack_db.get(atk_id)
                    if atk and _can_use_attack(mon, atk):
                        has_usable_attack = True
                        score += atk.damage * 0.2
                        break
            if has_usable_attack:
                score += 30.0

            score += mon.hp * 0.3
        elif opt.type == OptionType.YES:
            score -= 50.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _select_bench_pokemon(options: list[Option], state: State, my_index: int) -> list[int]:
    card_db = _card_db()
    scored = []
    for i, opt in enumerate(options):
        cid = _resolve_card_id(state, opt, my_index)
        cd = card_db.get(cid) if cid else None
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


def _select_discard(options: list[Option], state: State, my_index: int, min_count: int = 1) -> list[int]:
    card_db = _card_db()
    scored = []
    for i, opt in enumerate(options):
        cid = _resolve_card_id(state, opt, my_index)
        cd = card_db.get(cid) if cid else None
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
    return [s[1] for s in scored[:min(min_count, len(scored))]]


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
            elif opt.area == AreaType.BENCH and opt.index is not None and opt.index < len(me.bench):
                mon = me.bench[opt.index]
                if mon.maxHp > 0:
                    score += (1.0 - mon.hp / mon.maxHp) * 30.0
        else:
            score -= 100.0
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    return [scored[0][1]]


def _select_evolution(options: list[Option], state: State, my_index: int) -> list[int]:
    card_db = _card_db()
    scored = []
    for i, opt in enumerate(options):
        cid = _resolve_card_id(state, opt, my_index)
        cd = card_db.get(cid) if cid else None
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