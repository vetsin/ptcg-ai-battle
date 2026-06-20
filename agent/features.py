from __future__ import annotations

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
    AreaType,
    SpecialConditionType,
    to_observation_class,
)
from agent.evaluate import get_card_data, get_attack_data
from agent.opponent_model import get_opponent_model, DECK_ARCHETYPES

NUM_ENERGY_TYPES = 12
NUM_CARD_TYPES = 7
NUM_SELECT_TYPES = 11
NUM_OPTION_TYPES = 17
NUM_ARCHETYPES = len(DECK_ARCHETYPES)
NUM_STATE_FEATURES = 120 + NUM_ARCHETYPES
NUM_OPTION_FEATURES = 40


def encode_observation(obs: Observation) -> tuple[list[float], list[list[float]]]:
    if obs.current is None or obs.select is None:
        return _zero_state(), [_zero_option() for _ in range(max(len(obs.select.option), 1))] if obs.select else [_zero_option()]

    state_features = encode_state(obs.current)
    option_features = []
    for opt in obs.select.option:
        option_features.append(encode_option(opt, obs))
    return state_features, option_features


def encode_state(state: State) -> list[float]:
    features = [0.0] * NUM_STATE_FEATURES
    my_index = state.yourIndex
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    card_db = get_card_data()

    features[0] = float(state.turn) / 50.0
    features[1] = float(state.turnActionCount) / 10.0
    features[2] = float(state.yourIndex)
    features[3] = float(state.firstPlayer == my_index)
    features[4] = float(state.supporterPlayed)
    features[5] = float(state.stadiumPlayed)
    features[6] = float(state.energyAttached)
    features[7] = float(state.retreated)

    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    if my_active:
        features[8] = 1.0
        _encode_pokemon_features(my_active, features, 9, card_db)
    else:
        features[8] = 0.0

    has_usable_attack = 0.0
    if my_active and my_active.id in card_db:
        cd = card_db[my_active.id]
        if cd.attacks:
            from agent.evaluate import _can_use_attack
            attack_db = get_attack_data()
            for atk_id in cd.attacks:
                atk = attack_db.get(atk_id)
                if atk and _can_use_attack(my_active, atk):
                    has_usable_attack = 1.0
                    break
    features[27] = has_usable_attack

    if opp_active:
        features[28] = 1.0
        _encode_pokemon_features(opp_active, features, 29, card_db)
    else:
        features[28] = 0.0

    features[48] = float(len(me.bench)) / 5.0
    for i, mon in enumerate(me.bench[:5]):
        _encode_pokemon_features(mon, features, 49 + i * 9, card_db)

    features[94] = float(len(opp.bench)) / 5.0
    for i, mon in enumerate(opp.bench[:5]):
        features[95 + i * 1] = float(mon.hp) / 300.0 if mon.maxHp > 0 else 0.0

    my_prize_remaining = len(me.prize)
    opp_prize_remaining = len(opp.prize)
    features[100] = float(my_prize_remaining) / 6.0
    features[101] = float(6 - my_prize_remaining) / 6.0
    features[102] = float(opp_prize_remaining) / 6.0
    features[103] = float(6 - opp_prize_remaining) / 6.0

    features[104] = float(me.deckCount) / 60.0
    features[105] = float(me.handCount) / 30.0
    features[106] = float(opp.deckCount) / 60.0
    features[107] = float(opp.handCount) / 30.0

    features[108] = float(me.poisoned)
    features[109] = float(me.burned)
    features[110] = float(me.asleep)
    features[111] = float(me.paralyzed)
    features[112] = float(me.confused)
    features[113] = float(opp.poisoned)
    features[114] = float(opp.burned)
    features[115] = float(opp.asleep)
    features[116] = float(opp.paralyzed)
    features[117] = float(opp.confused)

    features[118] = float(len(me.discard)) / 60.0
    features[119] = float(len(opp.discard)) / 60.0

    opp_model = get_opponent_model(1 - my_index)
    opp_model.update(opp)
    archetype_probs = opp_model.get_archetype_probs()
    for i, name in enumerate(DECK_ARCHETYPES):
        features[120 + i] = archetype_probs.get(name, 0.0)

    return features


def _encode_pokemon_features(mon: Pokemon, features: list[float], offset: int, card_db: dict) -> None:
    if offset + 18 > len(features):
        return

    cd = card_db.get(mon.id)
    features[offset + 0] = float(mon.hp) / 400.0 if mon.maxHp > 0 else 0.0
    features[offset + 1] = float(mon.hp) / float(mon.maxHp) if mon.maxHp > 0 else 0.0
    features[offset + 2] = float(mon.maxHp) / 400.0
    features[offset + 3] = float(len(mon.energies)) / 5.0
    features[offset + 4] = float(mon.appearThisTurn)

    for i in range(NUM_ENERGY_TYPES):
        features[offset + 5 + i] = 0.0
    for e in mon.energies:
        e_int = int(e)
        if 0 <= e_int < NUM_ENERGY_TYPES:
            features[offset + 5 + e_int] += 1.0
    for i in range(NUM_ENERGY_TYPES):
        features[offset + 5 + i] = min(features[offset + 5 + i], 3.0) / 3.0

    if cd:
        features[offset + 17] = float(cd.ex)
        features[offset + 16] = float(cd.megaEx)
        features[offset + 15] = float(cd.basic)
        features[offset + 14] = float(cd.stage1)
        features[offset + 13] = float(cd.stage2)


def encode_option(opt: Option, obs: Observation) -> list[float]:
    features = [0.0] * NUM_OPTION_FEATURES

    for i, t in enumerate([OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE,
                           OptionType.ABILITY, OptionType.DISCARD, OptionType.RETREAT,
                           OptionType.ATTACK, OptionType.END]):
        features[i] = float(opt.type == t)

    card_db = get_card_data()
    attack_db = get_attack_data()

    if opt.cardId and opt.cardId in card_db:
        cd = card_db[opt.cardId]
        for i, ct in enumerate([CardType.POKEMON, CardType.ITEM, CardType.TOOL,
                                CardType.SUPPORTER, CardType.STADIUM,
                                CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY]):
            features[8 + i] = float(cd.cardType == ct)
        features[15] = float(cd.hp) / 400.0 if cd.hp else 0.0
        features[16] = float(cd.ex)
        features[17] = float(cd.megaEx)
        features[18] = float(cd.basic)
        features[19] = float(cd.stage1)
        features[20] = float(cd.stage2)
        features[21] = float(cd.retreatCost) / 4.0
        features[22] = float(len(cd.attacks)) / 4.0 if cd.attacks else 0.0
        features[23] = float(cd.tera)
        features[24] = float(cd.aceSpec)

    if opt.attackId and opt.attackId in attack_db:
        atk = attack_db[opt.attackId]
        features[25] = float(atk.damage) / 400.0
        features[26] = float(len(atk.energies)) / 5.0

    if opt.area is not None:
        for i, a in enumerate([AreaType.HAND, AreaType.BENCH, AreaType.ACTIVE,
                               AreaType.DISCARD, AreaType.PRIZE, AreaType.DECK]):
            features[27 + i] = float(opt.area == a)

    if opt.inPlayArea is not None:
        for i, a in enumerate([AreaType.ACTIVE, AreaType.BENCH]):
            features[33 + i] = float(opt.inPlayArea == a)

    features[35] = float(opt.number or 0) / 10.0 if opt.number is not None else 0.0
    features[36] = float(opt.playerIndex) if opt.playerIndex is not None else 0.0

    if opt.specialConditionType is not None:
        for i, s in enumerate([SpecialConditionType.POISON, SpecialConditionType.BURN,
                               SpecialConditionType.SLEEP, SpecialConditionType.PARALYZE,
                               SpecialConditionType.CONFUSE]):
            features[37 + i] = float(int(opt.specialConditionType) == int(s))

    return features


def _zero_state() -> list[float]:
    return [0.0] * NUM_STATE_FEATURES


def _zero_option() -> list[float]:
    return [0.0] * NUM_OPTION_FEATURES


def encode_training_step(step_dict: dict) -> tuple[list[float], list[list[float]], list[int]]:
    obs_dict = step_dict.get("obs_dict")
    if obs_dict is None:
        return _zero_state(), [_zero_option()], []

    obs = to_observation_class(obs_dict)
    state_features, option_features = encode_observation(obs)
    action = step_dict.get("action", [])

    action_indices = action if isinstance(action, list) else [action]

    return state_features, option_features, action_indices