from __future__ import annotations

import random
from cg.api import (
    Observation,
    State,
    PlayerState,
    Option,
    SelectData,
    Card,
    OptionType,
    SelectType,
    SelectContext,
    EnergyType,
    CardType,
    AreaType,
    Attack,
    CardData,
    to_observation_class,
)
from cg.api import all_card_data, all_attack


_card_data_cache: dict[int, CardData] | None = None
_attack_data_cache: dict[int, Attack] | None = None


def _get_card_db() -> dict[int, CardData]:
    global _card_data_cache
    if _card_data_cache is None:
        _card_data_cache = {c.cardId: c for c in all_card_data()}
    return _card_data_cache


def _get_attack_db() -> dict[int, Attack]:
    global _attack_data_cache
    if _attack_data_cache is None:
        _attack_data_cache = {a.attackId: a for a in all_attack()}
    return _attack_data_cache


def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        from agent.deck import load_deck_csv
        return load_deck_csv()

    options = obs.select.option
    min_count = obs.select.minCount
    max_count = obs.select.maxCount

    if max_count == 0:
        return []
    if len(options) == 1:
        return [0]
    if min_count == max_count and min_count > 0:
        return list(range(min_count))
    return random.sample(range(len(options)), min(max_count, len(options)))


def slightly_smart_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        from agent.deck import load_deck_csv
        return load_deck_csv()

    if obs.current is None:
        return random_agent(obs_dict)

    state = obs.current
    my_index = state.yourIndex
    select = obs.select
    options = select.option
    min_count = select.minCount
    max_count = select.maxCount

    if max_count == 0:
        return []
    if len(options) == 1:
        return [0]

    if select.type == SelectType.YES_NO:
        if select.context == SelectContext.IS_FIRST:
            return [_find_option_type(options, OptionType.YES)]
        return [_find_option_type(options, OptionType.YES)]

    if select.type == SelectType.ATTACK:
        return _pick_best_attack(options, state, my_index)

    if select.type == SelectType.MAIN:
        return _pick_main_action(options, state, my_index)

    if min_count == max_count and min_count > 0:
        return list(range(min_count))

    return random.sample(range(len(options)), min(max_count, len(options)))


def _find_option_type(options: list[Option], target: OptionType) -> int:
    for i, opt in enumerate(options):
        if opt.type == target:
            return i
    return 0


def _pick_best_attack(options: list[Option], state: State, my_index: int) -> list[int]:
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp else None

    attack_db = _get_attack_db()

    best_idx = 0
    best_dmg = -1
    for i, opt in enumerate(options):
        atk = attack_db.get(opt.attackId) if opt.attackId else None
        dmg = atk.damage if atk else 0
        if opp_active and dmg >= opp_active.hp:
            return [i]
        if dmg > best_dmg:
            best_dmg = dmg
            best_idx = i

    return [best_idx]


def _pick_main_action(options: list[Option], state: State, my_index: int) -> list[int]:
    me = state.players[my_index]
    card_db = _get_card_db()

    attack_idx = None
    attack_dmg = 0
    supporter_idx = None
    item_idx = None
    energy_idx = None
    evolve_idx = None
    end_idx = None

    attack_db = _get_attack_db()
    my_active = me.active[0] if me.active else None

    for i, opt in enumerate(options):
        if opt.type == OptionType.ATTACK:
            atk = attack_db.get(opt.attackId) if opt.attackId else None
            dmg = atk.damage if atk else 0
            if dmg > attack_dmg:
                attack_dmg = dmg
                attack_idx = i
        elif opt.type == OptionType.PLAY:
            cd = card_db.get(opt.cardId) if opt.cardId else None
            if cd:
                if cd.cardType == CardType.SUPPORTER and not state.supporterPlayed:
                    supporter_idx = i
                elif cd.cardType == CardType.ITEM:
                    item_idx = i
                elif cd.cardType == CardType.POKEMON and cd.basic:
                    if my_active is None:
                        return [i]
        elif opt.type == OptionType.ATTACH and not state.energyAttached:
            energy_idx = i
        elif opt.type == OptionType.EVOLVE:
            evolve_idx = i
        elif opt.type == OptionType.ABILITY:
            return [i]
        elif opt.type == OptionType.END:
            end_idx = i

    opp = state.players[1 - my_index]
    opp_active = opp.active[0] if opp else None
    if attack_idx is not None and opp_active and my_active:
        atk = attack_db.get(options[attack_idx].attackId) if options[attack_idx].attackId else None
        if atk and opp_active.hp > 0 and atk.damage >= opp_active.hp:
            return [attack_idx]

    if supporter_idx is not None:
        return [supporter_idx]
    if item_idx is not None:
        return [item_idx]
    if evolve_idx is not None:
        return [evolve_idx]
    if energy_idx is not None:
        return [energy_idx]
    if attack_idx is not None:
        return [attack_idx]
    if end_idx is not None:
        return [end_idx]

    return [0]


def get_opponent_fn(difficulty: float):
    return slightly_smart_agent


def load_competitive_decks() -> list[dict]:
    import json
    from pathlib import Path
    p = Path(__file__).parent.parent / "competitive_decks.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


DECK_TIERS = {
    "easy": [
        (794, 4),
        (1123, 4),
        (1086, 4),
        (1121, 4),
        (1119, 2),
        (2, 42),
    ],
    "medium": [
        (46, 4), (788, 4), (790, 2), (1079, 4), (1121, 4),
        (1182, 4), (1123, 4), (1086, 4), (2, 30),
    ],
    "hard": [
        (46, 4), (788, 4), (790, 2), (1079, 4), (1121, 4),
        (1182, 4), (1123, 4), (1086, 4), (1232, 2),
        (1119, 2), (1097, 2), (1227, 2), (2, 22),
    ],
    "expert": None,
}


def get_deck_for_tier(tier: str) -> list[int]:
    from agent.deck import load_deck_csv

    spec = DECK_TIERS.get(tier)
    if spec is None:
        return load_deck_csv()

    deck = []
    for card_id, count in spec:
        deck.extend([card_id] * count)

    while len(deck) < 60:
        deck.append(3)

    return deck[:60]


def get_tier_deck(level: int) -> list[int]:
    if level <= 0:
        return get_deck_for_tier("easy")
    elif level == 1:
        return get_deck_for_tier("medium")
    elif level == 2:
        return get_deck_for_tier("hard")
    else:
        return get_deck_for_tier("expert")