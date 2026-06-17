import os
import json
import random
import time

from cg.api import Observation, to_observation_class, SelectType, SelectContext
from agent.policy import choose_action
from agent.deck import load_deck_csv, get_default_deck, save_deck_csv
from agent.search import (
    mcts_search,
    predict_opponent_deck,
    predict_opponent_hand,
    predict_opponent_prize,
    predict_opponent_active,
)
from agent.evaluate import get_card_data

_deck_cache: list[int] | None = None
_use_search = True
_search_simulations = 30
_search_time_ms = 3000.0


def read_deck() -> list[int]:
    global _deck_cache
    if _deck_cache is not None:
        return _deck_cache

    deck = load_deck_csv()
    if len(deck) != 60:
        deck = get_default_deck()

    _deck_cache = deck
    return deck


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)

    if obs.select is None:
        deck = read_deck()
        return deck

    current = obs.current
    if current is None:
        return choose_action(obs)

    select = obs.select
    options = select.option
    min_count = select.minCount
    max_count = select.maxCount

    if len(options) <= 1 and max_count <= 1:
        if len(options) == 1:
            return [0]
        if max_count == 0:
            return []

    if select.type == SelectType.YES_NO:
        return choose_action(obs)

    if select.type == SelectType.COUNT:
        return choose_action(obs)

    if select.type == SelectType.SPECIAL_CONDITION:
        return choose_action(obs)

    select_context = select.context

    if _use_search and select.type == SelectType.MAIN and select_context == SelectContext.MAIN:
        action = _try_search(obs)
        if action is not None:
            return action

    if select.type == SelectType.MAIN:
        return choose_action(obs)

    if select.type == SelectType.ATTACK:
        return choose_action(obs)

    if select.type == SelectType.ENERGY:
        return choose_action(obs)

    if select.type in (SelectType.CARD, SelectType.ATTACHED_CARD, SelectType.CARD_OR_ATTACHED_CARD):
        return choose_action(obs)

    if select.type == SelectType.EVOLVE:
        return choose_action(obs)

    if select.type == SelectType.SKILL:
        return choose_action(obs)

    if min_count == max_count and min_count > 0:
        if min_count <= len(options):
            return list(range(min_count))

    if max_count > 0 and len(options) > 0:
        count = max(min_count, 1)
        return random.sample(range(len(options)), min(count, len(options)))

    return []


def _try_search(obs: Observation) -> list[int] | None:
    try:
        current = obs.current
        my_index = current.yourIndex
        me = current.players[my_index]
        opp = current.players[1 - my_index]

        all_card_ids = list(get_card_data().keys())

        my_prize = [p.id for p in me.prize if p is not None]
        my_deck_pred = [] if obs.select and obs.select.deck else [3] * me.deckCount

        opp_deck_pred = predict_opponent_deck(all_card_ids, opp)
        opp_prize_pred = predict_opponent_prize(opp)
        opp_hand_pred = predict_opponent_hand(opp)
        opp_active_pred = predict_opponent_active(opp)

        result = mcts_search(
            obs,
            my_deck=my_deck_pred,
            my_prize=my_prize,
            opp_deck=opp_deck_pred,
            opp_prize=opp_prize_pred,
            opp_hand=opp_hand_pred,
            opp_active=opp_active_pred,
            max_simulations=_search_simulations,
            max_time_ms=_search_time_ms,
        )
        return result
    except Exception:
        return None


if __name__ == "__main__":
    from cg.api import Observation, SelectData, Option, OptionType, State, PlayerState
    from cg.game import battle_start, battle_select, battle_finish

    deck0 = read_deck()
    deck1 = read_deck()

    obs0, start_data = battle_start(deck0, deck1)
    if obs0 is None:
        if start_data.errorPlayer == 0:
            print("Deck 0 is invalid")
        else:
            print("Deck 1 is invalid")
    else:
        print("Battle started successfully")

    battle_finish()
    print("Agent module loaded successfully")