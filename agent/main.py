import os
import random

from cg.api import Observation, to_observation_class, SelectType, SelectContext
from agent.policy import choose_action, set_my_strategy
from agent.deck import load_deck_csv, get_default_deck, save_deck_csv
from agent.opponent_model import reset_opponent_models
from agent.strategies import STRATEGIES, classify_my_deck

_deck_cache: list[int] | None = None
_last_game_id: int = -1
_models_loaded = False


def read_deck() -> list[int]:
    global _deck_cache
    if _deck_cache is not None:
        return _deck_cache

    deck = load_deck_csv()
    if len(deck) != 60:
        deck = get_default_deck()

    _deck_cache = deck
    return deck


def _ensure_models_loaded() -> None:
    global _models_loaded
    if _models_loaded:
        return
    _models_loaded = True

    from agent.policy import load_policy_model
    from agent.search import load_value_model

    load_policy_model()
    load_value_model()


def agent(obs_dict: dict) -> list[int]:
    global _last_game_id

    _ensure_models_loaded()

    if not obs_dict or 'select' not in obs_dict:
        deck = read_deck()
        reset_opponent_models()
        set_my_strategy(STRATEGIES.get(classify_my_deck(deck)))
        _last_game_id = -1
        return deck

    obs: Observation = to_observation_class(obs_dict)

    if obs.current is not None and obs.current.turn == 0 and _last_game_id != 0:
        reset_opponent_models()
        _last_game_id = 0

    if obs.select is None:
        deck = read_deck()
        return deck

    current = obs.current
    if current is None:
        return choose_action(obs)

    if current.result != -1:
        return []

    select = obs.select
    options = select.option
    min_count = select.minCount
    max_count = select.maxCount
    n_opts = len(options) if options else 0

    if n_opts == 0:
        return []

    if len(options) == 1 and max_count <= 1:
        return [0]

    action = choose_action(obs)

    if min_count > 0 and len(action) < min_count:
        if n_opts >= min_count:
            action = list(range(min_count))
        else:
            action = list(range(n_opts))
    if max_count > 0 and len(action) > max_count:
        action = action[:max_count]

    return action


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