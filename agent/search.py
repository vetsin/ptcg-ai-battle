import time
from cg.api import (
    Observation,
    PlayerState,
    SelectType,
)
from cg.api import search_begin, search_step, search_end
from agent.evaluate import evaluate_state, get_card_data
from agent.policy import choose_action


def mcts_search(
    obs: Observation,
    your_deck: list[int],
    your_prize: list[int],
    opponent_deck: list[int],
    opponent_prize: list[int],
    opponent_hand: list[int],
    opponent_active: list[int],
    max_simulations: int = 60,
    max_time_ms: float = 2000.0,
    manual_coin: bool = True,
) -> list[int] | None:
    if obs.current is None or obs.select is None:
        return None
    if obs.search_begin_input is None:
        return None

    my_index = obs.current.yourIndex

    try:
        root_state = search_begin(
            obs,
            your_deck,
            your_prize,
            opponent_deck,
            opponent_prize,
            opponent_hand,
            opponent_active,
            manual_coin=manual_coin,
        )
    except (ValueError, RuntimeError):
        return None

    root_obs = root_state.observation
    root_id = root_state.searchId

    if root_obs.select is None or root_obs.select.option is None or len(root_obs.select.option) == 0:
        search_end()
        return None

    if len(root_obs.select.option) == 1:
        search_end()
        return [0]

    n_options = len(root_obs.select.option)
    root_value = evaluate_state(obs.current, my_index)

    child_stats: dict[int, dict] = {}

    for i in range(min(n_options, 8)):
        action = _compute_action_for_option(root_obs, i)
        if action is None:
            continue

        try:
            child_state = search_step(root_id, action)
        except (ValueError, RuntimeError):
            continue

        child_obs = child_state.observation
        if child_obs is None:
            continue

        child_value = root_value
        if child_obs.current is not None:
            child_value = evaluate_state(child_obs.current, my_index)
            if child_obs.current.result != -1:
                child_value = 10000.0 if child_obs.current.result == my_index else -10000.0

        child_stats[i] = {
            "action": action,
            "value": child_value,
            "visits": 1,
            "search_id": child_state.searchId,
            "observation": child_obs,
            "is_terminal": child_obs.select is None or child_obs.current is None or child_obs.current.result != -1,
        }

    if not child_stats:
        search_end()
        return _get_heuristic_action(root_obs)

    start_time = time.time()
    simulations = 0

    while simulations < max_simulations:
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > max_time_ms:
            break

        best_idx = _select_root_child(child_stats, root_value, exploration=1.414)
        if best_idx is None:
            break

        stats = child_stats[best_idx]

        if stats["is_terminal"]:
            stats["visits"] += 1
            simulations += 1
            continue

        rollout_value = _rollout_from_node(
            stats["search_id"], stats["observation"], my_index, max_depth=5,
        )
        stats["visits"] += 1
        stats["value"] = stats["value"] * (stats["visits"] - 1) / stats["visits"] + rollout_value / stats["visits"]
        simulations += 1

    search_end()

    best_action = None
    best_visits = -1
    for idx, stats in child_stats.items():
        if stats["visits"] > best_visits:
            best_visits = stats["visits"]
            best_action = stats["action"]

    if best_action is not None:
        return best_action

    return _get_heuristic_action(root_obs)


def _select_root_child(child_stats: dict[int, dict], root_value: float, exploration: float = 1.414) -> int | None:
    total_visits = sum(s["visits"] for s in child_stats.values())
    if total_visits == 0:
        return next(iter(child_stats))

    best_idx = None
    best_ucb = -float("inf")

    for idx, stats in child_stats.items():
        if stats["visits"] == 0:
            return idx

        exploit = stats["value"]
        explore = exploration * (total_visits ** 0.5 / stats["visits"])
        ucb = exploit + explore

        if ucb > best_ucb:
            best_ucb = ucb
            best_idx = idx

    return best_idx


def _rollout_from_node(search_id: int, observation: Observation, my_index: int, max_depth: int = 5) -> float:
    current_obs = observation

    for _ in range(max_depth):
        if current_obs is None or current_obs.select is None or current_obs.current is None:
            break

        if current_obs.current.result != -1:
            if current_obs.current.result == my_index:
                return 10000.0
            return -10000.0

        options = current_obs.select.option
        if not options:
            break

        n_opts = len(options)
        if n_opts == 0:
            break

        action = _get_rollout_action(current_obs)
        if action is None:
            break

        try:
            next_state = search_step(search_id, action)
        except (ValueError, RuntimeError):
            break

        search_id = next_state.searchId
        current_obs = next_state.observation

    if current_obs and current_obs.current:
        return evaluate_state(current_obs.current, my_index)
    return 0.0


def _get_rollout_action(obs: Observation) -> list[int] | None:
    select = obs.select
    if select is None or not select.option:
        return None

    n = len(select.option)
    if n == 1:
        return [0]

    if select.type in (SelectType.MAIN, SelectType.ATTACK):
        heuristic = _get_heuristic_action(obs)
        if heuristic is not None:
            return heuristic

    if select.minCount > 0:
        return list(range(min(select.minCount, n)))

    return [0]


def _compute_action_for_option(obs: Observation, option_index: int) -> list[int] | None:
    select = obs.select
    if select is None:
        return None

    options = select.option
    if option_index >= len(options):
        return None

    min_count = select.minCount
    max_count = select.maxCount
    select_type = select.type

    if min_count == 0 and max_count == 0:
        return []

    if select_type in (SelectType.MAIN, SelectType.ATTACK, SelectType.YES_NO,
                       SelectType.COUNT, SelectType.SPECIAL_CONDITION, SelectType.SKILL):
        return [option_index]

    if select_type in (SelectType.CARD, SelectType.ATTACHED_CARD,
                       SelectType.CARD_OR_ATTACHED_CARD, SelectType.ENERGY, SelectType.EVOLVE):
        if max_count == 1:
            return [option_index]
        if min_count == max_count and min_count > 0:
            return list(range(min(min_count, len(options))))

    if max_count == 1:
        return [option_index]

    return [option_index]


def _get_heuristic_action(obs: Observation) -> list[int] | None:
    try:
        return choose_action(obs)
    except Exception:
        if obs.select and obs.select.option:
            n = len(obs.select.option)
            if n == 1:
                return [0]
            if obs.select.minCount > 0:
                return list(range(min(obs.select.minCount, n)))
            return [0]
        return None


def predict_opponent_deck(all_card_ids: list[int], opp: PlayerState) -> list[int]:
    known: set[int] = set()
    if opp.discard:
        for card in opp.discard:
            known.add(card.id)
    if opp.active:
        for mon in opp.active:
            if mon is not None:
                known.add(mon.id)
    if opp.bench:
        for mon in opp.bench:
            known.add(mon.id)

    remaining = 60 - len(known)
    pool = [cid for cid in all_card_ids if cid not in known]
    if not pool:
        return list(known) + [3] * max(0, remaining)

    repeats = (remaining // len(pool)) + 1
    filler = (pool * repeats)[:remaining]
    return list(known) + filler


def predict_opponent_hand(opp: PlayerState) -> list[int]:
    return [3] * opp.handCount


def predict_opponent_prize(opp: PlayerState) -> list[int]:
    result = []
    for p in opp.prize:
        if p is not None:
            result.append(p.id)
        else:
            result.append(3)
    return result


def predict_opponent_active(opp: PlayerState) -> list[int]:
    if opp.active and len(opp.active) > 0 and opp.active[0] is not None:
        return []
    return [3]


def build_search_inputs(obs: Observation) -> dict:
    if obs.current is None or obs.select is None:
        return {}

    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    opp = obs.current.players[1 - my_index]
    all_card_ids = list(get_card_data().keys())

    your_prize = [3 if p is None else p.id for p in me.prize]

    if obs.select.deck:
        your_deck_list = [c.id for c in obs.select.deck]
    else:
        your_deck_list = [3] * me.deckCount

    opp_deck = predict_opponent_deck(all_card_ids, opp)
    opp_prize = predict_opponent_prize(opp)
    opp_hand = predict_opponent_hand(opp)
    opp_active = predict_opponent_active(opp)

    return {
        "your_deck": your_deck_list,
        "your_prize": your_prize,
        "opponent_deck": opp_deck,
        "opponent_prize": opp_prize,
        "opponent_hand": opp_hand,
        "opponent_active": opp_active,
    }