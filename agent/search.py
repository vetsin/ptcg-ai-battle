import os
import random
import time
from cg.api import (
    Observation,
    PlayerState,
    SelectType,
    OptionType,
    CardType,
    EnergyType,
)
from cg.api import search_begin, search_step, search_end
from agent.evaluate import evaluate_state, get_card_data, get_attack_data
from agent.policy import choose_action
from agent.opponent_model import get_opponent_model
from agent.features import encode_observation

_value_model = None
_value_device = "cpu"
_value_net_enabled = False
_value_net_blend = 0.5
_value_net_scale = 400.0


def load_value_model(path: str | None = None, device: str = "cpu") -> None:
    """Opt-in: load a trained value head to replace MCTS rollouts (see SELFPLAY_PLAN.md A.6)."""
    global _value_model, _value_device, _value_net_enabled
    if path is None or not os.path.exists(path):
        _value_net_enabled = False
        return
    from agent.network import load_model
    try:
        _value_model = load_model(path, device=device)
        _value_device = device
        _value_net_enabled = True
    except Exception:
        _value_net_enabled = False


def _value_net_signal(obs: Observation, my_index: int) -> float | None:
    if not _value_net_enabled or _value_model is None:
        return None
    if obs is None or obs.current is None or obs.select is None:
        return None
    if obs.current.yourIndex != my_index:
        return None
    try:
        state_features, option_features = encode_observation(obs)
        if not state_features or not option_features:
            return None
        from agent.network import score_actions
        _, value = score_actions(_value_model, state_features, option_features, device=_value_device)
        return value
    except Exception:
        return None


def _static_value_eval(observation: Observation, my_index: int) -> float | None:
    obs = observation
    if obs is None or obs.current is None:
        return None
    if obs.current.result != -1:
        return 10000.0 if obs.current.result == my_index else -10000.0

    net_signal = _value_net_signal(obs, my_index)
    if net_signal is None:
        return None

    heuristic = evaluate_state(obs.current, my_index)
    scaled_net = (net_signal - 0.5) * 2.0 * _value_net_scale
    return (1.0 - _value_net_blend) * heuristic + _value_net_blend * scaled_net


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

    max_sims = _adaptive_sim_budget(obs, max_simulations)
    max_time = _adaptive_time_budget(obs, max_time_ms)

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

    for i in range(min(n_options, 10)):
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

        outcome_group = _classify_outcome(root_value, child_value)

        child_stats[i] = {
            "action": action,
            "value": child_value,
            "visits": 1,
            "search_id": child_state.searchId,
            "observation": child_obs,
            "is_terminal": child_obs.select is None or child_obs.current is None or child_obs.current.result != -1,
            "outcome_group": outcome_group,
        }

    if not child_stats:
        search_end()
        return _get_heuristic_action(root_obs)

    _prune_dominated(child_stats)

    if not child_stats:
        search_end()
        return _get_heuristic_action(root_obs)

    start_time = time.time()
    simulations = 0
    exploration = 1.414
    rollout_depth = _adaptive_rollout_depth(obs, 5)

    while simulations < max_sims:
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > max_time:
            break

        best_idx = _select_root_child(child_stats, root_value, exploration)
        if best_idx is None:
            break

        stats = child_stats[best_idx]

        if stats["is_terminal"]:
            stats["visits"] += 1
            simulations += 1
            continue

        rollout_value = _rollout_from_node(
            stats["search_id"], stats["observation"], my_index, max_depth=rollout_depth,
        )
        stats["visits"] += 1
        stats["value"] = stats["value"] * (stats["visits"] - 1) / stats["visits"] + rollout_value / stats["visits"]
        simulations += 1

        total_v = sum(s["visits"] for s in child_stats.values())
        if total_v >= 8:
            best_v = max(s["visits"] for s in child_stats.values())
            if best_v / total_v > 0.65:
                break

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


def _adaptive_sim_budget(obs: Observation, base: int) -> int:
    if obs.current is None:
        return base
    me = obs.current.players[obs.current.yourIndex]
    opp = obs.current.players[1 - obs.current.yourIndex]
    my_taken = 6 - len([p for p in me.prize if p is not None])
    opp_taken = 6 - len([p for p in opp.prize if p is not None])
    opp_active = opp.active[0] if opp.active else None

    if opp_taken >= 3 or my_taken >= 3:
        return int(base * 1.5)

    if opp_active and opp_active.maxHp > 0 and opp_active.hp * 2 <= opp_active.maxHp:
        return int(base * 1.2)

    if obs.select and obs.select.type == SelectType.ATTACK:
        return int(base * 1.3)

    if obs.select and len(obs.select.option) <= 3:
        return int(base * 0.6)

    return base


def _adaptive_time_budget(obs: Observation, base_ms: float) -> float:
    if obs.current is None:
        return base_ms
    me = obs.current.players[obs.current.yourIndex]
    opp = obs.current.players[1 - obs.current.yourIndex]
    my_taken = 6 - len([p for p in me.prize if p is not None])
    opp_taken = 6 - len([p for p in opp.prize if p is not None])

    if opp_taken >= 3 or my_taken >= 3:
        return base_ms * 1.5
    return base_ms


def _adaptive_rollout_depth(obs: Observation, base: int) -> int:
    if obs.current is None:
        return base
    me = obs.current.players[obs.current.yourIndex]
    opp = obs.current.players[1 - obs.current.yourIndex]
    my_taken = 6 - len([p for p in me.prize if p is not None])
    opp_taken = 6 - len([p for p in opp.prize if p is not None])

    if opp_taken >= 3 or my_taken >= 3:
        return base + 5
    if opp_taken >= 2 and my_taken >= 2:
        return base + 3
    return base


def _classify_outcome(root_value: float, child_value: float) -> str:
    delta = child_value - root_value
    if child_value >= 9000.0:
        return "win"
    if child_value <= -9000.0:
        return "loss"
    if delta > 500.0:
        return "big_gain"
    if delta > 100.0:
        return "small_gain"
    if delta < -500.0:
        return "big_loss"
    if delta < -100.0:
        return "small_loss"
    return "neutral"


def _prune_dominated(child_stats: dict[int, dict]) -> None:
    if len(child_stats) <= 4:
        return

    loss_indices = [i for i, s in child_stats.items() if s["outcome_group"] == "loss"]
    for idx in loss_indices:
        del child_stats[idx]

    if len(child_stats) <= 4:
        return

    big_loss_indices = [i for i, s in child_stats.items() if s["outcome_group"] == "big_loss"]
    for idx in big_loss_indices:
        if len(child_stats) > 4:
            del child_stats[idx]


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
    if _value_net_enabled:
        net_eval = _static_value_eval(observation, my_index)
        if net_eval is not None:
            return net_eval

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
        if random.random() < 0.7:
            heuristic = _get_heuristic_action(obs)
            if heuristic is not None:
                return heuristic
        else:
            return [random.randint(0, n - 1)]

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


def _energy_type_to_card_id(energy_type: int) -> int:
    mapping = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    }
    return mapping.get(int(energy_type), 2)


def _infer_opponent_energy(opp: PlayerState) -> int:
    card_db = get_card_data()
    energy_counts: dict[int, int] = {}

    pokemon_ids = set()
    if opp.active:
        for mon in opp.active:
            if mon is not None:
                pokemon_ids.add(mon.id)
    if opp.bench:
        for mon in opp.bench:
            pokemon_ids.add(mon.id)

    for pid in pokemon_ids:
        cd = card_db.get(pid)
        if cd and cd.attacks:
            attack_db = get_attack_data()
            for atk_id in cd.attacks:
                atk = attack_db.get(atk_id)
                if atk:
                    for e in atk.energies:
                        e_int = int(e)
                        if e_int in (1, 2, 3, 4, 5, 6, 7, 8):
                            energy_counts[e_int] = energy_counts.get(e_int, 0) + 1

    if energy_counts:
        return max(energy_counts, key=energy_counts.get)
    return 2


def _get_opponent_known_cards(opp: PlayerState) -> list[int]:
    known: list[int] = []
    if opp.discard:
        for card in opp.discard:
            known.append(card.id)
    if opp.active:
        for mon in opp.active:
            if mon is not None:
                known.append(mon.id)
    if opp.bench:
        for mon in opp.bench:
            known.append(mon.id)
    for p in opp.prize:
        if p is not None:
            known.append(p.id)
    return known


def predict_opponent_deck(all_card_ids: list[int], opp: PlayerState) -> list[int]:
    known_set: set[int] = set()
    known_list: list[int] = []

    if opp.discard:
        for card in opp.discard:
            known_set.add(card.id)
            known_list.append(card.id)
    if opp.active:
        for mon in opp.active:
            if mon is not None:
                known_set.add(mon.id)
                known_list.append(mon.id)
    if opp.bench:
        for mon in opp.bench:
            known_set.add(mon.id)
            known_list.append(mon.id)

    default_energy = _energy_type_to_card_id(_infer_opponent_energy(opp))

    filler_energy = [default_energy] * 15
    card_db = get_card_data()
    basic_pokemon = []
    for cid in all_card_ids:
        cd = card_db.get(cid)
        if cd and cd.cardType == CardType.POKEMON and cd.basic:
            if cid not in known_set:
                basic_pokemon.append(cid)

    filler_pokemon = []
    if basic_pokemon:
        step = max(1, len(basic_pokemon) // 8)
        for i in range(0, len(basic_pokemon), step):
            filler_pokemon.append(basic_pokemon[i])
        filler_pokemon = filler_pokemon[:3]

    remaining = 60 - len(known_list)
    filler = (filler_energy + filler_pokemon + [default_energy] * 20)
    while len(filler) < remaining:
        filler = filler + filler
    filler = filler[:remaining]

    return known_list + filler


def predict_opponent_hand(opp: PlayerState) -> list[int]:
    default_energy = _energy_type_to_card_id(_infer_opponent_energy(opp))
    n = opp.handCount
    if n <= 0:
        return []

    hand = [default_energy] * max(1, n * 2 // 3)
    hand.extend([default_energy] * (n - len(hand)))
    return hand[:n]


def predict_opponent_prize(opp: PlayerState) -> list[int]:
    default_energy = _energy_type_to_card_id(_infer_opponent_energy(opp))
    result = []
    for p in opp.prize:
        if p is not None:
            result.append(p.id)
        else:
            result.append(default_energy)
    return result


def predict_opponent_active(opp: PlayerState) -> list[int]:
    if opp.active and len(opp.active) > 0 and opp.active[0] is not None:
        return []
    if opp.bench:
        return [opp.bench[0].id]
    card_db = get_card_data()
    default_energy = _energy_type_to_card_id(_infer_opponent_energy(opp))
    basics = [cid for cid, cd in card_db.items()
              if cd.cardType == CardType.POKEMON and cd.basic and cd.energyType == int(EnergyType.FIRE)]
    if basics:
        return [basics[0]]
    return [1145]


def build_search_inputs(obs: Observation) -> dict:
    if obs.current is None or obs.select is None:
        return {}

    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    opp = obs.current.players[1 - my_index]

    opp_model = get_opponent_model(1 - my_index)
    opp_model.update(opp)

    default_energy = _energy_type_to_card_id(_infer_opponent_energy(opp))

    your_prize = [default_energy if p is None else p.id for p in me.prize]

    if obs.select.deck:
        your_deck_list = [c.id for c in obs.select.deck]
    else:
        your_deck_list = [default_energy] * me.deckCount

    opp_deck = opp_model.sample_deck(opp)
    opp_prize = opp_model.predict_prize(opp)
    opp_hand = opp_model.predict_hand(opp)
    opp_active = opp_model.predict_active(opp)

    return {
        "your_deck": your_deck_list,
        "your_prize": your_prize,
        "opponent_deck": opp_deck,
        "opponent_prize": opp_prize,
        "opponent_hand": opp_hand,
        "opponent_active": opp_active,
    }