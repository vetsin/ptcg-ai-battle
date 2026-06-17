import time
import random
from cg.api import (
    Observation,
    State,
    PlayerState,
    Pokemon,
    Option,
    SelectData,
    SearchState,
    ApiResult,
    OptionType,
    SelectType,
    SelectContext,
    EnergyType,
)
from cg.api import search_begin, search_step, search_end, search_release
from agent.evaluate import evaluate_state
from agent.policy import choose_action


class MCTSNode:
    __slots__ = ["search_id", "visit_count", "value_sum", "is_terminal", "observation", "parent_action"]

    def __init__(self, search_id: int | None = None):
        self.search_id = search_id
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_terminal = False
        self.observation: Observation | None = None
        self.parent_action: list[int] | None = None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def ucb(self) -> float:
        if self.visit_count == 0:
            return 1e9
        exploit = self.value
        explore = 1.414 * (0.0 ** 0.5 / self.visit_count) if self.visit_count > 0 else 0
        return exploit

    def update(self, value: float):
        self.visit_count += 1
        self.value_sum += value


def mcts_search(
    obs: Observation,
    my_deck: list[int],
    my_prize: list[int],
    opp_deck: list[int],
    opp_prize: list[int],
    opp_hand: list[int],
    opp_active: list[int],
    max_simulations: int = 50,
    max_time_ms: float = 4000.0,
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
            my_deck,
            my_prize,
            opp_deck,
            opp_prize,
            opp_hand,
            opp_active,
            manual_coin=manual_coin,
        )
    except (ValueError, RuntimeError):
        return None

    root_id = root_state.searchId
    root_obs = root_state.observation

    if root_obs.select is None or root_obs.select.option is None or len(root_obs.select.option) == 0:
        search_end()
        return None

    select = root_obs.select
    if len(select.option) == 1:
        search_end()
        return [0]

    if select.minCount == select.maxCount and select.minCount > 0:
        needs_full_selection = False
        if select.type in (
            SelectType.COUNT,
            SelectType.ENERGY,
            SelectType.CARD_OR_ATTACHED_CARD,
        ):
            if select.minCount > 1:
                needs_full_selection = True
        if not needs_full_selection:
            search_end()
            return None

    root_node = MCTSNode(search_id=root_id)
    root_node.observation = root_obs

    nodes: dict[int, MCTSNode] = {root_id: root_node}
    active_search_ids: set[int] = {root_id}

    start_time = time.time()
    simulations = 0

    try:
        while simulations < max_simulations:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > max_time_ms:
                break

            leaf = _select_leaf(nodes, root_id)
            if leaf is None:
                break

            if leaf.is_terminal or leaf.observation is None or leaf.observation.select is None:
                value = _evaluate_terminal(leaf, my_index)
                _backpropagate(nodes, leaf, value, my_index)
                simulations += 1
                continue

            actions = _get_actions_for_node(leaf, my_index)
            if not actions:
                leaf.is_terminal = True
                value = _evaluate_terminal(leaf, my_index)
                _backpropagate(nodes, leaf, value, my_index)
                simulations += 1
                continue

            action = actions[0]
            try:
                child_result = search_step(leaf.search_id, action)
                if child_result is None or child_result.error != 0:
                    leaf.is_terminal = True
                    value = evaluate_state(leaf.observation.current, my_index)
                    _backpropagate(nodes, leaf, value, my_index)
                    simulations += 1
                    continue

                child_obs = child_result.observation if child_result.state else None

                child_node = MCTSNode(search_id=leaf.search_id)
                child_node.parent_action = action
                child_node.observation = child_obs

                if child_obs is None or child_obs.select is None or child_obs.current is None:
                    child_node.is_terminal = True
                    value = evaluate_state(leaf.observation.current, my_index)
                    child_node.update(value)
                    _backpropagate(nodes, child_node, value, my_index)
                else:
                    value = evaluate_state(child_obs.current, my_index)
                    child_node.update(value)
                    _backpropagate(nodes, child_node, value, my_index)

            except (ValueError, RuntimeError):
                leaf.is_terminal = True
                value = evaluate_state(leaf.observation.current, my_index)
                _backpropagate(nodes, leaf, value, my_index)

            simulations += 1

    finally:
        search_end()

    return None


def _select_leaf(nodes: dict[int, MCTSNode], root_id: int) -> MCTSNode | None:
    root = nodes.get(root_id)
    if root is None:
        return None

    best = root
    best_ucb = -float("inf")

    for node in nodes.values():
        if node.visit_count == 0:
            return node
        ucb = node.ucb
        if ucb > best_ucb:
            best_ucb = ucb
            best = node

    return best


def _get_actions_for_node(node: MCTSNode, my_index: int) -> list[list[int]]:
    if node.observation is None or node.observation.select is None:
        return []

    obs = node.observation
    select = obs.select
    options = select.option

    if len(options) == 0:
        return []

    if len(options) == 1:
        return [[0]]

    action = choose_action(obs)
    if action:
        return [action]

    return [random.sample(range(len(options)), min(select.maxCount, len(options)))]


def _evaluate_terminal(node: MCTSNode, my_index: int) -> float:
    if node.observation and node.observation.current:
        return evaluate_state(node.observation.current, my_index)
    return 0.0


def _backpropagate(nodes: dict[int, MCTSNode], node: MCTSNode, value: float, my_index: int):
    node.visit_count += 1
    node.value_sum += value


def predict_opponent_deck(all_card_ids: list[int], opp: PlayerState) -> list[int]:
    known_opp_cards: set[int] = set()
    if opp.discard:
        for card in opp.discard:
            known_opp_cards.add(card.id)
    if opp.active:
        for mon in opp.active:
            if mon:
                known_opp_cards.add(mon.id)
    if opp.bench:
        for mon in opp.bench:
            known_opp_cards.add(mon.id)

    remaining = 60 - len(known_opp_cards)
    pool = [cid for cid in all_card_ids if cid not in known_opp_cards]
    if pool and remaining > 0:
        prediction = list(known_opp_cards) + (pool * ((remaining // len(pool)) + 1))[:remaining]
        return prediction[:60]
    return list(known_opp_cards) + [3] * max(0, remaining)


def predict_opponent_hand(opp: PlayerState) -> list[int]:
    return [3] * opp.handCount


def predict_opponent_prize(opp: PlayerState) -> list[int]:
    known = [p.id for p in opp.prize if p is not None]
    unknown_count = len([p for p in opp.prize if p is None])
    return known + [3] * unknown_count


def predict_opponent_active(opp: PlayerState) -> list[int]:
    if opp.active and opp.active[0] is not None:
        return []
    return [3]