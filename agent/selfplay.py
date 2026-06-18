from __future__ import annotations

import json
import os
import time
import random
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

from cg.api import (
    Observation,
    State,
    PlayerState,
    Option,
    SelectData,
    Log,
    LogType,
    OptionType,
    SelectType,
    SelectContext,
    to_observation_class,
)
from cg.game import battle_start, battle_select, battle_finish


@dataclass
class TrajectoryStep:
    obs_dict: dict
    action: list[int]
    select_type: int
    select_context: int
    num_options: int
    state_score: float
    turn: int
    my_index: int


@dataclass
class GameTrajectory:
    game_id: str
    player0_deck: list[int]
    player1_deck: list[int]
    my_index: int
    outcome: int
    my_prize_taken: int
    opp_prize_taken: int
    num_turns: int
    steps: list[TrajectoryStep] = field(default_factory=list)
    difficulty: float = 0.0
    reward: float = 0.0


@dataclass
class SelfPlayResult:
    games: list[GameTrajectory] = field(default_factory=list)
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    errors: int = 0
    avg_turns: float = 0.0
    avg_reward: float = 0.0


AgentFn = __import__("typing").Callable[[dict], list[int]]


def run_game(
    deck0: list[int],
    deck1: list[int],
    agent0_fn: AgentFn,
    agent1_fn: AgentFn,
    max_steps: int = 2000,
    game_id: str | None = None,
    collect_trajectory: bool = True,
    trajectory_player: int = 0,
) -> GameTrajectory:
    if game_id is None:
        game_id = f"game_{int(time.time() * 1000)}"

    obs_dict, start_data = battle_start(deck0, deck1)

    if obs_dict is None:
        return GameTrajectory(
            game_id=game_id,
            player0_deck=deck0,
            player1_deck=deck1,
            my_index=trajectory_player,
            outcome=-1,
            my_prize_taken=0,
            opp_prize_taken=0,
            num_turns=0,
        )

    agents = [agent0_fn, agent1_fn]
    decks = [deck0, deck1]

    trajectory = GameTrajectory(
        game_id=game_id,
        player0_deck=deck0,
        player1_deck=deck1,
        my_index=trajectory_player,
        outcome=-1,
        my_prize_taken=0,
        opp_prize_taken=0,
        num_turns=0,
    )

    from agent.evaluate import evaluate_state

    for step in range(max_steps):
        obs = to_observation_class(obs_dict)

        if obs.current is not None and obs.current.result != -1:
            trajectory.outcome = obs.current.result
            trajectory.num_turns = obs.current.turn
            _compute_game_stats(trajectory)
            try:
                battle_finish()
            except Exception:
                pass
            return trajectory

        current_player = obs.current.yourIndex if obs.current and obs.current.yourIndex is not None else 0

        if obs.select is None:
            action = decks[current_player]
        else:
            try:
                action = agents[current_player](obs_dict)
            except Exception:
                action = _random_action(obs)

        if collect_trajectory and current_player == trajectory_player and obs.select is not None and obs.current is not None:
            step_data = TrajectoryStep(
                obs_dict=obs_dict,
                action=action,
                select_type=int(obs.select.type),
                select_context=int(obs.select.context),
                num_options=len(obs.select.option),
                state_score=evaluate_state(obs.current, obs.current.yourIndex),
                turn=obs.current.turn,
                my_index=obs.current.yourIndex,
            )
            trajectory.steps.append(step_data)

        try:
            obs_dict = battle_select(action)
        except (IndexError, ValueError):
            try:
                battle_finish()
            except Exception:
                pass
            return trajectory

    try:
        battle_finish()
    except Exception:
        pass

    return trajectory


def run_self_play(
    deck: list[int],
    agent_fn: AgentFn,
    num_games: int = 100,
    opponent_fn: AgentFn | None = None,
    opponent_deck: list[int] | None = None,
    max_steps: int = 2000,
    collect_trajectory: bool = True,
) -> SelfPlayResult:
    if opponent_fn is None:
        opponent_fn = random_agent
    if opponent_deck is None:
        opponent_deck = deck

    result = SelfPlayResult()

    for i in range(num_games):
        deck0 = deck
        deck1 = opponent_deck

        trajectory_player = 0 if i % 2 == 0 else 1
        if trajectory_player == 1:
            deck0, deck1 = deck1, deck0

        game_id = f"selfplay_{i:04d}_{int(time.time())}"

        trajectory = run_game(
            deck0=deck0,
            deck1=deck1,
            agent0_fn=agent_fn,
            agent1_fn=opponent_fn,
            max_steps=max_steps,
            game_id=game_id,
            collect_trajectory=collect_trajectory,
            trajectory_player=trajectory_player,
        )

        result.total_games += 1

        if trajectory.outcome == trajectory_player:
            result.wins += 1
            trajectory.reward = 1.0
        elif trajectory.outcome == 1 - trajectory_player:
            result.losses += 1
            trajectory.reward = 0.0
        elif trajectory.outcome == 2:
            result.draws += 1
            trajectory.reward = 0.5
        else:
            result.errors += 1

        trajectory.my_index = trajectory_player

        if collect_trajectory:
            result.games.append(trajectory)

    valid = result.wins + result.losses + result.draws
    if valid > 0:
        result.avg_turns = sum(g.num_turns for g in result.games) / valid
        result.avg_reward = sum(g.reward for g in result.games) / valid

    return result


def compute_difficulty(trajectory: GameTrajectory) -> float:
    if not trajectory.steps:
        return 0.0

    scores = [s.state_score for s in trajectory.steps]
    if not scores:
        return 0.0

    avg_score = sum(scores) / len(scores)
    score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
    prize_diff = trajectory.my_prize_taken - trajectory.opp_prize_taken

    diff = 0.5
    if trajectory.outcome == trajectory.my_index:
        diff += 0.1 * min(prize_diff, 3)
    else:
        diff -= 0.1 * min(abs(prize_diff), 3)

    diff += score_variance / 10000.0

    return max(0.0, min(1.0, diff))


def filter_winning_trajectories(
    result: SelfPlayResult, min_steps: int = 5
) -> list[GameTrajectory]:
    return [
        g for g in result.games
        if g.reward >= 1.0 and len(g.steps) >= min_steps
    ]


def filter_trajectories_by_reward(
    result: SelfPlayResult, threshold: float = 0.5, min_steps: int = 3
) -> list[GameTrajectory]:
    return [
        g for g in result.games
        if g.reward >= threshold and len(g.steps) >= min_steps
    ]


def export_trajectories_jsonl(
    trajectories: list[GameTrajectory],
    output_path: str | Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for traj in trajectories:
            record = {
                "game_id": traj.game_id,
                "outcome": traj.outcome,
                "my_index": traj.my_index,
                "my_prize_taken": traj.my_prize_taken,
                "opp_prize_taken": traj.opp_prize_taken,
                "num_turns": traj.num_turns,
                "difficulty": traj.difficulty,
                "reward": traj.reward,
                "num_steps": len(traj.steps),
                "steps": [],
            }
            for step in traj.steps:
                obs = to_observation_class(step.obs_dict)
                current = obs.current
                me = current.players[step.my_index] if current else None
                opp = current.players[1 - step.my_index] if current else None

                step_record = {
                    "select_type": step.select_type,
                    "select_context": step.select_context,
                    "num_options": step.num_options,
                    "action": step.action,
                    "state_score": step.state_score,
                    "turn": step.turn,
                    "my_prize_remaining": len([p for p in me.prize if p is not None]) if me else 6,
                    "opp_prize_remaining": len([p for p in opp.prize if p is not None]) if opp else 6,
                    "my_active_hp": me.active[0].hp if me and me.active and me.active[0] else 0,
                    "opp_active_hp": opp.active[0].hp if opp and opp.active and opp.active[0] else 0,
                    "my_bench_count": len(me.bench) if me else 0,
                    "opp_bench_count": len(opp.bench) if opp else 0,
                }
                record["steps"].append(step_record)

            f.write(json.dumps(record) + "\n")


def export_training_data(
    trajectories: list[GameTrajectory],
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = []

    for traj in trajectories:
        for step in traj.steps:
            obs = to_observation_class(step.obs_dict)
            if obs.current is None or obs.select is None:
                continue

            record = {
                "game_id": traj.game_id,
                "reward": traj.reward,
                "outcome": traj.outcome,
                "my_index": step.my_index,
                "select_type": step.select_type,
                "select_context": step.select_context,
                "num_options": step.num_options,
                "action": step.action,
                "state_score": step.state_score,
                "turn": step.turn,
                "difficulty": traj.difficulty,
            }

            current = obs.current
            if current:
                me = current.players[step.my_index]
                opp = current.players[1 - step.my_index]
                record["my_prize_remaining"] = len([p for p in me.prize if p is not None])
                record["opp_prize_remaining"] = len([p for p in opp.prize if p is not None])
                record["my_active_hp"] = me.active[0].hp if me.active and me.active[0] else 0
                record["opp_active_hp"] = opp.active[0].hp if opp.active and opp.active[0] else 0
                record["my_bench_count"] = len(me.bench)
                record["opp_bench_count"] = len(opp.bench)
                record["turn_number"] = current.turn
                record["energy_attached"] = int(current.energyAttached)
                record["supporter_played"] = int(current.supporterPlayed)

            train_records.append(record)

    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for record in train_records:
            f.write(json.dumps(record) + "\n")

    metadata = {
        "total_games": len(trajectories),
        "total_steps": sum(len(t.steps) for t in trajectories),
        "total_records": len(train_records),
        "wins": sum(1 for t in trajectories if t.reward >= 1.0),
        "losses": sum(1 for t in trajectories if t.reward < 0.5),
        "avg_reward": sum(t.reward for t in trajectories) / len(trajectories) if trajectories else 0,
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _default_deck()

    if obs.current is not None and obs.current.result != -1:
        return []

    options = obs.select.option
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    n_opts = len(options) if options else 0

    if n_opts == 0:
        return []
    if max_count == 0:
        return []
    if len(options) == 1:
        return [0]
    if min_count == max_count and min_count > 0:
        return list(range(min(min_count, n_opts)))
    count = max(min_count, 1)
    return random.sample(range(n_opts), min(count, n_opts))


_deck_cache: list[int] | None = None


def _default_deck() -> list[int]:
    global _deck_cache
    if _deck_cache is None:
        from agent.deck import load_deck_csv
        _deck_cache = load_deck_csv()
    return _deck_cache


def _random_action(obs: Observation) -> list[int]:
    options = obs.select.option
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    n_opts = len(options) if options else 0

    if n_opts == 0:
        return []
    if max_count == 0:
        return []
    if len(options) == 1:
        return [0]
    if min_count == max_count and min_count > 0:
        return list(range(min(min_count, n_opts)))
    count = max(min_count, 1)
    return random.sample(range(n_opts), min(count, n_opts))


def _compute_game_stats(trajectory: GameTrajectory):
    if not trajectory.steps:
        return

    my_index = trajectory.my_index
    last_obs_dict = trajectory.steps[-1].obs_dict
    last_obs = to_observation_class(last_obs_dict)

    if last_obs.current:
        me = last_obs.current.players[my_index]
        opp = last_obs.current.players[1 - my_index]
        trajectory.my_prize_taken = 6 - len([p for p in me.prize if p is not None])
        trajectory.opp_prize_taken = 6 - len([p for p in opp.prize if p is not None])

    if trajectory.outcome == trajectory.my_index:
        trajectory.reward = 1.0
    elif trajectory.outcome == 1 - trajectory.my_index:
        trajectory.reward = 0.0
    else:
        trajectory.reward = 0.5

    trajectory.difficulty = compute_difficulty(trajectory)