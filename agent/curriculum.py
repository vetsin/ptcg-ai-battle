from __future__ import annotations

import json
import os
import time
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict

from agent.selfplay import (
    run_self_play,
    filter_winning_trajectories,
    filter_trajectories_by_reward,
    export_training_data,
    export_trajectories_jsonl,
    SelfPlayResult,
    GameTrajectory,
    compute_difficulty,
)
from agent.opponents import (
    random_agent,
    slightly_smart_agent,
    get_opponent_fn,
    get_deck_for_tier,
    DECK_TIERS,
)


@dataclass
class IterationStats:
    iteration: int
    tier: str
    num_games: int
    wins: int
    losses: int
    draws: int
    errors: int
    win_rate: float
    avg_reward: float
    avg_turns: float
    winning_trajectories: int
    filtered_trajectories: int
    elapsed_seconds: float


@dataclass
class CurriculumConfig:
    num_iterations: int = 3
    games_per_iteration: int = 50
    reward_threshold: float = 0.5
    min_trajectory_steps: int = 3
    difficulty_tiers: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    output_dir: str = "training_data"
    max_steps_per_game: int = 2000
    save_all_trajectories: bool = True
    save_winning_only: bool = True


def get_tier_deck(difficulty_idx: int) -> list[int]:
    tiers = list(DECK_TIERS.keys())
    idx = min(difficulty_idx, len(tiers) - 1)
    tier_name = tiers[idx]
    return get_deck_for_tier(tier_name)


def _get_agent_deck() -> list[int]:
    from agent.deck import load_deck_csv
    return load_deck_csv()


def run_iteration(
    iteration: int,
    agent_fn,
    tier: str,
    config: CurriculumConfig,
    num_games: int | None = None,
) -> tuple[list[GameTrajectory], IterationStats]:
    start_time = time.time()

    n_games = num_games or config.games_per_iteration
    difficulty_idx = config.difficulty_tiers.index(tier) if tier in config.difficulty_tiers else 0
    difficulty = difficulty_idx / max(len(config.difficulty_tiers) - 1, 1)
    opponent_fn = get_opponent_fn(difficulty)
    opponent_deck = get_tier_deck(difficulty_idx)

    my_deck = _get_agent_deck()

    result = run_self_play(
        deck=my_deck,
        agent_fn=agent_fn,
        num_games=n_games,
        opponent_fn=opponent_fn,
        opponent_deck=opponent_deck,
        max_steps=config.max_steps_per_game,
        collect_trajectory=True,
    )

    for traj in result.games:
        traj.difficulty = compute_difficulty(traj)

    winning = filter_winning_trajectories(result, min_steps=config.min_trajectory_steps)
    filtered = filter_trajectories_by_reward(result, threshold=config.reward_threshold, min_steps=config.min_trajectory_steps)

    elapsed = time.time() - start_time

    stats = IterationStats(
        iteration=iteration,
        tier=tier,
        num_games=result.total_games,
        wins=result.wins,
        losses=result.losses,
        draws=result.draws,
        errors=result.errors,
        win_rate=result.wins / max(result.total_games, 1),
        avg_reward=result.avg_reward,
        avg_turns=result.avg_turns,
        winning_trajectories=len(winning),
        filtered_trajectories=len(filtered),
        elapsed_seconds=elapsed,
    )

    tier_dir = Path(config.output_dir) / f"iteration_{iteration:03d}" / tier
    tier_dir.mkdir(parents=True, exist_ok=True)

    if config.save_all_trajectories:
        export_trajectories_jsonl(result.games, tier_dir / "all_trajectories.jsonl")

    if config.save_winning_only:
        export_trajectories_jsonl(winning, tier_dir / "winning_trajectories.jsonl")

    metadata = export_training_data(filtered, str(tier_dir))
    metadata["stats"] = asdict(stats)
    metadata["tier"] = tier
    metadata["iteration"] = iteration

    with open(tier_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(tier_dir / "stats.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)

    return filtered, stats


def run_curriculum(
    agent_fn,
    config: CurriculumConfig | None = None,
) -> list[tuple[list[GameTrajectory], IterationStats]]:
    if config is None:
        config = CurriculumConfig()

    all_results = []
    all_stats = []
    log_lines = []

    for iteration in range(config.num_iterations):
        for tier in config.difficulty_tiers:
            tier_label = f"Iter {iteration + 1}/{config.num_iterations}, Tier '{tier}'"
            print(f"[Curriculum] {tier_label}: Running {config.games_per_iteration} games...")

            filtered, stats = run_iteration(
                iteration=iteration,
                agent_fn=agent_fn,
                tier=tier,
                config=config,
            )

            summary = (
                f"[Curriculum] {tier_label}: "
                f"{stats.wins}W/{stats.losses}L/{stats.draws}D "
                f"(wr={stats.win_rate:.1%}, "
                f"reward={stats.avg_reward:.3f}, "
                f"{stats.filtered_trajectories} train steps, "
                f"{stats.elapsed_seconds:.1f}s)"
            )
            print(summary)
            log_lines.append(summary)

            all_results.append((filtered, stats))
            all_stats.append(stats)

    overall_path = Path(config.output_dir) / "curriculum_log.txt"
    with open(overall_path, "w") as f:
        f.write("\n".join(log_lines))

    return all_results


if __name__ == "__main__":
    from agent.main import agent

    config = CurriculumConfig(
        num_iterations=1,
        games_per_iteration=10,
        difficulty_tiers=["easy", "medium"],
        output_dir="training_data",
    )

    print("Starting curriculum self-play...")
    results = run_curriculum(agent, config)

    total_games = sum(s.num_games for _, s in results)
    total_wins = sum(s.wins for _, s in results)
    total_train = sum(s.filtered_trajectories for _, s in results)

    print(f"\nDone! {total_games} games, {total_wins} wins, {total_train} training steps")