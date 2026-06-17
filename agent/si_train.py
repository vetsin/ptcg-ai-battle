from __future__ import annotations

import json
import os
import time
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch

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
from agent.network import train_policy, PTCGDataset, collate_fn
from agent.features import encode_observation, NUM_STATE_FEATURES, NUM_OPTION_FEATURES
from cg.api import to_observation_class


@dataclass
class SITrainConfig:
    num_iterations: int = 3
    games_per_iteration: int = 100
    reward_threshold: float = 0.5
    min_trajectory_steps: int = 3
    difficulty_tiers: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    output_dir: str = "si_training"
    max_steps_per_game: int = 2000
    model_type: str = "simple"
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    max_options: int = 20
    device: str = "auto"
    policy_blend: float = 0.7


@dataclass
class SIIterationStats:
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
    training_steps: int
    train_loss: float = 0.0
    train_acc: float = 0.0
    elapsed_seconds: float = 0.0


def run_si_iteration(
    iteration: int,
    agent_fn,
    tier: str,
    config: SITrainConfig,
) -> tuple[str | None, SIIterationStats]:
    start_time = time.time()
    difficulty_idx = config.difficulty_tiers.index(tier) if tier in config.difficulty_tiers else 0
    difficulty = difficulty_idx / max(len(config.difficulty_tiers) - 1, 1)
    opponent_fn = get_opponent_fn(difficulty)
    opponent_deck = get_deck_for_tier(tier)

    from agent.deck import load_deck_csv
    my_deck = load_deck_csv()

    result = run_self_play(
        deck=my_deck,
        agent_fn=agent_fn,
        num_games=config.games_per_iteration,
        opponent_fn=opponent_fn,
        opponent_deck=opponent_deck,
        max_steps=config.max_steps_per_game,
        collect_trajectory=True,
    )

    for traj in result.games:
        traj.difficulty = compute_difficulty(traj)

    winning = filter_winning_trajectories(result, min_steps=config.min_trajectory_steps)
    filtered = filter_trajectories_by_reward(result, threshold=config.reward_threshold, min_steps=config.min_trajectory_steps)

    iter_dir = Path(config.output_dir) / f"iteration_{iteration:03d}" / tier
    iter_dir.mkdir(parents=True, exist_ok=True)

    if result.games:
        export_trajectories_jsonl(result.games, iter_dir / "all_trajectories.jsonl")
    if winning:
        export_trajectories_jsonl(winning, iter_dir / "winning_trajectories.jsonl")

    train_records = _prepare_train_records(filtered)
    export_training_data(filtered, str(iter_dir))

    # Also save full trajectory data for potential later use
    _save_trajectory_records(train_records, iter_dir / "train_records.jsonl")

    model_path = None
    train_loss = 0.0
    train_acc = 0.0

    if len(train_records) >= 10:
        all_data = _collect_all_training_data(Path(config.output_dir), max_previous_iterations=2)
        print(f"    [SI] Training on {len(all_data)} samples...")

        model_path = str(iter_dir / "model.pt")
        history = train_policy(
            train_data=all_data,
            model_type=config.model_type,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            max_options=config.max_options,
            device=config.device,
            save_path=model_path,
        )

        if history.get("train_loss"):
            train_loss = history["train_loss"][-1]
            train_acc = history["train_acc"][-1]

        if model_path:
            from agent.policy import load_policy_model, _policy_blend
            load_policy_model(model_path)
    else:
        model_path = None
        print(f"    [SI] Insufficient data ({len(train_records)} records), skipping training")

    elapsed = time.time() - start_time

    stats = SIIterationStats(
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
        training_steps=len(train_records),
        train_loss=train_loss,
        train_acc=train_acc,
        elapsed_seconds=elapsed,
    )

    with open(iter_dir / "stats.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)

    return model_path, stats


def run_SI_loop(
    agent_fn,
    config: SITrainConfig | None = None,
) -> list[SIIterationStats]:
    if config is None:
        config = SITrainConfig()

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    all_stats = []
    log_lines = []
    current_agent = agent_fn

    for iteration in range(config.num_iterations):
        for tier in config.difficulty_tiers:
            label = f"SI Iter {iteration+1}/{config.num_iterations} Tier '{tier}'"
            print(f"[{label}] Running {config.games_per_iteration} games...")

            model_path, stats = run_si_iteration(
                iteration=iteration,
                agent_fn=current_agent,
                tier=tier,
                config=config,
            )

            summary = (
                f"[{label}] "
                f"{stats.wins}W/{stats.losses}L/{stats.draws}D "
                f"(wr={stats.win_rate:.1%}, "
                f"steps={stats.filtered_trajectories}, "
                f"loss={stats.train_loss:.4f}, "
                f"acc={stats.train_acc:.3f}, "
                f"{stats.elapsed_seconds:.1f}s)"
            )
            print(summary)
            log_lines.append(summary)
            all_stats.append(stats)

    overall_path = Path(config.output_dir) / "si_log.txt"
    with open(overall_path, "w") as f:
        f.write("\n".join(log_lines))

    return all_stats


def _prepare_train_records(trajectories: list[GameTrajectory]) -> list[dict]:
    records = []

    for traj in trajectories:
        for step in traj.steps:
            obs = to_observation_class(step.obs_dict)
            if obs.current is None or obs.select is None:
                continue

            state_features, option_features = encode_observation(obs)

            action = step.action
            num_options = step.num_options

            if not option_features or num_options == 0:
                continue

            valid_action = []
            for a in action:
                if 0 <= a < num_options:
                    valid_action.append(a)

            if not valid_action:
                continue

            record = {
                "state_features": state_features,
                "option_features": option_features,
                "action": valid_action,
                "reward": traj.reward,
                "outcome": traj.outcome,
                "my_index": step.my_index,
                "state_score": step.state_score,
                "turn": step.turn,
                "select_type": step.select_type,
                "select_context": step.select_context,
                "num_options": num_options,
            }
            records.append(record)

    return records


def _save_trajectory_records(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _collect_all_training_data(base_dir: Path, max_previous_iterations: int = 2) -> list[dict]:
    all_records = []

    iter_dirs = sorted(base_dir.glob("iteration_*"))
    recent_iters = iter_dirs[-max_previous_iterations:] if len(iter_dirs) > max_previous_iterations else iter_dirs

    for iter_dir in recent_iters:
        for tier_dir in iter_dir.glob("*"):
            records_path = tier_dir / "train_records.jsonl"
            if not records_path.exists():
                continue

            with open(records_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except (json.JSONDecodeError, KeyError):
                        continue

    return all_records


if __name__ == "__main__":
    from agent.main import agent

    config = SITrainConfig(
        num_iterations=2,
        games_per_iteration=20,
        difficulty_tiers=["easy", "medium"],
        output_dir="si_training/run_001",
        epochs=10,
        hidden_dim=128,
    )

    print("Starting iterative self-improvement training...")
    stats = run_SI_loop(agent, config)

    print("\n=== SI Training Summary ===")
    for s in stats:
        print(f"  Iter {s.iteration} Tier {s.tier}: WR={s.win_rate:.1%} Loss={s.train_loss:.4f} Acc={s.train_acc:.3f}")

    total_games = sum(s.num_games for s in stats)
    total_wins = sum(s.wins for s in stats)
    total_steps = sum(s.filtered_trajectories for s in stats)
    print(f"\nTotal: {total_games} games, {total_wins} wins, {total_steps} training steps")