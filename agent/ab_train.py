from __future__ import annotations

import json
import os
import time
import random
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from agent.selfplay import run_game, GameTrajectory, SelfPlayResult
from agent.opponents import slightly_smart_agent, load_competitive_decks
from agent.deck import load_deck_csv
from agent.network import train_policy, PTCGDataset, collate_fn
from agent.features import encode_observation, NUM_STATE_FEATURES, NUM_OPTION_FEATURES
from agent.evaluate import evaluate_state
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish


@dataclass
class ABTrainConfig:
    num_iterations: int = 10
    games_per_iteration: int = 100
    validation_games: int = 20
    champion_threshold: float = 0.55
    mirror_ratio: float = 0.3
    min_trajectory_steps: int = 3
    model_type: str = "simple"
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    max_options: int = 20
    device: str = "auto"
    max_steps_per_game: int = 2000
    output_dir: str = "ab_training"


@dataclass
class ABIterationStats:
    iteration: int
    num_games: int
    wins: int
    losses: int
    draws: int
    errors: int
    win_rate: float
    avg_reward: float
    avg_turns: float
    training_samples: int
    champion_wr: float
    challenger_wr: float
    promoted: bool
    train_loss: float = 0.0
    train_acc: float = 0.0
    elapsed_seconds: float = 0.0


def _play_games_multi_deck(
    agent_fn,
    our_deck: list[int],
    opponent_decks: list[tuple[str, list[int]]],
    num_games: int,
    collect_trajectory: bool = True,
    max_steps: int = 2000,
    use_self_play: bool = True,
) -> SelfPlayResult:
    result = SelfPlayResult()

    mirror_games = int(num_games * 0.3)
    per_deck_games = (num_games - mirror_games) // max(len(opponent_decks), 1)
    remaining = num_games - mirror_games - per_deck_games * len(opponent_decks)

    game_schedule = []

    for _ in range(mirror_games):
        game_schedule.append(("mirror", our_deck))

    for name, deck in opponent_decks:
        for _ in range(per_deck_games):
            game_schedule.append((name, deck))

    for i in range(remaining):
        if opponent_decks:
            name, deck = opponent_decks[i % len(opponent_decks)]
            game_schedule.append((name, deck))

    random.shuffle(game_schedule)

    for g_idx, (opp_name, opp_deck) in enumerate(game_schedule):
        from agent.opponent_model import reset_opponent_models
        reset_opponent_models()

        traj_player = g_idx % 2

        if use_self_play:
            agent0_fn = agent_fn
            agent1_fn = agent_fn
        else:
            if traj_player == 0:
                agent0_fn = agent_fn
                agent1_fn = slightly_smart_agent
            else:
                agent0_fn = slightly_smart_agent
                agent1_fn = agent_fn

        if traj_player == 0:
            deck0 = our_deck
            deck1 = opp_deck
        else:
            deck0 = opp_deck
            deck1 = our_deck

        game_id = f"ab_{g_idx:04d}_{int(time.time())}"

        trajectory = run_game(
            deck0=deck0,
            deck1=deck1,
            agent0_fn=agent0_fn,
            agent1_fn=agent1_fn,
            max_steps=max_steps,
            game_id=game_id,
            collect_trajectory=collect_trajectory,
            trajectory_player=traj_player,
        )

        result.total_games += 1

        if trajectory.outcome == traj_player:
            result.wins += 1
            trajectory.reward = 1.0
        elif trajectory.outcome == 1 - traj_player:
            result.losses += 1
            trajectory.reward = 0.0
        elif trajectory.outcome == 2:
            result.draws += 1
            trajectory.reward = 0.5
        else:
            result.errors += 1
            trajectory.reward = 0.0

        trajectory.my_index = traj_player

        if collect_trajectory:
            result.games.append(trajectory)

    valid = result.wins + result.losses + result.draws
    if valid > 0:
        result.avg_turns = sum(g.num_turns for g in result.games) / valid
        result.avg_reward = sum(g.reward for g in result.games) / valid

    return result


def _trajectories_to_training_data(
    trajectories: list[GameTrajectory],
    min_steps: int = 3,
) -> list[dict]:
    records = []

    for traj in trajectories:
        if len(traj.steps) < min_steps:
            continue

        won = traj.outcome == traj.my_index

        for step in traj.steps:
            obs = to_observation_class(step.obs_dict)
            if obs.current is None or obs.select is None:
                continue

            state_features, option_features = encode_observation(obs)

            if not state_features or not option_features:
                continue

            action = step.action
            num_options = step.num_options

            valid_action = []
            for a in action:
                if 0 <= a < num_options:
                    valid_action.append(a)

            if not valid_action:
                continue

            outcome_label = 1.0 if won else 0.0

            record = {
                "state_features": state_features,
                "option_features": option_features,
                "action": valid_action,
                "reward": outcome_label,
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


def _validate_challenger(
    champion_fn,
    challenger_fn,
    our_deck: list[int],
    opponent_decks: list[tuple[str, list[int]]],
    num_games: int,
    max_steps: int = 2000,
) -> float:
    wins = 0
    total = 0

    for i in range(num_games):
        from agent.opponent_model import reset_opponent_models
        reset_opponent_models()

        deck_idx = i % max(len(opponent_decks), 1)
        opp_name, opp_deck = opponent_decks[deck_idx] if opponent_decks else ("mirror", our_deck)

        if i % 2 == 0:
            deck0 = our_deck
            deck1 = opp_deck
            agent0_fn = challenger_fn
            agent1_fn = champion_fn
            challenger_side = 0
        else:
            deck0 = opp_deck
            deck1 = our_deck
            agent0_fn = champion_fn
            agent1_fn = challenger_fn
            challenger_side = 1

        trajectory = run_game(
            deck0=deck0,
            deck1=deck1,
            agent0_fn=agent0_fn,
            agent1_fn=agent1_fn,
            max_steps=max_steps,
            game_id=f"val_{i:04d}",
            collect_trajectory=False,
        )

        total += 1
        if trajectory.outcome == challenger_side:
            wins += 1

    return wins / max(total, 1)


def run_ab_training(
    agent_fn,
    config: ABTrainConfig | None = None,
) -> list[ABIterationStats]:
    if config is None:
        config = ABTrainConfig()

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    our_deck = load_deck_csv()

    comp_decks_data = load_competitive_decks()
    opponent_decks = [
        (d["name"], d["deck"]) for d in comp_decks_data
    ]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    champion_model_path = None
    champion_wins = 0
    champion_total = 0

    from agent.policy import load_policy_model

    for iteration in range(config.num_iterations):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"AB Training — Iteration {iteration + 1}/{config.num_iterations}")
        print(f"{'='*60}")

        current_agent_fn = agent_fn

        result = _play_games_multi_deck(
            agent_fn=current_agent_fn,
            our_deck=our_deck,
            opponent_decks=opponent_decks,
            num_games=config.games_per_iteration,
            collect_trajectory=True,
            max_steps=config.max_steps_per_game,
            use_self_play=True,
        )

        win_rate = result.wins / max(result.total_games, 1)
        print(f"  Self-play: {result.wins}W/{result.losses}L/{result.draws}D "
              f"WR={win_rate:.1%} avg_turns={result.avg_turns:.1f}")

        all_trajectories = result.games
        winning = [g for g in all_trajectories if g.reward >= 1.0]
        losing = [g for g in all_trajectories if g.reward <= 0.0]

        print(f"  Trajectories: {len(winning)} wins, {len(losing)} losses, "
              f"{len(all_trajectories) - len(winning) - len(losing)} draws")

        all_records = _trajectories_to_training_data(
            all_trajectories,
            min_steps=config.min_trajectory_steps,
        )

        print(f"  Training samples: {len(all_records)}")

        iter_dir = output_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        model_path = None
        train_loss = 0.0
        train_acc = 0.0

        if len(all_records) >= 10:
            print(f"  Training model on {len(all_records)} samples...")

            model_path = str(iter_dir / "challenger.pt")

            history = train_policy(
                train_data=all_records,
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
            print(f"  Train: loss={train_loss:.4f} acc={train_acc:.3f}")

            challenger_fn = _make_challenger_fn(model_path, config.device)

            print(f"  Validating challenger vs champion...")
            challenger_wr = _validate_challenger(
                champion_fn=current_agent_fn,
                challenger_fn=challenger_fn,
                our_deck=our_deck,
                opponent_decks=opponent_decks,
                num_games=config.validation_games,
                max_steps=config.max_steps_per_game,
            )

            print(f"  Challenger WR vs champion: {challenger_wr:.1%}")

            if challenger_wr >= config.champion_threshold:
                print(f"  ✓ Challenger promoted! ({challenger_wr:.1%} >= {config.champion_threshold:.1%})")
                champion_model_path = model_path
                load_policy_model(model_path)
                promoted = True
            else:
                print(f"  ✗ Challenger not promoted ({challenger_wr:.1%} < {config.champion_threshold:.1%})")
                promoted = False
        else:
            print(f"  Insufficient data ({len(all_records)} records), skipping training")
            challenger_wr = 0.0
            promoted = False

        elapsed = time.time() - t0

        stats = ABIterationStats(
            iteration=iteration,
            num_games=result.total_games,
            wins=result.wins,
            losses=result.losses,
            draws=result.draws,
            errors=result.errors,
            win_rate=win_rate,
            avg_reward=result.avg_reward,
            avg_turns=result.avg_turns,
            training_samples=len(all_records),
            champion_wr=win_rate,
            challenger_wr=challenger_wr,
            promoted=promoted,
            train_loss=train_loss,
            train_acc=train_acc,
            elapsed_seconds=elapsed,
        )

        all_stats.append(stats)

        stats_path = iter_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(asdict(stats), f, indent=2)

        traj_path = iter_dir / "trajectories.jsonl"
        _save_trajectories(all_trajectories, traj_path)

        print(f"  Elapsed: {elapsed:.1f}s")

    final_model = champion_model_path or str(output_dir / "final_model.pt")
    if champion_model_path:
        import shutil
        shutil.copy2(champion_model_path, final_model)
        print(f"\nFinal champion model: {final_model}")
    else:
        print(f"\nNo champion was promoted. Original agent remains.")

    overall_path = output_dir / "ab_log.json"
    with open(overall_path, "w") as f:
        json.dump([asdict(s) for s in all_stats], f, indent=2)

    return all_stats


def _make_challenger_fn(model_path: str, device: str = "cpu"):
    from agent.network import load_model
    from agent.features import encode_observation
    from agent.policy import choose_action

    model = load_model(model_path, device=device)
    model.eval()

    def challenger_fn(obs_dict: dict) -> list[int]:
        return choose_action(to_observation_class(obs_dict))

    return challenger_fn


def _save_trajectories(trajectories: list[GameTrajectory], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
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
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    from agent.main import agent

    config = ABTrainConfig(
        num_iterations=6,
        games_per_iteration=100,
        validation_games=20,
        champion_threshold=0.55,
        hidden_dim=256,
        epochs=20,
        device="auto",
    )

    print("Starting A/B self-play training...")
    stats = run_ab_training(agent, config)

    print("\n=== A/B Training Summary ===")
    for s in stats:
        promoted = "✓" if s.promoted else "✗"
        print(f"  Iter {s.iteration}: WR={s.win_rate:.1%} "
              f"samples={s.training_samples} "
              f"challenger={s.challenger_wr:.1%} "
              f"{promoted} loss={s.train_loss:.4f} acc={s.train_acc:.3f} "
              f"({s.elapsed_seconds:.1f}s)")

    total_games = sum(s.num_games for s in stats)
    total_wins = sum(s.wins for s in stats)
    promoted_count = sum(1 for s in stats if s.promoted)
    print(f"\nTotal: {total_games} games, {total_wins} wins, "
          f"{promoted_count} promotions")