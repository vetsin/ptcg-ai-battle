from __future__ import annotations

import json
import os
import time
import random
import multiprocessing
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from agent.selfplay import GameTrajectory, SelfPlayResult
from agent.opponents import slightly_smart_agent, load_competitive_decks
from agent.deck import load_deck_csv
from agent.network import train_policy
from agent.features import encode_observation, NUM_STATE_FEATURES, NUM_OPTION_FEATURES
from cg.api import to_observation_class


def _run_single_game(args: tuple) -> dict:
    deck0, deck1, agent0_type, agent1_type, game_id, max_steps, collect_trajectory, traj_player, use_mcts = args

    from agent.opponent_model import reset_opponent_models
    from agent.opponents import slightly_smart_agent

    reset_opponent_models()

    if use_mcts:
        from agent.main import agent as agent_fn
    else:
        agent_fn = slightly_smart_agent

    fn0 = agent_fn if agent0_type == "agent" else slightly_smart_agent
    fn1 = agent_fn if agent1_type == "agent" else slightly_smart_agent

    from agent.selfplay import run_game

    trajectory = run_game(
        deck0=deck0,
        deck1=deck1,
        agent0_fn=fn0,
        agent1_fn=fn1,
        max_steps=max_steps,
        game_id=game_id,
        collect_trajectory=collect_trajectory,
        trajectory_player=traj_player,
    )

    result = {
        "outcome": trajectory.outcome,
        "reward": 0.0,
        "num_turns": trajectory.num_turns,
        "game_id": trajectory.game_id,
        "my_index": trajectory.my_index,
        "my_prize_taken": trajectory.my_prize_taken,
        "opp_prize_taken": trajectory.opp_prize_taken,
        "traj_player": traj_player,
        "steps_count": len(trajectory.steps),
    }

    if trajectory.outcome == traj_player:
        result["reward"] = 1.0
    elif trajectory.outcome == 1 - traj_player:
        result["reward"] = 0.0
    elif trajectory.outcome == 2:
        result["reward"] = 0.5

    if collect_trajectory:
        result["trajectory"] = trajectory
    else:
        result["trajectory"] = None

    return result


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
    num_workers: int = 1
    use_mcts_agent: bool = False


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
    num_workers: int = 1,
    use_mcts_agent: bool = False,
) -> SelfPlayResult:
    result = SelfPlayResult()

    mirror_games = int(num_games * 0.3)
    per_deck_games = (num_games - mirror_games) // max(len(opponent_decks), 1)
    remaining = num_games - mirror_games - per_deck_games * len(opponent_decks)

    game_args = []

    for i in range(mirror_games):
        traj_player = i % 2
        if traj_player == 0:
            deck0, deck1 = our_deck, our_deck
        else:
            deck0, deck1 = our_deck, our_deck
        agent0_type = "agent" if use_self_play else ("agent" if traj_player == 0 else "smart")
        agent1_type = "agent" if use_self_play else ("smart" if traj_player == 0 else "agent")
        game_args.append((deck0, deck1, agent0_type, agent1_type, f"ab_{i:04d}", max_steps, collect_trajectory, traj_player, use_mcts_agent))

    offset = mirror_games
    for name, deck in opponent_decks:
        for j in range(per_deck_games):
            idx = offset + j
            traj_player = idx % 2
            if traj_player == 0:
                deck0, deck1 = our_deck, deck
            else:
                deck0, deck1 = deck, our_deck
            agent0_type = "agent" if use_self_play else ("agent" if traj_player == 0 else "smart")
            agent1_type = "agent" if use_self_play else ("smart" if traj_player == 0 else "agent")
            game_args.append((deck0, deck1, agent0_type, agent1_type, f"ab_{idx:04d}", max_steps, collect_trajectory, traj_player, use_mcts_agent))
        offset += per_deck_games

    for i in range(remaining):
        if opponent_decks:
            name, deck = opponent_decks[i % len(opponent_decks)]
            idx = offset + i
            traj_player = idx % 2
            if traj_player == 0:
                deck0, deck1 = our_deck, deck
            else:
                deck0, deck1 = deck, our_deck
            agent0_type = "agent" if use_self_play else ("agent" if traj_player == 0 else "smart")
            agent1_type = "agent" if use_self_play else ("smart" if traj_player == 0 else "agent")
            game_args.append((deck0, deck1, agent0_type, agent1_type, f"ab_{idx:04d}", max_steps, collect_trajectory, traj_player, use_mcts_agent))

    random.shuffle(game_args)

    if num_workers > 1 and len(game_args) > 1:
        try:
            ctx = multiprocessing.get_context("fork")
            workers = min(num_workers, len(game_args))
            print(f"  [parallel] Starting {workers} workers for {len(game_args)} games")
            with ctx.Pool(processes=workers) as pool:
                results = pool.map(_run_single_game, game_args, chunksize=1)
        except Exception as e:
            print(f"  [parallel] Pool failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"  [parallel] Falling back to sequential")
            results = [_run_single_game(args) for args in game_args]
    else:
        results = [_run_single_game(args) for args in game_args]

    for res in results:
        result.total_games += 1
        reward = res["reward"]

        if reward >= 1.0:
            result.wins += 1
        elif reward <= 0.0:
            result.losses += 1
        elif reward > 0.0:
            result.draws += 1
        else:
            result.errors += 1

        if collect_trajectory and res["trajectory"] is not None:
            traj = res["trajectory"]
            traj.reward = reward
            traj.my_index = res["traj_player"]
            result.games.append(traj)

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
    num_workers: int = 1,
    champion_model_path: str | None = None,
) -> float:
    game_args = []

    for i in range(num_games):
        deck_idx = i % max(len(opponent_decks), 1)
        opp_name, opp_deck = opponent_decks[deck_idx] if opponent_decks else ("mirror", our_deck)

        if i % 2 == 0:
            deck0 = our_deck
            deck1 = opp_deck
            agent0_type = "challenger"
            agent1_type = "champion"
            challenger_side = 0
        else:
            deck0 = opp_deck
            deck1 = our_deck
            agent0_type = "champion"
            agent1_type = "challenger"
            challenger_side = 1

        game_args.append((deck0, deck1, agent0_type, agent1_type, f"val_{i:04d}", max_steps, False, challenger_side))

    val_fn = lambda args: _val_game(args, champion_model_path)

    if num_workers > 1 and len(game_args) > 1:
        try:
            ctx = multiprocessing.get_context("fork")
            workers = min(num_workers, len(game_args))
            with ctx.Pool(processes=workers) as pool:
                results = pool.map(val_fn, game_args)
        except Exception:
            results = [val_fn(args) for args in game_args]
    else:
        results = [val_fn(args) for args in game_args]

    wins = sum(1 for r in results if r["challenger_won"])
    total = len(results)
    return wins / max(total, 1)


def _val_game(game_args, champion_model_path: str | None):
    deck0, deck1, agent0_type, agent1_type, game_id, max_steps, _, traj_player = game_args

    from agent.opponent_model import reset_opponent_models
    from agent.selfplay import run_game

    if champion_model_path:
        from agent.policy import load_policy_model
        load_policy_model(champion_model_path)

    from agent.main import agent as agent_fn
    from agent.opponents import slightly_smart_agent

    reset_opponent_models()

    if agent0_type == "champion":
        fn0 = slightly_smart_agent
        fn1 = agent_fn
    else:
        fn0 = agent_fn
        fn1 = slightly_smart_agent

    trajectory = run_game(
        deck0=deck0,
        deck1=deck1,
        agent0_fn=fn0,
        agent1_fn=fn1,
        max_steps=max_steps,
        game_id=game_id,
        collect_trajectory=False,
    )

    return {"challenger_won": trajectory.outcome == traj_player}


def run_ab_training(
    agent_fn,
    config: ABTrainConfig | None = None,
) -> list[ABIterationStats]:
    if config is None:
        config = ABTrainConfig()

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    our_deck = load_deck_csv()

    from agent.deck import validate_deck_for_training

    our_errors = validate_deck_for_training(our_deck, name="our_deck")
    if our_errors:
        raise ValueError(f"Our deck failed pre-flight validation: {our_errors}")

    comp_decks_data = load_competitive_decks()
    opponent_decks = []
    for d in comp_decks_data:
        errors = validate_deck_for_training(d["deck"], name=d["name"])
        if errors:
            print(f"  [validation] Skipping '{d['name']}': {errors}")
            continue
        opponent_decks.append((d["name"], d["deck"]))

    print(f"  [validation] {len(opponent_decks)}/{len(comp_decks_data)} opponent decks passed pre-flight check")

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
            num_workers=config.num_workers,
            use_mcts_agent=config.use_mcts_agent,
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
                num_workers=config.num_workers,
                champion_model_path=model_path,
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

    final_model = str(output_dir / "final_model.pt")
    if champion_model_path:
        import shutil
        if os.path.abspath(champion_model_path) != os.path.abspath(final_model):
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
        games_per_iteration=200,
        validation_games=20,
        champion_threshold=0.55,
        hidden_dim=256,
        epochs=20,
        device="auto",
        num_workers=8,
        use_mcts_agent=False,
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