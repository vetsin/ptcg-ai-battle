"""Phase B/C of MULTI_DECK_ENSEMBLE_PLAN.md: cross-archetype round-robin
self-play across the 8 trained specialists, then distill the pooled
trajectories into one unified model.

Phase A (per-deck specialist training via agent/ab_train.py) is done — see
the plan doc's status table. This script:

  1. (Phase B) Plays many games where two of the 8 archetypes are picked
     uniformly at random (with replacement — mirrors allowed) and face off
     using their OWN specialist checkpoint on each side. Both sides'
     trajectories are recorded from the same playthrough (agent/selfplay.py's
     `run_game(..., collect_both=True)`).
  2. (Phase C) Picks whichever specialist validates best as a quick warm-start
     baseline, then trains one model on the full pooled trajectory set.
  3. Verifies the unified model round-robin against every specialist and
     against `slightly_smart_agent`/`random_agent` on every deck.

Run with: python3 -m agent.ensemble_train
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import random
import time
import multiprocessing
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
torch.set_num_threads(1)

from agent.selfplay import GameTrajectory, run_game
from agent.opponents import slightly_smart_agent, random_agent
from agent.network import train_policy, load_model
from agent.ab_train import _trajectories_to_training_data

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "ensemble_training"


def _archetypes() -> list[tuple[str, list[int], str]]:
    """(name, deck, specialist_checkpoint_path) for all 8 trained archetypes."""
    from agent.decks.mega_lucario import DECK as mega_lucario_ex
    from agent.decks.crustle import DECK as crustle
    from agent.decks.raging_bolt_ex import DECK as raging_bolt_ex
    from agent.decks.archaludon_ex import DECK as archaludon_ex
    from agent.decks.terapagos_ex import DECK as terapagos_ex
    from agent.decks.dark_trinity_ex import DECK as dark_trinity_ex
    from agent.decks.greninja_ex import DECK as greninja_ex
    from agent.decks.dragapult_dusknoir import DECK as dragapult_dusknoir

    decks = {
        "mega_lucario_ex": mega_lucario_ex,
        "crustle": crustle,
        "raging_bolt_ex": raging_bolt_ex,
        "archaludon_ex": archaludon_ex,
        "terapagos_ex": terapagos_ex,
        "dark_trinity_ex": dark_trinity_ex,
        "greninja_ex": greninja_ex,
        "dragapult_dusknoir": dragapult_dusknoir,
    }
    return [
        (name, deck, str(MODELS_DIR / f"specialist_{name}.pt"))
        for name, deck in decks.items()
    ]


def _suppress_default_model_autoload() -> None:
    """agent.main.agent() lazily calls _ensure_models_loaded() once per
    process, which loads agent/policy.py + agent/search.py's default
    model.pt into BOTH player slots — clobbering whatever per-player
    specialists we just loaded ourselves. Must be called after our own
    load_policy_model/load_value_model calls, before the first agent_fn(...)
    invocation in this process."""
    import agent.main as main_mod
    main_mod._models_loaded = True


def _play_cross_archetype_game(args: tuple) -> dict:
    name_a, deck_a, model_a, name_b, deck_b, model_b, game_id, max_steps = args

    from agent.opponent_model import reset_opponent_models
    from agent.policy import load_policy_model
    from agent.search import load_value_model

    reset_opponent_models()

    load_policy_model(model_a, index=0)
    load_policy_model(model_b, index=1)
    load_value_model(model_a, index=0)
    load_value_model(model_b, index=1)
    _suppress_default_model_autoload()

    from agent.main import agent as agent_fn

    try:
        traj_a, traj_b = run_game(
            deck0=deck_a,
            deck1=deck_b,
            agent0_fn=agent_fn,
            agent1_fn=agent_fn,
            max_steps=max_steps,
            game_id=game_id,
            collect_trajectory=True,
            collect_both=True,
        )
    except Exception:
        traceback.print_exc()
        return {"outcome": -1, "name_a": name_a, "name_b": name_b, "traj_a": None, "traj_b": None}

    traj_a.opponent_name = name_b
    traj_b.opponent_name = name_a
    traj_a.reward = 1.0 if traj_a.outcome == 0 else (0.5 if traj_a.outcome == 2 else 0.0)
    traj_b.reward = 1.0 if traj_b.outcome == 1 else (0.5 if traj_b.outcome == 2 else 0.0)

    return {
        "outcome": traj_a.outcome,
        "name_a": name_a,
        "name_b": name_b,
        "traj_a": traj_a,
        "traj_b": traj_b,
    }


@dataclass
class PhaseBResult:
    trajectories: list[GameTrajectory]
    total_games: int
    errors: int
    pair_wins: dict  # (name_a, name_b) -> [wins_a, wins_b, draws]


def run_phase_b(
    num_games: int = 400,
    max_steps: int = 1500,
    num_workers: int = 8,
    seed: int | None = None,
) -> PhaseBResult:
    if seed is not None:
        random.seed(seed)

    archetypes = _archetypes()
    game_args = []
    for i in range(num_games):
        a = random.choice(archetypes)
        b = random.choice(archetypes)
        game_args.append((a[0], a[1], a[2], b[0], b[1], b[2], f"ens_{i:05d}", max_steps))

    print(f"  [phase B] {len(game_args)} cross-archetype games across {len(archetypes)} archetypes")

    if num_workers > 1 and len(game_args) > 1:
        try:
            ctx = multiprocessing.get_context("fork")
            workers = min(num_workers, len(game_args))
            print(f"  [parallel] Starting {workers} workers")
            with ctx.Pool(processes=workers) as pool:
                results = pool.map(_play_cross_archetype_game, game_args, chunksize=1)
        except Exception as e:
            print(f"  [parallel] Pool failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"  [parallel] Falling back to sequential")
            results = [_play_cross_archetype_game(args) for args in game_args]
    else:
        results = [_play_cross_archetype_game(args) for args in game_args]

    trajectories: list[GameTrajectory] = []
    errors = 0
    pair_wins: dict = {}

    for res in results:
        if res["traj_a"] is None:
            errors += 1
            continue
        trajectories.append(res["traj_a"])
        trajectories.append(res["traj_b"])

        key = tuple(sorted((res["name_a"], res["name_b"])))
        pair_wins.setdefault(key, {"a_wins": 0, "b_wins": 0, "draws": 0})
        if res["outcome"] == 2:
            pair_wins[key]["draws"] += 1
        elif res["outcome"] == -1:
            pass
        elif (res["outcome"] == 0) == (key[0] == res["name_a"]):
            pair_wins[key]["a_wins"] += 1
        else:
            pair_wins[key]["b_wins"] += 1

    return PhaseBResult(
        trajectories=trajectories,
        total_games=len(game_args),
        errors=errors,
        pair_wins=pair_wins,
    )


def _select_warm_start(archetypes: list[tuple[str, list[int], str]], num_games: int = 10, max_steps: int = 1500) -> str:
    """Quick validation pass: each specialist plays its own deck vs
    slightly_smart_agent for a few games using its own checkpoint loaded on
    both the policy and value side. Returns the checkpoint path with the
    best win rate, to warm-start Phase C training from (MULTI_DECK_ENSEMBLE_PLAN.md
    Phase C: "starting from whichever Phase A checkpoint validates best")."""
    from agent.policy import load_policy_model
    from agent.search import load_value_model

    best_path = archetypes[0][2]
    best_wr = -1.0

    for name, deck, model_path in archetypes:
        if not Path(model_path).exists():
            continue
        load_policy_model(model_path, index=0)
        load_value_model(model_path, index=0)
        _suppress_default_model_autoload()
        from agent.main import agent as agent_fn

        wins = 0
        for i in range(num_games):
            traj = run_game(
                deck0=deck, deck1=deck,
                agent0_fn=agent_fn, agent1_fn=slightly_smart_agent,
                max_steps=max_steps, game_id=f"warmstart_{name}_{i}",
                collect_trajectory=False,
            )
            if traj.outcome == 0:
                wins += 1
        wr = wins / num_games
        print(f"  [warm-start probe] {name}: {wr:.1%} vs slightly_smart_agent")
        if wr > best_wr:
            best_wr = wr
            best_path = model_path

    print(f"  [warm-start] selected {best_path} ({best_wr:.1%})")
    return best_path


def run_phase_c(
    trajectories: list[GameTrajectory],
    warm_start_path: str,
    output_path: str,
    hidden_dim: int = 256,
    epochs: int = 20,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    records = _trajectories_to_training_data(trajectories, min_steps=3)
    print(f"  [phase C] {len(records)} training samples from {len(trajectories)} trajectories")

    if len(records) < 10:
        print("  [phase C] not enough samples, skipping training")
        return {"train_loss": [0.0], "train_acc": [0.0]}

    history = train_policy(
        train_data=records,
        model_type="simple",
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        save_path=output_path,
        warm_start_path=warm_start_path,
    )
    if history.get("train_loss"):
        print(f"  [phase C] final loss={history['train_loss'][-1]:.4f} acc={history['train_acc'][-1]:.3f}")
    return history


def _verify_unified_model(unified_path: str, archetypes: list[tuple[str, list[int], str]], num_games: int = 10, max_steps: int = 1500) -> dict:
    """Round-robin the unified model against (a) each specialist on its own
    deck, (b) slightly_smart_agent, (c) random_agent — per the plan's
    verification step."""
    from agent.policy import load_policy_model
    from agent.search import load_value_model

    report: dict = {}

    for name, deck, specialist_path in archetypes:
        load_policy_model(unified_path, index=0)
        load_value_model(unified_path, index=0)

        vs_results = {}

        for opp_label, opp_fn, opp_index_setup in [
            ("slightly_smart_agent", slightly_smart_agent, None),
            ("random_agent", random_agent, None),
        ]:
            _suppress_default_model_autoload()
            from agent.main import agent as agent_fn
            wins = 0
            for i in range(num_games):
                traj = run_game(
                    deck0=deck, deck1=deck,
                    agent0_fn=agent_fn, agent1_fn=opp_fn,
                    max_steps=max_steps, game_id=f"verify_{name}_{opp_label}_{i}",
                    collect_trajectory=False,
                )
                if traj.outcome == 0:
                    wins += 1
            vs_results[opp_label] = wins / num_games

        if Path(specialist_path).exists():
            load_policy_model(unified_path, index=0)
            load_value_model(unified_path, index=0)
            load_policy_model(specialist_path, index=1)
            load_value_model(specialist_path, index=1)
            _suppress_default_model_autoload()
            from agent.main import agent as agent_fn
            wins = 0
            for i in range(num_games):
                traj = run_game(
                    deck0=deck, deck1=deck,
                    agent0_fn=agent_fn, agent1_fn=agent_fn,
                    max_steps=max_steps, game_id=f"verify_{name}_specialist_{i}",
                    collect_trajectory=False,
                )
                if traj.outcome == 0:
                    wins += 1
            vs_results["own_specialist"] = wins / num_games

        report[name] = vs_results
        print(f"  [verify] {name}: " + ", ".join(f"{k}={v:.1%}" for k, v in vs_results.items()))

    return report


def run_ensemble_training(
    num_games_phase_b: int = 400,
    max_steps: int = 1500,
    num_workers: int = 8,
    hidden_dim: int = 256,
    epochs: int = 20,
    verify_games: int = 10,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archetypes = _archetypes()

    print("=" * 60)
    print("Phase B — cross-archetype round-robin self-play")
    print("=" * 60)
    t0 = time.time()
    phase_b_result = run_phase_b(num_games=num_games_phase_b, max_steps=max_steps, num_workers=num_workers)
    print(f"  Phase B done: {phase_b_result.total_games} games, {phase_b_result.errors} errors, "
          f"{len(phase_b_result.trajectories)} trajectories ({time.time() - t0:.1f}s)")
    for pair, rec in sorted(phase_b_result.pair_wins.items()):
        print(f"    {pair[0]} vs {pair[1]}: a={rec['a_wins']} b={rec['b_wins']} draw={rec['draws']}")

    print()
    print("=" * 60)
    print("Phase C — unification distillation")
    print("=" * 60)
    warm_start_path = _select_warm_start(archetypes)
    unified_path = str(OUTPUT_DIR / "unified_model.pt")
    run_phase_c(
        phase_b_result.trajectories,
        warm_start_path=warm_start_path,
        output_path=unified_path,
        hidden_dim=hidden_dim,
        epochs=epochs,
    )

    print()
    print("=" * 60)
    print("Verification — unified model round-robin")
    print("=" * 60)
    report = _verify_unified_model(unified_path, archetypes, num_games=verify_games, max_steps=max_steps)

    with open(OUTPUT_DIR / "verification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Unified model saved to: {unified_path}")
    print(f"Verification report saved to: {OUTPUT_DIR / 'verification_report.json'}")


if __name__ == "__main__":
    run_ensemble_training()
