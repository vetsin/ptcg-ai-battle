#!/usr/bin/env python3
"""Kaggle smoke test for PTCG AI Battle agent.

Validates the agent works correctly in the Kaggle simulation environment.
Run this before submitting to catch issues early.

Usage:
    python smoke_test.py
    python smoke_test.py --verbose
    python smoke_test.py --quick       # Skip slow tests
"""

import sys
import time
import traceback
from dataclasses import dataclass

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    elapsed: float = 0.0

results: list[TestResult] = []


def run_test(name: str, func, verbose: bool = False):
    """Run a single test and record results."""
    t0 = time.time()
    try:
        msg = func(verbose=verbose)
        elapsed = time.time() - t0
        results.append(TestResult(name, True, msg or "OK", elapsed))
        print(f"  [PASS] {name}: {msg or 'OK'} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        results.append(TestResult(name, False, str(e), elapsed))
        print(f"  [FAIL] {name}: {e}")
        if verbose:
            traceback.print_exc()
    return results[-1]


# ── Tests ──

def test_import(verbose=False):
    from main import agent
    assert callable(agent), "agent must be callable"
    return "agent function imported"


def test_deck_return(verbose=False):
    from main import agent
    result = agent({})
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 60, f"Expected 60 cards, got {len(result)}"
    assert all(isinstance(c, int) for c in result), "All cards must be int"
    assert all(c > 0 for c in result), "Card IDs must be positive"
    return f"{len(result)} cards returned"


def test_deck_csv_format(verbose=False):
    with open("deck.csv") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 60, f"deck.csv must have 60 lines, got {len(lines)}"
    assert all(l.isdigit() for l in lines), "All lines must be integers"
    return f"{len(lines)} valid card IDs"


def test_model_loads(verbose=False):
    import agent.policy as policy_mod
    policy_mod.load_policy_model("model.pt")
    assert policy_mod._policy_enabled, "Model failed to load"
    return f"model loaded (device={policy_mod._policy_device})"


def test_full_game(verbose=False):
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class
    from main import agent
    from agent.deck import load_deck_csv

    deck = load_deck_csv()
    n_games = 3 if verbose else 1
    winners = []

    for g in range(n_games):
        obs_dict, _ = battle_start(deck, deck)
        obs = to_observation_class(obs_dict)
        steps = 0
        for step in range(500):
            if obs.current is None or (obs.current and obs.current.result != -1):
                break
            if obs.select is None:
                obs_dict = battle_select(deck)
                obs = to_observation_class(obs_dict)
                continue
            action = agent(obs_dict)
            obs_dict = battle_select(action)
            obs = to_observation_class(obs_dict)
            steps += 1
        result = obs.current.result if obs.current else -1
        winners.append(result)
        battle_finish()

    return f"{n_games} games completed, {steps} steps last game"


def test_action_validation(verbose=False):
    """Test that all actions satisfy minCount/maxCount/index bounds."""
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class, SelectType
    from main import agent
    from agent.deck import load_deck_csv

    deck = load_deck_csv()
    obs_dict, _ = battle_start(deck, deck)
    obs = to_observation_class(obs_dict)
    violations = []
    steps = 0

    for step in range(300):
        if obs.current is None or (obs.current and obs.current.result != -1):
            break
        if obs.select is None:
            obs_dict = battle_select(deck)
            obs = to_observation_class(obs_dict)
            continue

        n_opts = len(obs.select.option) if obs.select and obs.select.option else 0
        min_c = obs.select.minCount if obs.select else 0
        max_c = obs.select.maxCount if obs.select else 0

        action = agent(obs_dict)

        if not isinstance(action, list):
            violations.append(f"Step {step}: action is {type(action)}, not list")
            break
        if max_c == 0 and len(action) != 0:
            violations.append(f"Step {step}: expected empty, got {action}")
            break
        if max_c > 0 and len(action) > max_c:
            violations.append(f"Step {step}: {len(action)} > maxCount {max_c}")
        if min_c > 0 and len(action) < min_c and n_opts >= min_c:
            violations.append(f"Step {step}: {len(action)} < minCount {min_c}")
        if any(i < 0 or i >= n_opts for i in action):
            violations.append(f"Step {step}: indices out of range [0, {n_opts})")

        obs_dict = battle_select(action)
        obs = to_observation_class(obs_dict)
        steps += 1

    battle_finish()
    assert not violations, f"{len(violations)} violations: {violations[:3]}"
    return f"{steps} steps, all actions valid"


def test_select_types(verbose=False):
    """Test that all SelectType variants are handled without crashing."""
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class, SelectType
    from main import agent
    from agent.deck import load_deck_csv

    deck = load_deck_csv()
    obs_dict, _ = battle_start(deck, deck)
    obs = to_observation_class(obs_dict)
    seen_types = set()

    for step in range(300):
        if obs.current is None or (obs.current and obs.current.result != -1):
            break
        if obs.select is None:
            obs_dict = battle_select(deck)
            obs = to_observation_class(obs_dict)
            continue
        if obs.select:
            seen_types.add(obs.select.type)
        action = agent(obs_dict)
        obs_dict = battle_select(action)
        obs = to_observation_class(obs_dict)

    battle_finish()
    type_names = {0:'MAIN',1:'CARD',2:'ATTACHED',3:'CARD_OR_ATTACHED',4:'ENERGY',
                  5:'SKILL',6:'ATTACK',7:'EVOLVE',8:'COUNT',9:'YES_NO',10:'SPECIAL'}
    names = sorted([type_names.get(t, str(t)) for t in seen_types])
    return f"handled: {names}"


def test_vs_random(verbose=False):
    """Test win rate against random agent."""
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class
    from main import agent
    from agent.opponents import random_agent
    from agent.deck import load_deck_csv
    from agent.opponent_model import reset_opponent_models

    deck = load_deck_csv()
    n_games = 10 if verbose else 10
    wins = 0
    total = 0

    for g in range(n_games):
        agent_player = g % 2
        reset_opponent_models()
        obs_dict, _ = battle_start(deck, deck)
        obs = to_observation_class(obs_dict)

        for step in range(500):
            if obs.current is None or (obs.current and obs.current.result != -1):
                break
            if obs.select is None:
                obs_dict = battle_select(deck)
                obs = to_observation_class(obs_dict)
                continue
            current = obs.current.yourIndex if obs.current else 0
            fn = agent if current == agent_player else random_agent
            action = fn(obs_dict)
            obs_dict = battle_select(action)
            obs = to_observation_class(obs_dict)

        result = obs.current.result if obs.current else -1
        if result == agent_player:
            wins += 1
        total += 1
        battle_finish()

    wr = wins / total * 100
    assert wr >= 20, f"Win rate {wr:.0f}% is below 20%"
    return f"WR={wr:.0f}% ({wins}/{total})"


def test_submission_files(verbose=False):
    """Check all required submission files exist."""
    import os
    required = [
        "main.py", "deck.csv",
        "cg/__init__.py", "cg/api.py", "cg/game.py", "cg/sim.py", "cg/utils.py",
        "cg/cg.dll", "cg/libcg.so",
        "agent/__init__.py", "agent/main.py", "agent/policy.py",
        "agent/search.py", "agent/evaluate.py", "agent/features.py",
        "agent/network.py", "agent/deck.py",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    assert not missing, f"Missing files: {missing}"
    return f"{len(required)} files present"


# ── Main ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PTCG AI Battle smoke test")
    parser.add_argument("--verbose", "-v", action="store_true", help="More test iterations")
    parser.add_argument("--quick", "-q", action="store_true", help="Skip slow tests")
    args = parser.parse_args()

    print("=" * 60)
    print("PTCG AI Battle — Kaggle Smoke Test")
    print("=" * 60)

    t_start = time.time()

    run_test("1. Submission files", test_submission_files, args.verbose)
    run_test("2. Import agent", test_import, args.verbose)
    run_test("3. Deck return (60 cards)", test_deck_return, args.verbose)
    run_test("4. Deck CSV format", test_deck_csv_format, args.verbose)
    run_test("5. Model loads", test_model_loads, args.verbose)

    if not args.quick:
        run_test("6. Full game", test_full_game, args.verbose)
        run_test("7. Action validation", test_action_validation, args.verbose)
        run_test("8. Select types", test_select_types, args.verbose)
        run_test("9. vs Random agent", test_vs_random, args.verbose)
    else:
        print("  [SKIP] Slow tests (--quick mode)")

    elapsed = time.time() - t_start
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    print("-" * 60)

    if failed > 0:
        print("FAILED — fix errors before submitting")
        for r in results:
            if not r.passed:
                print(f"  ✗ {r.name}: {r.message}")
        sys.exit(1)
    else:
        print("ALL PASSED — ready to submit")
        sys.exit(0)


if __name__ == "__main__":
    main()