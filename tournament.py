#!/usr/bin/env python3
"""Tournament: our agent vs all 8 NAIC 2026 competitive decks.

Usage:
    python tournament.py                  # default: 10 games per matchup
    python tournament.py --games 20        # 20 games per matchup
    python tournament.py --games 5 --quick # quick test run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent.selfplay import run_game, SelfPlayResult
from agent.opponents import slightly_smart_agent, random_agent
from agent.opponent_model import reset_opponent_models
from agent.deck import load_deck_csv


def load_competitive_decks() -> list[dict]:
    p = Path(__file__).parent / "competitive_decks.json"
    with open(p) as f:
        return json.load(f)


def run_tournament(
    agent_fn,
    our_deck: list[int],
    games_per_matchup: int = 10,
    opponent_fn=None,
):
    if opponent_fn is None:
        opponent_fn = slightly_smart_agent

    decks = load_competitive_decks()
    results = {}
    total_wins = 0
    total_games = 0

    for deck_info in decks:
        name = deck_info["name"]
        placing = deck_info["placing"]
        opp_deck = deck_info["deck"]
        label = f"{name} (#{placing})"

        wins = 0
        played = 0
        draws = 0
        errors = 0

        for i in range(games_per_matchup):
            reset_opponent_models()

            if i % 2 == 0:
                traj = run_game(
                    our_deck, opp_deck,
                    agent_fn, opponent_fn,
                    max_steps=2000,
                    collect_trajectory=False,
                )
                if traj.outcome == 0:
                    wins += 1
                elif traj.outcome == 2:
                    draws += 1
                elif traj.outcome == -1:
                    errors += 1
            else:
                traj = run_game(
                    opp_deck, our_deck,
                    opponent_fn, agent_fn,
                    max_steps=2000,
                    collect_trajectory=False,
                )
                if traj.outcome == 1:
                    wins += 1
                elif traj.outcome == 2:
                    draws += 1
                elif traj.outcome == -1:
                    errors += 1
            played += 1

        wr = wins / played if played > 0 else 0
        results[label] = {
            "wins": wins,
            "draws": draws,
            "errors": errors,
            "played": played,
            "win_rate": wr,
        }
        total_wins += wins
        total_games += played
        status = "W" if wr > 0.5 else ("L" if wr < 0.4 else "~")
        print(f"  vs {label:30s} WR={wr:.1%} ({wins}/{played}) {status}")

    overall_wr = total_wins / total_games if total_games > 0 else 0
    results["_overall"] = {
        "wins": total_wins,
        "played": total_games,
        "win_rate": overall_wr,
    }
    print(f"\n  Overall: WR={overall_wr:.1%} ({total_wins}/{total_games})")

    out_path = Path(__file__).parent / "tournament_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10, help="Games per matchup")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Competitive Deck Tournament — {args.games} games/matchup")
    print("=" * 60)

    t0 = time.time()

    from agent.main import agent
    our_deck = load_deck_csv()

    results = run_tournament(agent, our_deck, games_per_matchup=args.games)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")