from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path

from agent.deck import validate_deck, load_deck_csv
from agent.selfplay import run_game, SelfPlayResult
from agent.opponents import random_agent, slightly_smart_agent, get_deck_for_tier
from cg.api import all_card_data, CardType

_card_db = None


def _get_card_db():
    global _card_db
    if _card_db is None:
        _card_db = {c.cardId: c for c in all_card_data()}
    return _card_db


BASE_DECK = [
    (46, 4),     # Gouging Fire ex
    (788, 4),    # Charmander
    (790, 2),    # Mega Charizard X ex
    (1079, 4),   # Rare Candy
    (1121, 4),   # Ultra Ball
    (1182, 4),   # Boss's Orders
    (1123, 4),   # Switch
    (1086, 4),   # Buddy-Buddy Poffin
    (1227, 2),   # Lillie's Determination
    (1097, 2),   # Night Stretcher
    (1232, 2),   # Firebreather
    (1119, 2),   # Energy Search
    (2, 22),     # Basic Fire Energy
]

VARIANTS = {
    "boss_orders_3": {
        "changes": {(1182, 3), (1227, 3)},
        "name": "Boss's Orders x3, Lillie's Determination x3",
    },
    "boss_orders_2": {
        "changes": {(1182, 2), (1227, 4)},
        "name": "Boss's Orders x2, Lillie's Determination x4",
    },
    "fire_20": {
        "changes": {(2, 20), (1119, 4)},
        "name": "Fire Energy x20, Energy Search x4",
    },
    "fire_24": {
        "changes": {(2, 24), (1097, 0)},
        "name": "Fire Energy x24, no Night Stretcher",
    },
    "charizard_ex": {
        "changes": {(46, 3), (1158, 1), (790, 3), (2, 22)},
        "name": "Charizard ex x1, Mega Charizard x3, Gouging Fire x3",
    },
    "poffin_3": {
        "changes": {(1086, 3), (1232, 3)},
        "name": "Poffin x3, Firebreather x3",
    },
    "no_energy_search": {
        "changes": {(1119, 0), (1182, 5)},
        "name": "Boss's Orders x5, no Energy Search",
    },
    "poffin_5": {
        "changes": {(1086, 5), (1232, 1)},
        "name": "Poffin x5, Firebreather x1",
    },
    "switch_3": {
        "changes": {(1123, 3), (1097, 3)},
        "name": "Switch x3, Night Stretcher x3",
    },
    "night_stretcher_3": {
        "changes": {(1097, 3), (1119, 1)},
        "name": "Night Stretcher x3, Energy Search x1",
    },
}

COMMON_CARDS = {
    "supporters": [1182, 1227, 1206, 1183, 1184, 1186],
    "items": [1079, 1121, 1123, 1086, 1232, 1119, 1097, 1087],
    "basics": [788, 43, 47, 76, 1145, 1205],
}

TECH_CARDS = {
    1097: "Night Stretcher",
    1119: "Energy Search",
    1232: "Firebreather",
    1227: "Lillie's Determination",
    1182: "Boss's Orders",
    1123: "Switch",
    1086: "Buddy-Buddy Poffin",
}


def build_variant_deck(base: list[tuple[int, int]], changes: set[tuple[int, int]]) -> list[int]:
    deck_map = dict(base)
    for card_id, count in changes:
        if count == 0:
            deck_map.pop(card_id, None)
        else:
            deck_map[card_id] = count

    deck = []
    for card_id, count in deck_map.items():
        deck.extend([card_id] * count)

    if len(deck) < 60:
        deck.extend([2] * (60 - len(deck)))
    elif len(deck) > 60:
        excess_energy = [i for i, c in enumerate(deck) if c == 2]
        while len(deck) > 60 and excess_energy:
            idx = excess_energy.pop()
            deck.pop(idx)

    return deck[:60]


def generate_variant_decks() -> dict[str, list[int]]:
    variants = {"base": _deck_from_spec(BASE_DECK)}

    for name, spec in VARIANTS.items():
        deck = build_variant_deck(BASE_DECK, spec["changes"])
        errors = validate_deck(deck)
        if not errors:
            variants[name] = deck

    return variants


def _deck_from_spec(spec: list[tuple[int, int]]) -> list[int]:
    deck = []
    for card_id, count in spec:
        deck.extend([card_id] * count)
    return deck[:60]


def evaluate_deck(
    deck: list[int],
    agent_fn,
    opponent_lineups: list[tuple[str, list[int]]] | None = None,
    games_per_matchup: int = 30,
) -> dict:
    if opponent_lineups is None:
        opponent_lineups = [
            ("mirror", _deck_from_spec(BASE_DECK)),
            ("hard", get_deck_for_tier("hard")),
            ("easy", get_deck_for_tier("easy")),
        ]

    total_wins = 0
    total_games = 0
    matchup_results = {}

    for opp_name, opp_deck in opponent_lineups:
        wins = 0
        played = 0
        for i in range(games_per_matchup):
            from agent.opponent_model import reset_opponent_models
            reset_opponent_models()

            if i % 2 == 0:
                traj = run_game(
                    deck, opp_deck, agent_fn, slightly_smart_agent,
                    max_steps=2000, collect_trajectory=False,
                )
                if traj.outcome == 0:
                    wins += 1
            else:
                traj = run_game(
                    opp_deck, deck, slightly_smart_agent, agent_fn,
                    max_steps=2000, collect_trajectory=False,
                )
                if traj.outcome == 1:
                    wins += 1
            played += 1

        wr = wins / played if played > 0 else 0
        matchup_results[opp_name] = {"wins": wins, "played": played, "win_rate": wr}
        total_wins += wins
        total_games += played

    overall_wr = total_wins / total_games if total_games > 0 else 0

    return {
        "overall_win_rate": overall_wr,
        "total_wins": total_wins,
        "total_games": total_games,
        "matchups": matchup_results,
    }


def optimize_deck(
    agent_fn,
    games_per_matchup: int = 20,
    top_k: int = 3,
) -> dict:
    variants = generate_variant_decks()
    print(f"Generated {len(variants)} deck variants: {list(variants.keys())}")

    results = {}
    for name, deck in variants.items():
        errors = validate_deck(deck)
        if errors:
            print(f"  {name}: INVALID - {errors}")
            continue

        print(f"Evaluating {name}...")
        start = time.time()
        result = evaluate_deck(deck, agent_fn, games_per_matchup=games_per_matchup)
        elapsed = time.time() - start
        results[name] = result
        wr = result["overall_win_rate"]
        print(f"  {name}: WR={wr:.1%} ({result['total_wins']}/{result['total_games']}) [{elapsed:.1f}s]")
        for matchup_name, mr in result["matchups"].items():
            print(f"    vs {matchup_name}: {mr['win_rate']:.1%} ({mr['wins']}/{mr['played']})")

    sorted_results = sorted(results.items(), key=lambda x: -x[1]["overall_win_rate"])
    print(f"\nTop {top_k} decks:")
    for name, result in sorted_results[:top_k]:
        print(f"  {name}: WR={result['overall_win_rate']:.1%}")

    best_name = sorted_results[0][0]
    best_deck = variants[best_name]

    best_path = Path("best_deck.json")
    import json
    with open(best_path, "w") as f:
        json.dump({"name": best_name, "deck": best_deck, "results": results}, f, indent=2)

    return {
        "best_name": best_name,
        "best_deck": best_deck,
        "best_wr": sorted_results[0][1]["overall_win_rate"],
        "all_results": results,
    }


if __name__ == "__main__":
    from agent.main import agent
    import json

    result = optimize_deck(agent, games_per_matchup=20)
    print(f"\nBest deck: {result['best_name']} (WR={result['best_wr']:.1%})")

    if result["best_wr"] > 0.5:
        from agent.deck import save_deck_csv
        save_deck_csv(result["best_deck"], "deck.csv")
        print("Updated deck.csv with best deck")