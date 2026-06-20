"""Mega Kangaskhan ex / Crustle deck (MULTI_DECK_ENSEMBLE_PLAN.md, deck #2).

Real archetype matching limitlesstcg.com/decks/341 ("Crustle") and the
competitive_decks.json entry of the same name — re-exported here rather than
rebuilt so the opponent pool's copy and our own trainable copy never drift.

Crustle (345) walls behind its ability ("prevent all damage done to this
Pokémon by attacks from your opponent's Pokémon {ex}") while Mega Kangaskhan ex
(756) accrues card advantage via "Run Errand" (draw 2/turn while Active) and
eventually finishes with Rapid-Fire Combo.
"""

import json
from pathlib import Path

_COMPETITIVE_DECKS_PATH = Path(__file__).parent.parent.parent / "competitive_decks.json"


def _load_crustle_deck() -> list[int]:
    with open(_COMPETITIVE_DECKS_PATH) as f:
        decks = json.load(f)
    entry = next(d for d in decks if d["name"] == "Crustle")
    return entry["deck"]


DECK = _load_crustle_deck()
