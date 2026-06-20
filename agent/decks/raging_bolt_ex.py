"""Raging Bolt ex deck (MULTI_DECK_ENSEMBLE_PLAN.md, deck #3).

Re-exports the "Raging Bolt ex" entry from competitive_decks.json so the
opponent pool's copy and our own trainable copy never drift.

An "Ancient" Lightning/Dragon hybrid: Raging Bolt ex (63) and Iron Thorns ex
(37) are the ex-tier finishers (Bellowing Thunder / Volt Cyclone), while
Roaring Moon (61) and Koraidon (62) are single-prize basics that trade
respectably on their own (140 HP, real damage) before the ex's are powered up.
Burst Roar (71, discard hand/draw 6) is a no-damage setup attack worth using
early rather than late.
"""

import json
from pathlib import Path

_COMPETITIVE_DECKS_PATH = Path(__file__).parent.parent.parent / "competitive_decks.json"


def _load_deck() -> list[int]:
    with open(_COMPETITIVE_DECKS_PATH) as f:
        decks = json.load(f)
    entry = next(d for d in decks if d["name"] == "Raging Bolt ex")
    return entry["deck"]


DECK = _load_deck()
