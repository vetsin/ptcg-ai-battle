"""Archaludon ex deck (MULTI_DECK_ENSEMBLE_PLAN.md, deck #4).

Re-exports the "Archaludon ex" entry from competitive_decks.json so the
opponent pool's copy and our own trainable copy never drift.

A single evolution-line Metal control deck: Duraludon (839, basic) evolves
into Archaludon ex (190, 300hp) either naturally or via Rare Candy. Metal
Defender (253, 220 dmg) grants weakness immunity on top of the damage, and
Assemble Alloy (Archaludon ex's ability) reattaches discarded Metal energy
when it evolves in from hand — rewarding holding evolutions for a burst turn
rather than evolving the moment Duraludon hits the board.
"""

import json
from pathlib import Path

_COMPETITIVE_DECKS_PATH = Path(__file__).parent.parent.parent / "competitive_decks.json"


def _load_deck() -> list[int]:
    with open(_COMPETITIVE_DECKS_PATH) as f:
        decks = json.load(f)
    entry = next(d for d in decks if d["name"] == "Archaludon ex")
    return entry["deck"]


DECK = _load_deck()
