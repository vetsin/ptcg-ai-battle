"""Greninja ex deck (MULTI_DECK_ENSEMBLE_PLAN.md, deck #7).

Re-exports the "Greninja ex" entry from competitive_decks.json so the
opponent pool's copy and our own trainable copy never drift.

A single evolution-line Water deck: Froakie (33) -> Frogadier (34) ->
Greninja ex (40, stage2, 310hp), accelerated by Rare Candy to skip straight
from Froakie. Shinobi Blade (170 dmg for 1 Water energy + deck search) is
an extremely efficient finisher; Mirage Barrage discards 2 of Greninja's
own energy to spread 120 damage across 2 of the opponent's Pokemon.
"""

import json
from pathlib import Path

_COMPETITIVE_DECKS_PATH = Path(__file__).parent.parent.parent / "competitive_decks.json"


def _load_deck() -> list[int]:
    with open(_COMPETITIVE_DECKS_PATH) as f:
        decks = json.load(f)
    entry = next(d for d in decks if d["name"] == "Greninja ex")
    return entry["deck"]


DECK = _load_deck()
