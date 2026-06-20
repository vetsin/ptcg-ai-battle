"""Terapagos ex deck (MULTI_DECK_ENSEMBLE_PLAN.md, deck #5).

Re-exports the "Terapagos ex" entry from competitive_decks.json so the
opponent pool's copy and our own trainable copy never drift.

A Tera-type control deck with two attacker lines: Terapagos ex (176, basic,
230hp) pays its costs with any energy type and either scales with bench size
(Unified Beatdown) or hits for 180 with next-turn damage immunity (Crown
Opal); Chansey (124, basic, single-prize) evolves into Blissey ex (125,
300hp) for a second 180-damage attacker that also draws back to 6 cards.
All attack costs are paid with colorless-counting Basic Darkness Energy (7).
"""

import json
from pathlib import Path

_COMPETITIVE_DECKS_PATH = Path(__file__).parent.parent.parent / "competitive_decks.json"


def _load_deck() -> list[int]:
    with open(_COMPETITIVE_DECKS_PATH) as f:
        decks = json.load(f)
    entry = next(d for d in decks if d["name"] == "Terapagos ex")
    return entry["deck"]


DECK = _load_deck()
