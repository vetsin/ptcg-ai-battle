"""Dragapult Dusknoir deck (MULTI_DECK_ENSEMBLE_PLAN.md, deck #8).

Re-exports the "Dragapult Dusknoir" entry from competitive_decks.json so the
opponent pool's copy and our own trainable copy never drift.

Two evolution lines share the board: Dreepy (119) -> Drakloak (120) ->
Dragapult ex (121, 320hp) hits for 200 + 6 spread damage counters via
Phantom Dive, or pokes for free with Jet Headbutt; Duskull (131) ->
Dusclops (132) -> Dusknoir (133) hits for 150 + retreat-lock via Shadow
Bind, and both Dusclops/Dusknoir can self-KO their "Cursed Blast" ability
to snipe 50/130 damage onto a single opposing target. Munkidori (112),
Fezandipiti ex (140), Budew (235), and Meowth ex (1071) are single-copy
disruption/utility techs rounding out the toolbox.
"""

import json
from pathlib import Path

_COMPETITIVE_DECKS_PATH = Path(__file__).parent.parent.parent / "competitive_decks.json"


def _load_deck() -> list[int]:
    with open(_COMPETITIVE_DECKS_PATH) as f:
        decks = json.load(f)
    entry = next(d for d in decks if d["name"] == "Dragapult Dusknoir")
    return entry["deck"]


DECK = _load_deck()
