"""Dark Trinity ex deck (MULTI_DECK_ENSEMBLE_PLAN.md, deck #6).

Re-exports the "Dark Trinity ex" entry from competitive_decks.json so the
opponent pool's copy and our own trainable copy never drift.

Four interchangeable basic Darkness ex attackers (no evolutions): Okidogi ex
(138) ramps energy then hits for 130/260-if-poisoned, Munkidori ex (139)
hits for 190 every other turn, Fezandipiti ex (140) snipes the bench for 100
and draws 3 after a KO trade, Pecharunt ex (141) can poison-switch a
benched Darkness Pokemon into Active and scales its own attack with prizes
already lost. Binding Mochi (1162) adds +40 damage off a Poisoned target.
"""

import json
from pathlib import Path

_COMPETITIVE_DECKS_PATH = Path(__file__).parent.parent.parent / "competitive_decks.json"


def _load_deck() -> list[int]:
    with open(_COMPETITIVE_DECKS_PATH) as f:
        decks = json.load(f)
    entry = next(d for d in decks if d["name"] == "Dark Trinity ex")
    return entry["deck"]


DECK = _load_deck()
