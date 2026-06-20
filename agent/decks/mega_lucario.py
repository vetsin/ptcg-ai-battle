"""Mega Lucario ex deck (MEGA_LUCARIO_STRATEGY_PLAN.md §6).

Hybrid control->combo / single-prize-simulator: Solrock/Lunatone apply early
single-prize pressure and build consistency while Riolu evolves into Mega
Lucario ex (backed by Hariyama) as the multi-prize finisher.
"""

from agent.deck import build_deck_from_description

_DECK_DESCRIPTION = [
    (974, 4),   # Riolu
    (678, 4),   # Mega Lucario ex
    (673, 3),   # Makuhita
    (674, 3),   # Hariyama
    (676, 3),   # Solrock
    (675, 3),   # Lunatone
    (1121, 4),  # Ultra Ball
    (1182, 4),  # Boss's Orders
    (1227, 4),  # Lillie's Determination
    (1152, 4),  # Poké Pad
    (1142, 4),  # Fighting Gong
    (1213, 4),  # Judge
    (1092, 1),  # Secret Box (ACE SPEC)
    (6, 12),    # Basic Fighting Energy
    (20, 3),    # Rock Fighting Energy
]

DECK = build_deck_from_description(_DECK_DESCRIPTION)
