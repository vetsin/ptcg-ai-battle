# Multi-Deck Strategy Ensemble → Unified Champion (8-deck plan)

## Context

`MEGA_LUCARIO_STRATEGY_PLAN.md` (done; see status below) built one deck-specific
`StrategyProfile` and proved the architecture: strategy-conditioned PUCT priors +
heuristic bias + rollout nudging, gated entirely on `get_my_strategy() is not None`,
feeding the existing `train_policy` distillation. None of that machinery is
Lucario-specific — `STRATEGIES: dict[str, StrategyProfile]` and `classify_my_deck`
were already written as a registry/dispatch, not a single hardcoded check.

**Goal of this plan:** repeat that process for 7 more top-meta decks (8 total),
train each as an independent specialist, then cross-pollinate all 8 against each
other and distill everything into one unified, generalist champion model.

**Sourcing methodology (same as Lucario):** derive each profile from a human
strategy write-up + a real decklist, but ground every card reference in IDs
verified to exist in *this* engine's 1267-card DB (`agent.evaluate.get_card_data()`).
Newer-set cards a source mentions that aren't in our DB get dropped/substituted —
see Crustle below for a worked example of this reconciliation.

## Status of deck #1 — Mega Lucario ex

Implemented and training (`agent/strategies.py:MEGA_LUCARIO`,
`agent/decks/mega_lucario.py`). A/B-verified gains vs both `slightly_smart_agent`
and `random_agent`. A 10-iteration `ab_train` run against `deck.csv` is what
"the current plan" refers to in chat — once that finishes, deck.csv gets restored
to its original contents and this plan begins.

## Prerequisite engineering: per-player strategy state (do this first)

`agent/policy.py`'s `_my_strategy` is currently a single global value, and
`agent/selfplay.py`'s `run_game` sets it once via
`STRATEGIES.get(classify_my_deck(deck0)) or STRATEGIES.get(classify_my_deck(deck1))`.
That OR-fallback is harmless for Lucario-vs-non-Lucario games (bias terms key off
specific card IDs, so the non-matching side's options just never trigger any bias —
a no-op, not a wrong answer). It silently breaks once **both** sides have a real
profile (e.g. Lucario vs Crustle in the cross-training phase in §3): only one
side's bias would ever apply.

**Fix before building deck #2:**
- `agent/policy.py`: replace `_my_strategy: StrategyProfile | None` with
  `_my_strategy: list[StrategyProfile | None]` (len 2, indexed by player). New
  `set_my_strategy(profile, index: int | None = None)` — `index=None` sets both
  slots (preserves every existing call site's behavior byte-for-byte). New
  `get_my_strategy(my_index: int)` takes the asking player's index.
- `agent/search.py`: thread `my_index` into the two `get_my_strategy()` call sites
  (`_compute_root_priors`, `_get_rollout_action` — the latter needs
  `obs.current.yourIndex`, already available).
- `agent/selfplay.py`: `run_game` calls
  `set_my_strategy(STRATEGIES.get(classify_my_deck(deck0)), index=0)` and the
  same for `deck1`/`index=1`, dropping the OR-fallback.
- `agent/main.py`: unaffected — pass `index=None` (our own seat is the only one
  we ever know the deck for; the opponent's deck is genuinely unknown in real
  play, so there's nothing to set on that side anyway).

Verify with a quick scripted Lucario-vs-Crustle game and printed root priors on
both sides before moving on — confirm each side's bias only ever fires for its
own deck's card IDs.

## classify_my_deck: multi-archetype dispatch

Replace the single `if 678 in deck` check with a small ordered table of
(anchor_card_id → archetype_name) and a `STRATEGIES` registry keyed the same way
— anchor IDs below are chosen to be unique to that archetype in our card pool
(checked against `get_card_data()`, not guessed):

| Archetype | Anchor ID | Card |
|---|---|---|
| `mega_lucario_ex` | 678 | Mega Lucario ex |
| `crustle` | 756 | Mega Kangaskhan ex |
| `raging_bolt_ex` | 63 | Raging Bolt ex |
| `archaludon_ex` | 190 | Archaludon ex |
| `terapagos_ex` | 176 | Terapagos ex |
| `dark_trinity_ex` | 138 | Okidogi ex |
| `greninja_ex` | 40 | Greninja ex |
| `dragapult_dusknoir` | 121 | Dragapult ex (paired with Dusknoir 133) |

`classify_my_deck(deck) -> str | None` checks each anchor in order, returns the
first match (decks are disjoint enough in practice that order shouldn't matter,
but Dragapult-family decks in `competitive_decks.json` share base Pokémon —
`dragapult_dusknoir`'s anchor is the *combination* signal in practice since plain
`Dragapult ex` (121) alone also appears in `Dragapult`/`Dragapult Blaziken`; if
this collides, disambiguate with a secondary check for 133 Dusknoir's presence).

## The 8 decks

Picked for type/role diversity (covers Fighting, Grass/wall-control, Lightning,
Metal, Colorless/Tera, Darkness, Water, Psychic) and because 7 of 8 already have
validated real decklists sitting in `competitive_decks.json` — only Crustle's
*profile* (not decklist) needed reconciling against a newer-set source.

**Status (2026-06-20): Phase A complete, all 8 decks trained.** Bias-validation
note: the original plan to gate each profile on beating
`random_agent`/uniform priors by a margin was tested rigorously (Raging Bolt
ex, Terapagos ex) and dropped — see `feedback_strategy_bias_validation`
memory — `strategy_priors` doesn't show a reproducible edge over uniform at
current MCTS depth for any deck tested, so profiles ship for
documentation/future use without a promotion-rate or win-rate gate.

Final `ab_train` promotion counts (out of 10 iterations) and self-play win
totals (out of 1300 games):

| Deck | Promotions | Self-play wins |
|---|---|---|
| Mega Lucario ex | 10/10 | — (earlier session) |
| Crustle | 10/10 | 843/1300 |
| Raging Bolt ex | 1/10 | 578/1300 (weak — bias unvalidated but kept, see memory) |
| Archaludon ex | 10/10 | 910/1300 |
| Terapagos ex | 8/10 | 606/1300 |
| Dark Trinity ex | 10/10 | 700/1300 |
| Greninja ex | 10/10 | 731/1300 |
| Dragapult Dusknoir | 7/10 | 553/1300 |

All 8 specialist checkpoints archived under `models/specialist_<name>.pt`.

**Phase B/C complete (2026-06-20).** `agent/ensemble_train.py` required one
real prerequisite beyond what the plan anticipated: the policy/value nets in
`agent/policy.py`/`agent/search.py` were single global slots, so two
specialists couldn't actually play each other with their own weights in the
same process. Refactored both to per-player lists (`_policy_model[index]`,
`_value_model[index]`, mirroring the `_my_strategy` fix), and extended
`agent/selfplay.py`'s `run_game` with `collect_both=True` to record both
sides' trajectories from one playthrough instead of replaying the matchup
twice. Also added `train_policy(..., warm_start_path=...)` so Phase C could
warm-start from a real checkpoint instead of random init.

Phase B: 800 cross-archetype games (uniform random pairs, mirrors allowed),
0 errors, 1600 trajectories → 89,614 training samples. Warm-start probe
(10 games/specialist vs `slightly_smart_agent`) picked `terapagos_ex` (100%).
Phase C: 20 epochs, final train acc 73.8%. Verification round-robin
(20 games/matchup) on `ensemble_training/unified_model.pt`:

| Deck | vs slightly_smart | vs random | vs own specialist |
|---|---|---|---|
| Mega Lucario ex | 85% | 55% | 50% |
| Crustle | 50% | 45% | 60% |
| Raging Bolt ex | 25% | 50% | 50% |
| Archaludon ex | 60% | 55% | 85% |
| Terapagos ex | 80% | 35% | 20% |
| Dark Trinity ex | 95% | 95% | 30% |
| Greninja ex | 80% | 25% | 75% |
| Dragapult Dusknoir | 70% | 65% | 55% |

Takeaways: the unified model clearly generalizes (it's not just memorizing
one archetype) and even beats its own specialist outright on Archaludon ex
(85%) and Greninja ex (75%). Raging Bolt ex stays weak, consistent with its
Phase A result. One open anomaly: `vs random` is *lower* than `vs
slightly_smart_agent` on several decks (Terapagos 35% vs 80%, Greninja 25%
vs 80%) — backwards from the usual expectation that a weaker opponent is
easier to beat. Not yet investigated; candidates are n=20 noise or a real
quirk in how `random_agent`'s deck-building/mulligan choices interact with
the unified model's heuristic+MCTS path. Flagged for follow-up, not fixed.

### 1. Mega Lucario ex — done (see above)

### 2. Crustle (Mega Kangaskhan ex / Crustle) — done

**Sources:** limitlesstcg.com/decks/341 (confirms "Crustle" is a real, currently
winning archetype — 22 Regional Top 8s incl. 2 wins, NAIC 2026 result) +
20cards.com/meta/crustle-ex (strategy write-up).

**Reconciliation:** 20cards.com describes a *different, newer* "Crustle ex"
printing (`Boulder Crush`, `Heavy Helmet`, `Skull Fossil`) that does not exist
anywhere in this engine's 1267-card DB (verified — no card named "Crustle ex",
no attack "Boulder Crush"). What **does** exist, and exactly matches
limitlesstcg's archetype name plus the existing validated
`competitive_decks.json` "Crustle" entry, is **Mega Kangaskhan ex / Crustle**:
Crustle (id 345) tanks behind its ability —
*"Prevent all damage done to this Pokémon by attacks from your opponent's
Pokémon {ex}"* — literally the same ex-immunity mechanic `agent/evaluate.py`'s
`_has_ex_immunity`/`_wall_penalty_score` already special-case for opponents,
now relevant for *our own* wall. Mega Kangaskhan ex (756, megaEx, 300 HP) sits
behind it with the "Run Errand" ability (draw 2/turn while Active) building card
advantage, eventually attacking with Rapid-Fire Combo (1092, 200+ dmg, 3
colorless, coin-flip scaling). Dwebble (344) self-evolves via Ascension (478,
search deck for evolution, no Rare Candy needed).

| Role | Cards (engine IDs) | Knob |
|---|---|---|
| Wall / damage-immune blocker | Crustle `345` (skill: prevent ex damage) | keep Active vs ex/megaEx opponents; never retreat it out voluntarily into a non-wall |
| Win condition / card-advantage engine | Mega Kangaskhan ex `756` (Run Errand ability: draw 2/turn; Rapid-Fire Combo `1092` = 200+/[0,0,0]) | only attack once board is safe — the deck wants to *not* race |
| Evolution base | Dwebble `344` (Ascension `478` = self-evolve search) | prioritize laddering into 345 turn 1-2 |
| Disruption | Boss's Orders `1182`, Team Rocket's Petrel `1219` | mid-game, especially to break through opposing walls once Kangaskhan is set up |
| Consistency | Lillie's Determination `1227`, Pokégear 3.0 `1122`, Buddy-Buddy Poffin `1086` | early |
| Energy | Grow Grass Energy `18`, Mist Energy `11`, Spiky Energy `14` | attach toward whichever attacker is live |

**Phase weighting:** unlike Lucario's "race to the finisher," this is a patience
deck — bias should favor *keeping Crustle Active and not attacking* early/mid
game (it has no attack worth using), then favor Mega Kangaskhan ex's Rapid-Fire
Combo once it has 3+ energy and card advantage is established (e.g. hand size
≥ opponent's, or opponent's board is already thin). This needs a different shape
of phase gate than Lucario's `_is_finisher_phase` (energy-on-active threshold) —
add a `prefer_passive_wall_ids` concept to `StrategyProfile` (cards that should
actively be biased *against* attacking/retreating while they're Active), since
none of the existing fields capture "stay put and do nothing."

**Deck:** already exists at `competitive_decks.json` (name `"Crustle"`) — for
this deck only, `agent/decks/crustle.py` should just re-export that list rather
than rebuild it from a description tuple, to avoid drift between the opponent
pool's copy and our own trainable copy.

### 3-8. Remaining six (sketched now, detailed pass deferred to each one's turn)

All six already have validated 60-card lists in `competitive_decks.json`; the
work per deck is (a) write the `StrategyProfile`, (b) `agent/decks/<name>.py`
re-exporting that list, (c) add to `STRATEGIES`/anchor table, (d) one
`ab_train` pass. Sketches below are grounded (every ID below was read directly
from `get_card_data()`/`get_attack_data()`, not guessed) but not yet
human-source-verified the way Lucario/Crustle were — do that verification pass
when each deck's turn comes, same as the other two.

- **Raging Bolt ex** (`raging_bolt_ex`): key attacker 63 (Raging Bolt ex, 240hp,
  Bellowing Thunder `72` = 0/[4,6] — almost certainly a damage-counter/ability
  attack, check text), paired with Iron Thorns ex (37, Volt Cyclone `29` =
  140/[4,0,0], plus a Rule-Box lock ability worth checking for synergy/anti-synergy
  with our own ex Pokémon). Lightning-type control/big-hit hybrid.
- **Archaludon ex** (`archaludon_ex`): Duraludon (839) → Archaludon ex (190,
  300hp, Metal Defender `253` = 220/[8,8,8], plus an ability triggered on
  evolving-from-hand worth checking — likely a Stage-2-style search/draw). Rare
  Candy (1079) ladder. Metal control deck.
- **Terapagos ex** (`terapagos_ex`): single key attacker 176 (230hp, Crown Opal
  `233` = 180/[1,3,4] — odd mixed-energy cost, check text for a Tera-specific
  rule), Unified Beatdown `232` (0 dmg, likely scales with prior KOs). Colorless/
  Tera, single-card-engine deck — simplest profile of the eight.
- **Dark Trinity ex** (`dark_trinity_ex`): Okidogi ex (138, 250hp, Chain-Crazed
  `181` = 130/[7,7,0]) + Fezandipiti ex (140, 210hp, ability triggers off KOs
  during opponent's turn — disruption/revenge-kill shape). Darkness aggro-control.
- **Greninja ex** (`greninja_ex`): Froakie(33)→Frogadier(34)→Greninja ex(40,
  310hp, Shinobi Blade `33`[sic, different namespace from attack ids above] =
  170/[3], Mirage Barrage `34` = 0/[3,0,0] — check text, likely bench-damage
  spread). Rare Candy ladder. Water rapid-evolution glass cannon.
- **Dragapult Dusknoir** (`dragapult_dusknoir`): two evolution lines — Dreepy(119)
  →Drakloak(120)→Dragapult ex(121, 320hp stage2, Phantom Dive `154` = 200/[2,5])
  and Duskull(131)→Dusclops(132)→Dusknoir(133, 160hp, Shadow Bind `172` =
  150/[5,5,0], retreat-lock disruption). Crushing Hammer `1120` (energy
  disruption) in the list too — this is the most disruption-heavy of the eight.

## Training pipeline

### Phase A — per-deck specialist training (sequential, ~8x the Lucario run)

For each deck, same loop already proven on Lucario:
`save_deck_csv(deck)` → `python3 -m agent.ab_train` (10 iterations, curriculum,
20 epochs) → save `ab_training/final_model.pt` to a per-deck path (e.g.
`models/specialist_<name>.pt`) before the next deck overwrites `ab_training/`.
Each run produces a checkpoint that's good at *that* archetype because its
self-play trajectories were generated under that archetype's strategy-guided
PUCT, not because the network architecture differs (it's the same
`PTCGSimpleNet`/`PTCGPolicyNet` every time).

At ~70-200s/iteration observed for Lucario (post prize-count-fix), budget
~15-35 min/deck → roughly 2-4.5 hours for all 7 remaining decks sequentially.
Don't parallelize decks against each other in this phase — `num_workers=8`
already saturates the 16-core box per deck.

### Phase B — cross-archetype round-robin self-play

New script `agent/ensemble_train.py`:
- Load all 8 specialist checkpoints + decks.
- Each game: pick two of the 8 (with replacement — mirrors allowed) uniformly at
  random. Side A plays with specialist checkpoint A loaded + `set_my_strategy(profile_A, index=0)`;
  side B with checkpoint B + `index=1`. This is exactly why the per-player
  strategy fix above is a hard prerequisite, not a nice-to-have.
- Collect trajectories from both sides (not just one `trajectory_player`) since
  every specialist's perspective is useful signal here — `run_game`'s existing
  `trajectory_player` parameter only records one side; either call it twice (once
  per side) or extend it to record both in one pass.
- This produces a single large mixed-archetype trajectory pool.

### Phase C — unification distillation

One `train_policy` call (or a short `ab_train`-style A/B loop) over the full
Phase B trajectory pool, starting from whichever Phase A checkpoint validates
best as the initial weights (warm start, not random init) to speed convergence.
Validate the result by round-robin win rate across all 8 decks against (a) each
specialist checkpoint, (b) `slightly_smart_agent`, (c) `random_agent` — the
unified model should not regress below any specialist on its own archetype by
more than a small margin, and should beat every specialist when playing a deck
that *isn't* its specialty (that's the whole point of unifying).

## New/changed files

- **Prerequisite:** `agent/policy.py` (per-player `_my_strategy`), `agent/search.py`
  (thread `my_index` through `get_my_strategy` call sites), `agent/selfplay.py`
  (`run_game` sets both sides' strategies independently).
- **New per deck:** `agent/decks/<name>.py` (7x), `StrategyProfile` entries in
  `agent/strategies.py` (7x), anchor-table entries in `classify_my_deck`.
- **New orchestration:** `agent/ensemble_train.py` (Phase B/C), plus a thin
  driver script that loops Phase A over all 7 remaining decks unattended
  (mirrors what's being done by hand for Lucario right now: swap `deck.csv`,
  run `ab_train`, archive the checkpoint, restore, repeat).
- **`StrategyProfile` schema growth:** add `prefer_passive_wall_ids: list[int] = field(default_factory=list)`
  for Crustle's "don't attack yet" pattern; make all existing list fields default
  to `field(default_factory=list)` so future profiles don't need to populate
  categories that don't apply to them (Crustle has no `setup_attack_ids`, etc.).

## Verification (per deck, before folding into Phase B)

Same as the Lucario plan: deck builds + `validate_deck`/`validate_deck_for_training`
pass, `classify_my_deck` round-trips, PUCT priors visibly favor the archetype's
own cards in a real game, A/B vs `slightly_smart_agent` and `random_agent` both
≥ break-even against the plain-heuristic baseline (n≥30, ideally n≥50 given how
noisy n=30 turned out to be for Lucario vs `random_agent`).

## Execution order (per chat instruction)

1. Finish the in-flight Mega Lucario `ab_train` run; restore `deck.csv`.
2. Land the per-player strategy-state prerequisite refactor + its own quick
   verification.
3. Build Crustle (`agent/decks/crustle.py`, `StrategyProfile`, anchor entry,
   `prefer_passive_wall_ids` schema addition) → verify → `ab_train` run.
4. Repeat step 3 for the remaining six, in the order listed above.
5. Phase B (`agent/ensemble_train.py`) → Phase C unification → verification.
