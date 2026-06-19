# PTCG AI Battle — Self-Play Training & Multi-Deck Plan

## Problem Statement

Our agent has **83.8% WR vs competitive decks** but only **42% WR vs random**. The random WR is a red herring — Kaggle opponents are real agents, not random. The real issue: our model was trained on **winning trajectories against random/heuristic opponents**, producing a value head that outputs ~0.95 for every state (trained on unbalanced data). We need self-play against strong opponents to get balanced, discriminative training data.

Additionally, our agent only plays one deck (Gouging Fire ex) and only trains against opponents using that same deck. We need multi-deck coverage and a larger competitive deck dataset.

## Current Architecture Recap

- **Agent decision path**: `choose_action()` → MCTS (if ≥4 options) → heuristic fallback
- **MCTS**: 8 children expanded, 60 sims, 1500ms budget, 5-step heuristic rollouts
- **Training**: `si_train.py` runs agent vs `random_agent` (easy) / `slightly_smart_agent` (hard), collects winning trajectories only, trains PTCGSimpleNet (128 hidden)
- **Model**: PTCGSimpleNet with action+value heads. Currently barely used — MCTS heuristic is the main driver.
- **Decks**: 1 deck (Gouging Fire ex), 8 opponent decks in `competitive_decks.json`

---

## Phase A: A/B Self-Play Training Loop

### A.1: Core Self-Play Infrastructure

**New file**: `agent/ab_train.py`

The loop:

```
1. Save current agent as champion (model_A.pt)
2. For each iteration:
   a. Run N games: champion vs champion (both sides, alternating)
   b. Collect trajectories from BOTH players (winners AND losers)
   c. Label trajectories: winner actions = +1, loser actions = -1
   d. Train challenger model_B on all labeled trajectories
   e. Run M validation games: challenger_B vs champion_A
   f. If challenger wins > threshold (e.g., 55%), B becomes new champion
   g. Otherwise, discard B, champion stays
3. Repeat for K iterations
```

**Key differences from current `si_train.py`:**

| Current SI Training | A/B Self-Play |
|---|---|
| Trains vs random/slightly_smart | Trains vs itself |
| Only winning trajectories | Both winner and loser trajectories |
| Reward labels: always 1.0 | Reward labels: +1 for winner, 0 for loser |
| Value head sees only "won" states | Value head sees balanced win/loss states |
| Single model, no validation | Champion/challenger with validation gate |
| No deck variety | Games across multiple opponent decks |

**Implementation details:**

- Use `selfplay.run_game()` with both `agent0_fn` and `agent1_fn` = current champion
- Collect trajectories for **both sides** (`trajectory_player` alternates by game, but collect for both)
- Training label for each step: `reward = 1.0` if the player who made this action won, `0.0` if they lost
- Value head target: `1.0` for winner states, `0.0` for loser states — this fixes the "always 0.95" problem
- Validation: 20 games per iteration, champion retains if challenger WR < 55%

### A.2: Multi-Deck Self-Play

Self-play against mirror matches is insufficient. We need games vs diverse decks.

**Approach**: In each iteration, split games across opponent decks:

```python
DECKS = [
    load_deck_csv(),  # our Gouging Fire ex (mirror)
    *load_competitive_decks(),  # 8 NAIC decks
]

# Per iteration: 100 games total
# 30 games mirror (champion vs champion, same deck)
# 70 games across other decks (champion vs champion, different decks)
# Each opponent deck: ~9 games
```

The agent always plays its own deck (Gouging Fire ex). The opponent plays each competitive deck. Both sides use the champion agent for decision-making. This teaches the model to handle diverse matchups.

### A.3: Stochastic Opponent Rollouts in MCTS

During self-play training games, the rollout opponent should play realistically but with some randomness. This improves MCTS exploration.

**Change in `search.py` `_rollout_from_node`** and `_get_rollout_action`:

```python
# During MCTS rollouts, mix heuristic + random:
# 70% heuristic action, 30% random valid action
# This prevents MCTS from assuming perfect opponent play
```

This is a separate change from A/B training but complements it — the agent learns to handle both rational and unpredictable opponents.

### A.4: Training Curriculum

| Iteration | Games | Opponent | Purpose |
|---|---|---|---|
| 1-3 | 100 each | Champion vs competitive decks | Bootstrap from current agent |
| 4-6 | 200 each | Champion vs competitive decks | Deepen with more data |
| 7-10 | 100 each | Champion vs champion (recent) | Refine against strongest opponent |

Total: ~1000-1200 games. At ~0.5s/game (no MCTS for training opponent), this takes ~10 minutes.

**Status (2026-06-19)**: Implemented. `mirror_ratio` was declared on `ABTrainConfig` but never actually wired in (hardcoded 0.3) — fixed. Added `_curriculum_for_iteration()` in `agent/ab_train.py`, phase boundaries as fractions of `num_iterations` (0-30% bootstrap @ mirror_ratio=0, 30-60% deepen @ 0.1, 60-100% refine @ 1.0 mirror), so it scales to any `num_iterations`. Default `__main__` config now runs all 10 iterations.

### A.5: Model Architecture Changes

Current `PTCGSimpleNet` (128 hidden) is too small for diverse states. Changes:

1. **Increase hidden_dim to 256** for better representation
2. **Train value head with MSE loss on outcome labels** (0 or 1, not constant)
3. **Weight loser trajectories equally** — critical for balanced value learning
4. **Add deck archetype as input feature** (one-hot of 9 archetypes) so the model learns matchup patterns

**Status (2026-06-19)**: All implemented. (1)-(3) were already done by the A/B training loop. (4): reused the existing `OpponentModel` archetype inference (30 archetypes, not 9 — that count predates the richer `opponent_model.py`) rather than building a separate one-hot encoder; `encode_state()` now appends a 30-dim probability vector and calls `opp_model.update()` directly so the belief stays fresh even off the MCTS path. `NUM_STATE_FEATURES` 120->150 — this invalidates every prior checkpoint (`model.pt`, `ab_training/*`), so a full retrain followed.

### A.6: Value Head Integration with MCTS

After training produces a model with discriminative value output:

1. Replace `_rollout_from_node` with `_value_network_evaluate` (our original plan)
2. Use blended value: `0.5 * heuristic + 0.5 * scaled_net_value`
3. Scale network output: `net_signal = clamp(net_value, -1, 1) * scale_factor` where `scale_factor` is tuned to match heuristic range
4. This gives ~10x MCTS speedup (one forward pass vs 5 rollout steps × 2-3 sim calls each)

** gating**: Only enable value network in MCTS if model validation WR > champion WR (i.e., only if the model actually helps)

**Status (2026-06-19)**: Implemented in `agent/search.py` (`load_value_model`, `_static_value_eval`, gated behind `_value_net_enabled`, default off — zero behavior change unless explicitly loaded). Head-to-head eval (`eval_value_net.py`) of `ab_training/final_model.pt` vs the heuristic-rollout baseline across 30 games over all 16 competitive decks: **value-net MCTS lost, 40% WR vs baseline's 60%**. Per the gating rule above, left disabled. Likely cause: the value head was trained on self-play games using `slightly_smart_agent` (no MCTS) as the policy, so it learned a shallower notion of "good state" than the heuristic `evaluate_state` already encodes under full MCTS search. Re-attempt only after training on MCTS-driven self-play trajectories (`use_mcts_agent=True` in `ab_train.py`, currently unused because it's much slower), or after tuning `_value_net_blend`/`_value_net_scale`.

---

## Phase B: Expand Competitive Deck Dataset

### B.1: Current Coverage

8 NAIC 2026 decks in `competitive_decks.json`:

1. Lillie's Clefairy (#1st)
2. Dragapult Dusknoir (#2nd)
3. Dragapult (#3rd)
4. Slowking (#4th)
5. Crustle (#5th)
6. Dragapult Blaziken (#6th)
7. Rocket's Mewtwo (#7th)
8. N's Zoroark (#10th)

Gaps:
- Missing #8, #9, and all placements below #10
- Only 2 non-ex-heavy decks (Crustle, N's Zoroark)
- No Grass/Water/Electric archetype representation
- No stall/control archetype

### B.2: NAIC 2026 Top 16+ Decks

Source: NAIC 2026 results (available on limitlesstcg.com, pokebeach.com)

Add top 16 decklists, prioritizing:
- Missing archetypes (Grass, Water, Electric, Metal, Colorless)
- Decks with ex-immunity (like Crustle) since they're our weakness
- Meta-representative builds for each type

For each deck:
1. Look up the actual card list from tournament results
2. Map card names to card IDs using `all_card_data()` lookup
3. Handle missing cards with substitutions (as done for Special Red Card → Hand Trimmer)
4. Validate: exactly 60 cards, ≤4 of each (except basic energy), ≥1 basic Pokémon
5. Test: run 10 games vs our agent, verify the deck is functional

**Target**: 16-20 competitive decks covering all major archetypes.

### B.3: Auto-Deck Generation (Future)

Generate random but legal decks for broader coverage:
1. Pick a Pokemon line (1-2 evolution lines, 4-8 basics)
2. Add matching energy (12-16)
3. Add trainers (supporters, items, tools) — pick from meta staples
4. Validate: 60 cards, legal
5. Filter: only keep decks that have ≥2 attack options and ≥4 basic Pokemon

This is secondary to NAIC deck expansion but provides long-tail coverage.

### B.4: Validation Pipeline

For each new deck added:

```python
def validate_deck_for_training(deck: list[int], name: str) -> bool:
    """Verify deck is functional before using in training."""
    # 1. Legal deck check (already in deck.py)
    # 2. Can start a game without errors
    # 3. Can complete a 100-turn game vs random without crash
    # 4. Has at least one Pokemon with attack
    pass
```

**Status (2026-06-19)**: Implemented in `agent/deck.py`, all 4 checks present (the 3rd runs 3 trials with majority-vote since `random_agent` play is stochastic — a single crash can be a fluke rather than a structurally broken deck). Also extended the static legality check itself: it only compared per-card-ID counts, missing the real TCG rule of max-4-copies-*per name* across reprints. That gap is exactly what caused the "0% vs Hydrapple ex / Archaludon ex" entries in `NOTES.md` — both decks were illegal (not bad matchups), silently rejected by `battle_start`, recorded as instant losses. Fixed both decklists in `competitive_decks.json`. Wired into `run_ab_training()`: validates `our_deck` (fatal if broken) and filters `opponent_decks` (skip + warn) before any training games run.

---

## Phase C: Multi-Deck Training Integration

### C.1: Self-Play with Diverse Opponent Decks

Modify `run_self_play` to accept a list of opponent decks:

```python
def run_multi_deck_self_play(
    agent_fn,
    our_deck: list[int],
    opponent_decks: list[list[int]],
    games_per_deck: int = 10,
) -> SelfPlayResult:
    """Run games against multiple opponent decks."""
    all_results = SelfPlayResult()
    for opp_deck in opponent_decks:
        result = run_self_play(
            deck=our_deck,
            agent_fn=agent_fn,
            opponent_fn=agent_fn,  # self-play: both sides use champion
            opponent_deck=opp_deck,
            num_games=games_per_deck,
        )
        all_results.games.extend(result.games)
        all_results.wins += result.wins
        all_results.losses += result.losses
        ...
    return all_results
```

### C.2: Opponent-Model-Aware Training

During training games, the agent uses `opponent_model.py` for MCTS. But the opponent model's archetype priors should reflect which deck the opponent is actually playing. Currently `opponent_model.py` infers this from observed cards, but during self-play we know the deck upfront.

**Change**: In training mode, seed the opponent model with the actual deck archetype to make MCTS search more realistic.

### C.3: Reward Shaping for Matchup Difficulty

Not all wins are equal. Weight training rewards by matchup difficulty:

```python
# Wins against hard matchups (Crustle, Slowking) are worth more
MATCHUP_WEIGHTS = {
    "crustle": 1.5,     # our worst matchup
    "slowking": 1.3,    # 80% WR but challenging
    "lillies_clefairy": 1.2,  # top performer
    "default": 1.0,
}
```

This ensures the model learns more from hard matchups.

**Status (2026-06-19)**: Implemented in `agent/ab_train.py` (`MATCHUP_WEIGHTS`, `_matchup_weight()`). Required threading a real per-sample weight through training for the first time — `GameTrajectory` gained an `opponent_name` field (set in `_run_single_game`), `_trajectories_to_training_data()` tags each record with its matchup weight, and `agent/network.py`'s `PTCGDataset`/`collate_fn`/`train_policy` now carry and apply that weight in the loss (previously every sample was weighted equally with no mechanism to do otherwise).

---

## Implementation Order

| Step | Task | Files | Effort | Impact | Status |
|---|---|---|---|---|---|
| A.1 | A/B self-play loop | `agent/ab_train.py` (new) | 1 day | Critical | Done |
| A.2 | Multi-deck self-play games | `agent/ab_train.py` | 0.5 day | High | Done |
| B.2 | Expand competitive decks to 16-20 | `competitive_decks.json` | 1 day | High | Done (16) |
| A.3 | Stochastic MCTS rollouts | `agent/search.py` | 0.5 day | Medium | Done |
| A.4 | Training curriculum | `agent/ab_train.py` | 0.5 day | Medium | Done |
| B.4 | Deck validation pipeline | `agent/deck.py` | 0.5 day | Medium | Done |
| A.5 | Model architecture (256 hidden, deck feature) | `agent/network.py`, `agent/features.py` | 0.5 day | Medium | Done |
| C.1 | Multi-deck training integration | `agent/ab_train.py`, `agent/selfplay.py` | 0.5 day | Medium | Done (via A.2) |
| A.6 | Value head integration with MCTS | `agent/search.py` | 1 day | High (after A.1-A.4) | Done, disabled (underperformed) |
| C.3 | Matchup difficulty weighting | `agent/ab_train.py` | 0.5 day | Low | Done |

**Total estimated effort**: ~6 days

**Minimum viable path**: A.1 + A.2 + B.2 (2.5 days) — this gives us self-play training against diverse decks, which addresses the core problem (unbalanced training data).

---

## Expected Impact

| Metric | Current | After A/B Self-Play | After Full Plan |
|---|---|---|---|
| vs competitive decks | 83.8% | 85-88% | 88-92% |
| vs random | 42% | 50-60% | 60-70% |
| Value head discrimination | ~0.95 always | 0.3-0.7 range | 0.2-0.8 range |
| MCTS speed (with value net) | 0.5s/game | Same | 0.05s/game |
| Matchup coverage | 8 decks | 8 decks | 16-20 decks |
| Training data balance | 95% win labels | 50/50 win/loss | 50/50 + weighted |