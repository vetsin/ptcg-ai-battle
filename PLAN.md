# PTCG AI Battle — Improvement Plan

Based on analysis of the PokeAgent Challenge (arXiv:2603.15563) winning solutions and our current architecture.

## Current Performance (after all phases)

| Metric | Before | After |
|---|---|---|
| Win rate vs easy random | ~80% | **86%** |
| Win rate vs hard smart | ~70% | **78%** |
| Win rate vs mirror smart | ~50% | **53%** |
| Overall | ~67% | **75%** |

### Phase 1 (COMPLETE) — MCTS + Eval improvements
- Outcome-grouped MCTS: classify children into win/loss/big_gain/neutral groups, prune dominated
- Adaptive simulation budget: 1.5x in late game, 0.6x for simple decisions
- Adaptive time budget: 1.5x when prizes taken >= 3
- Adaptive rollout depth: +5 depth in late game
- Early termination: stop if best action has >65% visits
- Evaluation: prize acceleration, knockout threat, deck-out detection, multi-status penalty

### Phase 2 (COMPLETE) — Bayesian opponent prediction
- Deck archetype database (16 archetypes from competitive meta)
- OpponentModel class: Bayesian updates from observed cards
- Energy-type inference from opponent's Pokemon attacks
- Sampled deck construction from best archetype + known cards
- Hand/prize/active prediction using archetype-appropriate cards
- Integrated into MCTS search via build_search_inputs()

### Phase 3 (COMPLETE) — Deck optimization
- 9 deck variants evaluated in tournament (20 games x 3 matchups each)
- Base deck (Gouging Fire ex) confirmed best at 81.7% overall
- Boss's Orders x2 variant close at 80%

### Phase 4 (COMPLETE) — Verification
- All smoke tests pass (9/9)
- Comprehensive evaluation: 86% easy, 78% hard smart, 53% mirror
| Win rate vs mirror (self) | ~50% |
| SI training (2 iter × 3 tiers) | loss 1.28→1.10, acc 0.71→0.72 |
| MCTS per decision (smart gating) | ~20ms |
| Game time without MCTS | ~0.01s |
| Game time with MCTS | ~0.19s |

## Phase 1: Outcome-Grouped MCTS (FoulPlay-inspired)

**Problem:** Current MCTS explores every coin flip, damage roll, and search branch individually. FoulPlay won Gen 9 OU by grouping damage outcomes by knockout potential instead of exploring all 32 outcome variations.

**Implementation:** `agent/search.py`

### 1.1 Outcome clustering in search_step expansion

Current behavior: expand up to 8 children from root, then rollout with heuristic policy for 5 steps. This explores each branch independently.

Change:
- After expanding root children, classify each into outcome groups:
  - **Knockout**: opponent active would be KO'd (HP ≤ 0 after damage)
  - **High damage**: opponent HP reduced below 50%
  - **Low damage**: everything else
- Within each group, pick the highest-value child and skip redundant siblings
- For MAIN decisions, use our existing `evaluate_state` as the group classifier

### 1.2 Prior-knowledge rollout policy

Current `_rollout_from_node` uses the heuristic policy for random selections. Change:
- Use the neural model (if loaded) for rollout decisions instead of random selection
- This makes rollouts higher-quality, so fewer simulations needed
- Fallback to heuristic when model unavailable

### 1.3 Adaptive simulation budget

Current: fixed 60 simulations, 2000ms timeout.

Change:
- If root has only 2-3 options, reduce to 20 simulations (less branching = less need)
- If root has 8+ options, keep 60 simulations
- Early termination: if best action has >60% of visits, stop early

### 1.4 Deeper evaluation at terminal nodes

Current `evaluate_state` uses hand-crafted weights. Change:
- Add knockout bonus scaling (prizes remaining × prize advantage amplification)
- Add deck-out risk detection (if deck < 5 cards, penalize heavily)
- Cache card_data and attack_data lookups (already cached but verify hot path)

**Expected impact:** 2-5% win rate improvement from better MCTS decisions, possibly faster due to pruning.

---

## Phase 2: Bayesian Opponent Prediction (FoulPlay + PokéChamp)

**Problem:** Currently `predict_opponent_deck/hand/active` fills unknowns with basic energy (card ID 3). This is terrible — the MCTS search world becomes unrealistic, making search results unreliable.

**FoulPlay approach:** Maintain a probability distribution over opponent sets, progressively narrowing from observed damage, move patterns, and game actions.

**Implementation:** New file `agent/opponent_model.py`

### 2.1 Opponent deck prediction

Create a deck archetype database from our card data:

```python
# From the 1267 unique cards, build common deck archetypes
ARCHETYPES = {
    "gouging_fire_ex": [(46, 4), (788, 4), (790, 2), ...],  # our deck
    "charizard_ex": [...],
    "miraidon_ex": [...],
    "dragapult_ex": [...],
    # ... one per competitive archetype
}
```

During a game, maintain a belief distribution over which archetype the opponent is playing:
- **Prior:** uniform over archetypes, or weighted by meta-game frequency
- **Update rule:** each time we observe an opponent card (active, bench, discard, prize revealed), compute `P(archetype | observed_cards)` using Bayes' rule
- **Prediction:** sample a deck from `P(archetype)` for `search_begin`

### 2.2 Opponent hand prediction

Current: `[3] * opp.handCount` (all basic water energy).

Change:
- Weight towards cards that are in the archetype but not yet seen
- Prioritize supporter and Pokémon cards over energy for hand prediction
- Use deck archetype probabilities to weight

### 2.3 Opponent active prediction (face-down)

Current: `[3]` (single basic water energy) when opponent's active is face-down.

Change:
- Sample from the remaining cards in the most likely archetype
- If opponent has benched Pokémon, we know their evolutionary lines — predict active from those

### 2.4 Integration with MCTS

In `build_search_inputs()`:
- Call opponent model to get sampled deck/hand/prize/active predictions
- Optionally: run MCTS multiple times with different sampled predictions (2-3 samples), take majority vote

**Expected impact:** 5-10% win rate improvement. This is the biggest single change — realistic opponent modeling makes MCTS actually useful instead of misleading.

---

## Phase 3: Deck Optimization via Tournament Selection (PA-Agent)

**Problem:** Our deck is hand-crafted. PA-Agent evaluated candidate teams against 50+ lineups and kept only decks with >60% win rate.

**Implementation:** New file `agent/deck_optimize.py`

### 3.1 Deck variant generation

Start from our current deck. Generate variants by:
- Swapping Pokémon lines (e.g., more Charmander x4 → less Gouging Fire)
- Adjusting supporter counts (Boss's Orders x3 vs x4)
- Adjusting energy counts (20 vs 22 vs 24 Fire Energy)
- Swapping tech cards (Night Stretcher vs. Pal Pad vs. Iono)
- Using a card importance scoring from card data (HP, damage, abilities)

### 3.2 Tournament evaluation

For each candidate deck:
1. Play 50 games against each of N opponent deck lineups
2. Keep decks with >55% overall win rate
3. Discard the rest
4. Repeat for 10-20 generations with mutation

Opponent lineups to test against:
- Our current deck (mirror match)
- The top competitive archetypes from card data
- Random decks as baseline

### 3.3 Deck validation

Every generated deck must pass `validate_deck()`:
- Exactly 60 cards
- Max 4 of each card (except basic energy)
- At most 1 ACE SPEC
- Must contain at least 1 basic Pokémon

### 3.4 Tournament format

```python
def evaluate_deck(deck, opponent_lineups, games_per_matchup=50):
    total_wins = 0
    total_games = 0
    for opp_deck in opponent_lineups:
        result = run_self_play(deck, agent, games_per_matchup, opponent_deck=opp_deck)
        total_wins += result.wins
        total_games += result.total_games
    return total_wins / max(total_games, 1)
```

**Expected impact:** 3-8% win rate improvement from better deck construction. Mirror match performance especially sensitive to deck tuning.

---

## Phase 4: Verify against PokéAgent Paper Techniques

After re-reading arXiv:2603.15563, verify we haven't missed:

### 4.1 Team preview / matchup adaptation
- **Paper:** PA-Agent uses tournament selection evaluating candidate teams against 50+ lineups
- **PTCG:** We submit a fixed 60-card deck, no team preview. Not applicable.

### 4.2 Data weighting / curriculum
- **Paper:** PA-Agent uses dynamic data weighting (100% human → 10% human over 6 rounds)
- **PTCG:** We don't have human demonstration data, but our SI curriculum (easy → medium → hard) serves the same purpose. Could add explicit data weighting where recent tiers get higher loss weight.

### 4.3 DUCT (Decoupled UCB for Trees)
- **Paper:** FoulPlay uses DUCT for simultaneous move selection (both players act at once)
- **PTCG:** Turn-based, not simultaneous. Standard UCB is correct. Not applicable.

### 4.4 Damage roll grouping
- **Paper:** FoulPlay clusters damage outcomes into knockout groups (2 groups instead of 32)
- **PTCG:** We use `manual_coin=True` in `search_begin`, which lets the engine handle coin flips deterministically or we control them. Outcome grouping for PTCG means: grouping coin flip outcomes (heads vs tails) and damage roll variations. Our MCTS currently doesn't do this — each `search_step` call includes the engine's outcome. Phase 1's grouping addresses this.

### 4.5 Set prediction from observations
- **Paper:** FoulPlay infers opponent stats from observed damage, move priority, weather duration
- **PTCG:** We see the opponent's active, bench, discard, and any revealed prize cards. Our `predict_opponent_deck/hand/active` is naive. Phase 2 addresses this.

### 4.6 Two-phase curriculum learning
- **Paper:** Team Q uses Mechanics Phase (vs heuristic bots) → Strategy Phase (vs coach + self-play)
- **PTCG:** We already do this with easy/medium/hard tiers. Our "easy" tier is random play, "medium" is 50/50 random/smart, "hard" is slightly_smart. Could add a "mechanics" phase that only tests basic rules compliance.

### 4.7 LLM-based evaluation
- **Paper:** PokéChamp uses LLM as position evaluator in minimax
- **PTCG:** Not practical for Kaggle (no LLM access in evaluation). Our `evaluate_state` heuristic is the equivalent, and it's fast.

### 4.8 20M+ trajectory dataset
- **Paper:** Releases 4M human + 18M synthetic battles
- **PTCG:** This is for the video game, not TCG. Not directly usable, but the format (RL trajectories with private info) is exactly what our SI training generates.

### 4.9 Search depth management
- **Paper:** FoulPlay transitions from expectiminimax (5-turn depth limit) to MCTS (10+ turns) mid-game
- **PTCG:** Our MCTS does flat expansion from root (8 children) + 5-step rollout. Could add deeper search in late-game (when fewer options per turn, more critical decisions). Implement as: increase `max_simulations` from 40 to 80 and `max_depth` from 5 to 10 when `prizes_taken >= 3`.

### 4.10 Custom engine for speed
- **Paper:** FoulPlay built `poke-engine` in Rust for 100x speedup over Python
- **PTCG:** Our `cg/` engine is already compiled C++. Can't modify or replace it (Kaggle provides it). Speed gains must come from reducing calls (which we've done with smart MCTS gating).

---

## Implementation Order

| Phase | Task | Priority | Estimated Effort | Expected WR Gain |
|---|---|---|---|---|
| 1.1 | MCTS outcome pruning | high | 1 day | 1-2% |
| 1.2 | Model-guided rollouts | high | 0.5 day | 1-2% |
| 1.3 | Adaptive sim budget | medium | 0.5 day | 0-1% |
| 1.4 | Evaluation improvements | medium | 0.5 day | 1-3% |
| 2.1 | Opponent deck archetypes | high | 2 days | 5-10% |
| 2.2 | Opponent hand prediction | high | 1 day | 2-3% |
| 2.3 | Opponent active prediction | medium | 0.5 day | 1-2% |
| 2.4 | Multi-sample MCTS | medium | 1 day | 1-2% |
| 3.1 | Deck variant generation | medium | 1 day | — |
| 3.2 | Tournament evaluation | medium | 1 day | 3-8% |
| 3.3 | Deck validation | low | 0.5 day | — |
| 3.4 | Full tournament | medium | 2 days | — |
| 4.9 | Late-game deeper search | low | 0.5 day | 1-2% |

**Total estimated improvement:** 15-30% win rate improvement across all phases.

**Critical path:** Phase 2 (opponent prediction) is the highest-impact single change. Our current MCTS is searching against unrealistic opponent models, making it barely better than heuristic. With realistic opponent prediction, MCTS should improve substantially.

## Testing Strategy

After each phase:
1. Run `python smoke_test.py --verbose` (quick regression)
2. Run self-play evaluation: 50 games vs random, 30 games vs slightly_smart, 30 games mirror
3. Compare win rates against baseline
4. If WR drops, bisect and fix before proceeding