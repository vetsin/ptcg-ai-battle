# PTCG AI Battle — Notes

## Process Scaling Benchmark

16-game runs with `slightly_smart_agent`, `fork` context, 16-core machine:

| Workers | Total Time | per-game | Speedup |
|---------|------------|----------|---------|
| 1       | 8.5s       | 0.533s   | 1.0x    |
| 2       | 6.9s       | 0.430s   | 1.2x    |
| 4       | 2.8s       | 0.176s   | 3.0x    |
| 8       | 5.8s*      | 0.361s*  | 1.5x*   |
| 12      | 1.4s       | 0.088s   | 6.1x    |
| 16      | 1.6s       | 0.097s   | 5.3x    |

*Noisy due to game-length variance. Clean benchmark with 8 workers got 7.2x.

**Recommendation**: 8 workers. Sweet spot is 4-12; beyond 12, memory overhead and game-length imbalance eat the gains.

`fork` context is used (not `spawn`) because:
- `cg/` engine is process-safe but not thread-safe (global `Battle.battle_ptr`)
- `fork` copies existing process memory (fast, no re-import of torch/cg)
- `spawn` re-imports everything per worker (very slow startup)


## ChessTransformer Evaluation

Source: https://github.com/tchauffi/ChessTransformer

### What it does
- 11.7M parameter transformer, trained **purely on human chess games** (no self-play, no RL)
- AlphaZero-style MCTS (PUCT) using policy priors + value head
- ~2100 Elo vs Stockfish
- MCTS adds **+850 Elo** over raw policy (1327 → 2175 from 25 → 800 sims)

### Key techniques we should adopt
1. **Policy head as MCTS prior** — our MCTS currently doesn't use the network at all for selection
2. **Value head replacing rollouts** — one forward pass instead of 5-step heuristic rollout
3. **Batched MCTS leaf evaluation** — evaluate multiple leaves in one GPU forward pass
4. **Tree reuse** — re-root MCTS tree under the move actually played
5. **PUCT with FPU** — first-play-urgency (0.2) and c_puct=1.0 tuned for exploitation

### What transfers to PTCG
| Technique | Chess | PTCG | Notes |
|-----------|-------|------|-------|
| Policy+value MCTS | Yes | Yes | Our highest-priority next step |
| Batched leaf eval | Yes | Yes | GPU forward pass for multiple leaves |
| Tree reuse | Yes | Partial | PTCG state changes more between turns; limited benefit |
| PUCT search | Yes | Yes | Already using UCB, should add policy priors |
| Train on human data | Yes | NO | Zero human PTCG data exists; must self-play |

### What does NOT transfer
- **Human game training**: No PTCG game datasets exist. Must generate via self-play.
- **Fixed action space**: Chess has 4672 max moves (64×73). PTCG has 1-40 variable options per turn.
- **Perfect information**: Chess is perfect-info. PTCG has hidden info (opponent hand, deck, face-down cards).
- **Fast simulation**: Chess microsecond-per-move. PTCG ~0.5s/game with MCTS.


## Current Agent Status

| Metric | Value |
|--------|-------|
| vs competitive decks (16) | 71.2% WR |
| vs slightly_smart | 67% WR |
| vs random | 40-50% (variance) |
| MCTS budget | 60 sims, 1500ms, 5-step heuristic rollouts |
| Stochastic rollouts | 70% heuristic / 30% random |
| Model | PTCGSimpleNet (128 hidden, untrained value head) |
| Deck | Gouging Fire ex (all-ex attackers) |


## Known Weaknesses

- **40% vs Crustle** (ex-immunity wall)
- **40% vs Slowking**: partly a real hard matchup (psychic control), but also partly an intermittent engine bug — `battle_start` returns null for the Slowking deck roughly 30-50% of the time after a process has already run other battles (0/30-40 crashes when it's the very first battle in a fresh process; clearly nonzero, clustered crashes after prior games). Static legality passes (`validate_deck`), so this isn't a deck-construction bug like Hydrapple/Archaludon — it's inside the closed-source `libcg.so`, not fixable from this codebase. `validate_deck_for_training`'s majority-vote pre-flight check correctly catches this and skips the deck for that training run when it trips (working as intended), but it means Slowking training coverage is flaky run-to-run. True skill-based WR vs Slowking is likely higher than the recorded 40% once crash-losses are excluded — don't trust that number at face value.
- **MCTS rollouts assume rational opponent** (42% vs random)

### Fixed

- **0% vs Hydrapple ex / Archaludon ex**: these weren't matchup losses — both decks were illegal (Hydrapple ex had 5x Rare Candy; Archaludon ex had 8x "Duraludon" across two printings, ids 839+169, violating the real max-4-per-name rule). `battle_start` silently rejected them, recorded as `outcome=-1, turns=0`. Fixed in `competitive_decks.json` + added name-based dedup check to `agent/deck.py::validate_deck` (previously only checked per-card-ID counts, not per-name across reprints).
- **Value head outputs ~0.95 always**: fixed by `agent/ab_train.py` A/B self-play (see SELFPLAY_PLAN.md Phase A).