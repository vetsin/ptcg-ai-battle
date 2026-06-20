# Strategy-Guided MCTS for Mega Lucario ex (NudgeRL-inspired)

## Context

**Origin:** Review of arXiv 2605.15726 — *NudgeRL: "Nudging Beyond the Comfort Zone: Efficient
Strategy-Guided Exploration for RLVR."* That paper's specific algorithm (GRPO/LLM reasoning, verifiable
rewards) **does not transfer** to this AlphaZero-style MCTS + value/policy-net stack. The **transferable
idea** is "strategy nudging": condition exploration on a lightweight, explicit strategy context so search
produces diverse *on-strategy* trajectories, then distill that behavior back into the unconditioned net.

**This plan** applies that idea to a single deck — **Mega Lucario ex** — as a vertical-slice proof of
concept. We derive an explicit strategy profile from two human sources, use it to (1) **seed MCTS priors
(full PUCT)** and (2) **bias self-play trajectory generation**, so the existing `train_policy` distillation
absorbs the strategy. All changes are gated behind "is the agent piloting Mega Lucario?" so non-Lucario
play is unchanged.

**Strategy sources (derived, not parsed):**
- pokemon.com official strategy article for Mega Lucario ex
- limitlesstcg.com/decks/345 (decklist shape)

**Why:** The MCTS today (`agent/search.py`) is flat UCB1 with **no action priors**; expansion is the first
≤10 options and rollouts are 70% heuristic / 30% *uniform* random. There is no notion of the agent's own
game plan. Adding an explicit, card-grounded strategy gives search a meaningful prior and turns the random
rollout branch into a directed "nudge."

## Derived strategy (grounded in engine IDs — verified to exist in the card DB)

Mega Lucario ex is a **hybrid control→combo / single-prize-simulator**: lead with single-prize attackers to
deny easy 2-prize trades, build the combo in the background, finish with multi-prize hits.

| Role | Cards (engine IDs) | Knob |
|---|---|---|
| Primary finishers (win condition) | Mega Lucario ex `678` (Mega Brave `983` = 270/[F][F]), Hariyama `674` (Wild Press `978` = 210/[F][F][F]) | high attack-aggression late / when powered |
| Setup attack (power-up) | Mega Lucario Aura Jab `982` (130/[F]) | medium; enables the finisher |
| Single-prize early pressure | Solrock `676` (Cosmic Beam `980` = 70), Lunatone `675` (Power Gem `979`) | high early-game priority |
| Draw/consistency engine | Solrock+Lunatone combo, Lillie's Determination `1227`, Poké Pad `1152`, Fighting Gong `1142`, Ultra Ball `1121` | establish turns 1–3 |
| Evolution lines | Riolu `333/677/974`→`678`; Makuhita `673`→Hariyama `674` | prioritize evolving toward finishers |
| Energy | basic Fighting `6`, Rock Fighting `20` (blocks damage-counters / retreat-lock) | attach to active finisher first |
| Disruption | Judge `1213`, Boss's Orders `1182`, Hariyama Heave-Ho ability | mid-game weight |
| ACE SPEC | Secret Box `1092` | consistency |

**Phase weighting** (key NudgeRL principle — keep it a soft, turn-gated nudge, never a hard override):
early game favors single-prize attacker IDs; once a finisher is powered (≥2 Fighting attached) or prizes
are low, shift weight to `678`/`674` finisher attacks.

## Plan

### 1. New module `agent/strategies.py`
- `StrategyProfile` dataclass holding the table above: `key_attacker_ids`, `single_prize_ids`,
  `setup_engine_ids`, `evolution_targets`, `energy_ids`, `finisher_attack_ids`, `setup_attack_ids`,
  `disruption_ids`, and phase weights.
- `MEGA_LUCARIO = StrategyProfile(...)` populated with the IDs above. Registry `STRATEGIES: dict[str, StrategyProfile]`.
- `classify_my_deck(deck: list[int]) -> str | None` — for this scope, returns `"mega_lucario_ex"` when card
  `678` is present; else `None`. (Reuse the signature-matching idea from `opponent_model.archetype_signatures`.)
- `strategy_action_bias(opt, state, my_index, profile) -> float` — additive score bias for one option
  (attack/play/evolve/attach), phase-gated.
- `strategy_priors(options, state, my_index, profile) -> list[float]` — normalized prior over options for PUCT.

### 2. Set "my strategy" once per game
- Module-level `_my_strategy` + `set_my_strategy(profile|None)` in `agent/policy.py`.
- Runtime: `agent/main.py` (`_ensure_models_loaded` / deck-request path) calls
  `set_my_strategy(STRATEGIES.get(classify_my_deck(deck)))` at game start, alongside `reset_opponent_models()`.
- Self-play: the harness sets it per game from each player's deck (see §5).

### 3. Bias the heuristic scorers (self-play + MCTS rollout-heuristic path)
Add an additive `strategy_action_bias(...)` term, gated on `_my_strategy is not None`, weighted by a new
constant `STRATEGY_BIAS_WEIGHT`, into the existing scorers in `agent/policy.py`:
- `_score_main_option` (`policy.py:270`), `_score_attack_option`/`_handle_attack` (`:307`,`:497`),
  `_score_attach` (`:417`), `_score_evolve` (`:437`), and evolution/switch selectors as needed.
Because `choose_action` is the single decision entry point for both runtime and self-play, this biases
trajectory generation everywhere with one set of edits.

### 4. Full PUCT in MCTS — `agent/search.py`
- **Priors:** in `mcts_search` expansion (`search.py:137`), compute a per-option prior =
  blend of `strategy_priors(...)` and the **policy-net softmax** (reuse `network.score_actions` exactly as
  `_value_net_signal` already loads `_value_model`). Store `prior` in each `child_stats[i]`.
- **Expansion order:** rank options by prior and expand the **top-10** (replacing the current first-10),
  so strategy decides *which* children exist.
- **Selection:** rewrite `_select_root_child` (`search.py:314`) from UCB1 to PUCT:
  `score = value_norm + c_puct * prior * sqrt(total_visits)/(1+visits)`.
  **Subtlety to handle:** child `value` is heuristic-scaled (hundreds–thousands), so normalize it
  (e.g. `tanh(value/_value_net_scale)` or running min–max over children) before adding the prior term,
  or the prior is drowned out. New constants `_c_puct`, prior-blend weight.
- **Rollout nudge:** in `_get_rollout_action` (`search.py:379`), replace the 30% *uniform* random branch
  with strategy-weighted sampling over options when `_my_strategy` is set (keep 70% heuristic). This is the
  literal "nudge" — diverse but on-strategy rollouts.

### 5. Self-play wiring & distillation
- In `agent/ab_train.py` / `agent/selfplay.py`, call `set_my_strategy(...)` per game from the
  trajectory-player's deck. For this scope only the Lucario side gets a profile; opponents stay on current
  heuristics. (Optional later: profile both sides.)
- **No training-code change needed:** winning on-strategy trajectories flow through the existing
  `filter_winning_trajectories` → `train_policy` path, distilling the strategy into the net; PUCT then blends
  net + strategy priors at inference. C.3 matchup weighting in `ab_train` is unaffected.

### 6. Build the deck
- Add `agent/decks/mega_lucario.py` (or `decks/mega_lucario.csv`) constructing the 60-card list from the
  IDs above (≈3 Riolu/3 Mega Lucario, 2 Makuhita/2 Hariyama, Solrock/Lunatone, 10 Fighting `6` + 3 Rock
  Fighting `20`, the trainer suite, 1 Secret Box `1092`).
- Validate with existing `agent/deck.validate_deck` (60 cards, ≤4 copies/name, ≤1 ACE SPEC) and
  `validate_deck_for_training` (has attacker + trial games).

### Tunables (all conservative defaults, keep nudge soft)
`STRATEGY_BIAS_WEIGHT`, `STRATEGY_PRIOR_WEIGHT` (net-vs-strategy blend), `_c_puct`, value normalization scale.

## Critical files
- **New:** `agent/strategies.py`, `agent/decks/mega_lucario.py` (or `.csv`)
- **Edit:** `agent/search.py` (PUCT: `mcts_search` `:137`, `_select_root_child` `:314`, `_get_rollout_action` `:379`),
  `agent/policy.py` (`set_my_strategy` + bias in scorers `:270`–`:497`), `agent/main.py` (set strategy at game start),
  `agent/ab_train.py` / `agent/selfplay.py` (set strategy per self-play game)
- **Reuse unchanged:** `agent/network.score_actions` (priors/value), `agent/evaluate.evaluate_state`,
  `agent/deck.validate_deck*`, `train_policy` (distillation)

## Verification
1. **Deck builds:** `python3 -c "from agent.deck import validate_deck; from agent.decks.mega_lucario import DECK; print(validate_deck(DECK))"` → 60 cards, valid.
2. **Classification:** `classify_my_deck(DECK) == "mega_lucario_ex"`; a non-Lucario deck → `None` (gating works).
3. **PUCT sanity:** with strategy set, log root-child priors for an ATTACK select — finisher/single-prize
   options should carry higher prior; confirm `_select_root_child` still returns a valid index and search
   completes within the time budget.
4. **A/B head-to-head** (extend the `eval_value_net.py` harness): Lucario **with** strategy+PUCT vs Lucario
   **without** (UCB1, no profile) over N games — mirror + vs `competitive_decks.json`. Expect ≥ break-even,
   ideally a win-rate gain. **Also run vs `random_agent`** to guard against the A.6-style regression
   (must not drop below the plain-MCTS baseline).
5. **Self-play loop:** run a short `ab_train` iteration; confirm trajectories collect, `set_my_strategy`
   fires per game, and `train_policy` runs to completion producing a checkpoint.
6. **Smoke:** `python3 smoke_test.py` (or one `run_game`) with the Lucario deck — no exceptions, game finishes.

## Notes for the implementer (Sonnet)
- Basic Fighting energy is card id `6` (per `ENERGY_CARD_MAP` / the Mamoswine archetype `energy_card`);
  Rock Fighting Energy is `20`. All Pokémon/trainer IDs in the table above were confirmed present in the
  1267-card engine DB via `agent.evaluate.get_card_data()`.
- `CardData` fields: `cardId, name, cardType, retreatCost, hp, weakness, resistance, energyType, basic,
  stage1, stage2, ex, megaEx, tera, aceSpec, evolvesFrom, skills, attacks`. Note `cardId` (not `id`);
  the `get_card_data()` dict is keyed by id. Fighting `energyType == 6`.
- Keep all new behavior gated on `_my_strategy is not None` so the existing Gouging Fire / other decks and
  the current champion `model.pt` continue to behave exactly as today.
