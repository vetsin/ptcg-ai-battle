# PTCG AI Battle — Submission Guide

## Architecture

```
main.py ──► agent/main.py :: agent(obs_dict) → list[int]
                 │
                 ├─► agent/policy.py :: choose_action()
                 │       ├─► MCTS (agent/search.py) for important MAIN/ATTACK
                 │       ├─► Neural model (agent/network.py) for MAIN/ATTACK/CARD
                 │       └─► Heuristic fallback (agent/policy.py)
                 │
                 ├─► agent/evaluate.py :: board evaluation heuristic
                 ├─► agent/features.py :: 120-dim state + 40-dim option
                 └─► agent/deck.py :: deck loading/validation
```

**Decision flow per selection:**

1. If `obs.select is None` → return deck (60 card IDs)
2. If game over (`current.result != -1`) → return `[]`
3. If 0 options → return `[]`
4. If 1 option → return `[0]`
5. If MAIN/ATTACK with ≥4 options and attack/play present → MCTS search (40 sims, 1.5s timeout)
6. If model loaded and MAIN/ATTACK/CARD → blend model (70%) / heuristic (30%)
7. Fallback → heuristic policy

## Submitting to Kaggle

### 1. Run the smoke test

```bash
python smoke_test.py
```

Expected output:
```
============================================================
PTCG AI Battle — Kaggle Smoke Test
============================================================
  [PASS] 1. Submission files
  [PASS] 2. Import agent
  [PASS] 3. Deck return (60 cards)
  [PASS] 4. Deck CSV format
  [PASS] 5. Model loads
  [PASS] 6. Full game
  [PASS] 7. Action validation
  [PASS] 8. Select types
  [PASS] 9. vs Random agent
ALL PASSED — ready to submit
```

For verbose output (more games, more detail):
```bash
python smoke_test.py --verbose
```

To skip slow tests:
```bash
python smoke_test.py --quick
```

### 2. Package the submission

```bash
./submit.sh pack
```

This creates `ptcg-agent-submission.tar.gz` (1.4 MB) containing:

| File | Purpose |
|---|---|
| `main.py` | Entry point — exports `agent()` |
| `deck.csv` | 60 card IDs (Gouging Fire ex) |
| `model.pt` | Trained policy network (313 KB) |
| `cg/*` | Game engine bindings (provided by Kaggle) |
| `agent/main.py` | Decision routing, game-over handling |
| `agent/policy.py` | Heuristic + model + MCTS action selection |
| `agent/search.py` | MCTS via `search_begin/search_step/search_end` API |
| `agent/evaluate.py` | Board state evaluation heuristic |
| `agent/features.py` | Feature encoding for neural model |
| `agent/network.py` | PTCGSimpleNet, training, inference |
| `agent/deck.py` | Deck loading/validation |

### 3. Upload to Kaggle

**Option A: Web upload**

1. Go to the competition page
2. Click "Submit" → "Upload Submission"
3. Upload the `ptcg-agent-submission.tar.gz` file

**Option B: Kaggle CLI** (if installed)

```bash
pip install kaggle
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

kaggle competitions submit -c ptcg-ai-battle \
    -f ptcg-agent-submission.tar.gz \
    -m "Gouging Fire ex + MCTS + SI-trained model v1"
```

### 4. Kaggle simulation environment

The Kaggle runner expects:

- **`/kaggle_simulations/agent/main.py`** — must define `agent(obs_dict: dict) -> list[int]`
- **`/kaggle_simulations/agent/deck.csv`** — 60 card IDs, one per line
- **`/kaggle_simulations/agent/cg/`** — game engine (provided by Kaggle, but we ship our copy)
- **`/kaggle_simulations/agent/model.pt`** — our trained model (optional, falls back to heuristic)

The `deck.csv` loading path in `agent/deck.py` handles both local and Kaggle paths.

### 5. Validate on Kaggle

After submission, check the Kaggle output log for:

- No `ImportError` or `ModuleNotFoundError`
- No `IndexError` from invalid action indices
- Game completes within time limits
- Agent returns valid actions for all `SelectType` variants

## Training

### Quick training (GPU recommended)

```bash
python -m agent.si_train
```

Default: 2 iterations × 2 tiers (easy, medium), 20 games each.

### Full training

```python
from agent.si_train import run_SI_loop, SITrainConfig
from agent.main import agent

config = SITrainConfig(
    num_iterations=3,
    games_per_iteration=100,
    difficulty_tiers=["easy", "medium", "hard"],
    output_dir="si_training/run_003",
    epochs=15,
    device="cuda",  # use GPU
)

stats = run_SI_loop(agent, config)
```

Output goes to `si_training/run_003/` with `model.pt` per iteration/tier. Copy the latest to the project root:

```bash
cp si_training/run_003/iteration_002/hard/model.pt model.pt
```

### Performance benchmarks

| Config | Speed | Win Rate vs Hard |
|---|---|---|
| No MCTS | 0.01s/game | ~45% |
| MCTS (min_options=4, smart gating) | 0.19s/game | ~50% |
| MCTS (min_options=3, all MAIN) | ~1.0s/game | ~50% |

MCTS adds ~20x overhead but only improves WR by ~5pp. For training runs, disable MCTS with:

```python
import agent.policy as policy_mod
policy_mod._mcts_enabled = False
```

## Key files to edit

| Goal | Edit |
|---|---|
| Change deck | `deck.csv` and `agent/deck.py` → `_DEFAULT_DECK_DESCRIPTION` |
| Tune MCTS | `agent/policy.py` → `_mcts_enabled`, `_mcts_max_time_ms`, `_mcts_min_options` |
| Tune heuristic | `agent/policy.py` → `_handle_main`, `_handle_attack`, etc. |
| Change model blend | `agent/policy.py` → `_policy_blend` (0.0–1.0) |
| Change network | `agent/network.py` → `PTCGSimpleNet` or `PTCGPolicyNet` |
| Change features | `agent/features.py` → `NUM_STATE_FEATURES`, `NUM_OPTION_FEATURES` |

## Troubleshooting

| Issue | Fix |
|---|---|
| `ImportError: No module named 'cg'` | Run from project root, or add project root to `PYTHONPATH` |
| `IndexError` from `battle_select` | Agent returned invalid action — check minCount/maxCount |
| MCTS timeout | Reduce `_mcts_max_time_ms` or `_mcts_simulations` in `search.py` |
| Model not loading | Check `model.pt` exists in project root, or set `load_policy_model(path)` |
| Low win rate | Run more SI training iterations, or tune heuristic in `policy.py` |