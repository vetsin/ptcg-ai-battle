# PTCG AI Battle — Run & Train Guide

## Setup

```bash
# Python 3.13+, PyTorch 2.0+, NumPy 1.24+
pip install torch numpy
```

The `cg/` directory contains the game engine shared library (`cg.dll` / `libcg.so`) and Python bindings. No additional install needed — just make sure the project root is on `PYTHONPATH` or you run commands from the project root.

## Project Structure

```
ptcg-ai-battle/
├── main.py              # Kaggle entry point — delegates to agent.main.agent()
├── deck.csv             # 60-line deck list (one card ID per line)
├── model.pt             # Trained policy network (~313KB)
├── cg/                  # Game engine bindings
│   ├── api.py           # Observation, State, Option, SelectType, search_begin/step/end
│   ├── game.py          # battle_start, battle_select, battle_finish
│   ├── sim.py           # Low-level ctypes bridge
│   └── utils.py         # Dataclass conversion helpers
├── agent/
│   ├── main.py          # agent() function — decision routing, game-over handling
│   ├── policy.py        # choose_action() — heuristic + model + MCTS blending
│   ├── search.py        # MCTS search via cg search_begin/step/end API
│   ├── evaluate.py      # evaluate_state() — heuristic board evaluation
│   ├── features.py      # encode_observation() — 120-dim state + 40-dim option features
│   ├── network.py       # PTCGPolicyNet, PTCGSimpleNet, train_policy(), score_actions()
│   ├── selfplay.py      # run_game(), run_self_play() — trajectory collection
│   ├── opponents.py     # random_agent, slightly_smart_agent, tier decks
│   ├── curriculum.py    # Curriculum-only self-play (no model training)
│   ├── si_train.py      # Iterative self-improvement loop (self-play → filter → train)
│   └── deck.py          # Deck loading, validation, default deck definition
├── data/                 # Card data CSVs, sample submission
└── si_training/          # Output from training runs
```

## Quick Run — Single Game

```python
from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
from agent.main import agent
from agent.deck import load_deck_csv

deck = load_deck_csv()
obs_dict, _ = battle_start(deck, deck)
obs = to_observation_class(obs_dict)

for step in range(500):
    if obs.current is None or (obs.current and obs.current.result != -1):
        break
    if obs.select is None:
        obs_dict = battle_select(deck)
        obs = to_observation_class(obs_dict)
        continue
    action = agent(obs_dict)
    obs_dict = battle_select(action)
    obs = to_observation_class(obs_dict)

battle_finish()
print(f"Winner: {obs.current.result}")
```

## Self-Play Evaluation

```python
from agent.selfplay import run_self_play
from agent.main import agent
from agent.opponents import random_agent, slightly_smart_agent, get_deck_for_tier
from agent.deck import load_deck_csv

deck = load_deck_csv()

# vs random (easy)
result = run_self_play(deck, agent, num_games=50, opponent_fn=random_agent)
print(f"Win rate: {result.wins / result.total_games:.1%}")

# vs slightly_smart (hard) with a different deck
result = run_self_play(deck, agent, num_games=50,
                       opponent_fn=slightly_smart_agent,
                       opponent_deck=get_deck_for_tier("hard"))
```

## Training

### Iteration 0 — Collect trajectories, no model yet

The agent starts as a pure heuristic (no model). Self-play generates trajectories, winning games are filtered, and a model is trained on them.

### Iterative Self-Improvement (SI Training)

This is the main training loop. It runs self-play at increasing difficulty tiers, filters winning trajectories, and fine-tunes the policy network.

```bash
# Full SI loop: 3 iterations × 3 tiers, 50 games each
python -m agent.si_train
```

Or programmatically:

```python
from agent.si_train import run_SI_loop, SITrainConfig
from agent.main import agent

config = SITrainConfig(
    num_iterations=3,           # Number of SI iterations
    games_per_iteration=50,     # Games per tier per iteration
    difficulty_tiers=["easy", "medium", "hard"],
    output_dir="si_training/run_003",
    epochs=15,                 # Training epochs per tier
    hidden_dim=128,            # Network hidden dim
    reward_threshold=0.5,      # Min reward to include trajectory
    max_steps_per_game=2000,
)

stats = run_SI_loop(agent, config)
for s in stats:
    print(f"  Iter {s.iteration} Tier {s.tier}: "
          f"WR={s.win_rate:.1%} "
          f"Steps={s.filtered_trajectories} "
          f"Loss={s.train_loss:.4f} "
          f"Acc={s.train_acc:.3f}")
```

### SI Train Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_iterations` | 3 | Number of outer SI loops |
| `games_per_iteration` | 100 | Games played per tier per iteration |
| `reward_threshold` | 0.5 | Minimum reward for trajectory inclusion |
| `min_trajectory_steps` | 3 | Min steps for a trajectory to be usable |
| `difficulty_tiers` | easy/medium/hard | Tier progression order |
| `output_dir` | si_training | Where models/data are saved |
| `model_type` | simple | "simple" (PTCGSimpleNet) or "attention" (PTCGPolicyNet) |
| `hidden_dim` | 128 | Network hidden dimension |
| `learning_rate` | 1e-3 | Adam LR |
| `epochs` | 20 | Training epochs per tier |
| `batch_size` | 64 | Training batch size |
| `max_options` | 20 | Max options padded per state |
| `device` | auto | "auto", "cpu", or "cuda" |
| `policy_blend` | 0.7 | Model vs heuristic blend ratio |

### Curriculum-Only (No Model Training)

If you just want to collect trajectory data without training a model:

```python
from agent.curriculum import run_curriculum, CurriculumConfig
from agent.main import agent

config = CurriculumConfig(
    num_iterations=1,
    games_per_iteration=20,
    difficulty_tiers=["easy", "medium"],
    output_dir="training_data",
)
results = run_curriculum(agent, config)
```

### Training Output

Each run produces:

```
si_training/run_XXX/
├── iteration_000/
│   ├── easy/
│   │   ├── all_trajectories.jsonl
│   │   ├── winning_trajectories.jsonl
│   │   ├── train.jsonl
│   │   ├── train_records.jsonl
│   │   ├── metadata.json
│   │   ├── model.pt
│   │   └── stats.json
│   ├── medium/ ...
│   └── hard/ ...
├── iteration_001/ ...
└── si_log.txt
```

The latest `model.pt` from the hardest tier of the last iteration is the best model.

## MCTS Search

The agent uses MCTS for MAIN and ATTACK selections with ≥3 options. It runs a flat expansion from root (up to 8 children), then does rollouts using the heuristic policy.

```python
from agent.search import mcts_search, build_search_inputs
from cg.api import to_observation_class

obs = to_observation_class(obs_dict)
if obs.search_begin_input is not None:
    inputs = build_search_inputs(obs)
    result = mcts_search(obs, max_simulations=40, max_time_ms=1500, **inputs)
    # result is a list[int] of option indices, or None
```

MCTS parameters are configurable in `agent/policy.py`:
- `_mcts_enabled = True`
- `_mcts_max_time_ms = 1500.0` (ms per decision)
- `_mcts_min_options = 3` (minimum options to trigger MCTS)

## Opponent Tiers

| Tier | Agent | Deck |
|---|---|---|
| easy | Random agent | Basic Fire Deck |
| medium | 50% random / 50% smart | Gouging Fire ex (our deck) |
| hard | Slightly smart heuristic | Gouging Fire ex + supporters |
| expert | Our agent (mirror) | Our deck |

## Model Integration

The policy blends model and heuristic decisions:

```python
import agent.policy as policy_mod
policy_mod.load_policy_model("model.pt")  # or None for auto-discovery
policy_mod._policy_blend = 0.7  # 70% model, 30% heuristic
policy_mod._mcts_enabled = True  # Enable MCTS for MAIN/ATTACK
```

## Deck

The default deck is a Gouging Fire ex competitive deck defined in `agent/deck.py`. To change the deck, edit `deck.csv` (one card ID per line, 60 lines). Current composition:

| Card | ID | Count |
|---|---|---|
| Gouging Fire ex | 46 | 4 |
| Charmander | 788 | 4 |
| Mega Charizard X ex | 790 | 2 |
| Rare Candy | 1079 | 4 |
| Ultra Ball | 1121 | 4 |
| Boss's Orders | 1182 | 4 |
| Switch | 1123 | 4 |
| Buddy-Buddy Poffin | 1086 | 4 |
| Firebreather | 1232 | 2 |
| Energy Search | 1119 | 2 |
| Night Stretcher | 1097 | 2 |
| Lillie's Determination | 1227 | 2 |
| Basic Fire Energy | 2 | 22 |

## Kaggle Submission

The submission format is:

```
submission/
├── main.py       # Must define agent(obs_dict) -> list[int]
├── deck.csv       # 60 card IDs, one per line
├── model.pt       # Trained policy (optional)
└── cg/            # Game engine (provided by Kaggle)
```

`main.py` delegates to `agent.main.agent()`. When `obs_dict` has no `select` key or `obs.select is None`, the agent returns the deck list. Otherwise it routes through MCTS → model → heuristic.

## Key Design Decisions

- **Policy blending**: 70% neural model / 30% heuristic when model is loaded, pure heuristic otherwise
- **MCTS**: Flat 8-branch expansion from root, heuristic rollouts, UCB selection, 40 simulations max, 1.5s time limit
- **Feature dims**: 120 state features, 40 option features
- **Network**: PTCGSimpleNet (128 hidden, ~78K params)
- **Training**: Cross-entropy on action selection + MSE on value prediction
- **Self-improvement**: Easy-to-hard curriculum; winning trajectories are used for supervised fine-tuning