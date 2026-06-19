"""Head-to-head: value-net MCTS vs heuristic-rollout MCTS (SELFPLAY_PLAN.md A.6 gating check)."""
import sys
import time

from agent.deck import load_deck_csv
from agent.opponents import load_competitive_decks
from agent.selfplay import run_game
import agent.search as search


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "ab_training/final_model.pt"
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    search.load_value_model(model_path)
    print(f"Value net loaded: {search._value_net_enabled} (path={model_path})")

    from agent.main import agent as base_agent_fn

    def value_net_agent(obs_dict):
        search._value_net_enabled = True
        return base_agent_fn(obs_dict)

    def baseline_agent(obs_dict):
        search._value_net_enabled = False
        return base_agent_fn(obs_dict)

    our_deck = load_deck_csv()
    opp_decks = [(d["name"], d["deck"]) for d in load_competitive_decks()]

    value_net_wins = 0
    baseline_wins = 0
    draws = 0

    t0 = time.time()
    for i in range(num_games):
        opp_name, opp_deck = opp_decks[i % len(opp_decks)]
        value_net_side = i % 2

        if value_net_side == 0:
            deck0, deck1 = our_deck, opp_deck
            fn0, fn1 = value_net_agent, baseline_agent
        else:
            deck0, deck1 = opp_deck, our_deck
            fn0, fn1 = baseline_agent, value_net_agent

        search._value_net_enabled = True
        traj = run_game(
            deck0=deck0, deck1=deck1, agent0_fn=fn0, agent1_fn=fn1,
            max_steps=2000, game_id=f"vn_{i:03d}", collect_trajectory=False,
        )

        if traj.outcome == value_net_side:
            value_net_wins += 1
        elif traj.outcome == 2:
            draws += 1
        else:
            baseline_wins += 1

        print(f"  game {i}: vs {opp_name}, outcome={traj.outcome}, value_net_side={value_net_side}, turns={traj.num_turns}")

    elapsed = time.time() - t0
    print(f"\nValue-net WR: {value_net_wins}/{num_games} ({value_net_wins/num_games:.1%})")
    print(f"Baseline WR:  {baseline_wins}/{num_games} ({baseline_wins/num_games:.1%})")
    print(f"Draws: {draws}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/num_games:.2f}s/game)")


if __name__ == "__main__":
    main()
