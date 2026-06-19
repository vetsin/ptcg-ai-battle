import csv
import os

from cg.api import CardData, CardType, EnergyType, all_card_data

_card_db: dict[int, CardData] | None = None


def get_deck_card_db() -> dict[int, CardData]:
    global _card_db
    if _card_db is None:
        _card_db = {c.cardId: c for c in all_card_data()}
    return _card_db


def load_card_csv(path: str | None = None) -> list[dict]:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "EN_Card_Data.csv")
    cards = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards.append(row)
    return cards


def validate_deck(deck: list[int]) -> list[str]:
    card_db = get_deck_card_db()
    errors = []

    if len(deck) != 60:
        errors.append(f"Deck must have exactly 60 cards, got {len(deck)}")

    from collections import Counter
    counts = Counter(deck)

    ace_spec_ids = set()
    name_counts: dict[str, int] = {}
    name_types: dict[str, CardType] = {}
    for cid, count in counts.items():
        cd = card_db.get(cid)
        if cd is None:
            errors.append(f"Unknown card ID: {cid}")
            continue
        if cd.aceSpec:
            ace_spec_ids.add(cid)
        if count > 4 and cd.cardType != CardType.BASIC_ENERGY:
            errors.append(f"Card {cid} ({cd.name}): max 4 copies allowed, got {count}")
        if cd.cardType != CardType.BASIC_ENERGY:
            name_counts[cd.name] = name_counts.get(cd.name, 0) + count
            name_types[cd.name] = cd.cardType

    if len(ace_spec_ids) > 1:
        errors.append(f"Only 1 ACE SPEC card allowed in deck, got {len(ace_spec_ids)}: {ace_spec_ids}")

    for name, total in name_counts.items():
        if total > 4:
            errors.append(f"Card name '{name}': max 4 copies allowed across all printings, got {total}")

    return errors


def validate_deck_for_training(deck: list[int], name: str = "deck", max_steps: int = 300) -> list[str]:
    """Pre-flight check before using a deck in training/self-play (SELFPLAY_PLAN.md B.4).

    Catches decks that pass `validate_deck`'s static legality check but still fail
    at runtime (engine-level rejection, no attackers, etc.) so they don't silently
    consume training games as instant losses (battle_start returns None -> outcome=-1).
    """
    errors = validate_deck(deck)
    if errors:
        return [f"[{name}] {e}" for e in errors]

    card_db = get_deck_card_db()
    has_attacker = False
    for cid in set(deck):
        cd = card_db.get(cid)
        if cd and cd.cardType == CardType.POKEMON and cd.attacks:
            has_attacker = True
            break
    if not has_attacker:
        errors.append(f"[{name}] No Pokemon with attacks found in deck")

    from cg.game import battle_start, battle_finish

    try:
        obs_dict, _ = battle_start(deck, deck)
    except Exception as e:
        return [f"[{name}] battle_start raised {type(e).__name__}: {e}"]

    if obs_dict is None:
        errors.append(f"[{name}] battle_start rejected the deck (engine-level legality failure)")
        return errors

    try:
        battle_finish()
    except Exception:
        pass

    from agent.selfplay import run_game
    from agent.opponents import random_agent

    # random_agent play is stochastic, so a single crashing trial can be a rare edge
    # case rather than a structurally broken deck. Run a few trials and only flag the
    # deck if a majority crash immediately (outcome=-1, turns=0 means battle_start/engine
    # rejected it outright, not just a slow/incomplete game).
    trials = 3
    crashes = 0
    for t in range(trials):
        try:
            traj = run_game(
                deck0=deck, deck1=deck,
                agent0_fn=random_agent, agent1_fn=random_agent,
                max_steps=max_steps, game_id=f"validate_{name}_{t}",
                collect_trajectory=False,
            )
        except Exception as e:
            errors.append(f"[{name}] run_game raised {type(e).__name__}: {e}")
            return errors

        if traj.outcome == -1 and traj.num_turns == 0:
            crashes += 1

    if crashes > trials // 2:
        errors.append(f"[{name}] game crashed immediately in {crashes}/{trials} trials (outcome=-1, turns=0)")

    return errors


def build_charizard_ex_deck() -> list[int]:
    deck = [
        721, 721, 721,  # Pidgey x3
        722, 722,       # Pidgeotto x2
        723, 723, 723, 723,  # Pidgeot ex x4
        1145, 1145, 1145, 1145,  # Charmander x4
        1205, 1205,     # Charmeleon x2
        1158,            # Charizard ex x1 (actually let me check)
        1227, 1227, 1227, 1227,  # Rare Candy x4
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # Basic Fire Energy x10
    ]
    return deck


def build_deck_from_description(description: list[tuple[int, int]]) -> list[int]:
    deck = []
    for card_id, count in description:
        deck.extend([card_id] * count)
    return deck


def get_default_deck() -> list[int]:
    return _DEFAULT_DECK


_DEFAULT_DECK_DESCRIPTION = [
    (46, 4),    # Gouging Fire ex (main attacker)
    (788, 4),   # Charmander (starter/backup)
    (790, 2),   # Mega Charizard X ex (late game)
    (1079, 4),  # Rare Candy
    (1121, 4),  # Ultra Ball
    (1182, 4),  # Boss's Orders
    (1123, 4),  # Switch
    (1086, 4),  # Buddy-Buddy Poffin
    (1232, 2),  # Firebreather
    (1119, 2),  # Energy Search
    (1097, 2),  # Night Stretcher
    (1227, 2),  # Lillie's Determination
    (2, 22),    # Basic Fire Energy
]

_DEFAULT_DECK = build_deck_from_description(_DEFAULT_DECK_DESCRIPTION)


def save_deck_csv(deck: list[int], path: str = "deck.csv"):
    with open(path, "w") as f:
        for card_id in deck:
            f.write(f"{card_id}\n")


def load_deck_csv(path: str = "deck.csv") -> list[int]:
    if not os.path.exists(path):
        alt_path = os.path.join("/kaggle_simulations/agent", path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            return get_default_deck()
    deck = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
    return deck