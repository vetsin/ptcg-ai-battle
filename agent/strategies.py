"""Strategy-guided MCTS / self-play nudging (NudgeRL-inspired, see MEGA_LUCARIO_STRATEGY_PLAN.md).

A StrategyProfile is a small, explicit, card-grounded game plan for a single deck
archetype. It is used to (1) bias the heuristic action scorers in agent/policy.py
and (2) compute PUCT priors / rollout sampling weights in agent/search.py. It is
deliberately a *soft* nudge: callers add a weighted bias on top of existing scores
rather than replacing them.

Note on engine semantics: for PLAY / ATTACH / EVOLVE options, the engine leaves
`Option.cardId` unset. The real card identity must be resolved via `Option.index`
into the acting player's hand (for the card being played/attached/evolved) and via
`Option.inPlayArea`/`Option.inPlayIndex` into the player's board (for the Pokemon
being attached-to or evolved). Only ATTACK options carry a populated `attackId`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cg.api import State, Option, OptionType, AreaType
from agent.evaluate import get_card_data


@dataclass
class StrategyProfile:
    name: str
    key_attacker_ids: list[int] = field(default_factory=list)
    finisher_attack_ids: list[int] = field(default_factory=list)
    setup_attack_ids: list[int] = field(default_factory=list)
    single_prize_ids: list[int] = field(default_factory=list)
    single_prize_attack_ids: list[int] = field(default_factory=list)
    setup_engine_ids: list[int] = field(default_factory=list)
    evolution_targets: list[tuple[int, int]] = field(default_factory=list)
    energy_ids: list[int] = field(default_factory=list)
    disruption_ids: list[int] = field(default_factory=list)
    # Cards that want to sit Active and *not* attack/retreat (e.g. an
    # ex-immune wall) — see MULTI_DECK_ENSEMBLE_PLAN.md's Crustle profile.
    prefer_passive_wall_ids: list[int] = field(default_factory=list)
    powered_energy_threshold: int = 2
    low_prize_threshold: int = 2
    early_turn_cutoff: int = 3


MEGA_LUCARIO = StrategyProfile(
    name="mega_lucario_ex",
    key_attacker_ids=[678, 674],          # Mega Lucario ex, Hariyama
    finisher_attack_ids=[983, 978],       # Mega Brave (270), Wild Press (210)
    setup_attack_ids=[982],               # Aura Jab (130) — powers up while attacking
    single_prize_ids=[676, 675],          # Solrock, Lunatone
    single_prize_attack_ids=[980, 979],   # Cosmic Beam, Power Gem
    setup_engine_ids=[1227, 1152, 1142, 1121],  # Lillie's Determination, Poké Pad, Fighting Gong, Ultra Ball
    evolution_targets=[(333, 678), (677, 678), (974, 678), (673, 674)],
    energy_ids=[6, 20],                   # Basic Fighting, Rock Fighting
    disruption_ids=[1213, 1182],          # Judge, Boss's Orders
)

CRUSTLE = StrategyProfile(
    name="crustle",
    key_attacker_ids=[756],                # Mega Kangaskhan ex (Run Errand: draw 2/turn while Active)
    finisher_attack_ids=[1092],            # Rapid-Fire Combo (200+, coin-flip scaling)
    single_prize_ids=[344],                # Dwebble (self-evolves via Ascension, no Rare Candy needed)
    setup_engine_ids=[1227, 1122, 1086],   # Lillie's Determination, Pokégear 3.0, Buddy-Buddy Poffin
    evolution_targets=[(344, 345)],        # Dwebble -> Crustle
    energy_ids=[18, 11, 14],               # Grow Grass, Mist, Spiky Energy
    disruption_ids=[1182, 1219],           # Boss's Orders, Team Rocket's Petrel
    prefer_passive_wall_ids=[345],         # Crustle: prevents all ex/megaEx damage while Active
)

# Note on bias validation: a drafted profile's bias was tested across many
# configurations (mirror + fixed-baseline, multiple MCTS sim/time budgets,
# multiple softmax temperatures) for both Raging Bolt ex and Terapagos ex and
# never showed a reproducible edge over uniform priors (bounced +-17pp on
# n=30-40 batches — noise, not signal). This appears to be a structural limit
# of root-only prior reordering at shallow MCTS depth rather than a per-deck
# authoring problem (Terapagos's profile lost to *uniform* priors too, despite
# both beating slightly_smart_agent heavily on the strength of the deck alone).
# Per user direction: ship profiles without gating on the validation bar —
# they're empirically harmless, and keeping them costs nothing pending a fix
# to the underlying mechanism (see [[feedback_strategy_bias_validation]]).

RAGING_BOLT = StrategyProfile(
    name="raging_bolt_ex",
    key_attacker_ids=[63, 37],              # Raging Bolt ex, Iron Thorns ex
    finisher_attack_ids=[72],               # Bellowing Thunder (discard energy to scale dmg)
    setup_attack_ids=[71],                  # Burst Roar (0 dmg, discard hand/draw 6)
    single_prize_ids=[61, 62],              # Roaring Moon, Koraidon
    single_prize_attack_ids=[67, 68, 69, 70],  # their attacks
    setup_engine_ids=[1227, 1121, 1152, 1086, 1097],  # Lillie's, Ultra Ball, Poké Pad, Buddy-Buddy Poffin, Night Stretcher
    energy_ids=[4, 20],                     # Basic Lightning, Rock Fighting (pays Bellowing Thunder's [L][F])
    disruption_ids=[1182, 1120],            # Boss's Orders, Crushing Hammer
)

ARCHALUDON = StrategyProfile(
    name="archaludon_ex",
    key_attacker_ids=[190],                 # Archaludon ex
    finisher_attack_ids=[253],              # Metal Defender (220 dmg + weakness immunity)
    evolution_targets=[(839, 190)],         # Duraludon -> Archaludon ex
    setup_engine_ids=[1079, 1121, 1086, 1227, 1097],  # Rare Candy, Ultra Ball, Buddy-Buddy Poffin, Lillie's, Night Stretcher
    energy_ids=[8],                         # Basic Metal Energy
    disruption_ids=[1182],                  # Boss's Orders
)

TERAPAGOS = StrategyProfile(
    name="terapagos_ex",
    key_attacker_ids=[176, 125],            # Terapagos ex, Blissey ex
    finisher_attack_ids=[233, 159],         # Crown Opal (180+immunity), Return (180+draw)
    setup_attack_ids=[232],                 # Unified Beatdown (scales w/ bench)
    single_prize_ids=[124],                 # Chansey
    single_prize_attack_ids=[157, 158],     # Lucky Attachment, Boundless Power
    setup_engine_ids=[1121, 1086, 1152, 1227, 1097, 1119, 1146],  # Ultra Ball, Buddy-Buddy Poffin, Poké Pad, Lillie's, Night Stretcher, Energy Search, Wondrous Patch
    evolution_targets=[(124, 125)],         # Chansey -> Blissey ex
    energy_ids=[7],                         # Basic Darkness Energy
    disruption_ids=[1182, 1120],            # Boss's Orders, Crushing Hammer
)

DARK_TRINITY = StrategyProfile(
    name="dark_trinity_ex",
    key_attacker_ids=[138, 139, 140, 141],  # Okidogi ex, Munkidori ex, Fezandipiti ex, Pecharunt ex
    finisher_attack_ids=[181, 182, 184],     # Chain-Crazed, Dirty Headbutt, Irritated Outburst
    setup_attack_ids=[180],                  # Poisonous Musculature (energy ramp, 0 dmg)
    single_prize_attack_ids=[183],           # Cruel Arrow (100 dmg bench snipe, ignores weak/resist)
    setup_engine_ids=[1121, 1086, 1152, 1227, 1097, 1216],  # Ultra Ball, Buddy-Buddy Poffin, Poké Pad, Lillie's, Night Stretcher, Team Rocket's Ariana
    energy_ids=[7],                          # Basic Darkness Energy
    disruption_ids=[1182, 1162],             # Boss's Orders, Binding Mochi (+40 dmg vs Poisoned target)
)

GRENINJA = StrategyProfile(
    name="greninja_ex",
    key_attacker_ids=[40],                   # Greninja ex
    finisher_attack_ids=[33],                # Shinobi Blade (170 dmg + deck search, 1 energy)
    setup_attack_ids=[34],                   # Mirage Barrage (discard 2 energy, 120 dmg to 2 mons)
    single_prize_ids=[33, 34],               # Froakie, Frogadier (pre-evolution board pieces)
    single_prize_attack_ids=[23, 24, 25],    # Flock, Flop, Numbing Water
    setup_engine_ids=[1079, 1121, 1086, 1227, 1097, 1152, 1240],  # Rare Candy, Ultra Ball, Buddy-Buddy Poffin, Lillie's, Night Stretcher, Poké Pad, Rosa's Encouragement
    evolution_targets=[(33, 34), (34, 40), (33, 40)],  # Froakie -> Frogadier -> Greninja ex (or direct via Rare Candy)
    energy_ids=[3],                          # Basic Water Energy
    disruption_ids=[1182],                   # Boss's Orders
)

DRAGAPULT_DUSKNOIR = StrategyProfile(
    name="dragapult_dusknoir",
    key_attacker_ids=[121, 133],             # Dragapult ex, Dusknoir
    finisher_attack_ids=[154, 172],          # Phantom Dive (200+spread), Shadow Bind (150+lock)
    setup_attack_ids=[153],                  # Jet Headbutt (free 70 dmg, tempo)
    single_prize_ids=[119, 120, 131, 132, 112, 235],  # Dreepy, Drakloak, Duskull, Dusclops, Munkidori, Budew
    single_prize_attack_ids=[150, 151, 152, 169, 170, 171, 141, 323],  # Petty Grudge, Bite, Dragon Headbutt, Come and Get You, Mumble, Will-O-Wisp, Mind Bend, Itchy Pollen
    setup_engine_ids=[1121, 1086, 1152, 1227, 1097, 1198, 1231],  # Ultra Ball, Buddy-Buddy Poffin, Poké Pad, Lillie's, Night Stretcher, Crispin, Dawn
    evolution_targets=[(119, 120), (120, 121), (119, 121), (131, 132), (132, 133), (131, 133)],
    energy_ids=[2, 5, 7],                    # Basic Fire, Psychic, Darkness Energy
    disruption_ids=[1182, 1120, 1080, 1087],  # Boss's Orders, Crushing Hammer, Unfair Stamp, Hand Trimmer
)

STRATEGIES: dict[str, StrategyProfile] = {
    "mega_lucario_ex": MEGA_LUCARIO,
    "crustle": CRUSTLE,
    "raging_bolt_ex": RAGING_BOLT,
    "archaludon_ex": ARCHALUDON,
    "terapagos_ex": TERAPAGOS,
    "dark_trinity_ex": DARK_TRINITY,
    "greninja_ex": GRENINJA,
    "dragapult_dusknoir": DRAGAPULT_DUSKNOIR,
}

# Anchor card -> archetype name, checked in order (MULTI_DECK_ENSEMBLE_PLAN.md).
# Each anchor was checked against get_card_data() to be unique to its archetype.
# (Fezandipiti ex/140 and Pecharunt ex/141 are NOT unique to this deck — they
# splash into Dragapult/N's Zoroark variants — so Okidogi ex/138 is the anchor.
# Likewise Dragapult ex/121 splashes into the plain "Dragapult" and "Dragapult
# Blaziken" decks, so Dusknoir/133 — unique to this deck — is the anchor here.)
_DECK_ANCHORS: list[tuple[int, str]] = [
    (678, "mega_lucario_ex"),  # Mega Lucario ex
    (756, "crustle"),          # Mega Kangaskhan ex
    (63, "raging_bolt_ex"),    # Raging Bolt ex
    (190, "archaludon_ex"),    # Archaludon ex
    (176, "terapagos_ex"),     # Terapagos ex
    (138, "dark_trinity_ex"),  # Okidogi ex
    (40, "greninja_ex"),       # Greninja ex
    (133, "dragapult_dusknoir"),  # Dusknoir
]


def classify_my_deck(deck: list[int]) -> str | None:
    deck_set = set(deck)
    for anchor_id, name in _DECK_ANCHORS:
        if anchor_id in deck_set:
            return name
    return None


def _is_finisher_phase(state: State, my_index: int, profile: StrategyProfile) -> bool:
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    opp_prizes_remaining = len(opp.prize)
    if opp_prizes_remaining <= profile.low_prize_threshold:
        return True
    my_active = me.active[0] if me.active else None
    if my_active is not None and my_active.id in profile.key_attacker_ids:
        if len(my_active.energies) >= profile.powered_energy_threshold:
            return True
    return False


def _is_early_phase(state: State, profile: StrategyProfile) -> bool:
    return state.turn <= profile.early_turn_cutoff


def _hand_card_id(state: State, my_index: int, index: int | None) -> int | None:
    if index is None:
        return None
    me = state.players[my_index]
    hand = me.hand
    if not hand or index >= len(hand):
        return None
    return hand[index].id


def _in_play_pokemon_id(state: State, my_index: int, area: AreaType | None, index: int | None) -> int | None:
    me = state.players[my_index]
    if area == AreaType.ACTIVE:
        mon = me.active[0] if me.active else None
        return mon.id if mon else None
    if area == AreaType.BENCH:
        if me.bench and index is not None and index < len(me.bench):
            return me.bench[index].id
    return None


def strategy_action_bias(opt: Option, state: State | None, my_index: int, profile: StrategyProfile) -> float:
    if state is None:
        return 0.0

    finisher_phase = _is_finisher_phase(state, my_index, profile)
    early = _is_early_phase(state, profile)

    if opt.type == OptionType.ATTACK:
        attack_id = opt.attackId
        if attack_id in profile.finisher_attack_ids:
            return 90.0 if finisher_phase else 30.0
        if attack_id in profile.setup_attack_ids:
            return 15.0 if finisher_phase else 40.0
        if attack_id in profile.single_prize_attack_ids:
            return 60.0 if early else 10.0
        return 0.0

    if opt.type == OptionType.PLAY:
        cid = _hand_card_id(state, my_index, opt.index)
        if cid is None:
            return 0.0
        if cid in profile.single_prize_ids:
            return 50.0 if early else 5.0
        if cid in profile.setup_engine_ids:
            return 35.0 if early else 10.0
        if cid in profile.disruption_ids:
            return 5.0 if early else 25.0
        if cid in profile.key_attacker_ids:
            return 25.0 if finisher_phase else 10.0
        return 0.0

    if opt.type == OptionType.EVOLVE:
        cid = _hand_card_id(state, my_index, opt.index)
        if cid is not None and (cid in profile.key_attacker_ids or cid in profile.prefer_passive_wall_ids):
            return 45.0
        return 0.0

    if opt.type == OptionType.ATTACH:
        target_id = _in_play_pokemon_id(state, my_index, opt.inPlayArea, opt.inPlayIndex)
        cid = _hand_card_id(state, my_index, opt.index)
        bias = 0.0
        if opt.inPlayArea == AreaType.ACTIVE:
            bias += 15.0
        if target_id is not None and target_id in profile.key_attacker_ids:
            bias += 25.0
        if cid is not None and cid in profile.energy_ids:
            bias += 5.0
        return bias

    if opt.type == OptionType.RETREAT and profile.prefer_passive_wall_ids:
        me = state.players[my_index]
        my_active = me.active[0] if me.active else None
        if my_active is not None and my_active.id in profile.prefer_passive_wall_ids:
            if _opp_active_is_ex(state, my_index):
                return -120.0
        return 0.0

    return 0.0


def _opp_active_is_ex(state: State, my_index: int) -> bool:
    opp = state.players[1 - my_index]
    opp_active = opp.active[0] if opp.active else None
    if opp_active is None:
        return False
    cd = get_card_data().get(opp_active.id)
    return cd is not None and (cd.ex or cd.megaEx)


def strategy_priors(options: list[Option], state: State | None, my_index: int, profile: StrategyProfile) -> list[float]:
    n = len(options)
    if n == 0:
        return []
    biases = [strategy_action_bias(opt, state, my_index, profile) for opt in options]
    max_bias = max(biases)
    temperature = 40.0
    exps = [math.exp((b - max_bias) / temperature) for b in biases]
    total = sum(exps)
    if total <= 0:
        return [1.0 / n] * n
    return [e / total for e in exps]
