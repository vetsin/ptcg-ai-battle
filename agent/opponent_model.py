from __future__ import annotations

import random
from cg.api import (
    PlayerState,
    Pokemon,
    CardData,
    CardType,
    EnergyType,
)
from agent.evaluate import get_card_data, get_attack_data

ENERGY_CARD_MAP = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
}

DECK_ARCHETYPES: dict[str, dict] = {
    "gouging_fire_ex": {
        "name": "Gouging Fire ex",
        "energy_type": EnergyType.FIRE,
        "energy_card": 2,
        "core": [46, 788, 790, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 22,
    },
    "charizard_ex": {
        "name": "Charizard ex",
        "energy_type": EnergyType.FIRE,
        "energy_card": 2,
        "core": [1145, 1205, 1158, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
    "ruling_bolt_ex": {
        "name": "Raging Bolt ex",
        "energy_type": EnergyType.DRAGON,
        "energy_card": 2,
        "core": [63, 788, 790, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 22,
    },
    "dragapult_ex": {
        "name": "Dragapult ex",
        "energy_type": EnergyType.PSYCHIC,
        "energy_card": 5,
        "core": [121, 1158, 1079, 1121, 1182, 1123, 1086, 1232],
        "energy_count": 14,
    },
    "iron_thorns_ex": {
        "name": "Iron Thorns ex",
        "energy_type": EnergyType.LIGHTNING,
        "energy_card": 4,
        "core": [37, 1079, 1121, 1182, 1123, 1086, 1232],
        "energy_count": 14,
    },
    "terapagos_ex": {
        "name": "Terapagos ex",
        "energy_type": EnergyType.COLORLESS,
        "energy_card": 2,
        "core": [176, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 18,
    },
    "mamoswine_ex": {
        "name": "Mamoswine ex",
        "energy_type": EnergyType.FIGHTING,
        "energy_card": 6,
        "core": [283, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 16,
    },
    "hydreigon_ex": {
        "name": "Hydreigon ex",
        "energy_type": EnergyType.DARKNESS,
        "energy_card": 7,
        "core": [229, 1079, 1121, 1182, 1123, 1086, 1097, 1232],
        "energy_count": 13,
    },
    "palafin_ex": {
        "name": "Palafin ex",
        "energy_type": EnergyType.WATER,
        "energy_card": 3,
        "core": [107, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 16,
    },
    "generic_fire": {
        "name": "Generic Fire",
        "energy_type": EnergyType.FIRE,
        "energy_card": 2,
        "core": [46, 788, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
    "generic_water": {
        "name": "Generic Water",
        "energy_type": EnergyType.WATER,
        "energy_card": 3,
        "core": [47, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
    "generic_grass": {
        "name": "Generic Grass",
        "energy_type": EnergyType.GRASS,
        "energy_card": 1,
        "core": [42, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
    "generic_lightning": {
        "name": "Generic Lightning",
        "energy_type": EnergyType.LIGHTNING,
        "energy_card": 4,
        "core": [37, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
    "generic_psychic": {
        "name": "Generic Psychic",
        "energy_type": EnergyType.PSYCHIC,
        "energy_card": 5,
        "core": [121, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
    "generic_fighting": {
        "name": "Generic Fighting",
        "energy_type": EnergyType.FIGHTING,
        "energy_card": 6,
        "core": [283, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
    "generic_darkness": {
        "name": "Generic Darkness",
        "energy_type": EnergyType.DARKNESS,
        "energy_card": 7,
        "core": [138, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
    "generic_metal": {
        "name": "Generic Metal",
        "energy_type": EnergyType.METAL,
        "energy_card": 8,
        "core": [85, 1079, 1121, 1182, 1123, 1086],
        "energy_count": 20,
    },
}


class OpponentModel:
    def __init__(self):
        self.archetype_log_probs: dict[str, float] = {}
        self.observed_cards: set[int] = set()
        self.initialized = False
        for name in DECK_ARCHETYPES:
            self.archetype_log_probs[name] = 0.0
        self._observed_energy_types: dict[int, int] = {}

    def update(self, opp: PlayerState) -> None:
        new_cards: set[int] = set()

        if opp.active:
            for mon in opp.active:
                if mon is not None and mon.id not in self.observed_cards:
                    new_cards.add(mon.id)
        if opp.bench:
            for mon in opp.bench:
                if mon.id not in self.observed_cards:
                    new_cards.add(mon.id)
        if opp.discard:
            for card in opp.discard:
                if card.id not in self.observed_cards:
                    new_cards.add(card.id)
        for p in opp.prize:
            if p is not None and p.id not in self.observed_cards:
                new_cards.add(p.id)

        if not new_cards and self.initialized:
            return

        self.observed_cards.update(new_cards)

        card_db = get_card_data()

        for card_id in new_cards:
            cd = card_db.get(card_id)
            if cd is None:
                continue

            e_int = int(cd.energyType) if cd.energyType is not None else 0
            if e_int in (1, 2, 3, 4, 5, 6, 7, 8):
                self._observed_energy_types[e_int] = self._observed_energy_types.get(e_int, 0) + 1

            for name, spec in DECK_ARCHETYPES.items():
                if card_id in spec["core"]:
                    self.archetype_log_probs[name] += 1.5
                else:
                    etype = spec.get("energy_type")
                    etype_int = int(etype) if etype is not None else 0
                    if cd.cardType == CardType.POKEMON:
                        if cd.energyType == etype:
                            self.archetype_log_probs[name] += 0.8
                        elif cd.energyType == EnergyType.COLORLESS or etype == EnergyType.COLORLESS:
                            self.archetype_log_probs[name] += 0.3
                        else:
                            self.archetype_log_probs[name] -= 0.2

                    elif cd.cardType == CardType.BASIC_ENERGY:
                        if card_id == spec["energy_card"]:
                            self.archetype_log_probs[name] += 1.0
                        else:
                            self.archetype_log_probs[name] -= 0.5

        self.initialized = True

    def get_best_archetype(self) -> str:
        if not self.initialized:
            return "gouging_fire_ex"
        best = max(self.archetype_log_probs, key=self.archetype_log_probs.get)
        return best

    def get_archetype_probs(self) -> dict[str, float]:
        import math
        max_lp = max(self.archetype_log_probs.values()) if self.archetype_log_probs else 0
        probs = {}
        total = 0.0
        for name, lp in self.archetype_log_probs.items():
            p = math.exp(lp - max_lp)
            probs[name] = p
            total += p
        if total > 0:
            for name in probs:
                probs[name] /= total
        return probs

    def sample_deck(self, opp: PlayerState) -> list[int]:
        archetype_name = self.get_best_archetype()
        spec = DECK_ARCHETYPES[archetype_name]

        deck: list[int] = []

        if opp.active:
            for mon in opp.active:
                if mon is not None:
                    deck.append(mon.id)
        for mon in opp.bench or []:
            deck.append(mon.id)
        for card in opp.discard or []:
            deck.append(card.id)
        for p in opp.prize:
            if p is not None:
                deck.append(p.id)

        core = spec["core"]
        remaining = 60 - len(deck)

        core_cards = []
        for cid in core:
            if cid not in self.observed_cards and cid not in deck:
                core_cards.extend([cid] * 4)

        needed = min(remaining, len(core_cards))
        deck.extend(core_cards[:needed])
        remaining = 60 - len(deck)

        energy_id = spec["energy_card"]
        deck.extend([energy_id] * remaining)

        return deck[:60]

    def predict_hand(self, opp: PlayerState) -> list[int]:
        archetype_name = self.get_best_archetype()
        spec = DECK_ARCHETYPES[archetype_name]
        n = opp.handCount
        if n <= 0:
            return []

        energy_id = spec["energy_card"]
        core = spec["core"]

        supporter_ids = [cid for cid in core if _is_supporter(cid)]

        hand = [energy_id] * max(1, n * 2 // 3)
        if supporter_ids:
            hand.extend([supporter_ids[i % len(supporter_ids)] for i in range(n - len(hand))])
        while len(hand) < n:
            hand.append(energy_id)
        return hand[:n]

    def predict_active(self, opp: PlayerState) -> list[int]:
        if opp.active and len(opp.active) > 0 and opp.active[0] is not None:
            return []

        if opp.bench:
            return [opp.bench[0].id]

        archetype_name = self.get_best_archetype()
        spec = DECK_ARCHETYPES[archetype_name]
        return [spec["core"][0]]

    def predict_prize(self, opp: PlayerState) -> list[int]:
        archetype_name = self.get_best_archetype()
        spec = DECK_ARCHETYPES[archetype_name]
        energy_id = spec["energy_card"]

        result = []
        for p in opp.prize:
            if p is not None:
                result.append(p.id)
            else:
                result.append(energy_id)
        return result


def _is_supporter(card_id: int) -> bool:
    cd = get_card_data().get(card_id)
    return cd is not None and cd.cardType == CardType.SUPPORTER


_opponent_model_cache: dict[int, OpponentModel] = {}


def get_opponent_model(player_index: int = 1) -> OpponentModel:
    if player_index not in _opponent_model_cache:
        _opponent_model_cache[player_index] = OpponentModel()
    return _opponent_model_cache[player_index]


def reset_opponent_models() -> None:
    _opponent_model_cache.clear()