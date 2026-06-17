from cg.api import (
    Observation,
    State,
    PlayerState,
    Pokemon,
    Option,
    SelectData,
    Card,
    OptionType,
    SelectType,
    SelectContext,
    EnergyType,
    CardType,
    SpecialConditionType,
    Attack,
    CardData,
    AreaType,
)
from cg.api import all_card_data, all_attack

_card_data_cache: dict[int, CardData] | None = None
_attack_data_cache: dict[int, Attack] | None = None


def get_card_data() -> dict[int, CardData]:
    global _card_data_cache
    if _card_data_cache is None:
        cards = all_card_data()
        _card_data_cache = {c.cardId: c for c in cards}
    return _card_data_cache


def get_attack_data() -> dict[int, Attack]:
    global _attack_data_cache
    if _attack_data_cache is None:
        attacks = all_attack()
        _attack_data_cache = {a.attackId: a for a in attacks}
    return _attack_data_cache


def evaluate_state(state: State, my_index: int) -> float:
    if state.result != -1:
        if state.result == my_index:
            return 10000.0
        elif state.result == 1 - my_index:
            return -10000.0
        else:
            return 0.0

    me = state.players[my_index]
    opp = state.players[1 - my_index]

    score = 0.0
    score += _prize_score(me, opp)
    score += _board_score(me, opp)
    score += _hp_score(me, opp)
    score += _energy_score(me)
    score += _hand_score(me, opp, state)
    score += _special_condition_score(me, opp)

    return score


def _prize_score(me: PlayerState, opp: PlayerState) -> float:
    my_remaining = len([p for p in me.prize if p is not None])
    opp_remaining = len([p for p in opp.prize if p is not None])
    my_taken = 6 - my_remaining
    opp_taken = 6 - opp_remaining
    if opp_taken >= 6:
        return 5000.0
    if my_taken >= 6:
        return -5000.0
    return (opp_taken - my_taken) * 200.0 + opp_taken * 80.0


def _board_score(me: PlayerState, opp: PlayerState) -> float:
    score = 0.0
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    if opp_active is None and len(opp.bench) == 0:
        score += 5000.0
    elif opp_active is None:
        score += 300.0

    if my_active is None and len(me.bench) == 0:
        score -= 5000.0
    elif my_active is None:
        score -= 300.0

    score += len(me.bench) * 30.0
    score -= len(opp.bench) * 20.0

    card_db = get_card_data()
    if my_active is not None:
        cd = card_db.get(my_active.id)
        if cd:
            if cd.ex or cd.megaEx:
                score += 40.0
            if cd.stage2:
                score += 30.0
            elif cd.stage1:
                score += 15.0

    return score


def _hp_score(me: PlayerState, opp: PlayerState) -> float:
    score = 0.0
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    if my_active is not None and my_active.maxHp > 0:
        hp_ratio = my_active.hp / my_active.maxHp
        score += hp_ratio * 60.0
    if opp_active is not None and opp_active.maxHp > 0:
        opp_hp_ratio = opp_active.hp / opp_active.maxHp
        score -= opp_hp_ratio * 20.0
        score += (1.0 - opp_hp_ratio) * 40.0

    for mon in me.bench:
        if mon.maxHp > 0:
            hp_ratio = mon.hp / mon.maxHp
            score += hp_ratio * 5.0

    return score


def _can_use_attack(pokemon: Pokemon, attack: Attack) -> bool:
    energy_counts: dict[int, int] = {}
    for e in pokemon.energies:
        energy_counts[int(e)] = energy_counts.get(int(e), 0) + 1

    colorless_needed = 0
    remaining = dict(energy_counts)
    for e in attack.energies:
        e_int = int(e)
        if e_int == int(EnergyType.COLORLESS):
            colorless_needed += 1
        else:
            if remaining.get(e_int, 0) <= 0:
                return False
            remaining[e_int] -= 1

    total_available = sum(remaining.values())
    return total_available >= colorless_needed


def _energy_score(me: PlayerState) -> float:
    score = 0.0
    my_active = me.active[0] if me.active else None
    if my_active is not None:
        card_db = get_card_data()
        cd = card_db.get(my_active.id)
        if cd and cd.attacks:
            attack_db = get_attack_data()
            best_ready_dmg = 0
            total_energy = len(my_active.energies)
            for atk_id in cd.attacks:
                atk = attack_db.get(atk_id)
                if atk and _can_use_attack(my_active, atk):
                    if atk.damage > best_ready_dmg:
                        best_ready_dmg = atk.damage
            score += best_ready_dmg * 0.5
            score += total_energy * 3.0
    return score


def _has_active(me: PlayerState) -> bool:
    return len(me.active) > 0 and me.active[0] is not None


def _hand_score(me: PlayerState, opp: PlayerState, state: State) -> float:
    score = 0.0
    card_db = get_card_data()
    if me.hand is not None:
        supporter_count = 0
        energy_count = 0
        basic_count = 0
        for card in me.hand:
            cd = card_db.get(card.id)
            if cd:
                if cd.cardType == CardType.SUPPORTER:
                    supporter_count += 1
                elif cd.cardType == CardType.BASIC_ENERGY:
                    energy_count += 1
                elif cd.cardType == CardType.POKEMON and cd.basic:
                    basic_count += 1
        if not state.energyAttached and _has_active(me):
            score += energy_count * 5.0
        score += supporter_count * 3.0
        if not _has_active(me):
            score += basic_count * 20.0

    score -= opp.handCount * 2.0

    deck_ratio = me.deckCount / 60.0 if me.deckCount > 0 else 0
    if deck_ratio < 0.1:
        score -= 50.0

    return score


def _special_condition_score(me: PlayerState, opp: PlayerState) -> float:
    score = 0.0
    if me.poisoned:
        score -= 15.0
    if me.burned:
        score -= 10.0
    if me.asleep:
        score -= 8.0
    if me.paralyzed:
        score -= 20.0
    if me.confused:
        score -= 8.0

    if opp.poisoned:
        score += 15.0
    if opp.burned:
        score += 10.0
    if opp.asleep:
        score += 8.0
    if opp.paralyzed:
        score += 20.0
    if opp.confused:
        score += 8.0

    return score