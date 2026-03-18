from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PrescriptiveInput:
    round: int
    hand: float
    foot: float
    dropping: float
    opp_hand: float
    opp_foot: float
    opp_dropping: float
    round_score: float
    light: float
    reprimand: float
    serious_total: float


def _push_unique(target: List[str], message: str) -> None:
    if message not in target:
        target.append(message)


def evaluate_prescriptive(payload: PrescriptiveInput) -> Dict[str, List[str]]:
    offense: List[str] = []
    defense: List[str] = []
    discipline: List[str] = []
    tactical: List[str] = []

    hand = payload.hand
    foot = payload.foot
    dropping = payload.dropping
    opp_hand = payload.opp_hand
    opp_foot = payload.opp_foot
    opp_dropping = payload.opp_dropping
    total_strikes = hand + foot + dropping
    opp_total_strikes = opp_hand + opp_foot + opp_dropping
    round_score = payload.round_score

    # Hand Strike Dominance
    if hand >= 5:
        _push_unique(offense, "Great hand pressure - maintain inside exchanges.")
    if hand <= 2:
        _push_unique(offense, "Increase punching output to avoid losing close-range exchanges.")
    if opp_hand > hand:
        _push_unique(defense, "Tighten guard - opponent winning hand exchanges.")

    # Foot Strike Control
    if foot >= 4:
        _push_unique(offense, "Maintain long-range control using kicks.")
    if foot < 2:
        _push_unique(offense, "Increase kick usage to break opponent rhythm.")
    if opp_foot > foot:
        _push_unique(defense, "Respond with stronger kicking offense.")

    # Dropping Techniques
    if dropping >= 2:
        _push_unique(offense, "Excellent use of dropping techniques - maintain balance control.")
    if dropping == 0:
        _push_unique(offense, "Attempt dropping techniques when the opponent loses balance.")
    if opp_dropping > dropping:
        _push_unique(defense, "Maintain balance and defend against throws.")

    # Match Tempo
    if total_strikes >= 5:
        _push_unique(tactical, "Maintain offensive pressure.")
    if total_strikes <= 2:
        _push_unique(tactical, "Increase tempo and scoring attempts.")
    if opp_total_strikes > total_strikes:
        _push_unique(tactical, "Match or exceed opponent pace.")

    # Attack Diversity
    if hand > 0 and foot > 0:
        _push_unique(tactical, "Maintain variety of attacks.")
    if hand == 0 or foot == 0:
        _push_unique(tactical, "Introduce more technique variation.")

    # Round Scoring Evaluation
    if round_score >= 10:
        _push_unique(tactical, "Excellent scoring output - maintain performance.")
    if round_score < 5:
        _push_unique(tactical, "Increase offensive openings.")

    # Violation Monitoring
    if payload.light >= 1:
        _push_unique(discipline, "Maintain discipline and clean techniques.")
    if payload.reprimand >= 1:
        _push_unique(discipline, "Avoid actions that may lead to point deductions.")
    if payload.serious_total >= 1:
        _push_unique(discipline, "Control aggression to avoid disqualification risk.")

    return {
        "offense": offense or ["None"],
        "defense": defense or ["None"],
        "discipline": discipline or ["None"],
        "tactical": tactical or ["None"],
    }

