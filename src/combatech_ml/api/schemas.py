from pydantic import BaseModel


class WinPredictRequest(BaseModel):
    CUM_OFFENSE: float
    CUM_VIOLATION: float
    OFFENSE_ADV: float
    VIOLATION_ADV: float
    ROUND: float


class BehaviorPredictRequest(BaseModel):
    ROUND_OFFENSE: float
    CUM_OFFENSE: float
    CUM_VIOLATION: float
    TOTAL_RAW_VIOLATION_COUNT: float


class MatchAnalyticsRequest(BaseModel):
    eventName: str
    matchId: int


class TrainRequest(BaseModel):
    csvPath: str


class PrescriptivePredictRequest(BaseModel):
    round: int = 1
    hand: float = 0
    foot: float = 0
    dropping: float = 0
    opp_hand: float = 0
    opp_foot: float = 0
    opp_dropping: float = 0
    round_score: float = 0
    light: float = 0
    reprimand: float = 0
    serious_total: float = 0
