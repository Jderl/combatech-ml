from pydantic import BaseModel


class WinPredictRequest(BaseModel):
    CUM_OFFENSE: float
    CUM_VIOLATION: float
    OFFENSE_ADV: float
    VIOLATION_ADV: float
    ROUND: float


class PrematchWinPredictRequest(BaseModel):
    PM_MATCHES_PLAYED: float = 0
    PM_WIN_RATE: float = 0.5
    PM_AVG_OFFENSE: float = 0
    PM_AVG_VIOLATION: float = 0
    PM_OPP_MATCHES_PLAYED: float = 0
    PM_OPP_WIN_RATE: float = 0.5
    PM_OPP_AVG_OFFENSE: float = 0
    PM_OPP_AVG_VIOLATION: float = 0


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
