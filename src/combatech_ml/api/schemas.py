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
