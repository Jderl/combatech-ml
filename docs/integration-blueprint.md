# End-to-End Integration Blueprint (React + Axios + Python ML)

## 1) System Design

### Recommended (Direct from Frontend)

`combatech-frontend (Axios) -> combatech-ml (FastAPI)`

- `combatech-frontend` keeps using DAL pattern (`src/dal/*`).
- Add a new DAL module for ML calls.
- `combatech-ml` serves win probability + behavior predictions.

### Optional (via Spring Boot Gateway)

`combatech-frontend -> combatech-backend -> combatech-ml`

Use this if you want all APIs behind Spring security and one domain.

## 2) ML Service API Contract

Base URL (local): `http://localhost:8001`

Model mapping:
- `LSVM (LinearSVC)` real-time per round -> `/predict/win` and `/predict/win/realtime`
- `LSVM + Platt scaling (sigmoid calibration)` pre-match -> `/predict/win/prematch`
- `Gaussian Naive Bayes` behavior classification -> `/predict/behavior`

### `POST /predict/win` (realtime alias)
Request:
```json
{
  "CUM_OFFENSE": 12,
  "CUM_VIOLATION": 1,
  "OFFENSE_ADV": 3,
  "VIOLATION_ADV": 0,
  "ROUND": 2
}
```
Response:
```json
{ "winProbability": 0.7421 }
```

### `POST /predict/win/prematch`
Request:
```json
{
  "PM_MATCHES_PLAYED": 3,
  "PM_WIN_RATE": 0.66,
  "PM_AVG_OFFENSE": 11,
  "PM_AVG_VIOLATION": 1,
  "PM_OPP_MATCHES_PLAYED": 1,
  "PM_OPP_WIN_RATE": 0.5,
  "PM_OPP_AVG_OFFENSE": 8,
  "PM_OPP_AVG_VIOLATION": 2
}
```
Response:
```json
{
  "winProbability": 0.5379,
  "calibration": "platt-scaling-sigmoid"
}
```

### `POST /predict/behavior`
Request:
```json
{
  "ROUND_OFFENSE": 5,
  "CUM_OFFENSE": 12,
  "CUM_VIOLATION": 1,
  "TOTAL_RAW_VIOLATION_COUNT": 1
}
```
Response:
```json
{
  "predBehavior": "AGGRESSIVE",
  "classProbabilities": {
    "AGGRESSIVE": 0.81,
    "DEFENSIVE": 0.14,
    "PASSIVE": 0.02,
    "PENALTY_PRONE": 0.03
  }
}
```

### `POST /predict/match-analytics`
Request:
```json
{
  "eventName": "2025 BATANGAS ATHLETE DISTRICT MEET",
  "matchId": 5
}
```
Response:
```json
{
  "rows": [
    {
      "EVENT_NAME": "...",
      "MATCH_ID": 5,
      "ROUND": 1,
      "CORNER": "BLUE",
      "PLAYER_NAME": "...",
      "WIN_PROBABILITY": 0.62,
      "PRED_BEHAVIOR": "AGGRESSIVE",
      "ROUND_SCORE": 10,
      "TOTAL_RAW_VIOLATION_COUNT": 1,
      "NET_SCORE": 9,
      "CUM_NET_SCORE": 9
    }
  ]
}
```

## 3) Frontend Integration (Existing DAL style)

Create: `combatech-frontend/src/dal/mlAnalytics.ts`

```ts
import api from "./api";

export interface WinPredictPayload {
  CUM_OFFENSE: number;
  CUM_VIOLATION: number;
  OFFENSE_ADV: number;
  VIOLATION_ADV: number;
  ROUND: number;
}

export interface PrematchWinPredictPayload {
  PM_MATCHES_PLAYED?: number;
  PM_WIN_RATE?: number;
  PM_AVG_OFFENSE?: number;
  PM_AVG_VIOLATION?: number;
  PM_OPP_MATCHES_PLAYED?: number;
  PM_OPP_WIN_RATE?: number;
  PM_OPP_AVG_OFFENSE?: number;
  PM_OPP_AVG_VIOLATION?: number;
}

export interface BehaviorPredictPayload {
  ROUND_OFFENSE: number;
  CUM_OFFENSE: number;
  CUM_VIOLATION: number;
  TOTAL_RAW_VIOLATION_COUNT: number;
}

export const PredictWin = async (payload: WinPredictPayload) => {
  const res = await api.post("http://localhost:8001/predict/win", payload);
  return res.data;
};

export const PredictPrematchWin = async (payload: PrematchWinPredictPayload) => {
  const res = await api.post("http://localhost:8001/predict/win/prematch", payload);
  return res.data;
};

export const PredictBehavior = async (payload: BehaviorPredictPayload) => {
  const res = await api.post("http://localhost:8001/predict/behavior", payload);
  return res.data;
};

export const GetMatchAnalytics = async (eventName: string, matchId: number) => {
  const res = await api.post("http://localhost:8001/predict/match-analytics", {
    eventName,
    matchId,
  });
  return res.data;
};
```

Then call it from:
- `combatech-frontend/src/components/event_matches/GenerateMatchDialog.tsx`
- or `combatech-frontend/src/pages/ScoringPages/CoachAlgoViewPage.tsx`

## 4) Backend Integration (Optional)

If using Spring Boot gateway, add a controller in `combatech-backend` that forwards requests to `http://localhost:8001` via `RestTemplate`/`WebClient`.

Use when you need:
- centralized auth
- API key/JWT enforcement
- one public API URL

## 5) Local Run Order

1. Start ML API
```powershell
cd combatech-ml
.venv\Scripts\activate
$env:PYTHONPATH="src"
python scripts/train_models.py --csv original_files_ml/game_data_cleaned.csv --out models/artifacts
uvicorn combatech_ml.api.main:app --reload --port 8001
```

2. Start Spring Boot (if used)
```powershell
cd combatech-backend
./mvnw spring-boot:run
```

3. Start React
```powershell
cd combatech-frontend
npm run dev
```

## 6) CORS and Security

- `combatech-ml` currently allows `http://localhost:5173` and `http://localhost:3000`.
- For production, restrict origins and put ML behind backend gateway or API key protection.
