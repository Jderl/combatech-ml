from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from combatech_ml.api.schemas import (
    BehaviorPredictRequest,
    MatchAnalyticsRequest,
    TrainRequest,
    WinPredictRequest,
)
from combatech_ml.core.combined_pipeline import (
    build_match_analytics,
    load_artifacts,
    load_dataset,
    predict_behavior,
    predict_win,
    train_all,
)

BASE_DIR = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
DEFAULT_DATASET = BASE_DIR / "original_files_ml" / "game_data_cleaned.csv"

app = FastAPI(title="Combatech ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models: Optional[object] = None
dataset_df: Optional[pd.DataFrame] = None


def _try_load_models() -> None:
    global models
    if (ARTIFACTS_DIR / "win_model.joblib").exists() and (ARTIFACTS_DIR / "behavior_model.joblib").exists():
        models = load_artifacts(ARTIFACTS_DIR)


def _try_load_dataset() -> None:
    global dataset_df
    if DEFAULT_DATASET.exists():
        dataset_df = load_dataset(DEFAULT_DATASET)


_try_load_models()
_try_load_dataset()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "modelsLoaded": models is not None,
        "datasetLoaded": dataset_df is not None,
    }


@app.post("/train")
def train(payload: TrainRequest) -> dict:
    global models, dataset_df
    path = Path(payload.csvPath)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {path}")

    models = train_all(path, ARTIFACTS_DIR)
    dataset_df = load_dataset(path)
    return {"message": "Training completed", "artifactsDir": str(ARTIFACTS_DIR)}


@app.post("/predict/win")
def predict_win_endpoint(payload: WinPredictRequest) -> dict:
    if models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded. Run /train first.")

    probability = predict_win(models.win_model, payload.model_dump())
    return {"winProbability": probability}


@app.post("/predict/behavior")
def predict_behavior_endpoint(payload: BehaviorPredictRequest) -> dict:
    if models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded. Run /train first.")

    pred, probs = predict_behavior(models.behavior_model, payload.model_dump())
    return {"predBehavior": pred, "classProbabilities": probs}


@app.post("/predict/match-analytics")
def predict_match_analytics(payload: MatchAnalyticsRequest) -> dict:
    if models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded. Run /train first.")
    if dataset_df is None:
        raise HTTPException(status_code=503, detail="Dataset is not loaded. Run /train first.")

    rows = build_match_analytics(
        dataset_df,
        event_name=payload.eventName,
        match_id=payload.matchId,
        win_model=models.win_model,
        behavior_model=models.behavior_model,
    )
    return {"rows": rows}
