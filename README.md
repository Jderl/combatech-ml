# Combatech ML

This folder now contains a consolidated Python implementation for both notebooks:

- `FINAL_ML_V7-Copy1.ipynb` -> win probability model
- `N8_UPDATE_V2.ipynb` -> behavior classification model

## Current Model Mapping

- `LSVM + Platt Scaling` -> pre-match win probability (`/predict/win/prematch`)
- `LSVM` -> real-time win probability per round (`/predict/win` and `/predict/win/realtime`)
- `Naive Bayes` -> player behavior classification during match (`/predict/behavior`)

## Structure

- `src/combatech_ml/core/combined_pipeline.py`: single merged training/inference pipeline
- `src/combatech_ml/api/main.py`: FastAPI service for React/Axios
- `scripts/train_models.py`: CLI model training
- `models/artifacts/`: exported `.joblib` model files
- `docs/integration-blueprint.md`: end-to-end integration with frontend/backend

## Run

```powershell
cd combatech-ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

$env:PYTHONPATH="src"
python scripts/train_models.py --csv original_files_ml/game_data_cleaned.csv --out models/artifacts
uvicorn combatech_ml.api.main:app --reload --port 8001
```

## Endpoints

- `GET /health`
- `POST /train`
- `POST /predict/win` (realtime alias)
- `POST /predict/win/realtime`
- `POST /predict/win/prematch`
- `POST /predict/behavior`
- `POST /predict/match-analytics`

See `docs/integration-blueprint.md` for React Axios integration details.
