from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WIN_FEATURES = [
    "CUM_OFFENSE",
    "CUM_VIOLATION",
    "OFFENSE_ADV",
    "VIOLATION_ADV",
    "ROUND",
]

BEHAVIOR_FEATURES = [
    "ROUND_OFFENSE",
    "CUM_OFFENSE",
    "CUM_VIOLATION",
    "TOTAL_RAW_VIOLATION_COUNT",
]


@dataclass
class TrainedArtifacts:
    win_model: Any
    behavior_model: Any


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    )
    if "ROUND" in df.columns:
        df["ROUND"] = pd.to_numeric(df["ROUND"], errors="coerce")
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    if "MATCH_KEY" not in df.columns:
        df["MATCH_KEY"] = df["EVENT_NAME"].astype(str) + "-" + df["MATCH_ID"].astype(str)
    return df.sort_values(["MATCH_KEY", "CORNER", "ROUND"])


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    return _normalize_columns(pd.read_csv(csv_path))


def _ensure_round_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ROUND_OFFENSE"] = (
        d.get("NUM_HAND_STRIKE", 0)
        + d.get("NUM_FOOT_STRIKE", 0)
        + d.get("NUM_DROPING_SCORE", 0)
    )
    d["CUM_OFFENSE"] = d.groupby(["MATCH_KEY", "CORNER"])["ROUND_OFFENSE"].cumsum()
    d["CUM_VIOLATION"] = d.groupby(["MATCH_KEY", "CORNER"])["TOTAL_RAW_VIOLATION_COUNT"].cumsum()
    return d


def _build_win_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_round_features(df)
    opp = d[["MATCH_KEY", "ROUND", "CORNER", "CUM_OFFENSE", "CUM_VIOLATION"]].copy()
    opp = opp.rename(
        columns={
            "CORNER": "CORNER_OPP",
            "CUM_OFFENSE": "CUM_OFFENSE_OPP",
            "CUM_VIOLATION": "CUM_VIOLATION_OPP",
        }
    )
    d = d.merge(opp, on=["MATCH_KEY", "ROUND"], how="inner")
    d = d[d["CORNER"] != d["CORNER_OPP"]].copy()
    d["OFFENSE_ADV"] = d["CUM_OFFENSE"] - d["CUM_OFFENSE_OPP"]
    d["VIOLATION_ADV"] = d["CUM_VIOLATION_OPP"] - d["CUM_VIOLATION"]
    if "WIN_STATUS" in d.columns:
        d["TARGET_WIN"] = d["WIN_STATUS"].map({"WIN": 1, "LOSE": 0})
    return d


def train_win_model(df: pd.DataFrame, random_state: int = 42) -> Any:
    d = _build_win_frame(df)
    d = d.dropna(subset=["TARGET_WIN"]).copy()

    x = d[WIN_FEATURES].fillna(0)
    y = d["TARGET_WIN"]
    groups = d["MATCH_KEY"]

    gss = GroupShuffleSplit(test_size=0.30, n_splits=1, random_state=random_state)
    train_idx, _ = next(gss.split(x, y, groups))
    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def _assign_behavior_labels(df_behavior: pd.DataFrame) -> pd.DataFrame:
    d = df_behavior.copy()
    off_hi = d["CUM_OFFENSE"].quantile(0.70)
    off_lo = d["CUM_OFFENSE"].quantile(0.20)

    def assign(row: pd.Series) -> str:
        if row["TOTAL_RAW_VIOLATION_COUNT"] >= 5:
            return "PENALTY_PRONE"
        if row["ROUND_OFFENSE"] <= 0 and row["CUM_OFFENSE"] <= off_lo:
            return "PASSIVE"
        if row["ROUND_OFFENSE"] >= 5 or row["CUM_OFFENSE"] >= off_hi:
            return "AGGRESSIVE"
        return "DEFENSIVE"

    d["BEHAVIOR_LABEL"] = d.apply(assign, axis=1)
    return d


def train_behavior_model(df: pd.DataFrame, random_state: int = 42) -> Any:
    d = _ensure_round_features(df)

    behavior_df = (
        d.groupby(["MATCH_KEY", "EVENT_NAME", "MATCH_ID", "CORNER"], as_index=False)
        .agg(
            {
                "ROUND_OFFENSE": "sum",
                "CUM_OFFENSE": "max",
                "CUM_VIOLATION": "max",
                "TOTAL_RAW_VIOLATION_COUNT": "sum",
                "ROUND": "max",
            }
        )
    )
    behavior_df = _assign_behavior_labels(behavior_df)

    x = behavior_df[BEHAVIOR_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    y = behavior_df["BEHAVIOR_LABEL"]
    groups = behavior_df["MATCH_KEY"]

    gss = GroupShuffleSplit(test_size=0.30, n_splits=1, random_state=random_state)
    train_idx, _ = next(gss.split(x, y, groups))
    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_all(csv_path: str | Path, artifacts_dir: str | Path) -> TrainedArtifacts:
    df = load_dataset(csv_path)
    win_model = train_win_model(df)
    behavior_model = train_behavior_model(df)

    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(win_model, out / "win_model.joblib")
    joblib.dump(behavior_model, out / "behavior_model.joblib")
    joblib.dump(WIN_FEATURES, out / "win_features.joblib")
    joblib.dump(BEHAVIOR_FEATURES, out / "behavior_features.joblib")

    return TrainedArtifacts(win_model=win_model, behavior_model=behavior_model)


def load_artifacts(artifacts_dir: str | Path) -> TrainedArtifacts:
    base = Path(artifacts_dir)
    return TrainedArtifacts(
        win_model=joblib.load(base / "win_model.joblib"),
        behavior_model=joblib.load(base / "behavior_model.joblib"),
    )


def _payload_to_frame(payload: Dict[str, float], features: List[str]) -> pd.DataFrame:
    row = {feature: float(payload.get(feature, 0)) for feature in features}
    return pd.DataFrame([row], columns=features)


def predict_win(win_model: Any, payload: Dict[str, float]) -> float:
    x = _payload_to_frame(payload, WIN_FEATURES)
    return float(win_model.predict_proba(x)[0, 1])


def predict_behavior(behavior_model: Any, payload: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    x = _payload_to_frame(payload, BEHAVIOR_FEATURES)
    pred = str(behavior_model.predict(x)[0])
    proba: Dict[str, float] = {}
    if hasattr(behavior_model, "predict_proba"):
        classes = behavior_model.named_steps["model"].classes_
        probs = behavior_model.predict_proba(x)[0]
        proba = {str(c): float(v) for c, v in zip(classes, probs)}
    return pred, proba


def build_match_analytics(
    df: pd.DataFrame,
    event_name: str,
    match_id: int,
    win_model: Any,
    behavior_model: Any,
) -> List[Dict[str, Any]]:
    d = _normalize_columns(df)

    win_frame = _build_win_frame(d)
    win_frame["WIN_PROBABILITY"] = win_model.predict_proba(win_frame[WIN_FEATURES].fillna(0))[:, 1]

    round_frame = _ensure_round_features(d)
    round_frame["PRED_BEHAVIOR"] = behavior_model.predict(round_frame[BEHAVIOR_FEATURES].fillna(0))
    round_frame["NET_SCORE"] = round_frame.get("ROUND_SCORE", 0) - round_frame.get(
        "TOTAL_RAW_VIOLATION_COUNT", 0
    )
    round_frame["CUM_NET_SCORE"] = round_frame.groupby(["EVENT_NAME", "MATCH_ID", "CORNER"])[
        "NET_SCORE"
    ].cumsum()

    win_filtered = win_frame[
        (win_frame["EVENT_NAME"] == event_name) & (win_frame["MATCH_ID"] == match_id)
    ].copy()
    behavior_filtered = round_frame[
        (round_frame["EVENT_NAME"] == event_name) & (round_frame["MATCH_ID"] == match_id)
    ].copy()

    merge_keys = ["MATCH_KEY", "ROUND", "CORNER", "EVENT_NAME", "MATCH_ID"]
    merged = behavior_filtered.merge(
        win_filtered[merge_keys + ["WIN_PROBABILITY"]],
        on=merge_keys,
        how="left",
    )

    output_cols = [
        "EVENT_NAME",
        "MATCH_ID",
        "ROUND",
        "CORNER",
        "PLAYER_NAME",
        "WIN_PROBABILITY",
        "PRED_BEHAVIOR",
        "ROUND_SCORE",
        "TOTAL_RAW_VIOLATION_COUNT",
        "NET_SCORE",
        "CUM_NET_SCORE",
    ]

    for col in output_cols:
        if col not in merged.columns:
            merged[col] = None

    merged = merged[output_cols].sort_values(["ROUND", "CORNER"])
    return merged.to_dict(orient="records")
