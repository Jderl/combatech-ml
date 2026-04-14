from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

REALTIME_WIN_FEATURES = [
    "CUM_OFFENSE",
    "CUM_VIOLATION",
    "OFFENSE_ADV",
    "VIOLATION_ADV",
    "ROUND",
]

PREMATCH_PROFILE_FEATURES = [
    "PM_MATCHES_PLAYED",
    "PM_WIN_RATE",
    "PM_AVG_OFFENSE",
    "PM_AVG_VIOLATION",
]

PREMATCH_OPP_PROFILE_FEATURES = [
    "PM_OPP_MATCHES_PLAYED",
    "PM_OPP_WIN_RATE",
    "PM_OPP_AVG_OFFENSE",
    "PM_OPP_AVG_VIOLATION",
]

PREMATCH_BASE_FEATURES = PREMATCH_PROFILE_FEATURES + PREMATCH_OPP_PROFILE_FEATURES

PREMATCH_FEATURES = PREMATCH_BASE_FEATURES + [
    "PM_EXPERIENCE_ADV",
    "PM_WIN_RATE_ADV",
    "PM_OFFENSE_ADV",
    "PM_VIOLATION_ADV",
]

BEHAVIOR_FEATURES = [
    "ROUND_OFFENSE",
    "CUM_OFFENSE",
    "CUM_VIOLATION",
    "TOTAL_RAW_VIOLATION_COUNT",
]

# Backward compatibility aliases
WIN_FEATURES = REALTIME_WIN_FEATURES


@dataclass
class TrainedArtifacts:
    prematch_win_model: Any
    realtime_win_model: Any
    behavior_model: Any

    @property
    def win_model(self) -> Any:
        return self.realtime_win_model


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    )
    if "ROUND" in df.columns:
        df["ROUND"] = pd.to_numeric(df["ROUND"], errors="coerce")
    if {"YEAR", "MONTH", "DAY"}.issubset(df.columns):
        date_expr = (
            df["YEAR"].astype(str).str.strip()
            + "-"
            + df["MONTH"].astype(str).str.strip()
            + "-"
            + df["DAY"].astype(str).str.strip()
        )
        df["EVENT_DATE"] = pd.to_datetime(date_expr, errors="coerce")
    else:
        df["EVENT_DATE"] = pd.NaT
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    if "MATCH_KEY" not in df.columns:
        df["MATCH_KEY"] = df["EVENT_NAME"].astype(str) + "-" + df["MATCH_ID"].astype(str)
    sort_cols = [
        col
        for col in ["EVENT_DATE", "EVENT_NAME", "MATCH_ID", "MATCH_KEY", "CORNER", "ROUND"]
        if col in df.columns
    ]
    return df.sort_values(sort_cols) if sort_cols else df


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


def _extract_final_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "WIN_STATUS" not in df.columns:
        return pd.DataFrame(columns=["MATCH_KEY", "CORNER", "TARGET_WIN"])

    labels = df[["MATCH_KEY", "CORNER", "WIN_STATUS"]].copy()
    labels["TARGET_WIN"] = labels["WIN_STATUS"].map({"WIN": 1, "LOSE": 0})
    labels = labels.dropna(subset=["TARGET_WIN"]).drop_duplicates(
        subset=["MATCH_KEY", "CORNER"], keep="last"
    )
    return labels[["MATCH_KEY", "CORNER", "TARGET_WIN"]]


def _build_realtime_win_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_round_features(df)
    labels = _extract_final_match_labels(d)
    d = d.merge(labels, on=["MATCH_KEY", "CORNER"], how="left")

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
    return d


def _rolling_mean_shifted(series: pd.Series, default_value: float = 0.0) -> pd.Series:
    return series.shift().expanding().mean().fillna(default_value)


def _build_prematch_win_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_round_features(df)
    labels = _extract_final_match_labels(d)

    grouped = (
        d.groupby(
            ["MATCH_KEY", "EVENT_NAME", "MATCH_ID", "EVENT_DATE", "CORNER", "PLAYER_NAME"],
            as_index=False,
        )
        .agg(
            {
                "ROUND_OFFENSE": "sum",
                "TOTAL_RAW_VIOLATION_COUNT": "sum",
            }
        )
        .rename(
            columns={
                "ROUND_OFFENSE": "MATCH_OFFENSE",
                "TOTAL_RAW_VIOLATION_COUNT": "MATCH_VIOLATION",
            }
        )
    )

    grouped = grouped.merge(labels, on=["MATCH_KEY", "CORNER"], how="left")
    grouped = grouped.sort_values(["PLAYER_NAME", "EVENT_DATE", "MATCH_ID", "MATCH_KEY"])
    player_groups = grouped.groupby("PLAYER_NAME", dropna=False)

    grouped["PM_MATCHES_PLAYED"] = player_groups.cumcount().astype(float)
    grouped["PM_WIN_RATE"] = player_groups["TARGET_WIN"].transform(
        lambda s: _rolling_mean_shifted(s, default_value=0.5)
    )
    grouped["PM_AVG_OFFENSE"] = player_groups["MATCH_OFFENSE"].transform(
        lambda s: _rolling_mean_shifted(s, default_value=0.0)
    )
    grouped["PM_AVG_VIOLATION"] = player_groups["MATCH_VIOLATION"].transform(
        lambda s: _rolling_mean_shifted(s, default_value=0.0)
    )

    opponent = grouped[["MATCH_KEY", "CORNER", *PREMATCH_PROFILE_FEATURES]].copy()
    opponent = opponent.rename(
        columns={
            "CORNER": "CORNER_OPP",
            "PM_MATCHES_PLAYED": "PM_OPP_MATCHES_PLAYED",
            "PM_WIN_RATE": "PM_OPP_WIN_RATE",
            "PM_AVG_OFFENSE": "PM_OPP_AVG_OFFENSE",
            "PM_AVG_VIOLATION": "PM_OPP_AVG_VIOLATION",
        }
    )

    prematch = grouped.merge(opponent, on=["MATCH_KEY"], how="inner")
    prematch = prematch[prematch["CORNER"] != prematch["CORNER_OPP"]].copy()
    prematch["PM_EXPERIENCE_ADV"] = (
        prematch["PM_MATCHES_PLAYED"] - prematch["PM_OPP_MATCHES_PLAYED"]
    )
    prematch["PM_WIN_RATE_ADV"] = prematch["PM_WIN_RATE"] - prematch["PM_OPP_WIN_RATE"]
    prematch["PM_OFFENSE_ADV"] = prematch["PM_AVG_OFFENSE"] - prematch["PM_OPP_AVG_OFFENSE"]
    prematch["PM_VIOLATION_ADV"] = (
        prematch["PM_OPP_AVG_VIOLATION"] - prematch["PM_AVG_VIOLATION"]
    )
    return prematch


def _split_train_data(
    frame: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    x = frame[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = frame[target_col].astype(int)
    groups = frame["MATCH_KEY"]

    if y.nunique() < 2:
        raise ValueError("Training data must contain at least two classes.")

    gss = GroupShuffleSplit(test_size=0.30, n_splits=1, random_state=random_state)
    train_idx, _ = next(gss.split(x, y, groups))
    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]

    if y_train.nunique() < 2:
        raise ValueError("Training split produced a single class. Adjust data or split strategy.")

    return x_train, y_train


def _build_calibrated_lsvm(random_state: int, y_train: pd.Series) -> CalibratedClassifierCV:
    min_class_samples = int(y_train.value_counts().min())
    cv_folds = min(5, min_class_samples)
    if cv_folds < 2:
        raise ValueError("Not enough samples per class for Platt scaling calibration.")

    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",  # Platt scaling
        cv=cv_folds,
    )


def train_realtime_win_model(df: pd.DataFrame, random_state: int = 42) -> Any:
    d = _build_realtime_win_frame(df)
    d = d.dropna(subset=["TARGET_WIN"]).copy()

    x_train, y_train = _split_train_data(
        d,
        feature_cols=REALTIME_WIN_FEATURES,
        target_col="TARGET_WIN",
        random_state=random_state,
    )

    model = _build_calibrated_lsvm(random_state=random_state, y_train=y_train)
    model.fit(x_train, y_train)
    return model


def _build_win_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Backward compatibility for legacy imports.
    return _build_realtime_win_frame(df)


def train_win_model(df: pd.DataFrame, random_state: int = 42) -> Any:
    # Backward compatibility for legacy imports.
    return train_realtime_win_model(df, random_state=random_state)


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
    behavior_df = _assign_behavior_labels(d)

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


def train_prematch_win_model(df: pd.DataFrame, random_state: int = 42) -> Any:
    d = _build_prematch_win_frame(df)
    d = d.dropna(subset=["TARGET_WIN"]).copy()

    x_train, y_train = _split_train_data(
        d,
        feature_cols=PREMATCH_FEATURES,
        target_col="TARGET_WIN",
        random_state=random_state,
    )

    model = _build_calibrated_lsvm(random_state=random_state, y_train=y_train)
    model.fit(x_train, y_train)
    return model


def train_all(csv_path: str | Path, artifacts_dir: str | Path) -> TrainedArtifacts:
    df = load_dataset(csv_path)
    prematch_win_model = train_prematch_win_model(df)
    realtime_win_model = train_realtime_win_model(df)
    behavior_model = train_behavior_model(df)

    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(prematch_win_model, out / "prematch_win_model.joblib")
    joblib.dump(realtime_win_model, out / "realtime_win_model.joblib")
    joblib.dump(behavior_model, out / "behavior_model.joblib")
    joblib.dump(PREMATCH_FEATURES, out / "prematch_features.joblib")
    joblib.dump(REALTIME_WIN_FEATURES, out / "realtime_win_features.joblib")
    joblib.dump(BEHAVIOR_FEATURES, out / "behavior_features.joblib")

    # Legacy artifacts retained for existing integrations.
    joblib.dump(realtime_win_model, out / "win_model.joblib")
    joblib.dump(REALTIME_WIN_FEATURES, out / "win_features.joblib")

    return TrainedArtifacts(
        prematch_win_model=prematch_win_model,
        realtime_win_model=realtime_win_model,
        behavior_model=behavior_model,
    )


def load_artifacts(artifacts_dir: str | Path) -> TrainedArtifacts:
    base = Path(artifacts_dir)
    realtime_model_path = (
        base / "realtime_win_model.joblib"
        if (base / "realtime_win_model.joblib").exists()
        else base / "win_model.joblib"
    )
    prematch_model_path = (
        base / "prematch_win_model.joblib"
        if (base / "prematch_win_model.joblib").exists()
        else realtime_model_path
    )

    return TrainedArtifacts(
        prematch_win_model=joblib.load(prematch_model_path),
        realtime_win_model=joblib.load(realtime_model_path),
        behavior_model=joblib.load(base / "behavior_model.joblib"),
    )


def _payload_to_frame(payload: Dict[str, float], features: List[str]) -> pd.DataFrame:
    row = {feature: float(payload.get(feature, 0)) for feature in features}
    return pd.DataFrame([row], columns=features)


def _predict_probabilities(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1].astype(float)
        return probs.ravel().astype(float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(x), dtype=float)


def _build_prematch_payload(payload: Dict[str, float]) -> Dict[str, float]:
    prematch_payload = {
        feature: float(payload.get(feature, 0.5 if "WIN_RATE" in feature else 0.0))
        for feature in PREMATCH_BASE_FEATURES
    }
    prematch_payload["PM_EXPERIENCE_ADV"] = (
        prematch_payload["PM_MATCHES_PLAYED"] - prematch_payload["PM_OPP_MATCHES_PLAYED"]
    )
    prematch_payload["PM_WIN_RATE_ADV"] = (
        prematch_payload["PM_WIN_RATE"] - prematch_payload["PM_OPP_WIN_RATE"]
    )
    prematch_payload["PM_OFFENSE_ADV"] = (
        prematch_payload["PM_AVG_OFFENSE"] - prematch_payload["PM_OPP_AVG_OFFENSE"]
    )
    prematch_payload["PM_VIOLATION_ADV"] = (
        prematch_payload["PM_OPP_AVG_VIOLATION"] - prematch_payload["PM_AVG_VIOLATION"]
    )
    return prematch_payload


def predict_realtime_win(realtime_win_model: Any, payload: Dict[str, float]) -> float:
    x = _payload_to_frame(payload, REALTIME_WIN_FEATURES)
    return float(_predict_probabilities(realtime_win_model, x)[0])


def predict_prematch_win(prematch_win_model: Any, payload: Dict[str, float]) -> float:
    prematch_payload = _build_prematch_payload(payload)
    x = _payload_to_frame(prematch_payload, PREMATCH_FEATURES)
    return float(_predict_probabilities(prematch_win_model, x)[0])


def predict_win(win_model: Any, payload: Dict[str, float]) -> float:
    # Backward compatibility: existing callers still use predict_win for realtime.
    return predict_realtime_win(win_model, payload)


def predict_behavior(behavior_model: Any, payload: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    x = _payload_to_frame(payload, BEHAVIOR_FEATURES)
    pred = str(behavior_model.predict(x)[0])
    proba: Dict[str, float] = {}
    if hasattr(behavior_model, "predict_proba"):
        classes = getattr(behavior_model, "classes_", None)
        if classes is None and hasattr(behavior_model, "named_steps"):
            classes = behavior_model.named_steps["model"].classes_
        probs = behavior_model.predict_proba(x)[0]
        if classes is not None:
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

    win_frame = _build_realtime_win_frame(d)
    win_frame["WIN_PROBABILITY"] = _predict_probabilities(
        win_model,
        win_frame[REALTIME_WIN_FEATURES].fillna(0),
    )

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
