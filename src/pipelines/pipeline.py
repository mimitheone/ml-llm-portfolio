from pathlib import Path
from typing import Any, Dict, Literal

import joblib
import pandas as pd

from src.core.io import load_csv
from src.core.metrics import classification_metrics, regression_metrics


def split_Xy(df: pd.DataFrame, target: str, features: list[str] | None = None):
    if features is None:
        features = [c for c in df.columns if c != target]
    return df[features], df[target]


def _compute_metrics(model, X, y, task: str) -> Dict[str, Any]:
    y_pred = model.predict(X)
    if task == "regression":
        return regression_metrics(y, y_pred)
    y_proba = None
    try:
        raw = model.predict_proba(X)
        y_proba = raw[:, 1] if raw.shape[1] == 2 else raw
    except AttributeError:
        pass
    return classification_metrics(y, y_pred, y_proba)


def train(
    cfg: Dict[str, Any],
    algo_builder,
    model_dir: Path,
    task: Literal["regression", "classification"],
) -> Dict[str, Any]:
    train_df = load_csv(cfg["data"]["train"])
    valid_df = load_csv(cfg["data"]["valid"])
    target = cfg["target"]
    features = cfg.get("features")

    X_train, y_train = split_Xy(train_df, target, features)
    X_valid, y_valid = split_Xy(valid_df, target, features)

    model = algo_builder(cfg.get("model", {}))
    model.fit(X_train, y_train)

    metrics = _compute_metrics(model, X_valid, y_valid, task)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")

    return metrics


def evaluate(
    cfg: Dict[str, Any],
    model_path: Path,
    task: Literal["regression", "classification"],
) -> Dict[str, Any]:
    df = load_csv(cfg["data"]["test"])
    target = cfg["target"]
    features = cfg.get("features")
    model = joblib.load(model_path)
    X, y = split_Xy(df, target, features)
    return _compute_metrics(model, X, y, task)


def predict(cfg: Dict[str, Any], model_path: Path, input_csv: str, output_csv: str) -> str:
    in_df = load_csv(input_csv)
    features = cfg.get("features") or list(in_df.columns)
    model = joblib.load(model_path)
    preds = model.predict(in_df[features])
    out = in_df.copy()
    out["prediction"] = preds
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv
