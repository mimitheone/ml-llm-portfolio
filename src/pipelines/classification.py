from pathlib import Path
from typing import Dict, Any
import pandas as pd
import joblib

from src.core.io import load_csv
from src.core.metrics import classification_metrics

def split_Xy(df: pd.DataFrame, target: str, features: list[str] | None = None):
    if features is None:
        features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]
    return X, y

def train_random_forest(cfg: Dict[str, Any], algo_builder, model_dir: Path):
    # Data
    train_df = load_csv(cfg["data"]["train"])
    valid_df = load_csv(cfg["data"]["valid"])
    target = cfg["target"]
    features = cfg.get("features")

    X_train, y_train = split_Xy(train_df, target, features)
    X_valid, y_valid = split_Xy(valid_df, target, features)

    # Model
    model = algo_builder(cfg.get("model", {}))
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_valid)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_valid)
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
    except Exception:
        pass

    metrics = classification_metrics(y_valid, y_pred, y_proba)

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")

    return metrics

def evaluate(cfg: Dict[str, Any], model_path: Path):
    import joblib
    df = load_csv(cfg["data"]["test"])
    target = cfg["target"]
    features = cfg.get("features")
    model = joblib.load(model_path)

    X, y = split_Xy(df, target, features)
    y_pred = model.predict(X)
    y_proba = None
    try:
        y_proba = model.predict_proba(X)
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
    except Exception:
        pass
    return classification_metrics(y, y_pred, y_proba)

def predict(cfg: Dict[str, Any], model_path: Path, input_csv: str, output_csv: str):
    import joblib
    in_df = load_csv(input_csv)
    features = cfg.get("features") or list(in_df.columns)
    model = joblib.load(model_path)
    preds = model.predict(in_df[features])
    out = in_df.copy()
    out["prediction"] = preds
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv
