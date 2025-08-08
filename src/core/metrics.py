from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def classification_metrics(
    y_true, y_pred, y_proba=None, average: str = "binary"
) -> Dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average)),
    }
    # ROC AUC for binary or multiclass (ovo)
    if y_proba is not None:
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            else:
                out["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovo")
                )
        except Exception:
            pass
    return out


def regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Calculate regression metrics for continuous target variables."""
    out = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return out
