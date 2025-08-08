from dataclasses import dataclass
from typing import Any, Dict
from sklearn.linear_model import LinearRegression


@dataclass
class LinearRegressionConfig:
    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: int | None = None
    positive: bool = False


def build_model(cfg: Dict[str, Any]) -> LinearRegression:
    params = LinearRegressionConfig(**cfg).__dict__
    return LinearRegression(**params)
