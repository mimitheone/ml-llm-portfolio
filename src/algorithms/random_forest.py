from dataclasses import dataclass
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier

@dataclass
class RandomForestConfig:
    n_estimators: int = 300
    max_depth: int | None = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42

def build_model(cfg: Dict[str, Any]) -> RandomForestClassifier:
    params = RandomForestConfig(**cfg).__dict__
    return RandomForestClassifier(**params)
