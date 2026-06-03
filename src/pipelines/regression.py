from src.pipelines.pipeline import evaluate as _evaluate
from src.pipelines.pipeline import predict  # noqa: F401 — re-exported for callers
from src.pipelines.pipeline import train as _train


def train_linear_regression(cfg, algo_builder, model_dir):
    return _train(cfg, algo_builder, model_dir, task="regression")


def evaluate(cfg, model_path):
    return _evaluate(cfg, model_path, task="regression")
