import argparse
from pathlib import Path

import yaml

from src.algorithms import gradient_descent, linear_regression, random_forest
from src.core.utils import set_seed
from src.pipelines.pipeline import evaluate, predict, train

REGISTRY = {
    "random_forest": random_forest,
    "linear_regression": linear_regression,
    "gradient_descent": gradient_descent,
}


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ML Portfolio CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    model_choices = list(REGISTRY)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--model", required=True, choices=model_choices)
    p_train.add_argument("--config", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate a model")
    p_eval.add_argument("--model", required=True, choices=model_choices)
    p_eval.add_argument("--config", required=True)

    p_pred = sub.add_parser("predict", help="Predict with a model")
    p_pred.add_argument("--model", required=True, choices=model_choices)
    p_pred.add_argument("--config", required=True)
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg.get("seed", 42))

    module = REGISTRY[args.model]
    task = module.TASK
    model_dir = Path("models") / args.model / "models"
    model_path = model_dir / "model.pkl"

    if args.cmd == "train":
        metrics = train(cfg, module.build_model, model_dir, task)
        print("Training done. Validation metrics:", metrics)

    elif args.cmd == "evaluate":
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        print("Test metrics:", evaluate(cfg, model_path, task))

    elif args.cmd == "predict":
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"Predictions saved to: {predict(cfg, model_path, args.input, args.output)}")


if __name__ == "__main__":
    main()
