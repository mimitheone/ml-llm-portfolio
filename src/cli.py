import argparse
from pathlib import Path
import yaml
from src.core.utils import set_seed
from src.pipelines.classification import (
    train_random_forest,
    evaluate as evaluate_classification,
    predict as predict_classification,
)
from src.pipelines.regression import (
    train_linear_regression,
    evaluate as evaluate_regression,
    predict as predict_regression,
)
from src.algorithms.random_forest import build_model as build_rf
from src.algorithms.linear_regression import build_model as build_lr


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ML Portfolio CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument(
        "--model", required=True, choices=["random_forest", "linear_regression"]
    )
    p_train.add_argument(
        "--task", required=True, choices=["classification", "regression"]
    )
    p_train.add_argument("--config", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate a model")
    p_eval.add_argument(
        "--model", required=True, choices=["random_forest", "linear_regression"]
    )
    p_eval.add_argument(
        "--task", required=True, choices=["classification", "regression"]
    )
    p_eval.add_argument("--config", required=True)

    p_pred = sub.add_parser("predict", help="Predict with a model")
    p_pred.add_argument(
        "--model", required=True, choices=["random_forest", "linear_regression"]
    )
    p_pred.add_argument(
        "--task", required=True, choices=["classification", "regression"]
    )
    p_pred.add_argument("--config", required=True)
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()
    cfg = load_cfg(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # New structure: models/{model_name}/models/model.pkl
    model_dir = Path("models") / f"{args.model}" / "models"
    model_path = model_dir / "model.pkl"

    if args.cmd == "train":
        if args.model == "random_forest" and args.task == "classification":
            metrics = train_random_forest(cfg, build_rf, model_dir)
            print("Training done. Validation metrics:", metrics)
        elif args.model == "linear_regression" and args.task == "regression":
            metrics = train_linear_regression(cfg, build_lr, model_dir)
            print("Training done. Validation metrics:", metrics)
        else:
            raise ValueError(
                f"Unsupported model-task combination: {args.model}-{args.task}"
            )

    elif args.cmd == "evaluate":
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if args.task == "classification":
            metrics = evaluate_classification(cfg, model_path)
        elif args.task == "regression":
            metrics = evaluate_regression(cfg, model_path)
        else:
            raise ValueError(f"Unsupported task: {args.task}")
        print("Test metrics:", metrics)

    elif args.cmd == "predict":
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if args.task == "classification":
            out_path = predict_classification(cfg, model_path, args.input, args.output)
        elif args.task == "regression":
            out_path = predict_regression(cfg, model_path, args.input, args.output)
        else:
            raise ValueError(f"Unsupported task: {args.task}")
        print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
