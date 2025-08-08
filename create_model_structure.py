#!/usr/bin/env python3
"""
Script to create new model structures in the organized folder layout.
This script creates the standard folder structure for any new model.
"""

import os
from pathlib import Path
import argparse


def create_model_structure(model_name: str, model_type: str = "regression"):
    """
    Create the standard folder structure for a new model.

    Args:
        model_name: Name of the model (e.g., 'ridge_regression', 'svm')
        model_type: Type of model ('regression' or 'classification')
    """

    # Define the base path
    base_path = Path("models") / model_name

    # Create the folder structure
    folders = [
        base_path / "data",
        base_path / "configs",
        base_path / "reports",
        base_path / "models",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {folder}")

    # Create a basic config file
    config_template = f"""# models/{model_name}/configs/{model_name}.yaml

seed: 42

data:
  train: models/{model_name}/data/train.csv
  valid: models/{model_name}/data/valid.csv
  test:  models/{model_name}/data/test.csv

target: "target"

features:
  - feature1
  - feature2
  - feature3

model:
  # Add model-specific parameters here
  random_state: 42

metrics:
  # Add metrics based on model type
  - accuracy
  - f1
  - roc_auc

reports:
  dir: "models/{model_name}/reports"
"""

    config_path = base_path / "configs" / f"{model_name}.yaml"
    with open(config_path, "w") as f:
        f.write(config_template)

    print(f"✅ Created: {config_path}")

    # Create a README for the model
    readme_template = f"""# {model_name.replace('_', ' ').title()}

This folder contains all files related to the {model_name} model.

## Structure

```
models/{model_name}/
├── data/           # Model-specific datasets
├── configs/        # Configuration files
├── reports/        # Output reports and predictions
└── models/         # Trained model files
```

## Usage

```bash
# Train the model
python -m src.cli train --model {model_name} --task {model_type} --config models/{model_name}/configs/{model_name}.yaml

# Evaluate the model
python -m src.cli evaluate --model {model_name} --task {model_type} --config models/{model_name}/configs/{model_name}.yaml

# Make predictions
python -m src.cli predict --model {model_name} --task {model_type} --config models/{model_name}/configs/{model_name}.yaml --input models/{model_name}/data/inference.csv --output models/{model_name}/reports/predictions.csv
```

## Files

- `data/`: Training, validation, test, and inference datasets
- `configs/{model_name}.yaml`: Model configuration
- `reports/`: Generated reports and predictions
- `models/model.pkl`: Trained model file
"""

    readme_path = base_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_template)

    print(f"✅ Created: {readme_path}")

    print(f"\n🎉 Model structure created successfully!")
    print(f"📁 Model folder: {base_path}")
    print(f"🔧 Next steps:")
    print(f"   1. Add your data to {base_path}/data/")
    print(f"   2. Update {base_path}/configs/{model_name}.yaml")
    print(f"   3. Implement the model in src/algorithms/{model_name}.py")
    print(f"   4. Add training pipeline in src/pipelines/{model_type}.py")
    print(f"   5. Update src/cli.py to support the new model")


def main():
    parser = argparse.ArgumentParser(description="Create new model structure")
    parser.add_argument(
        "model_name", help="Name of the model (e.g., 'ridge_regression')"
    )
    parser.add_argument(
        "--type",
        choices=["regression", "classification"],
        default="regression",
        help="Type of model (default: regression)",
    )

    args = parser.parse_args()
    create_model_structure(args.model_name, args.type)


if __name__ == "__main__":
    main()
