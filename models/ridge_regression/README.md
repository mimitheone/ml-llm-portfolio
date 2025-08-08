# Ridge Regression

This folder contains all files related to the ridge_regression model.

## Structure

```
models/ridge_regression/
├── data/           # Model-specific datasets
├── configs/        # Configuration files
├── reports/        # Output reports and predictions
└── models/         # Trained model files
```

## Usage

```bash
# Train the model
python -m src.cli train --model ridge_regression --task regression --config models/ridge_regression/configs/ridge_regression.yaml

# Evaluate the model
python -m src.cli evaluate --model ridge_regression --task regression --config models/ridge_regression/configs/ridge_regression.yaml

# Make predictions
python -m src.cli predict --model ridge_regression --task regression --config models/ridge_regression/configs/ridge_regression.yaml --input models/ridge_regression/data/inference.csv --output models/ridge_regression/reports/predictions.csv
```

## Files

- `data/`: Training, validation, test, and inference datasets
- `configs/ridge_regression.yaml`: Model configuration
- `reports/`: Generated reports and predictions
- `models/model.pkl`: Trained model file
