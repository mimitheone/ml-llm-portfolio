#!/usr/bin/env python3
"""
Linear Regression Demo - ML & LLM Finance Portfolio

This script demonstrates Linear Regression, the simplest and most fundamental
machine learning algorithm. It's perfect for beginners to understand the basics
of supervised learning.

THEORY:
Linear Regression finds the best straight line (or hyperplane in higher dimensions)
that fits the data. The equation is: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Where:
- y is the target variable (what we want to predict)
- β₀ is the intercept (y-intercept when all features are 0)
- β₁, β₂, ..., βₙ are the coefficients (slopes for each feature)
- x₁, x₂, ..., xₙ are the features (input variables)
- ε is the error term (noise)

The algorithm finds the best coefficients by minimizing the Mean Squared Error (MSE):
MSE = (1/n) * Σ(y_true - y_pred)²

This is called "Ordinary Least Squares" (OLS) regression.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.utils import set_seed
from src.algorithms.linear_regression import build_model
from src.pipelines.regression import train_linear_regression, evaluate, predict
import yaml

def main():
    print("🎯 Linear Regression Demo - ML & LLM Finance Portfolio")
    print("=" * 60)
    print("\n📚 THEORY:")
    print("Linear Regression finds the best straight line that fits the data.")
    print("Equation: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε")
    print("Goal: Minimize Mean Squared Error (MSE)")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config_path = "configs/regression/linear_regression.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"\n📊 Configuration loaded from: {config_path}")
    print(f"🎯 Target variable: {cfg['target']}")
    print(f"🔧 Features: {cfg['features']}")
    print(f"🌐 Model parameters: {cfg['model']}")
    
    # Load and examine data
    print(f"\n📈 Data Analysis:")
    train_df = pd.read_csv(cfg['data']['train'])
    print(f"   - Training samples: {len(train_df)}")
    print(f"   - Features: {list(train_df.columns[:-1])}")
    print(f"   - Target: {train_df.columns[-1]}")
    
    # Show data statistics
    print(f"\n📊 Data Statistics:")
    print(train_df.describe().round(3))
    
    # Show correlations
    print(f"\n🔗 Feature Correlations with Target:")
    correlations = train_df.corr()[cfg['target']].sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != cfg['target']:
            print(f"   - {feature}: {corr:.3f}")
    
    # Train model
    print(f"\n🚀 Training Linear Regression model...")
    model_dir = Path("models/linear_regression")
    metrics = train_linear_regression(cfg, build_model, model_dir)
    
    print("✅ Training completed!")
    print(f"📈 Validation metrics:")
    for metric, value in metrics.items():
        print(f"   - {metric.upper()}: {value:.4f}")
    
    # Load the trained model to examine coefficients
    import joblib
    model_path = model_dir / "model.pkl"
    model = joblib.load(model_path)
    
    print(f"\n🔍 Model Coefficients (β values):")
    print(f"   - Intercept (β₀): {model.intercept_:.4f}")
    for feature, coef in zip(cfg['features'], model.coef_):
        print(f"   - {feature} (β): {coef:.4f}")
    
    # Compare with true coefficients
    print(f"\n🎯 True vs Predicted Coefficients:")
    true_coeffs = {'feature1': 2.0, 'feature2': 1.5, 'feature3': -0.5}
    for feature in cfg['features']:
        pred_coef = model.coef_[cfg['features'].index(feature)]
        true_coef = true_coeffs[feature]
        error = abs(pred_coef - true_coef)
        print(f"   - {feature}: True={true_coef:.1f}, Predicted={pred_coef:.4f}, Error={error:.4f}")
    
    # Evaluate model
    print(f"\n🔍 Evaluating model on test set...")
    test_metrics = evaluate(cfg, model_path)
    
    print("✅ Evaluation completed!")
    print(f"📊 Test metrics:")
    for metric, value in test_metrics.items():
        print(f"   - {metric.upper()}: {value:.4f}")
    
    # Make predictions
    print(f"\n🎯 Making predictions on inference data...")
    input_file = "data/processed/regression_inference.csv"
    output_file = "reports/demo_linear_regression_predictions.csv"
    
    if os.path.exists(input_file):
        predictions_path = predict(cfg, model_path, input_file, output_file)
        print(f"✅ Predictions saved to: {predictions_path}")
        
        # Show predictions
        predictions_df = pd.read_csv(predictions_path)
        print(f"\n📋 Predictions preview:")
        print(predictions_df.to_string(index=False))
        
        # Show manual calculation for first prediction
        print(f"\n🧮 Manual calculation for first prediction:")
        first_row = predictions_df.iloc[0]
        manual_pred = model.intercept_
        for feature in cfg['features']:
            coef = model.coef_[cfg['features'].index(feature)]
            value = first_row[feature]
            manual_pred += coef * value
            print(f"   - {feature}: {coef:.4f} × {value:.1f} = {coef * value:.4f}")
        print(f"   - Intercept: {model.intercept_:.4f}")
        print(f"   - Total: {manual_pred:.4f}")
        print(f"   - Model prediction: {first_row['prediction']:.4f}")
    else:
        print(f"⚠️  Inference file not found: {input_file}")
    
    print(f"\n🎉 Linear Regression Demo completed successfully!")
    print(f"\n📁 Generated files:")
    print(f"   - Model: {model_path}")
    print(f"   - Predictions: {output_file}")
    
    print(f"\n🔧 To run the full CLI:")
    print(f"   python -m src.cli train --model linear_regression --task regression --config configs/regression/linear_regression.yaml")
    print(f"   python -m src.cli evaluate --model linear_regression --task regression --config configs/regression/linear_regression.yaml")
    print(f"   python -m src.cli predict --model linear_regression --task regression --config configs/regression/linear_regression.yaml --input data/processed/regression_inference.csv --output reports/predictions.csv")
    
    print(f"\n💡 Key Takeaways:")
    print(f"   - Linear Regression is the foundation of machine learning")
    print(f"   - It finds the best straight line through the data")
    print(f"   - Coefficients show the impact of each feature")
    print(f"   - R² close to 1.0 means the model fits well")
    print(f"   - MSE and RMSE measure prediction error")

if __name__ == "__main__":
    main() 