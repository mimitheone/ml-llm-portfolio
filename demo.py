#!/usr/bin/env python3
"""
Demo script for the ML & LLM Finance Portfolio project.
This script demonstrates the basic functionality of the project.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.utils import set_seed
from src.algorithms.random_forest import build_model
from src.pipelines.classification import train_random_forest, evaluate, predict
import yaml

def main():
    print("🤖💹 ML & LLM Finance Portfolio Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config_path = "configs/classification/random_forest.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"📊 Configuration loaded from: {config_path}")
    print(f"🎯 Target variable: {cfg['target']}")
    print(f"🔧 Features: {cfg['features']}")
    print(f"🌳 Model parameters: {cfg['model']}")
    
    # Train model
    print("\n🚀 Training Random Forest model...")
    model_dir = Path("models/random_forest")
    metrics = train_random_forest(cfg, build_model, model_dir)
    
    print("✅ Training completed!")
    print(f"📈 Validation metrics: {metrics}")
    
    # Evaluate model
    print("\n🔍 Evaluating model on test set...")
    model_path = model_dir / "model.pkl"
    test_metrics = evaluate(cfg, model_path)
    
    print("✅ Evaluation completed!")
    print(f"📊 Test metrics: {test_metrics}")
    
    # Make predictions
    print("\n🎯 Making predictions on inference data...")
    input_file = "data/processed/inference.csv"
    output_file = "reports/demo_predictions.csv"
    
    if os.path.exists(input_file):
        predictions_path = predict(cfg, model_path, input_file, output_file)
        print(f"✅ Predictions saved to: {predictions_path}")
        
        # Show predictions
        import pandas as pd
        predictions_df = pd.read_csv(predictions_path)
        print(f"\n📋 Predictions preview:")
        print(predictions_df.to_string(index=False))
    else:
        print(f"⚠️  Inference file not found: {input_file}")
    
    print("\n🎉 Demo completed successfully!")
    print("\n📁 Generated files:")
    print(f"   - Model: {model_path}")
    print(f"   - Predictions: {output_file}")
    
    print("\n🔧 To run the full CLI:")
    print("   python -m src.cli train --model random_forest --task classification --config configs/classification/random_forest.yaml")
    print("   python -m src.cli evaluate --model random_forest --task classification --config configs/classification/random_forest.yaml")
    print("   python -m src.cli predict --model random_forest --task classification --config configs/classification/random_forest.yaml --input data/processed/inference.csv --output reports/predictions.csv")

if __name__ == "__main__":
    main() 