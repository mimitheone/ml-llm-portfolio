#!/usr/bin/env python3
"""
Script to create sample regression data for testing linear regression.
This creates a simple dataset where the target is a linear combination of features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_regression_data():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create features
    n_samples = 1000
    
    # Generate features with some correlation
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.normal(0, 1, n_samples)
    
    # Create target as linear combination of features + noise
    # y = 2*feature1 + 1.5*feature2 - 0.5*feature3 + noise
    target = 2 * feature1 + 1.5 * feature2 - 0.5 * feature3 + np.random.normal(0, 0.1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'target': target
    })
    
    # Split data
    train_size = int(0.7 * n_samples)
    valid_size = int(0.15 * n_samples)
    
    train_df = df[:train_size]
    valid_df = df[train_size:train_size + valid_size]
    test_df = df[train_size + valid_size:]
    
    # Create directories
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    train_df.to_csv(data_dir / "regression_train.csv", index=False)
    valid_df.to_csv(data_dir / "regression_valid.csv", index=False)
    test_df.to_csv(data_dir / "regression_test.csv", index=False)
    
    # Create inference data
    inference_df = pd.DataFrame({
        'feature1': [0.5, -0.3, 1.2, -0.8],
        'feature2': [0.1, 0.7, -0.4, 0.9],
        'feature3': [0.3, -0.2, 0.6, -0.1]
    })
    inference_df.to_csv(data_dir / "regression_inference.csv", index=False)
    
    print(f"✅ Created regression datasets:")
    print(f"   - Training: {len(train_df)} samples")
    print(f"   - Validation: {len(valid_df)} samples")
    print(f"   - Test: {len(test_df)} samples")
    print(f"   - Inference: {len(inference_df)} samples")
    
    print(f"\n📊 Data statistics:")
    print(f"   - Target mean: {df['target'].mean():.3f}")
    print(f"   - Target std: {df['target'].std():.3f}")
    print(f"   - Feature1 mean: {df['feature1'].mean():.3f}")
    print(f"   - Feature2 mean: {df['feature2'].mean():.3f}")
    print(f"   - Feature3 mean: {df['feature3'].mean():.3f}")
    
    print(f"\n🎯 True coefficients (for reference):")
    print(f"   - feature1: 2.0")
    print(f"   - feature2: 1.5")
    print(f"   - feature3: -0.5")

if __name__ == "__main__":
    create_regression_data() 