"""
Tests for core functionality
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.utils import set_seed  # noqa: E402
from src.core.io import load_csv, save_csv  # noqa: E402
from src.core.metrics import classification_metrics, regression_metrics  # noqa: E402


class TestUtils:
    def test_set_seed(self):
        """Test that set_seed works without errors"""
        set_seed(42)
        # Should not raise any exceptions
        assert True


class TestIO:
    def test_save_and_load_csv(self):
        """Test save_csv and load_csv functions"""
        # Create test data
        test_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "target": [0, 1, 0],
            }
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_csv(test_data, temp_path)

            # Load the data back
            loaded_data = load_csv(temp_path)

            # Check if data is the same
            pd.testing.assert_frame_equal(test_data, loaded_data)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestMetrics:
    def test_classification_metrics(self):
        """Test classification metrics calculation"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1])

        metrics = classification_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_regression_metrics(self):
        """Test regression metrics calculation"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        metrics = regression_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert isinstance(metrics["mse"], float)
        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["r2"], float)


class TestAlgorithms:
    def test_linear_regression_import(self):
        """Test that linear regression can be imported"""
        try:
            from src.algorithms.linear_regression import build_model

            assert callable(build_model)
        except ImportError:
            pytest.skip("Linear regression not available")

    def test_random_forest_import(self):
        """Test that random forest can be imported"""
        try:
            from src.algorithms.random_forest import build_model

            assert callable(build_model)
        except ImportError:
            pytest.skip("Random forest not available")


class TestPipelines:
    def test_regression_pipeline_import(self):
        """Test that regression pipeline can be imported"""
        try:
            from src.pipelines.regression import (
                train_linear_regression,
                evaluate,
                predict,
            )

            assert callable(train_linear_regression)
            assert callable(evaluate)
            assert callable(predict)
        except ImportError:
            pytest.skip("Regression pipeline not available")

    def test_classification_pipeline_import(self):
        """Test that classification pipeline can be imported"""
        try:
            from src.pipelines.classification import (
                train_random_forest,
                evaluate,
                predict,
            )

            assert callable(train_random_forest)
            assert callable(evaluate)
            assert callable(predict)
        except ImportError:
            pytest.skip("Classification pipeline not available")
