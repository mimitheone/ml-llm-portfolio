from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass
class GradientDescentConfig:
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    random_state: int = 42


class GradientDescentRegressor(BaseEstimator, RegressorMixin):
    """
    Gradient Descent implementation for linear regression.
    
    This is a custom implementation of gradient descent optimization
    for linear regression problems.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, random_state: int = 42):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientDescentRegressor':
        """
        Fit the model using gradient descent.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Fitted model
        """
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred = self._predict(X)
            
            # Compute gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost
            cost = self._compute_cost(y_pred, y)
            self.cost_history.append(cost)
            
            # Check convergence
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break
                
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        return self._predict(X)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Internal prediction method."""
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute mean squared error cost."""
        return np.mean((y_pred - y_true) ** 2)
    
    def get_cost_history(self) -> List[float]:
        """Get the cost history from training."""
        return self.cost_history.copy()


def build_model(cfg: Dict[str, Any]) -> GradientDescentRegressor:
    """
    Build a GradientDescentRegressor model from configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        GradientDescentRegressor: Configured model
    """
    params = GradientDescentConfig(**cfg).__dict__
    return GradientDescentRegressor(**params)
