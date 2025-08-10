# Random Forest

## Overview
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes predicted by individual trees (classification) or the mean prediction of individual trees (regression). It's known for its robustness, interpretability, and ability to handle high-dimensional data.

## Mathematical Foundation

### Ensemble Learning
Random Forest combines predictions from multiple decision trees:

**Classification**:
```
ŷ = mode({ŷ₁, ŷ₂, ..., ŷₙ})
```

**Regression**:
```
ŷ = (1/n) * Σᵢ₌₁ⁿ ŷᵢ
```

Where:
- `ŷᵢ` = prediction from tree i
- `n` = number of trees in the forest

### Tree Construction
Each tree is built using:
1. **Bootstrap sampling** from training data
2. **Random feature selection** at each split
3. **Greedy optimization** of split criteria

### Split Criteria
**Gini Index** (Classification):
```
Gini = 1 - Σᵢ₌₁ᶜ pᵢ²
```

**Information Gain** (Classification):
```
IG = H(S) - Σᵢ₌₁ᵛ (|Sᵥ|/|S|) * H(Sᵥ)
```

**Mean Squared Error** (Regression):
```
MSE = (1/n) * Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

Where:
- `pᵢ` = probability of class i
- `H(S)` = entropy of set S
- `Sᵥ` = subset of S for value v

## Banking Applications

### 1. Credit Risk Assessment
- **Input**: Financial ratios, credit history, demographic data
- **Output**: Default probability, risk score
- **Use Case**: Loan approval, credit limit setting

### 2. Fraud Detection
- **Input**: Transaction patterns, user behavior, location data
- **Output**: Fraud probability, anomaly score
- **Use Case**: Real-time fraud monitoring

### 3. Customer Churn Prediction
- **Input**: Transaction history, product usage, service interactions
- **Output**: Churn probability, retention score
- **Use Case**: Customer retention strategies

### 4. Market Risk Modeling
- **Input**: Market indicators, economic data, volatility measures
- **Output**: Risk level, VaR estimates
- **Use Case**: Portfolio risk management

### 5. AML (Anti-Money Laundering)
- **Input**: Transaction patterns, customer profiles, network analysis
- **Output**: Suspicious activity score
- **Use Case**: Regulatory compliance

## Implementation in Banking

### Credit Scoring Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np

# Prepare features
features = [
    'income', 'age', 'employment_years', 'credit_history_length',
    'debt_to_income', 'payment_history', 'credit_utilization',
    'number_of_accounts', 'recent_inquiries'
]

X = credit_data[features]
y = credit_data['default_flag']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Model evaluation
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nFeature Importance:")
print(feature_importance)
```

## Key Parameters

### Tree Parameters
- **n_estimators**: Number of trees (default: 100)
- **max_depth**: Maximum depth of trees (default: None)
- **min_samples_split**: Minimum samples to split (default: 2)
- **min_samples_leaf**: Minimum samples in leaf (default: 1)

### Feature Selection
- **max_features**: Features to consider for splits
  - `'sqrt'`: √n features
  - `'log2'`: log₂(n) features
  - `int`: Specific number of features
  - `float`: Fraction of features

### Sampling Parameters
- **bootstrap**: Use bootstrap sampling (default: True)
- **max_samples**: Sample size for bootstrap (default: None)

## Model Interpretability

### 1. Feature Importance
```python
# Global feature importance
importance = rf_model.feature_importances_
feature_names = X.columns

# Plot feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(range(len(importance)), importance)
plt.yticks(range(len(importance)), feature_names)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

### 2. Partial Dependence Plots
```python
from sklearn.inspection import partial_dependence

# Partial dependence for income
pdp_income = partial_dependence(
    rf_model, X_train, [0], percentiles=(0.05, 0.95)
)

plt.figure(figsize=(8, 6))
plt.plot(pdp_income[1][0], pdp_income[0][0])
plt.xlabel('Income')
plt.ylabel('Partial Dependence')
plt.title('Partial Dependence Plot: Income')
plt.show()
```

### 3. Individual Tree Predictions
```python
# Get predictions from individual trees
tree_predictions = []
for tree in rf_model.estimators_:
    tree_pred = tree.predict(X_test)
    tree_predictions.append(tree_pred)

# Analyze prediction consistency
tree_predictions = np.array(tree_predictions)
prediction_consistency = np.std(tree_predictions, axis=0)
```

## Banking-Specific Considerations

### 1. Regulatory Compliance
- **EU AI Act**: High-risk classification for credit decisions
- **Basel III**: Model validation requirements
- **GDPR**: Right to explanation
- **Fair Lending**: Anti-discrimination requirements

### 2. Model Validation
- **Backtesting**: Historical performance validation
- **Stress Testing**: Extreme scenario testing
- **Out-of-Time Testing**: Future data validation
- **Out-of-Sample Testing**: Unseen data validation

### 3. Risk Management
- **Model Risk**: Validate assumptions and limitations
- **Operational Risk**: Monitor model performance
- **Reputational Risk**: Ensure fair and ethical decisions

## Model Evaluation

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve

### Regression Metrics
- **R² Score**: Coefficient of determination
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Root Mean Squared Error (RMSE)**: Standard deviation of residuals

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(
    rf_model, X_train, y_train, 
    cv=5, scoring='roc_auc'
)

print(f"CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## Best Practices

### 1. Data Preprocessing
- Handle missing values appropriately
- Encode categorical variables
- Scale numerical features if needed
- Remove outliers that may affect splits

### 2. Hyperparameter Tuning
- Use grid search or random search
- Cross-validate parameter combinations
- Consider business constraints
- Monitor overfitting

### 3. Feature Engineering
- Create domain-specific features
- Handle multicollinearity
- Consider feature interactions
- Validate feature stability

### 4. Model Monitoring
- Track performance metrics over time
- Monitor feature importance stability
- Validate predictions on new data
- Retrain models periodically

## Advantages
- ✅ Robust to overfitting
- ✅ Handles high-dimensional data
- ✅ Provides feature importance
- ✅ Works with missing values
- ✅ Fast training and prediction
- ✅ Good interpretability

## Limitations
- ❌ Less interpretable than single trees
- ❌ May not capture complex interactions
- ❌ Requires sufficient training data
- ❌ Computationally intensive for large forests
- ❌ Black-box nature for individual predictions

## Future Directions
- **Explainable AI**: SHAP values, LIME
- **Federated Learning**: Distributed training
- **Online Learning**: Incremental updates
- **Quantum Random Forest**: Quantum computing integration
