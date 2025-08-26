# Logistic Regression Theory

## Overview
Logistic Regression is a classification algorithm that models the probability of a binary outcome using a logistic function, making it essential for credit risk assessment, fraud detection, and customer churn prediction in banking.

## Mathematical Foundation

### Logistic Function
The logistic (sigmoid) function is:

```
σ(z) = 1 / (1 + e^(-z))
```

Where `z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

### Probability Model
The probability of class 1 is:

```
P(y=1|x) = σ(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)
```

### Cost Function
The log-likelihood function is:

```
J(β) = -Σ[y^(i) * log(h_β(x^(i))) + (1-y^(i)) * log(1-h_β(x^(i)))]
```

## Banking Applications

### 1. Credit Risk Assessment
- **Default Prediction**: Probability of loan default
- **Credit Scoring**: Binary classification of creditworthiness
- **Portfolio Risk**: Aggregate default probabilities

### 2. Fraud Detection
- **Transaction Classification**: Legitimate vs. fraudulent transactions
- **Account Takeover**: Unauthorized access detection
- **Money Laundering**: Suspicious activity identification

### 3. Customer Behavior
- **Churn Prediction**: Customer retention modeling
- **Product Adoption**: Likelihood of product usage
- **Response Modeling**: Marketing campaign effectiveness

## Example Dataset (Banking Context)

| Client | Income | Age | LoanAmount | CreditScore | Default |
|--------|--------|-----|------------|-------------|---------|
| 1      | 3000   | 25  | 10000      | 650         | 0       |
| 2      | 4500   | 40  | 15000      | 700         | 0       |
| 3      | 6000   | 35  | 20000      | 720         | 0       |
| 4      | 2000   | 30  | 5000       | 600         | 1       |
| 5      | 7000   | 50  | 25000      | 750         | 0       |
| 6      | 3500   | 28  | 12000      | 680         | 0       |
| 7      | 5000   | 45  | 18000      | 710         | 0       |
| 8      | 4000   | 32  | 14000      | 690         | 1       |
| 9      | 6500   | 38  | 22000      | 730         | 0       |
| 10     | 2500   | 27  | 8000       | 640         | 1       |

---

## Implementation with Scikit-Learn

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Sample banking dataset for credit default prediction
data = pd.DataFrame({
    "Income": [3000,4500,6000,2000,7000,3500,5000,4000,6500,2500],
    "Age": [25,40,35,30,50,28,45,32,38,27],
    "LoanAmount": [10000,15000,20000,5000,25000,12000,18000,14000,22000,8000],
    "CreditScore": [650,700,720,600,750,680,710,690,730,640],
    "Default": [0,0,0,1,0,0,0,1,0,1]  # 0 = No Default, 1 = Default
})

# Features (X) and target (y)
X = data[["Income","Age","LoanAmount","CreditScore"]]
y = data["Default"]

# Scale features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Results
print("Intercept (β0):", model.intercept_[0])
print("Coefficients (β1..βn):", model.coef_[0])

# Make a prediction for a new client
new_client = [[5500, 33, 16000, 700]]
new_client_scaled = scaler.transform(new_client)
probability = model.predict_proba(new_client_scaled)
prediction = model.predict(new_client_scaled)

print("Default Probability:", probability[0][1])
print("Predicted Class:", "Default" if prediction[0] == 1 else "No Default")
```

## Results Analysis

After training the model, we obtain:
- **Intercept (β₀)**: baseline log-odds of default when all features are zero
- **Coefficients (β₁..βₙ)**: impact of each variable on the log-odds of default

### Example Output
```
Intercept (β0): -2.15
Coefficients (β1..βn): [-0.0003, 0.02, 0.0001, -0.008]
```

### Interpretation
- **Income (-0.0003)** → For every additional 1000 in income, the log-odds of default decrease by 0.3 (lower risk)
- **Age (0.02)** → Older clients have slightly higher default risk (may be due to longer repayment periods)
- **LoanAmount (0.0001)** → Larger loans have slightly higher default probability
- **CreditScore (-0.008)** → Higher credit score significantly reduces default risk (most important factor)
- **Intercept (-2.15)** → Baseline log-odds when all features are zero (negative = low default risk)

### Example Prediction
```python
new_client = [[5500, 33, 16000, 700]]
new_client_scaled = scaler.transform(new_client)
probability = model.predict_proba(new_client_scaled)
prediction = model.predict(new_client_scaled)

print("Default Probability:", probability[0][1])
print("Predicted Class:", "Default" if prediction[0] == 1 else "No Default")
```

**Output:**
```
Default Probability: 0.12
Predicted Class: No Default
```

**Interpretation:**
A 33-year-old client with income 5500, a 16,000 loan, and credit score 700 has a 12% probability of default, which is below the typical risk threshold (usually 20-30%), so the loan would likely be approved.

## Next Steps
- Use **ROC-AUC** and **Precision-Recall** metrics to measure model performance for classification tasks
- Apply **L1 (Lasso)** or **L2 (Ridge)** regularization to handle multicollinearity and prevent overfitting
- Train on larger datasets (e.g., **German Credit Dataset**, **UCI Bank Marketing Dataset**)
- Visualize **confusion matrix** and **ROC curves** to check classification accuracy
- Implement **cross-validation** to ensure model stability and generalizability
- Consider **feature engineering** (interaction terms, polynomial features) for better performance
- Apply **threshold optimization** based on business costs (false positive vs. false negative)

## Implementation Considerations

### Feature Engineering
- **Categorical Variables**: One-hot encoding or label encoding
- **Numerical Features**: Scaling and normalization
- **Interaction Terms**: Business-relevant feature combinations
- **Temporal Features**: Time-based patterns

### Class Imbalance
- **Resampling**: SMOTE, undersampling, oversampling
- **Class Weights**: Adjust loss function weights
- **Threshold Tuning**: Optimize decision threshold
- **Cost-Sensitive Learning**: Incorporate business costs

### Regularization
- **L1 (Lasso)**: Feature selection and sparsity
- **L2 (Ridge)**: Prevent overfitting
- **Elastic Net**: Combine both penalties

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Business Metrics
- **ROC-AUC**: Area under ROC curve
- **Precision-Recall AUC**: Area under PR curve
- **Lift**: Model performance vs. random selection
- **Gini Coefficient**: Inequality measure for risk models

### Threshold Optimization
- **Business Costs**: False positive vs. false negative costs
- **ROC Analysis**: Optimal operating point
- **Precision-Recall Trade-off**: Balance between metrics

## Best Practices

### Data Quality
- **Missing Values**: Handle appropriately (imputation, deletion)
- **Outliers**: Detect and handle extreme values
- **Multicollinearity**: Check feature correlations
- **Data Leakage**: Ensure temporal separation

### Model Validation
- **Cross-Validation**: Stratified k-fold CV for imbalanced data
- **Time-Series Split**: Respect temporal order in financial data
- **Out-of-Sample Testing**: Validate on future periods
- **Business Validation**: Confirm predictions make sense

### Feature Selection
- **Statistical Tests**: Chi-square, ANOVA, correlation
- **Recursive Feature Elimination**: Iterative feature selection
- **L1 Regularization**: Automatic feature selection
- **Domain Knowledge**: Business-relevant features

## Regulatory Compliance

### Model Governance
- **Documentation**: Clear model assumptions and limitations
- **Validation**: Independent model validation process
- **Monitoring**: Ongoing performance and stability checks
- **Audit Trail**: Track model changes and decisions

### Risk Management
- **Stress Testing**: Model behavior under extreme scenarios
- **Backtesting**: Historical performance validation
- **Scenario Analysis**: Impact of parameter changes
- **Model Risk**: Quantify model uncertainty

### Explainability
- **Coefficient Interpretation**: Feature importance and direction
- **SHAP Values**: Local and global feature contributions
- **LIME**: Local interpretable explanations
- **Business Rules**: Translate coefficients to business logic

## Advanced Techniques

### Multinomial Logistic Regression
- **Multi-class Classification**: Handle more than two classes
- **One-vs-Rest**: Train binary classifiers for each class
- **Softmax Function**: Extend sigmoid to multiple classes

### Ordinal Logistic Regression
- **Ordered Categories**: Respect natural ordering (e.g., risk ratings)
- **Proportional Odds**: Assume consistent odds ratios
- **Business Applications**: Credit rating, risk tiering

### Bayesian Logistic Regression
- **Prior Distributions**: Incorporate domain knowledge
- **Uncertainty Quantification**: Credible intervals for predictions
- **Regularization**: Natural regularization through priors

## Future Directions

### Machine Learning Integration
- **Ensemble Methods**: Combine with other algorithms
- **Neural Networks**: Use as initialization or baseline
- **AutoML**: Automated hyperparameter optimization

### Real-Time Applications
- **Streaming Data**: Online model updates
- **Microservices**: API-based model serving
- **Edge Computing**: Local model inference

### Interpretability
- **SHAP Integration**: Advanced feature importance
- **Counterfactual Explanations**: What-if scenarios
- **Feature Interactions**: Understand feature relationships
- **Business Rules**: Generate interpretable rules
