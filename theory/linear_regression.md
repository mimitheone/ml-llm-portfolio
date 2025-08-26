# Linear Regression

## Overview
Linear Regression is one of the simplest and most widely used algorithms in Machine Learning.  
It tries to describe how one **dependent variable** (the target) changes when one or more **independent variables** (the inputs/features) change.

- **y** → the **dependent variable**, also called the target.  
  In banking: this could be the **monthly loan payment**, **customer deposit amount**, or **default risk score**.  
- **x₁, x₂, ..., xₙ** → the **independent variables**, also called predictors or features.  
  In banking: this could include **income, age, loan amount, credit score, years with the bank**, etc.  

The model equation is:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

- **β₀** → intercept (baseline prediction when all features are zero)  
- **βᵢ** → coefficients showing the strength and direction of influence of each feature  
- **ε** → error term (the part of y that the model cannot explain)  

🎯 **What the model is trying to do:**  
Find the values of the coefficients (β) so that the predictions ŷ are as close as possible to the actual values of y.  
This is done by minimizing the **Mean Squared Error (MSE)**:

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

In simple words:  
> Linear Regression tries to draw the "best possible line (or plane)" through the data points, so that the difference between predicted and actual values is as small as possible.

---

## Mathematical Foundation

### Simple Linear Regression
For a single feature:
```
y = β₀ + β₁x + ε
```

Where:
- **y** = target variable (dependent variable)
- **x** = feature (independent variable)
- **β₀** = y-intercept (bias term)
- **β₁** = slope (coefficient)
- **ε** = error term

### Multiple Linear Regression
For multiple features:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

## Cost Function
Mean Squared Error (MSE):
```
J(β) = (1/2m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

Where:
- **m** = number of training examples
- **h(x⁽ⁱ⁾)** = predicted value
- **y⁽ⁱ⁾** = actual value

## Optimization
**Ordinary Least Squares (OLS)**: Minimizes the sum of squared residuals by solving the normal equation:
```
β = (XᵀX)⁻¹Xᵀy
```

## Banking Applications

### 1. Credit Scoring
- **Input**: Income, age, employment history, credit history
- **Output**: Credit score prediction
- **Use Case**: Loan approval decisions

### 2. Revenue Forecasting
- **Input**: Historical sales data, market indicators, seasonal factors
- **Output**: Future revenue predictions
- **Use Case**: Budget planning, risk assessment

### 3. Risk Assessment
- **Input**: Financial ratios, market conditions, economic indicators
- **Output**: Risk scores
- **Use Case**: Portfolio management, regulatory compliance

## Assumptions
1. **Linearity**: Relationship between features and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No Multicollinearity**: Features are not highly correlated

## Advantages
- ✅ Simple and interpretable
- ✅ Fast training and prediction
- ✅ Provides coefficient importance
- ✅ Works well with small datasets

## Limitations
- ❌ Assumes linear relationships
- ❌ Sensitive to outliers
- ❌ Cannot capture complex patterns
- ❌ Requires feature scaling

## Implementation in Banking
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Scale features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Get coefficients (feature importance)
coefficients = model.coef_
feature_importance = dict(zip(feature_names, coefficients))
```

## Model Evaluation
- **R² Score**: Proportion of variance explained
- **Mean Absolute Error (MAE)**: Average absolute difference  
- **Root Mean Squared Error (RMSE)**: Standard deviation of residuals
- **Residual Analysis**: Check assumptions

## Best Practices
1. **Feature Engineering**: Create meaningful features
2. **Data Preprocessing**: Handle missing values, outliers
3. **Cross-Validation**: Use k-fold cross-validation
4. **Regularization**: Consider Ridge/Lasso for overfitting
5. **Interpretability**: Explain coefficients to stakeholders

## Regulatory Considerations
- **EU AI Act**: Low-risk classification for simple models
- **Basel III**: Risk model validation requirements
- **GDPR**: Explainable AI requirements
- **Model Governance**: Document assumptions and limitations
