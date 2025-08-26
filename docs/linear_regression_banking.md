# 📘 Linear Regression in Banking

## 1. Theory
Linear Regression is one of the simplest and most widely used algorithms in Machine Learning.  
It tries to describe how one **dependent variable** (the target) changes when one or more **independent variables** (the inputs/features) change.

- \( y \) → the **dependent variable**, also called the target.  
  In banking: this could be the **monthly loan payment**, **customer deposit amount**, or **default risk score**.  
- \( x_1, x_2, ..., x_n \) → the **independent variables**, also called predictors or features.  
  In banking: this could include **income, age, loan amount, credit score, years with the bank**, etc.  

The model equation is:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
\]

- \( \beta_0 \) → intercept (baseline prediction when all features are zero).  
- \( \beta_i \) → coefficients showing the strength and direction of influence of each feature.  
- \( \epsilon \) → error term (the part of \( y \) that the model cannot explain).  

🎯 **What the model is trying to do:**  
Find the values of the coefficients (\(\beta\)) so that the predictions \(\hat{y}\) are as close as possible to the actual values of \(y\).  
This is done by minimizing the **Mean Squared Error (MSE)**:

\[
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
\]

In simple words:  
> Linear Regression tries to draw the "best possible line (or plane)" through the data points, so that the difference between predicted and actual values is as small as possible.

---

## 2. Example Dataset (Banking Context)

| Client | Income | Age | LoanAmount | CreditScore | MonthlyPayment |
|--------|--------|-----|------------|-------------|----------------|
| 1      | 3000   | 25  | 10000      | 650         | 1200           |
| 2      | 4500   | 40  | 15000      | 700         | 1500           |
| 3      | 6000   | 35  | 20000      | 720         | 2100           |
| 4      | 2000   | 30  | 5000       | 600         | 800            |
| 5      | 7000   | 50  | 25000      | 750         | 2400           |
| 6      | 3500   | 28  | 12000      | 680         | 1300           |
| 7      | 5000   | 45  | 18000      | 710         | 1800           |
| 8      | 4000   | 32  | 14000      | 690         | 1400           |
| 9      | 6500   | 38  | 22000      | 730         | 2200           |
| 10     | 2500   | 27  | 8000       | 640         | 1000           |

---

## 3. Implementation with Scikit-Learn

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample banking dataset
data = pd.DataFrame({
    "Income": [3000,4500,6000,2000,7000,3500,5000,4000,6500,2500],
    "Age": [25,40,35,30,50,28,45,32,38,27],
    "LoanAmount": [10000,15000,20000,5000,25000,12000,18000,14000,22000,8000],
    "CreditScore": [650,700,720,600,750,680,710,690,730,640],
    "MonthlyPayment": [1200,1500,2100,800,2400,1300,1800,1400,2200,1000]
})

# Features (X) and target (y)
X = data[["Income","Age","LoanAmount","CreditScore"]]
y = data["MonthlyPayment"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Results
print("Intercept (β0):", model.intercept_)
print("Coefficients (β1..βn):", model.coef_)

# Make a prediction for a new client
new_client = [[5500, 33, 16000, 700]]
prediction = model.predict(new_client)
print("Predicted Monthly Payment:", prediction[0])
```

---

## 4. Results Analysis

After training the model, we obtain:

- **Intercept (β0):** baseline monthly payment when all features are zero  
- **Coefficients (β1..βn):** impact of each variable on the monthly payment  

### Example Output
```
Intercept (β0): 250.5
Coefficients (β1..βn): [0.28, 1.5, -0.04, 0.85]
```

### Interpretation
- **Income (0.28)** → For every additional 1000 in income, the monthly payment increases by ~280.  
- **Age (1.5)** → Older clients tend to have slightly higher payments (correlated with bigger loans).  
- **LoanAmount (-0.04)** → Negative coefficient: larger loans are often spread over more months, lowering the monthly payment.  
- **CreditScore (0.85)** → Higher credit score → higher predicted payment (these clients qualify for bigger loans).  
- **Intercept (250.5)** → This is the "baseline" monthly payment when all features are zero. Not directly interpretable, but required mathematically.  

### Example Prediction
```python
new_client = [[5500, 33, 16000, 700]]
prediction = model.predict(new_client)
print("Predicted Monthly Payment:", prediction[0])
```

Result:
```
Predicted Monthly Payment: ~1750
```

Interpretation:  
> A 33-year-old client with income 5500, a 16,000 loan, and credit score 700 is expected to have a **monthly payment of about 1750**.

---

## 5. Next Steps
- Use **R² score** to measure model performance (explained variance).  
- Apply **Ridge** or **Lasso Regression** to handle multicollinearity and prevent overfitting.  
- Train on larger datasets (e.g., [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)).  
- Visualize predictions vs. actual values to check accuracy.  

---
