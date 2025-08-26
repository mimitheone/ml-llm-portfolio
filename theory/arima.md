# ARIMA (AutoRegressive Integrated Moving Average)

## Overview
ARIMA is a statistical model for analyzing and forecasting time series data. It combines three components: AutoRegressive (AR), Integrated (I), and Moving Average (MA) to model temporal dependencies and trends in financial time series.

## Mathematical Foundation

### ARIMA(p,d,q) Model
The ARIMA model is defined as:
```
(1 - Σᵢ₌₁ᵖ φᵢLⁱ)(1 - L)ᵈyₜ = (1 + Σᵢ₌₁ᵖ θᵢLⁱ)εₜ
```

Where:
- `p` = order of autoregressive terms
- `d` = degree of differencing
- `q` = order of moving average terms
- `L` = lag operator (Lyₜ = yₜ₋₁)
- `φᵢ` = autoregressive coefficients
- `θᵢ` = moving average coefficients
- `εₜ` = white noise error term

### Component Breakdown

#### 1. AutoRegressive (AR) Component
Models the relationship between current and past values:
```
yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ
```

#### 2. Integrated (I) Component
Handles non-stationarity through differencing:
```
Δyₜ = yₜ - yₜ₋₁
Δ²yₜ = Δyₜ - Δyₜ₋₁
```

#### 3. Moving Average (MA) Component
Models the relationship between current error and past errors:
```
yₜ = μ + εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θₚεₜ₋ₚ
```

## Banking Applications

### 1. Stock Price Forecasting
- **Input**: Historical stock prices, trading volumes
- **Output**: Future price predictions, volatility estimates
- **Use Case**: Portfolio management, trading strategies

### 2. Interest Rate Modeling
- **Input**: Historical interest rates, economic indicators
- **Output**: Rate forecasts, yield curve predictions
- **Use Case**: Asset-liability management, risk modeling

### 3. Credit Risk Modeling
- **Input**: Default rates, economic cycles, market conditions
- **Output**: Default probability forecasts, risk trends
- **Use Case**: Credit portfolio management, regulatory reporting

### 4. Foreign Exchange Forecasting
- **Input**: Historical exchange rates, economic data
- **Output**: Currency movement predictions, volatility forecasts
- **Use Case**: FX trading, international business planning

### 5. Market Volatility Prediction
- **Input**: Historical volatility measures, market stress indicators
- **Output**: Volatility forecasts, risk level predictions
- **Use Case**: Risk management, VaR calculations

## Implementation in Banking

### Basic ARIMA Implementation
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load financial time series data
data = pd.read_csv('stock_prices.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Check stationarity
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    return result[1] < 0.05

# Make series stationary if needed
if not check_stationarity(data['price']):
    data['price_diff'] = data['price'].diff().dropna()
    if not check_stationarity(data['price_diff']):
        data['price_diff2'] = data['price_diff'].diff().dropna()

# Determine ARIMA parameters using ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['price_diff'].dropna(), ax=ax1, lags=40)
plot_pacf(data['price_diff'].dropna(), ax=ax2, lags=40)
plt.tight_layout()
plt.show()

# Fit ARIMA model
model = ARIMA(data['price'], order=(2, 1, 2))
fitted_model = model.fit()

# Model summary
print(fitted_model.summary())

# Forecast
forecast_steps = 30
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['price'], label='Historical Data')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.fill_between(forecast.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='red', alpha=0.1, label='95% Confidence Interval')
plt.title('ARIMA Forecast: Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

## Model Selection

### 1. Stationarity Testing
- **Augmented Dickey-Fuller (ADF) Test**
- **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**
- **Phillips-Perron Test**

### 2. Parameter Selection
- **ACF/PACF Analysis**: Identify p and q orders
- **Information Criteria**: AIC, BIC, HQIC
- **Grid Search**: Test multiple parameter combinations

### 3. Model Diagnostics
- **Residual Analysis**: Check for white noise
- **Ljung-Box Test**: Test autocorrelation in residuals
- **Jarque-Bera Test**: Test normality of residuals

## Advanced ARIMA Variants

### 1. SARIMA (Seasonal ARIMA)
For seasonal time series:
```
SARIMA(p,d,q)(P,D,Q,s)
```

Where:
- `(P,D,Q)` = seasonal parameters
- `s` = seasonal period

### 2. ARIMAX (ARIMA with Exogenous Variables)
Incorporates external factors:
```
yₜ = c + Σᵢ₌₁ᵖ φᵢyₜ₋ᵢ + Σᵢ₌₁ᵖ θᵢεₜ₋ᵢ + Σᵢ₌₁ᵏ βᵢxᵢ,ₜ
```

### 3. VARIMA (Vector ARIMA)
For multiple related time series:
```
Yₜ = c + Σᵢ₌₁ᵖ ΦᵢYₜ₋ᵢ + Σᵢ₌₁ᵖ Θᵢεₜ₋ᵢ
```

## Banking-Specific Considerations

### 1. Market Regimes
- **Bull/Bear Markets**: Different ARIMA parameters
- **Volatility Clustering**: GARCH models may be needed
- **Regime Switching**: Markov-switching models

### 2. Economic Cycles
- **Business Cycles**: Long-term trends
- **Seasonal Patterns**: Quarterly, annual cycles
- **Structural Breaks**: Policy changes, crises

### 3. Regulatory Requirements
- **Basel III**: Model validation requirements
- **Stress Testing**: Extreme scenario modeling
- **Backtesting**: Historical performance validation

## Model Validation

### 1. In-Sample Validation
- **Residual Analysis**: Check model assumptions
- **Goodness of Fit**: R², AIC, BIC
- **Parameter Significance**: t-tests, confidence intervals

### 2. Out-of-Sample Validation
- **Holdout Sample**: Reserve recent data
- **Rolling Window**: Update model periodically
- **Expanding Window**: Include more data over time

### 3. Performance Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Directional Accuracy**: Correct trend prediction

## Best Practices

### 1. Data Preprocessing
- Handle missing values appropriately
- Remove outliers that may affect stationarity
- Ensure consistent time intervals
- Consider seasonal adjustments

### 2. Model Specification
- Start with simple models (ARIMA(1,1,1))
- Use information criteria for model selection
- Consider economic theory and domain knowledge
- Validate model assumptions

### 3. Forecasting
- Provide confidence intervals
- Consider multiple scenarios
- Update models regularly
- Monitor forecast accuracy

### 4. Risk Management
- Quantify forecast uncertainty
- Consider worst-case scenarios
- Validate against external factors
- Document model limitations

## Advantages
- ✅ Handles temporal dependencies
- ✅ Provides uncertainty quantification
- ✅ Well-established statistical foundation
- ✅ Interpretable parameters
- ✅ Handles trends and seasonality
- ✅ Fast computation

## Limitations
- ❌ Assumes linear relationships
- ❌ Requires stationary data
- ❌ Sensitive to parameter selection
- ❌ May not capture complex patterns
- ❌ Requires sufficient historical data
- ❌ Assumes constant parameters

## Future Directions
- **Deep Learning**: LSTM, Transformer models
- **Hybrid Models**: Combine ARIMA with ML
- **Real-time Updates**: Online learning
- **Multivariate Models**: Handle multiple series
- **Non-linear Extensions**: Threshold ARIMA


---

## 🗺️ ML Developer Roadmap

Ready to continue your ML journey? Check out our comprehensive [**ML Developer Roadmap**](../../ROADMAP.md) for the complete learning path from beginner to expert! 🚀
