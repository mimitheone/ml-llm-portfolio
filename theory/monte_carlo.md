# Monte Carlo Simulations 🎲

## Table of Contents
- [Simple Explanation (Like You're 5 Years Old)](#simple-explanation-like-youre-5-years-old)
- [What is Monte Carlo Simulation?](#what-is-monte-carlo-simulation)
- [When to Use Monte Carlo](#when-to-use-monte-carlo)
- [Example 1: Portfolio Risk Analysis](#example-1-portfolio-risk-analysis)
- [Example 2: Option Pricing (Black-Scholes)](#example-2-option-pricing-black-scholes)
- [Example 3: Project Timeline Estimation](#example-3-project-timeline-estimation)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation in Python](#implementation-in-python)
- [Advantages & Limitations](#advantages--limitations)
- [Common Mistakes](#common-mistakes)
- [Homework Assignments](#homework-assignments)
- [Quick Reference](#quick-reference)
- [ML Developer Roadmap](#ml-developer-roadmap)

---

## Simple Explanation (Like You're 5 Years Old) 🧒

Imagine you want to know how many candies you'll get if you shake a big jar! 🍬

**The Problem**: You can't count all the candies, but you want to guess!

**Monte Carlo Solution**: 
1. Take a small handful of candies (sample) 🖐️
2. Count how many you got
3. Put them back and shake again
4. Do this many, many times (like 1000 times!)
5. Look at all your results and make a good guess! 🎯

**Real Example**: 
- If you got 5, 7, 4, 6, 8 candies in your samples
- You can guess there are probably around 6 candies in the jar
- The more times you try, the better your guess becomes!

**Why "Monte Carlo"?** 🎰
Named after the famous casino in Monaco because it's like playing many games to see what usually happens!

---

## What is Monte Carlo Simulation? 🎲

Monte Carlo simulation is a computational technique that uses repeated random sampling to obtain numerical results. It's particularly useful for problems that are difficult or impossible to solve analytically.

**Key Concepts**:
- **Random Sampling**: Generate random inputs based on probability distributions
- **Repeated Trials**: Run the simulation thousands or millions of times
- **Statistical Analysis**: Analyze the results to understand probabilities and risks
- **Uncertainty Quantification**: Measure and visualize uncertainty in complex systems

**Core Idea**: Instead of trying to solve a complex problem exactly, we approximate the solution by running many random experiments and analyzing the patterns.

---

## When to Use Monte Carlo 🎯

### ✅ **Perfect For**:
- **Risk Assessment**: Portfolio risk, credit risk, operational risk
- **Financial Modeling**: Option pricing, VaR calculation, stress testing
- **Project Planning**: Timeline estimation, cost forecasting
- **Scientific Computing**: Particle physics, molecular dynamics
- **Game Theory**: Strategy optimization, decision analysis

### ❌ **Avoid When**:
- **Deterministic Problems**: When exact solutions exist
- **Real-time Applications**: When speed is critical
- **Simple Calculations**: When analytical solutions are straightforward
- **Low-dimensional Problems**: When brute force is feasible

---

## Example 1: Portfolio Risk Analysis 📊

### Problem Statement
Calculate the Value at Risk (VaR) for a portfolio with multiple assets using Monte Carlo simulation.

### Dataset Overview
```python
# Sample portfolio data
portfolio = {
    'stocks': {'weight': 0.6, 'return': 0.12, 'volatility': 0.20},
    'bonds': {'weight': 0.3, 'return': 0.05, 'volatility': 0.08},
    'commodities': {'weight': 0.1, 'return': 0.08, 'volatility': 0.25}
}
```

### Implementation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo_portfolio(portfolio, n_simulations=10000, time_horizon=252):
    """
    Monte Carlo simulation for portfolio risk analysis
    """
    results = []
    
    for _ in range(n_simulations):
        # Generate random returns for each asset
        portfolio_return = 0
        for asset, params in portfolio.items():
            # Simulate daily returns using normal distribution
            daily_return = np.random.normal(
                params['return'] / time_horizon, 
                params['volatility'] / np.sqrt(time_horizon)
            )
            portfolio_return += params['weight'] * daily_return
        
        results.append(portfolio_return)
    
    return np.array(results)

# Run simulation
simulation_results = monte_carlo_portfolio(portfolio)

# Calculate VaR (95% confidence level)
var_95 = np.percentile(simulation_results, 5)
print(f"95% VaR: {var_95:.4f}")

# Visualize results
plt.figure(figsize=(12, 6))
plt.hist(simulation_results, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: {var_95:.4f}')
plt.xlabel('Portfolio Return')
plt.ylabel('Frequency')
plt.title('Monte Carlo Portfolio Simulation Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Expected Results
- **95% VaR**: Around -0.015 to -0.020 (1.5% to 2% daily loss)
- **Distribution**: Approximately normal with slight skewness
- **Risk Metrics**: Standard deviation, maximum loss, expected shortfall

### Interpretation
- **Risk Assessment**: Portfolio has 5% chance of losing more than VaR amount daily
- **Capital Allocation**: Helps determine required capital reserves
- **Stress Testing**: Simulates extreme market conditions

---

## Example 2: Option Pricing (Black-Scholes) 💰

### Problem Statement
Price a European call option using Monte Carlo simulation and compare with Black-Scholes analytical solution.

### Implementation
```python
def monte_carlo_option_pricing(S0, K, T, r, sigma, n_simulations=100000):
    """
    Monte Carlo simulation for European call option pricing
    S0: Current stock price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    """
    # Generate random stock price paths
    Z = np.random.standard_normal(n_simulations)
    
    # Stock price at maturity using geometric Brownian motion
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Option payoff at maturity
    payoff = np.maximum(ST - K, 0)
    
    # Discount to present value
    option_price = np.exp(-r * T) * np.mean(payoff)
    
    return option_price

# Example parameters
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

# Monte Carlo price
mc_price = monte_carlo_option_pricing(S0, K, T, r, sigma)
print(f"Monte Carlo Option Price: ${mc_price:.4f}")

# Black-Scholes analytical solution (for comparison)
def black_scholes_call(S0, K, T, r, sigma):
    from scipy.stats import norm
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price

bs_price = black_scholes_call(S0, K, T, r, sigma)
print(f"Black-Scholes Price: ${bs_price:.4f}")
print(f"Difference: ${abs(mc_price - bs_price):.6f}")
```

### Expected Results
- **Monte Carlo Price**: Around $10.45
- **Black-Scholes Price**: Around $10.45
- **Difference**: Very small (< $0.01) with sufficient simulations

### Business Implications
- **Risk Management**: Accurate option pricing for trading strategies
- **Portfolio Hedging**: Determine optimal hedge ratios
- **Regulatory Compliance**: Meet capital requirements for options trading

---

## Example 3: Project Timeline Estimation ⏰

### Problem Statement
Estimate project completion time considering task dependencies and uncertainties.

### Implementation
```python
def monte_carlo_project_timeline(task_durations, dependencies, n_simulations=10000):
    """
    Monte Carlo simulation for project timeline estimation
    task_durations: dict of task names and their duration distributions
    dependencies: list of task dependencies
    """
    project_durations = []
    
    for _ in range(n_simulations):
        # Sample task durations
        sampled_durations = {}
        for task, dist_params in task_durations.items():
            if dist_params['type'] == 'normal':
                duration = np.random.normal(dist_params['mean'], dist_params['std'])
            elif dist_params['type'] == 'triangular':
                duration = np.random.triangular(
                    dist_params['min'], dist_params['mode'], dist_params['max']
                )
            sampled_durations[task] = max(0, duration)  # Duration can't be negative
        
        # Calculate critical path (simplified)
        project_duration = max(sampled_durations.values())
        project_durations.append(project_duration)
    
    return np.array(project_durations)

# Example project tasks
tasks = {
    'Planning': {'type': 'triangular', 'min': 5, 'mode': 7, 'max': 10},
    'Development': {'type': 'triangular', 'min': 15, 'mode': 20, 'max': 30},
    'Testing': {'type': 'normal', 'mean': 8, 'std': 2},
    'Deployment': {'type': 'triangular', 'min': 2, 'mode': 3, 'max': 5}
}

# Run simulation
project_results = monte_carlo_project_timeline(tasks, [])
print(f"Expected Project Duration: {np.mean(project_results):.1f} days")
print(f"90% Confidence Interval: {np.percentile(project_results, [5, 95])}")
```

### Expected Results
- **Expected Duration**: Around 35-40 days
- **90% Confidence**: 30-50 days range
- **Risk Assessment**: Probability of completing within budget timeline

---

## Mathematical Foundation 🧮

### Core Principles
1. **Law of Large Numbers**: As n → ∞, sample mean → population mean
2. **Central Limit Theorem**: Sum of random variables approaches normal distribution
3. **Random Number Generation**: Pseudo-random sequences for reproducibility

### Key Formulas
```
Expected Value: E[X] = (1/n) * Σ(x_i)
Variance: Var(X) = (1/n) * Σ(x_i - E[X])²
Standard Error: SE = √(Var(X) / n)
Confidence Interval: [E[X] - 1.96*SE, E[X] + 1.96*SE]
```

### Convergence Properties
- **Error**: Decreases as 1/√n
- **Precision**: Doubling simulations improves precision by √2
- **Trade-off**: More simulations = better accuracy but slower computation

---

## Implementation in Python 🐍

### Basic Monte Carlo Framework
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class MonteCarloSimulation:
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
        self.results = []
    
    def run_simulation(self, simulation_function, *args, **kwargs):
        """Run Monte Carlo simulation"""
        self.results = []
        for _ in range(self.n_simulations):
            result = simulation_function(*args, **kwargs)
            self.results.append(result)
        return np.array(self.results)
    
    def analyze_results(self):
        """Analyze simulation results"""
        results = np.array(self.results)
        analysis = {
            'mean': np.mean(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results),
            'percentiles': np.percentile(results, [5, 25, 50, 75, 95])
        }
        return analysis
    
    def plot_distribution(self, title="Monte Carlo Results"):
        """Plot results distribution"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.results, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(self.results), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.results):.4f}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
```

### Advanced Features
- **Variance Reduction**: Antithetic variates, control variates
- **Parallel Processing**: Multiprocessing for large simulations
- **Stratified Sampling**: Better coverage of probability space
- **Importance Sampling**: Focus on rare but important events

---

## Advantages & Limitations ⚖️

### ✅ **Advantages**:
- **Flexibility**: Can handle complex, non-linear systems
- **Intuitive**: Easy to understand and explain to stakeholders
- **Uncertainty Quantification**: Provides probability distributions
- **Scalability**: Can be parallelized for faster computation
- **Real-world Modeling**: Incorporates actual data distributions

### ❌ **Limitations**:
- **Computational Cost**: Can be slow for complex models
- **Randomness**: Results vary between runs (can be mitigated)
- **Model Risk**: Quality depends on input assumptions
- **Convergence**: May need many simulations for accuracy
- **Interpretation**: Requires statistical knowledge to analyze results

---

## Common Mistakes 🚫

### 1. **Insufficient Simulations**
- **Problem**: Too few runs lead to inaccurate results
- **Solution**: Start with 10,000+ simulations, increase for complex models

### 2. **Poor Random Number Generation**
- **Problem**: Using poor quality random numbers
- **Solution**: Use numpy.random or other high-quality generators

### 3. **Ignoring Dependencies**
- **Problem**: Assuming independent random variables when they're correlated
- **Solution**: Use copulas or correlation matrices

### 4. **Overlooking Model Validation**
- **Problem**: Not validating simulation results against known solutions
- **Solution**: Test with simple cases first, compare with analytical solutions

### 5. **Misinterpreting Results**
- **Problem**: Confusing mean with most likely outcome
- **Solution**: Always look at full distribution, not just point estimates

---

## Homework Assignments 📚

### Level 1: Beginner (1-2 days)
**Assignment 1**: Basic Monte Carlo Integration
- Calculate π using Monte Carlo method
- Compare accuracy with different numbers of simulations
- Visualize convergence of the estimate

**Assignment 2**: Simple Risk Simulation
- Simulate coin flips to understand probability
- Calculate probability of getting 5+ heads in 10 flips
- Compare with analytical solution

### Level 2: Intermediate (3-5 days)
**Assignment 3**: Financial Risk Modeling
- Implement portfolio VaR calculation
- Test with different asset correlations
- Analyze impact of distribution assumptions

**Assignment 4**: Project Planning Simulation
- Build project timeline estimator
- Include task dependencies and resource constraints
- Calculate probability of meeting deadlines

### Level 3: Advanced (1-2 weeks)
**Assignment 5**: Advanced Financial Modeling
- Implement full Black-Scholes Monte Carlo
- Add multiple underlying assets
- Include transaction costs and market impact

**Assignment 6**: Risk Management System
- Build comprehensive risk assessment tool
- Include stress testing and scenario analysis
- Create interactive dashboard for stakeholders

---

## Quick Reference 📋

### Key Concepts
```
Monte Carlo: Random sampling for numerical approximation
Law of Large Numbers: More samples = better accuracy
Central Limit Theorem: Sums approach normal distribution
```

### Python Implementation
```python
import numpy as np

# Basic Monte Carlo
def monte_carlo_basic(n_simulations):
    results = []
    for _ in range(n_simulations):
        # Your simulation logic here
        result = np.random.normal(0, 1)
        results.append(result)
    return np.array(results)

# Run simulation
simulation_results = monte_carlo_basic(10000)
mean_result = np.mean(simulation_results)
confidence_interval = np.percentile(simulation_results, [5, 95])
```

### Common Applications
- **Finance**: Risk assessment, option pricing, portfolio optimization
- **Engineering**: Reliability analysis, stress testing, quality control
- **Science**: Particle physics, molecular dynamics, climate modeling
- **Business**: Project planning, demand forecasting, supply chain optimization

---

## 🗺️ ML Developer Roadmap

Ready to continue your ML journey? Check out our comprehensive [**ML Developer Roadmap**](../../ROADMAP.md) for the complete learning path from beginner to expert! 🚀
