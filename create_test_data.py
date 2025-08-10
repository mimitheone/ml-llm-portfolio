#!/usr/bin/env python3
"""
Create comprehensive test data for all ML algorithms in the banking AI platform.
This script generates realistic banking datasets for testing all 21 algorithms.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_credit_risk_data(n_samples=1000):
    """Create credit risk dataset for regression and classification models."""
    
    # Customer demographics
    ages = np.random.normal(45, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    incomes = np.random.lognormal(10.5, 0.5, n_samples)
    incomes = np.clip(incomes, 20000, 200000)
    
    credit_scores = np.random.normal(700, 100, n_samples).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)
    
    debt_ratios = np.random.beta(2, 5, n_samples)
    debt_ratios = np.clip(debt_ratios, 0.1, 0.8)
    
    employment_length = np.random.exponential(5, n_samples)
    employment_length = np.clip(employment_length, 0, 30)
    
    # Loan characteristics
    loan_amounts = incomes * np.random.uniform(0.5, 3.0, n_samples)
    loan_amounts = np.clip(loan_amounts, 10000, 500000)
    
    interest_rates = np.random.normal(0.06, 0.02, n_samples)
    interest_rates = np.clip(interest_rates, 0.03, 0.15)
    
    loan_terms = np.random.choice([12, 24, 36, 48, 60], n_samples)
    
    # Risk factors
    gdp_growth = np.random.normal(0.025, 0.01, n_samples)
    unemployment_rate = np.random.normal(0.05, 0.02, n_samples)
    inflation_rate = np.random.normal(0.02, 0.01, n_samples)
    
    # Target variables
    # Probability of Default (0-1) for regression
    pd_score = (
        0.1 * (1 - credit_scores/850) +  # Credit score impact
        0.3 * debt_ratios +               # Debt ratio impact
        0.2 * (1 - employment_length/30) + # Employment stability
        0.1 * (1 - incomes/200000) +     # Income impact
        0.1 * (1 - gdp_growth/0.05) +   # Economic impact
        0.2 * unemployment_rate +         # Unemployment impact
        np.random.normal(0, 0.05, n_samples)  # Random noise
    )
    pd_score = np.clip(pd_score, 0, 1)
    
    # Default flag (0/1) for classification
    default_flag = (pd_score > 0.5).astype(int)
    
    # Credit rating (AAA to D) for multi-class
    rating_mapping = {
        0: 'AAA', 1: 'AA', 2: 'A', 3: 'BBB', 
        4: 'BB', 5: 'B', 6: 'CCC', 7: 'CC', 8: 'C', 9: 'D'
    }
    rating_numeric = np.digitize(pd_score, bins=np.linspace(0, 1, 10)) - 1
    credit_rating = [rating_mapping[r] for r in rating_numeric]
    
    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': ages,
        'income': incomes,
        'credit_score': credit_scores,
        'debt_ratio': debt_ratios,
        'employment_length': employment_length,
        'loan_amount': loan_amounts,
        'interest_rate': interest_rates,
        'loan_term': loan_terms,
        'gdp_growth': gdp_growth,
        'unemployment_rate': unemployment_rate,
        'inflation_rate': inflation_rate,
        'pd_score': pd_score,
        'default_flag': default_flag,
        'credit_rating': credit_rating
    })
    
    return data

def create_transaction_data(n_samples=2000):
    """Create transaction dataset for fraud detection and anomaly detection."""
    
    # Transaction characteristics
    amounts = np.random.lognormal(4, 1.5, n_samples)
    amounts = np.clip(amounts, 1, 10000)
    
    # Merchant categories
    merchant_categories = np.random.choice([
        'retail', 'travel', 'food', 'gas', 'entertainment', 
        'healthcare', 'utilities', 'insurance', 'investment'
    ], n_samples, p=[0.3, 0.1, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    
    # Location patterns
    locations = np.random.choice([
        'local', 'national', 'international'
    ], n_samples, p=[0.7, 0.2, 0.1])
    
    # Time patterns
    hours = np.random.choice(range(24), n_samples, p=[
        0.017, 0.008, 0.008, 0.008, 0.008, 0.008,  # 0-5 AM
        0.017, 0.042, 0.068, 0.085, 0.102, 0.127,  # 6-11 AM
        0.127, 0.102, 0.085, 0.068, 0.042, 0.017,  # 12-5 PM
        0.017, 0.008, 0.008, 0.008, 0.008, 0.012   # 6-11 PM
    ])
    
    # Customer behavior patterns
    customer_ids = np.random.choice(range(1, 501), n_samples)
    
    # Fraud indicators
    fraud_score = np.random.beta(1, 10, n_samples)  # Most transactions are legitimate
    
    # Create fraud flags based on suspicious patterns
    fraud_flag = np.zeros(n_samples)
    
    # High amount + international + unusual hour = suspicious
    suspicious_mask = (
        (amounts > np.percentile(amounts, 95)) & 
        (locations == 'international') & 
        ((hours < 6) | (hours > 22))
    )
    fraud_flag[suspicious_mask] = 1
    
    # Add some random fraud cases
    random_fraud = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    fraud_flag = np.maximum(fraud_flag, random_fraud)
    
    # Anomaly scores for anomaly detection
    anomaly_score = (
        0.3 * (amounts / np.max(amounts)) +
        0.2 * (fraud_flag) +
        0.2 * (locations == 'international').astype(int) +
        0.1 * ((hours < 6) | (hours > 22)).astype(int) +
        0.2 * np.random.random(n_samples)
    )
    
    data = pd.DataFrame({
        'transaction_id': range(1, n_samples + 1),
        'customer_id': customer_ids,
        'amount': amounts,
        'merchant_category': merchant_categories,
        'location': locations,
        'hour': hours,
        'fraud_flag': fraud_flag,
        'anomaly_score': anomaly_score
    })
    
    return data

def create_market_data(n_samples=500):
    """Create market dataset for time series and forecasting models."""
    
    # Date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Economic indicators with trends and seasonality
    time_trend = np.arange(n_samples)
    
    # GDP growth with trend and seasonality
    gdp_growth = (
        0.02 +  # Base growth
        0.0001 * time_trend +  # Slight upward trend
        0.01 * np.sin(2 * np.pi * time_trend / 365) +  # Annual seasonality
        0.005 * np.sin(2 * np.pi * time_trend / 90) +  # Quarterly seasonality
        np.random.normal(0, 0.005, n_samples)  # Random noise
    )
    
    # Interest rates with trend
    interest_rate = (
        0.05 +  # Base rate
        0.0002 * time_trend +  # Gradual increase
        0.01 * np.sin(2 * np.pi * time_trend / 365) +  # Annual cycle
        np.random.normal(0, 0.002, n_samples)  # Small random variation
    )
    
    # Unemployment rate
    unemployment_rate = (
        0.05 +  # Base rate
        -0.0001 * time_trend +  # Gradual decrease
        0.01 * np.sin(2 * np.pi * time_trend / 365) +  # Annual cycle
        np.random.normal(0, 0.003, n_samples)
    )
    
    # Market volatility
    volatility = (
        0.15 +  # Base volatility
        0.02 * np.sin(2 * np.pi * time_trend / 180) +  # Semi-annual cycle
        np.random.normal(0, 0.01, n_samples)
    )
    
    # Banking revenue (target variable for forecasting)
    revenue = (
        1000000 +  # Base revenue
        50000 * time_trend +  # Growth trend
        100000 * np.sin(2 * np.pi * time_trend / 365) +  # Annual seasonality
        50000 * np.sin(2 * np.pi * time_trend / 90) +  # Quarterly seasonality
        np.random.normal(0, 50000, n_samples)  # Random variation
    )
    
    # Cash flows
    cash_flow = (
        500000 +  # Base cash flow
        25000 * time_trend +  # Growth trend
        50000 * np.sin(2 * np.pi * time_trend / 365) +  # Annual seasonality
        25000 * np.sin(2 * np.pi * time_trend / 90) +  # Quarterly seasonality
        np.random.normal(0, 25000, n_samples)
    )
    
    data = pd.DataFrame({
        'date': dates,
        'gdp_growth': gdp_growth,
        'interest_rate': interest_rate,
        'unemployment_rate': unemployment_rate,
        'volatility': volatility,
        'revenue': revenue,
        'cash_flow': cash_flow
    })
    
    return data

def create_customer_segmentation_data(n_samples=1500):
    """Create customer dataset for clustering and segmentation."""
    
    # Customer demographics
    ages = np.random.normal(45, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    incomes = np.random.lognormal(10.5, 0.5, n_samples)
    incomes = np.clip(incomes, 20000, 200000)
    
    # Banking behavior
    account_balance = np.random.lognormal(8, 1, n_samples)
    account_balance = np.clip(account_balance, 100, 100000)
    
    transaction_frequency = np.random.poisson(15, n_samples)
    transaction_frequency = np.clip(transaction_frequency, 1, 50)
    
    avg_transaction_amount = np.random.lognormal(3, 0.8, n_samples)
    avg_transaction_amount = np.clip(avg_transaction_amount, 10, 2000)
    
    # Product usage
    num_products = np.random.poisson(3, n_samples)
    num_products = np.clip(num_products, 1, 8)
    
    credit_card_usage = np.random.uniform(0, 1, n_samples)
    investment_products = np.random.uniform(0, 1, n_samples)
    mortgage_products = np.random.uniform(0, 1, n_samples)
    
    # Customer satisfaction
    satisfaction_score = np.random.normal(7, 1.5, n_samples)
    satisfaction_score = np.clip(satisfaction_score, 1, 10)
    
    # Churn probability
    churn_probability = (
        0.3 * (1 - satisfaction_score/10) +
        0.2 * (1 - incomes/200000) +
        0.2 * (1 - account_balance/100000) +
        0.1 * (1 - num_products/8) +
        0.2 * np.random.random(n_samples)
    )
    churn_probability = np.clip(churn_probability, 0, 1)
    
    churn_flag = (churn_probability > 0.5).astype(int)
    
    # Customer lifetime value
    clv = (
        account_balance * 0.1 +
        transaction_frequency * avg_transaction_amount * 0.02 +
        num_products * 1000 +
        satisfaction_score * 1000
    )
    
    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': ages,
        'income': incomes,
        'account_balance': account_balance,
        'transaction_frequency': transaction_frequency,
        'avg_transaction_amount': avg_transaction_amount,
        'num_products': num_products,
        'credit_card_usage': credit_card_usage,
        'investment_products': investment_products,
        'mortgage_products': mortgage_products,
        'satisfaction_score': satisfaction_score,
        'churn_probability': churn_probability,
        'churn_flag': churn_flag,
        'customer_lifetime_value': clv
    })
    
    return data

def create_portfolio_data(n_samples=300):
    """Create portfolio dataset for risk management and clustering."""
    
    # Asset characteristics
    asset_types = np.random.choice([
        'government_bonds', 'corporate_bonds', 'equities', 
        'real_estate', 'commodities', 'currencies'
    ], n_samples, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
    
    # Portfolio weights (simplified - just one weight per portfolio)
    weights = np.random.uniform(0.1, 1.0, n_samples)
    
    # Risk metrics
    volatility = np.random.gamma(2, 0.1, n_samples)
    volatility = np.clip(volatility, 0.05, 0.5)
    
    beta = np.random.normal(1, 0.3, n_samples)
    beta = np.clip(beta, 0.1, 2.0)
    
    # Returns
    returns = np.random.normal(0.08, 0.15, n_samples)
    returns = np.clip(returns, -0.3, 0.4)
    
    # Correlation matrix (simplified)
    correlation = np.random.uniform(-0.8, 0.8, n_samples)
    
    # Risk-adjusted returns
    sharpe_ratio = returns / volatility
    
    # VaR (Value at Risk)
    var_95 = -1.645 * volatility  # 95% confidence level
    
    # Expected Shortfall
    es_95 = -1.645 * volatility * 1.25  # Simplified calculation
    
    # ESG scores
    esg_score = np.random.beta(3, 2, n_samples)
    
    # Liquidity scores
    liquidity_score = np.random.beta(2, 1, n_samples)
    
    data = pd.DataFrame({
        'portfolio_id': range(1, n_samples + 1),
        'asset_type': asset_types,
        'weight': weights,
        'volatility': volatility,
        'beta': beta,
        'returns': returns,
        'correlation': correlation,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'es_95': es_95,
        'esg_score': esg_score,
        'liquidity_score': liquidity_score
    })
    
    return data

def create_recommendation_data(n_samples=1000):
    """Create recommendation dataset for collaborative and content-based filtering."""
    
    # User preferences
    user_ids = range(1, 101)  # 100 users
    product_ids = range(1, 51)  # 50 products
    
    # Product categories
    product_categories = np.random.choice([
        'savings', 'checking', 'credit_card', 'mortgage', 
        'investment', 'insurance', 'loan', 'wealth_management'
    ], 50)
    
    # User-product interactions
    interactions = []
    
    for user_id in user_ids:
        # Each user interacts with 5-15 products
        num_interactions = np.random.randint(5, 16)
        selected_products = np.random.choice(product_ids, num_interactions, replace=False)
        
        for product_id in selected_products:
            # Rating (1-5) or interaction (0/1)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.25, 0.3, 0.2])
            interaction = 1 if rating >= 3 else 0
            
            interactions.append({
                'user_id': user_id,
                'product_id': product_id,
                'rating': rating,
                'interaction': interaction,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Product features
    product_features = pd.DataFrame({
        'product_id': product_ids,
        'category': product_categories,
        'risk_level': np.random.choice(['low', 'medium', 'high'], 50),
        'min_amount': np.random.choice([0, 1000, 5000, 10000, 50000], 50),
        'interest_rate': np.random.uniform(0.01, 0.15, 50),
        'term_length': np.random.choice([0, 12, 24, 36, 60, 120], 50)
    })
    
    # User features
    user_features = pd.DataFrame({
        'user_id': user_ids,
        'age_group': np.random.choice(['18-25', '26-35', '36-50', '51-65', '65+'], 100),
        'income_level': np.random.choice(['low', 'medium', 'high'], 100),
        'risk_tolerance': np.random.choice(['conservative', 'moderate', 'aggressive'], 100),
        'investment_experience': np.random.choice(['beginner', 'intermediate', 'advanced'], 100)
    })
    
    return interactions_df, product_features, user_features

def main():
    """Generate all test datasets."""
    
    print("🏦 Creating Banking AI Platform Test Data...")
    
    # Create test data directory
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # 1. Credit Risk Data (Regression & Classification)
    print("📊 Creating Credit Risk Dataset...")
    credit_data = create_credit_risk_data(1000)
    credit_data.to_csv(test_data_dir / "credit_risk_data.csv", index=False)
    
    # 2. Transaction Data (Fraud Detection & Anomaly Detection)
    print("💳 Creating Transaction Dataset...")
    transaction_data = create_transaction_data(2000)
    transaction_data.to_csv(test_data_dir / "transaction_data.csv", index=False)
    
    # 3. Market Data (Time Series & Forecasting)
    print("📈 Creating Market Dataset...")
    market_data = create_market_data(500)
    market_data.to_csv(test_data_dir / "market_data.csv", index=False)
    
    # 4. Customer Segmentation Data (Clustering)
    print("👥 Creating Customer Segmentation Dataset...")
    customer_data = create_customer_segmentation_data(1500)
    customer_data.to_csv(test_data_dir / "customer_segmentation_data.csv", index=False)
    
    # 5. Portfolio Data (Risk Management & Clustering)
    print("📊 Creating Portfolio Dataset...")
    portfolio_data = create_portfolio_data(300)
    portfolio_data.to_csv(test_data_dir / "portfolio_data.csv", index=False)
    
    # 6. Recommendation Data (Collaborative & Content-Based Filtering)
    print("🎯 Creating Recommendation Dataset...")
    interactions, products, users = create_recommendation_data(1000)
    interactions.to_csv(test_data_dir / "recommendation_interactions.csv", index=False)
    products.to_csv(test_data_dir / "recommendation_products.csv", index=False)
    users.to_csv(test_data_dir / "recommendation_users.csv", index=False)
    
    # Create dataset summary
    summary = {
        'Dataset': [
            'Credit Risk Data',
            'Transaction Data', 
            'Market Data',
            'Customer Segmentation',
            'Portfolio Data',
            'Recommendation Interactions',
            'Recommendation Products',
            'Recommendation Users'
        ],
        'Records': [
            len(credit_data),
            len(transaction_data),
            len(market_data),
            len(customer_data),
            len(portfolio_data),
            len(interactions),
            len(products),
            len(users)
        ],
        'Use Cases': [
            'Credit scoring, default prediction, rating classification',
            'Fraud detection, anomaly detection, transaction monitoring',
            'Revenue forecasting, cash flow prediction, economic modeling',
            'Customer segmentation, churn prediction, behavioral analysis',
            'Portfolio risk management, asset allocation, VaR calculation',
            'Product recommendations, collaborative filtering',
            'Product features for content-based filtering',
            'User profiles for personalization'
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(test_data_dir / "dataset_summary.csv", index=False)
    
    print("\n✅ All test datasets created successfully!")
    print(f"📁 Location: {test_data_dir.absolute()}")
    print("\n📊 Dataset Summary:")
    print(summary_df.to_string(index=False))
    
    print("\n🎯 Ready for testing all 21 ML algorithms!")
    print("   - Regression: Credit risk, revenue forecasting")
    print("   - Classification: Default prediction, fraud detection")
    print("   - Clustering: Customer segmentation, portfolio grouping")
    print("   - Anomaly Detection: Transaction monitoring, behavior analysis")
    print("   - Time Series: Cash flow forecasting, market prediction")
    print("   - Recommendation: Product matching, customer preferences")

if __name__ == "__main__":
    main()
