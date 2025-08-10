# 🤖💹 ML & LLM Portfolio (Python) — Financial Analysis, Machine Learning & Agentic Workflows

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚦 **Status & Quality**
[![GitHub Actions](https://github.com/mimitheone/ml-llm-portfolio/workflows/Python%20CI/badge.svg)](https://github.com/mimitheone/ml-llm-portfolio/actions/workflows/ci.yml)
[![Build Status](https://img.shields.io/github/actions/workflow/status/mimitheone/ml-llm-portfolio/ci.yml?branch=main&label=CI/CD)](https://github.com/mimitheone/ml-llm-portfolio/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/mimitheone/ml-llm-portfolio/branch/main/graph/badge.svg)](https://codecov.io/gh/mimitheone/ml-llm-portfolio)
[![Coverage](https://img.shields.io/badge/coverage-80%25+-brightgreen.svg)](https://codecov.io/gh/mimitheone/ml-llm-portfolio)

## 🛠️ **Tools & Standards**
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: flake8](https://img.shields.io/badge/linting-flake8-yellowgreen)](https://flake8.pycqa.org/)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-blue.svg)](https://docs.pytest.org/)

## 📊 **Repository Info**
[![Last Commit](https://img.shields.io/github/last-commit/mimitheone/ml-llm-portfolio)](https://github.com/mimitheone/ml-llm-portfolio/commits/main)
[![Repository Size](https://img.shields.io/github/repo-size/mimitheone/ml-llm-portfolio)](https://github.com/mimitheone/ml-llm-portfolio)
[![Issues](https://img.shields.io/github/issues/mimitheone/ml-llm-portfolio)](https://github.com/mimitheone/ml-llm-portfolio/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/mimitheone/ml-llm-portfolio)](https://github.com/mimitheone/ml-llm-portfolio/pulls)

## 🚀 About Me
I am a results-driven technology leader with **15+ years in IT** and **8+ years in management**, known for **strong leadership, strategic thinking, and mentoring talent**.  
I have led diverse teams, delivered mission-critical projects, and combined business insight with deep technical expertise.  

This portfolio project was **conceived, designed, and implemented entirely by me** — from the initial concept to the final code.  
It demonstrates my ability to **bridge business and technology**, delivering **end-to-end solutions** that integrate **Machine Learning, AI, LLMs, and financial analytics** into production-ready systems.

---

## 📊 Project Overview
This repository contains **20 Machine Learning algorithms** implemented in Python for real-world **financial data analysis**, integrated with **Large Language Model (LLM) agent workflows** using LangChain, LangGraph, and LlamaIndex.  

**Key features:**
- 📈 **Forecasting** (revenues, expenses, cash flows)  
- 🛡️ **Risk & fraud detection**  
- 👥 **Customer segmentation**  
- 🎯 **Recommendation systems**  
- 🖼️ **Dimensionality reduction & visualization**  
- 🤖 **LLM-based automated analysis & reporting** via LangChain + LangGraph + LlamaIndex

---

## 🧪 Implemented ML Algorithms (21)

### **A. 📉 Regression**
1. Linear Regression  
2. Ridge Regression  
3. Lasso Regression  
4. Polynomial Regression  
5. ARIMA (Time Series)
6. Gradient Descent

### **B. 🔍 Classification**
7. Logistic Regression  
8. Decision Tree Classifier  
9. Random Forest Classifier  
10. Gradient Boosting (XGBoost)  
11. Support Vector Machine (SVM)

### **C. 📊 Clustering**
12. K-Means  
13. DBSCAN  
14. Hierarchical Clustering

### **D. ⚠️ Anomaly Detection**
15. Isolation Forest  
16. One-Class SVM

### **E. 🔧 Dimensionality Reduction**
17. PCA  
18. t-SNE (Visualization)

### **F. 🎯 Recommendation Systems**
19. Collaborative Filtering (Matrix Factorization)  
20. Content-Based Filtering

### **G. 🧠 Neural Networks**
21. Multilayer Perceptron (MLP)

---

## 🧩 LLM Agentic Workflows
The project also integrates **Large Language Models** for:
- 🗣️ **Natural language analysis** of ML outputs  
- 📝 **Automated executive reports** for financial stakeholders  
- 🔍 **Interactive agent systems** that query and interpret structured data  
- 📚 **Data retrieval & enrichment** via LlamaIndex and RAG techniques

---

## 🏦 **Banking Use-Cases & KPIs**

This portfolio demonstrates **real-world banking applications** with industry-standard metrics and regulatory compliance:

| **Category** | **Key Metrics** | **ML/LLM Applications** |
|--------------|-----------------|-------------------------|
| **🛡️ Risk Management** | PD/LGD/EAD, NPL ratio, VaR, Stress tests | Credit scoring, portfolio risk modeling, stress testing automation |
| **📈 Performance** | ROTE, NIM, Cost-to-Income, Fee income | Revenue forecasting, cost optimization, profitability analysis |
| **💧 Liquidity** | LCR/NSFR, Cash gap analysis | Liquidity forecasting, ALM optimization, regulatory reporting |
| **🌱 ESG/Compliance** | ESG score, Explainability (SHAP), EU AI Act mapping | Risk category classification, data governance, transparency reporting |

### 🤖 **LLM Agent Example: KPI Analysis**

```python
# Example: Ask the agent to explain a KPI movement
from src.llm_agents.report_agent import analyze_metrics

# Analyze banking KPIs with SHAP explanations
analysis = analyze_metrics(
    kpis={
        "ROTE": 0.111, 
        "NIM": 0.027,
        "Cost-to-Income": 0.58,
        "NPL_Ratio": 0.023
    }, 
    shap_summary="Feature 'GDP_growth' contributed +0.8% to ROTE prediction",
    market_context="Q3 2024 - Economic recovery phase"
)

print(analysis[:600])
# Output: "ROTE increased to 11.1% (+0.8% from Q2) driven by GDP growth 
#         recovery and improved credit quality. NIM compression to 2.7% reflects 
#         competitive pricing environment. Cost-to-Income ratio at 58% shows 
#         operational efficiency gains..."
```

---

## 📁 Project Structure
```text
.
├─ configs/
├─ data/
├─ docs/
├─ models/
├─ notebooks/
├─ reports/
├─ src/
│  ├─ core/
│  ├─ data/
│  ├─ pipelines/
│  ├─ algorithms/
│  ├─ llm_agents/          # LangChain, LangGraph, LlamaIndex integrations
│  └─ cli.py
├─ tests/
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ README.md
```

---

## 🛠 Tech Stack
- **Languages & Frameworks:** Python, scikit-learn, XGBoost, statsmodels, PyTorch  
- **LLM & Agents:** LangChain, LangGraph, LlamaIndex  
- **Data Manipulation:** pandas, NumPy  
- **Visualization:** matplotlib, plotly  
- **Config & Utils:** YAML, joblib  
- **DevOps:** Docker, GitHub Actions CI/CD  

---

## 🎯 Additional Specialisations
- 🇪🇺 **EU AI Act Compliance** — Applying the European Union’s AI Act principles in financial ML and LLM systems, ensuring legal compliance, ethical standards, and transparency.  
- 🧾 **XAI (Explainable AI)** — Designing interpretable models and workflows using SHAP, LIME, and other explainability tools.  
- 🤝 **Agent-Oriented Design** — Building multi-agent workflows with LangChain, LangGraph, and LlamaIndex for complex decision-making.  

---

## ⚙️ Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

---

## 🚀 Quick Start
```bash
# Train a model
python -m src.cli train \
  --model random_forest \
  --task classification \
  --config configs/classification/random_forest.yaml

# Evaluate
python -m src.cli evaluate \
  --model random_forest \
  --task classification \
  --config configs/classification/random_forest.yaml

# Predict
python -m src.cli predict \
  --model random_forest \
  --task classification \
  --config configs/classification/random_forest.yaml \
  --input data/processed/inference.csv \
  --output reports/predictions_rf.csv
```

---

## 📈 Example Outputs

### ARIMA Forecast
![ARIMA Forecast](docs/images/arima_forecast_placeholder.png)

### Confusion Matrix
![Confusion Matrix](docs/images/confusion_matrix_placeholder.png)

### SHAP Feature Importance
![SHAP Plot](docs/images/shap_placeholder.png)

---

## 📝 Sample Output
```text
Model: Random Forest Classifier
Accuracy: 0.94
Precision: 0.92
Recall: 0.95
F1-Score: 0.935
ROC AUC: 0.97

Top 3 Features by Importance:
1. revenue_growth
2. operating_margin
3. debt_to_equity_ratio
```

---

## 🤖 Example LLM Workflow
**LangChain + LangGraph + LlamaIndex pipeline**:
1. Retrieve latest financial data from processed dataset  
2. Pass through ML models (e.g., ARIMA, XGBoost)  
3. Use LangGraph to orchestrate analysis agents  
4. LlamaIndex for contextual RAG over financial documents  
5. Generate human-readable insights and strategic recommendations  

---

## 🐳 Docker Support
```bash
docker build -t ml-finance .
docker run -it --rm ml-finance
```

---

## 🔄 CI/CD with GitHub Actions
- ✅ Automatic testing on pull requests  
- ✅ Linting & formatting checks (black, flake8)  
- ✅ Automatic build & push of Docker image  
- ☁️ Optional deployment to cloud  

---

## 📅 Roadmap
- [ ] Cross-validation & Optuna hyperparameter tuning  
- [ ] Explainability with SHAP & permutation importance  
- [ ] MLflow or Weights & Biases integration  
- [ ] Advanced time series with seasonality & holidays  
- [ ] REST API with FastAPI for online inference  
- [ ] Full integration of LangChain + LangGraph agent flows for automated financial reporting  

---

## 🪪 License
MIT License — free to use with attribution.

---

## 💬 Final Note
This is not just a code repository — it's a **demonstration of my ability to deliver complex, business-critical AI/ML projects from start to finish**.  
It merges **financial acumen**, **machine learning expertise**, **LLM agent design**, **EU AI Act compliance**, and **strong leadership** into a single, production-ready package.
