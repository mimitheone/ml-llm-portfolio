# 🤖💹 ML & LLM Portfolio (Python) — Financial Analysis, Machine Learning & Agentic Workflows

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/mimitheone/ml-llm-portfolio/workflows/Python%20CI/badge.svg)](https://github.com/mimitheone/ml-llm-portfolio/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: flake8](https://img.shields.io/badge/linting-flake8-yellowgreen)](https://flake8.pycqa.org/)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-blue.svg)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/mimitheone/ml-llm-portfolio)

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
- 🛡 **Risk & fraud detection**  
- 👥 **Customer segmentation**  
- 🎯 **Recommendation systems**  
- 🖼 **Dimensionality reduction & visualization**  
- 🤖 **LLM-based automated analysis & reporting** via LangChain + LangGraph + LlamaIndex

---

## 🧪 Implemented ML Algorithms (20)

### **A. 📉 Regression**
1. Linear Regression  
2. Ridge Regression  
3. Lasso Regression  
4. Polynomial Regression  
5. ARIMA (Time Series)

### **B. 🧩 Classification**
6. Logistic Regression  
7. Decision Tree Classifier  
8. Random Forest Classifier  
9. Gradient Boosting (XGBoost)  
10. Support Vector Machine (SVM)

### **C. 📊 Clustering**
11. K-Means  
12. DBSCAN  
13. Hierarchical Clustering

### **D. 🚨 Anomaly Detection**
14. Isolation Forest  
15. One-Class SVM

### **E. ✂️ Dimensionality Reduction**
16. PCA  
17. t-SNE (Visualization)

### **F. 🎯 Recommendation Systems**
18. Collaborative Filtering (Matrix Factorization)  
19. Content-Based Filtering

### **G. 🧠 Neural Networks**
20. Multilayer Perceptron (MLP)

---

## 🧩 LLM Agentic Workflows
The project also integrates **Large Language Models** for:
- 🗣 **Natural language analysis** of ML outputs  
- 📝 **Automated executive reports** for financial stakeholders  
- 🔍 **Interactive agent systems** that query and interpret structured data  
- 📚 **Data retrieval & enrichment** via LlamaIndex and RAG techniques

---

## 🗂️ Project Structure
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
