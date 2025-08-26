# K-Means Clustering

## Overview
K-Means is an unsupervised learning algorithm that partitions data into K clusters based on similarity. It's widely used in banking for customer segmentation, risk profiling, and portfolio grouping. The algorithm iteratively assigns data points to the nearest cluster centroid and updates centroids based on the mean of assigned points.

## Mathematical Foundation

### Objective Function
K-Means minimizes the within-cluster sum of squares (WCSS):
```
J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

Where:
- `k` = number of clusters
- `Cᵢ` = cluster i
- `μᵢ` = centroid of cluster i
- `x` = data point
- `||x - μᵢ||²` = squared Euclidean distance

### Algorithm Steps

#### 1. Initialization
Randomly select k initial centroids:
```
μᵢ⁽⁰⁾ = random point from dataset, i = 1, 2, ..., k
```

#### 2. Assignment Step
Assign each point to nearest centroid:
```
Cᵢ = {x : ||x - μᵢ||² ≤ ||x - μⱼ||² ∀j ≠ i}
```

#### 3. Update Step
Recalculate centroids as mean of assigned points:
```
μᵢ = (1/|Cᵢ|) * Σₓ∈Cᵢ x
```

#### 4. Convergence
Repeat steps 2-3 until centroids stabilize or max iterations reached.

## Banking Applications

### 1. Customer Segmentation
- **Input**: Transaction patterns, demographics, product usage
- **Output**: Customer segments (e.g., high-value, mass market, premium)
- **Use Case**: Marketing strategies, product development

### 2. Risk Profiling
- **Input**: Financial ratios, credit scores, behavioral data
- **Output**: Risk clusters (low, medium, high risk)
- **Use Case**: Credit scoring, portfolio management

### 3. Portfolio Grouping
- **Input**: Asset characteristics, returns, volatility
- **Output**: Portfolio clusters for diversification
- **Use Case**: Asset allocation, risk management

### 4. Fraud Detection
- **Input**: Transaction patterns, user behavior, location data
- **Output**: Normal vs. anomalous behavior clusters
- **Use Case**: Real-time fraud monitoring

### 5. Branch Performance Analysis
- **Input**: Branch metrics, customer demographics, location data
- **Output**: Branch performance clusters
- **Use Case**: Resource allocation, performance optimization

## Implementation in Banking

### Customer Segmentation Example
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load customer data
customer_data = pd.read_csv('customer_data.csv')

# Select features for clustering
features = [
    'age', 'income', 'credit_score', 'transaction_frequency',
    'average_transaction_amount', 'products_owned', 'tenure_years'
]

X = customer_data[features]

# Handle missing values
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow method
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Optimal k (from elbow method)
optimal_k = 5

# Fit K-Means model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to data
customer_data['cluster'] = cluster_labels

# Analyze clusters
cluster_analysis = customer_data.groupby('cluster')[features].mean()
cluster_sizes = customer_data['cluster'].value_counts().sort_index()

print("Cluster Analysis:")
print(cluster_analysis)
print("\nCluster Sizes:")
print(cluster_sizes)

# Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                     cmap='viridis', alpha=0.7, s=50)
plt.colorbar(scatter)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Customer Segments (PCA Visualization)')
plt.show()

# Cluster profiling
for cluster_id in range(optimal_k):
    cluster_data = customer_data[customer_data['cluster'] == cluster_id]
    
    print(f"\n=== Cluster {cluster_id} Profile ===")
    print(f"Size: {len(cluster_data)} customers")
    print(f"Average Income: ${cluster_data['income'].mean():,.0f}")
    print(f"Average Credit Score: {cluster_data['credit_score'].mean():.0f}")
    print(f"Average Age: {cluster_data['age'].mean():.1f} years")
    print(f"Products Owned: {cluster_data['products_owned'].mean():.1f}")
```

## Key Parameters

### Core Parameters
- **n_clusters**: Number of clusters (k)
- **init**: Initialization method ('k-means++', 'random')
- **n_init**: Number of initializations (default: 10)
- **max_iter**: Maximum iterations (default: 300)
- **tol**: Tolerance for convergence (default: 1e-4)

### Initialization Methods
1. **k-means++**: Smart initialization to avoid poor local optima
2. **random**: Random initialization (may lead to poor results)

## Model Evaluation

### 1. Inertia (Within-Cluster Sum of Squares)
Lower values indicate better clustering:
```
inertia = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

### 2. Silhouette Score
Measures cluster quality (-1 to 1, higher is better):
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- `a(i)` = average distance to points in same cluster
- `b(i)` = minimum average distance to points in other clusters

### 3. Calinski-Harabasz Index
Ratio of between-cluster to within-cluster dispersion:
```
CH = [tr(Bₖ) / (k-1)] / [tr(Wₖ) / (n-k)]
```

### 4. Davies-Bouldin Index
Average similarity measure of clusters (lower is better):
```
DB = (1/k) * Σᵢ₌₁ᵏ maxⱼ≠ᵢ (σᵢ + σⱼ) / d(μᵢ, μⱼ)
```

## Banking-Specific Considerations

### 1. Feature Selection
- **Demographic**: Age, income, education, location
- **Financial**: Credit score, income, debt levels
- **Behavioral**: Transaction patterns, product usage
- **Risk**: Default history, payment behavior

### 2. Data Quality
- **Missing Values**: Handle appropriately (imputation, removal)
- **Outliers**: May affect cluster centroids
- **Scaling**: Ensure features are comparable
- **Categorical Variables**: Encode appropriately

### 3. Interpretability
- **Cluster Profiles**: Understand characteristics of each segment
- **Business Meaning**: Translate clusters to actionable insights
- **Stability**: Ensure clusters are consistent over time

## Best Practices

### 1. Data Preprocessing
- Scale numerical features to similar ranges
- Handle missing values appropriately
- Remove or treat outliers
- Encode categorical variables

### 2. Model Selection
- Use Elbow method or silhouette analysis
- Consider business requirements for number of clusters
- Run multiple initializations
- Validate cluster stability

### 3. Feature Engineering
- Create meaningful derived features
- Consider domain knowledge
- Handle multicollinearity
- Validate feature importance

### 4. Validation and Monitoring
- Use holdout data for validation
- Monitor cluster stability over time
- Validate against business metrics
- Update models periodically

## Advantages
- ✅ Simple and interpretable
- ✅ Fast and scalable
- ✅ Guaranteed convergence
- ✅ Works well with spherical clusters
- ✅ Easy to implement and understand
- ✅ Good for initial data exploration

## Limitations
- ❌ Assumes spherical clusters
- ❌ Sensitive to initial centroids
- ❌ Requires specifying number of clusters
- ❌ May converge to local optima
- ❌ Assumes equal-sized clusters
- ❌ Sensitive to outliers

## Advanced Variants

### 1. K-Means++
Improved initialization to avoid poor local optima:
```
μ₁ = random point
μᵢ = point with probability proportional to minⱼ<ᵢ ||x - μⱼ||²
```

### 2. Mini-Batch K-Means
Process data in batches for large datasets:
```
Update centroids using mini-batch of data
```

### 3. Fuzzy K-Means
Soft clustering with membership probabilities:
```
uᵢⱼ = 1 / Σₖ₌₁ᵏ (||xⱼ - μᵢ|| / ||xⱼ - μₖ||)^(2/(m-1))
```

## Future Directions
- **Deep Learning**: Autoencoders for clustering
- **Online Learning**: Incremental clustering
- **Hierarchical**: Multi-level clustering
- **Ensemble**: Combine multiple clustering methods
- **Quantum**: Quantum K-Means algorithms


---

## 🗺️ ML Developer Roadmap

Ready to continue your ML journey? Check out our comprehensive [**ML Developer Roadmap**](../../ROADMAP.md) for the complete learning path from beginner to expert! 🚀
