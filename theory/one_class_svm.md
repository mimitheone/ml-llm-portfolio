# One-Class SVM Theory

## Overview
One-Class SVM is an unsupervised learning algorithm that learns a decision boundary around normal data points, classifying new points as either normal or anomalous, making it particularly valuable for fraud detection, outlier detection, and risk monitoring in banking applications.

## Mathematical Foundation

### One-Class SVM Formulation
The algorithm finds a hyperplane that separates normal data from the origin with maximum margin:

**Primal Problem**:
```
Minimize: (1/2) * ||w||² + (1/νn) * Σ(ξ_i) - ρ
```

Subject to:
```
w^T * φ(x_i) ≥ ρ - ξ_i, ξ_i ≥ 0
```

Where:
- `w` is the weight vector
- `φ(x)` is the feature mapping function
- `ξ_i` are slack variables
- `ρ` is the offset from origin
- `ν` controls the fraction of outliers (0 < ν ≤ 1)

### Dual Problem
```
Maximize: Σ(α_i * K(x_i, x_i)) - (1/2) * Σ(α_i * α_j * K(x_i, x_j))
```

Subject to:
```
0 ≤ α_i ≤ 1/(νn), Σ(α_i) = 1
```

### Decision Function
```
f(x) = sign(Σ(α_i * K(x_i, x)) - ρ)
```

## Banking Applications

### 1. Fraud Detection
- **Transaction Monitoring**: Detect anomalous transactions
- **Behavioral Analysis**: Identify unusual customer behavior
- **Account Takeover**: Detect unauthorized access patterns
- **Money Laundering**: Identify suspicious activity patterns

### 2. Risk Monitoring
- **Portfolio Risk**: Detect unusual risk patterns
- **Market Anomalies**: Identify market microstructure changes
- **Credit Risk**: Detect unusual credit behavior
- **Operational Risk**: Identify operational anomalies

### 3. Compliance Monitoring
- **Regulatory Violations**: Detect compliance breaches
- **Data Quality**: Identify data anomalies
- **System Monitoring**: Detect system failures
- **Audit Trail**: Identify unusual audit patterns

## Implementation Considerations

### Kernel Selection
- **RBF Kernel**: `K(x_i, x_j) = exp(-γ * ||x_i - x_j||²)`
  - Good for non-linear decision boundaries
  - γ controls the influence of each training point
- **Linear Kernel**: `K(x_i, x_j) = x_i^T * x_j`
  - Faster computation
  - Linear decision boundary
- **Polynomial Kernel**: `K(x_i, x_j) = (γ * x_i^T * x_j + r)^d`
  - Captures polynomial relationships
  - More parameters to tune

### Hyperparameter Tuning
- **ν (nu)**: Fraction of outliers (0 < ν ≤ 1)
  - Lower values: More sensitive to outliers
  - Higher values: Less sensitive to outliers
- **γ (gamma)**: Kernel coefficient for RBF/polynomial
  - Lower values: Broader influence of training points
  - Higher values: Narrower influence of training points

### Feature Engineering
- **Feature Scaling**: Essential for kernel performance
- **Feature Selection**: Choose relevant features for anomaly detection
- **Domain Knowledge**: Incorporate business expertise
- **Risk Factors**: Focus on regulatory risk factors

## Evaluation Metrics

### Anomaly Detection Metrics
- **Precision**: True anomalies / (True anomalies + False positives)
- **Recall**: True anomalies / (True anomalies + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

### Business Metrics
- **False Positive Rate**: Cost of false alarms
- **False Negative Rate**: Cost of missed anomalies
- **Business Impact**: Quantify anomaly detection value
- **Regulatory Compliance**: Meet compliance requirements

### Model Stability
- **Parameter Sensitivity**: Robustness to parameter changes
- **Feature Stability**: Consistency of feature importance
- **Performance Drift**: Monitor over time
- **Cross-Validation**: Robust performance estimates

## Best Practices

### Data Preparation
- **Normal Data**: Ensure training data contains only normal instances
- **Feature Scaling**: Standardize features for kernel performance
- **Outlier Removal**: Remove obvious outliers from training data
- **Feature Selection**: Choose relevant features for anomaly detection

### Model Selection
- **Kernel Choice**: Start with RBF kernel, then try others
- **Parameter Tuning**: Use grid search for hyperparameters
- **Business Validation**: Ensure anomalies make business sense
- **Regulatory Requirements**: Consider compliance needs

### Performance Monitoring
- **False Positive Rate**: Monitor false alarm frequency
- **False Negative Rate**: Monitor missed anomaly frequency
- **Business Impact**: Quantify anomaly detection value
- **Regulatory Compliance**: Ensure compliance requirements

## Regulatory Compliance

### Model Governance
- **Documentation**: Clear anomaly detection methodology
- **Validation**: Independent validation of detection results
- **Monitoring**: Ongoing performance monitoring
- **Audit Trail**: Track detection changes and decisions

### Risk Management
- **False Positive Risk**: Cost of false alarms
- **False Negative Risk**: Cost of missed anomalies
- **Business Impact**: Quantify detection business value
- **Regulatory Risk**: Compliance with regulatory requirements

### Explainability
- **Anomaly Characteristics**: Describe what makes anomalies anomalous
- **Feature Importance**: Identify key anomaly detection features
- **Decision Boundaries**: Visualize anomaly detection regions
- **Business Rules**: Translate anomalies to business logic

## Advanced Techniques

### Adaptive One-Class SVM
- **Online Learning**: Update model with new data
- **Concept Drift**: Handle changing data distributions
- **Dynamic Thresholds**: Adjust detection sensitivity
- **Business Constraints**: Incorporate domain knowledge

### Ensemble Methods
- **Multiple Kernels**: Combine results from different kernels
- **Multiple Parameters**: Aggregate results from different parameter settings
- **Bootstrap Anomaly Detection**: Robust detection with resampling
- **Stability Analysis**: Identify stable detection patterns

### Multi-Modal Anomaly Detection
- **Different Data Types**: Handle numerical, categorical, and text data
- **Feature Fusion**: Combine different feature representations
- **Domain Adaptation**: Adapt to different data domains
- **Transfer Learning**: Use knowledge from related domains

## Future Directions

### Machine Learning Integration
- **Deep Learning**: Neural network-based anomaly detection
- **AutoML**: Automated parameter optimization
- **Multi-task Learning**: Handle multiple anomaly detection tasks
- **Reinforcement Learning**: Dynamic parameter adjustment

### Real-Time Applications
- **Streaming Data**: Online anomaly detection
- **Real-Time Monitoring**: Live anomaly detection
- **Edge Computing**: Local anomaly detection
- **Dynamic Models**: Adaptive detection structures

### Interpretability
- **SHAP Integration**: Feature importance for anomaly detection
- **Counterfactual Analysis**: What-if scenarios for anomalies
- **Feature Interactions**: Understand anomaly relationships
- **Business Rules**: Generate interpretable anomaly descriptions
