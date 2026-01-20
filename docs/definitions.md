# Core Definitions (ML Foundations)

## Supervised Learning

Models are trained on labeled data `(X, y)` where the target variable is
known.

**Examples** - **Regression** (predict continuous values): Linear
Regression, Random Forest, Neural Networks\
- **Classification** (predict categories): Logistic Regression, Decision
Trees, Gradient Boosting

Example repo: https://github.com/example/supervised-ml-lab

**Pros** - High predictive accuracy when labels are reliable - Easier
evaluation (clear ground truth)

**Cons** - Requires labeled data (expensive) - Vulnerable to label noise
and bias

------------------------------------------------------------------------

## Unsupervised Learning

Models learn structure from unlabeled data by discovering patterns in
`X` alone.

**Examples** - Clustering (K-Means, DBSCAN, GMM) - Dimensionality
reduction (PCA, t-SNE, UMAP) - Density estimation (Gaussian Mixture
Models)

Example repo: https://github.com/example/unsupervised-ml-lab

**Pros** - Works when labels are unavailable - Useful for exploration
and anomaly detection

**Cons** - Harder to evaluate objectively - Results can be sensitive to
hyperparameters

------------------------------------------------------------------------

