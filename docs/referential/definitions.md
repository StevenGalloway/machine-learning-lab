# Core Definitions (ML Foundations)

## Supervised Learning

Models are trained on labeled data `(X, y)` where the target variable is
known.

**Examples** - **Regression** (predict continuous values): Linear
Regression, Random Forest, Neural Networks\
- **Classification** (predict categories): Logistic Regression, Decision
Trees, Gradient Boosting

Example repo: [Breast Cancer Identification (XGBoost)](case-studies\breast-cancer-xgb\scripts\train_eval.py)
Example repo: [Loan Approval (XGBoost)](case-studies\loan-approval-xgb\scripts\train_eval.py)
Example repo: [Football Points Prediction (Linear Regression)](case-studies\football-points-prediction-linear-reg\scripts\points-prediction-linear-reg.py)
Example repo: [Basketball Points Prediction (Linear Regression)](case-studies\basketball-points-prediction-linear-reg\scripts\points-prediction-linear-reg.py)

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

Example repo: (Insert Example Repo when completed)

**Pros** - Works when labels are unavailable - Useful for exploration
and anomaly detection

**Cons** - Harder to evaluate objectively - Results can be sensitive to
hyperparameters

------------------------------------------------------------------------

