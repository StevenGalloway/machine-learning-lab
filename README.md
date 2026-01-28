# Machine Learning Lab

This repository demonstrates practical machine learning across the full
lifecycle --- from problem framing and data design to evaluation,
deployment, monitoring, and iteration. The intent of this lab is to
serve both as a learning portfolio and a production-oriented reference
aligned with **FAANG-style expectations for Machine Learning Engineers**
(systems thinking, rigor, reproducibility, and impact).

------------------------------------------------------------------------

## What does this repository demonstrates

-   End-to-end ownership of ML systems (not just model building)
-   Strong fundamentals in math, statistics, and optimization
-   Clear thinking about data, leakage, and evaluation
-   Practical MLOps and production experience
-   Awareness of bias, fairness, and responsible AI
-   Ability to trade off performance, latency, cost, and maintainability

------------------------------------------------------------------------

## Focus Areas

-   Problem framing & success criteria
-   Feature engineering & data leakage prevention
-   Model selection & evaluation
-   Experiment reproducibility
-   Error analysis & iteration
-   Drift detection & retraining strategy
-   ML system design & MLOps
-   Responsible AI (bias, fairness, transparency)

------------------------------------------------------------------------

## Repository Structure

-   `case-studies/` -- End-to-end ML scenarios with business context
-   `src/` -- Reusable feature, training, and inference pipelines
-   `configs/` -- Experiment configurations (YAML)
-   `notebooks/` -- Exploratory Data Analysis(EDA), prototyping, and visualization
-   `tests/` -- Validation, unit tests, and data checks
-   `models/` -- Trained artifacts and model cards
-   `monitoring/` -- Drift detection and performance dashboards
-   `mlops/` -- CI/CD, feature stores, and deployment configs
-   `docs/` -- Design docs, referential docs, and architecture diagrams

------------------------------------------------------------------------

## Referential Documentation Index

Each topic below links directly to its detailed documentation in
`docs/`:

-   [Best Practices](docs/referential/best-practices.md)
-   [Bias & Fairness Tracking](docs/referential/bias-fairness-tracking.md)
-   [Clustered Models](docs/referential/clustered-models-examples.md)
-   [Data Leakage](docs/referential/data-leakage.md)
-   [Definitions](docs/referential/definitions.md)
-   [Feature Engineering](docs/referential/feature-engineering.md)
-   [Feature Stores](docs/referential/feature-stores.md)
-   [K-Means Model](docs/referential/kmeans-model.md)
-   [Mathematics for ML](docs/referential/mathematics-definitions.md)
-   [MLOps](docs/referential/mlops.md)
-   [Model Deployment Strategies](docs/referential/model-deployment-strategies.md)
-   [Model Evaluation Deep Dive](docs/referential/model-evaluation.md)
-   [Monte Carlo Simulations](docs/referential/monte-carlo-simulations.md)
-   [Naives Bayes Model](docs/referential/naives-bayes-model.md)
-   [Nueral Networks and Activation Functions](docs/referential/neural-networks-activation-functions.md)
-   [Non-Clustered Models](docs/referential/non-clustered-models-examples.md)
-   [Random Forest Models](docs/referential/random-forest-model.md)

------------------------------------------------------------------------

# Featured Highlights

-   [NFL Passing Yards Prediction Linear Regression (Jupyter)](notebooks/jupyter/nfl-passing-yards-prediction/README.md)
-   [Football Points Prediction Linear Regression Model (Python)](case-studies/football-points-prediction-linear-reg/README.md)
-   [Basketball Points Prediction Linear Regression Model (Python)](case-studies/basketball-points-prediction-linear-reg/README.md)
-   [Ravens & Steelers Game Prediction (NFL) Random Forest Model (Python)](case-studies/nfl-game-prediction/README.md)
-   [Loan Default Prediction Random Forest Model (Python)](case-studies/loan-default-prediction-rf/README.md)
-   [NBA Points Prediction Random Forest (Jupyter)](notebooks/jupyter/nba-points-prediction/README.md)
-   [Breast Cancer XGBoost Model (Python)](case-studies/breast-cancer-xgb/README.md)
-   [Loan Approval XGBoost Model (Python)](case-studies/loan-approval-xgb/README.md)
-   [Text Send Identification Naives Bayes (Python)](case-studies/text-sender-identification-nb/README.md)

------------------------------------------------------------------------

# Featured Model Results

-   [Ravens & Steelers Game Prediction (NFL) - Results](case-studies/nfl-game-prediction/results/baseline_results.md)
-   [Loan Default Prediction - Results)](case-studies/loan-default-prediction-rf/README.md)
-   [Breast Cancer XGBoost - Results](case-studies/breast-cancer-xgb/results/baseline_results.md)
-   [Loan Approval XGBoost - Results](case-studies/loan-approval-xgb/results/baseline_results.md)
-   [Football Points Prediction (NFL) - Results](case-studies/football-points-prediction-linear-reg/results/baseline_results_nfl.md)
-   [Basketball Points Prediction - Results](case-studies/basketball-points-prediction-linear-reg/results/baseline_results.md)
-   [Text Send Identification NB - Results](case-studies/text-sender-identification-nb/results/baseline_results.md)

------------------------------------------------------------------------

See each document for detailed explanations, diagrams, and examples.

# Tools & Stack Featured

This repository demonstrates proficiency across the full ML lifecycle
--- from data to modeling to production --- using a modern,
industry-aligned toolchain. The tools below are organized by capability
rather than category to emphasize **systems thinking, tradeoffs, and
end-to-end ownership**. This is not a complete list of skills, only a 
highlight of skills utilized or intended to be utilized in this 
repository.

------------------------------------------------------------------------

## Core Programming & ML Foundations

-   **Python** --- primary language for data, modeling, and pipelines\
    *Example:* [Ravens & Steelers Game Prediction (NFL) Random Forest Model](case-studies/nfl-game-prediction/scripts/nfl_game_prediction_random_forest.py)
    *Example:* [Breast Cancer Identification](case-studies/breast-cancer-xgb/scripts/train_eval.py)
-   **NumPy** --- numerical computing, linear algebra, and optimization\
    *Example:* [Loan Approval](case-studies/breast-cancer-xgb/scripts/train_eval.py)
    *Example:* [NFL Passing Yards](notebooks/jupyter/nfl-passing-yards-prediction/nfl-passing-yards-prediction_linear_reg.ipynb)
-   **Pandas** --- data manipulation, feature engineering, and analysis\
    *Example:* [Football Points Prediction](case-studies/football-points-prediction-linear-reg/scripts/points-prediction-linear-reg.py)\
    *Example:* [Text Sender Prediction](case-studies/text-sender-identification-nb/scripts/text_sender_identification_nb.py)

------------------------------------------------------------------------

## Modeling & Algorithms

-   **scikit-learn** --- baseline models, pipelines, evaluation, and CV\
    *Example:* [Breast Cancer Identification](case-studies/breast-cancer-xgb/scripts/train_eval.py)
    *Example:* [Ravens & Steelers Game Prediction (NFL) Random Forest Model](case-studies/nfl-game-prediction/scripts/nfl_game_prediction_random_forest.py)
-   **XGBoost / LightGBM** --- high-performance tabular modeling\
    *Example:* [Loan Approval](case-studies/breast-cancer-xgb/scripts/train_eval.py)
-   *Example:* [Breast Cancer XGBoost Model (Python)](case-studies/breast-cancer-xgb/README.md)
-   **PyTorch** --- representation learning and deep models (when
    appropriate)\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Feature Engineering & Data Quality

-   **scikit-learn Pipelines** --- reproducible preprocessing +
    training\
    *Example:* [Text Sender Prediction](case-studies/text-sender-identification-nb/scripts/text_sender_identification_nb.py)
    *Example:* [NBA Points Scoring](notebooks/jupyter/nba-points-prediction/nba-points-prediction_rf.ipynb)
-   **Feature Selection (Filter/Wrapper/Embedded)** --- dimensionality
    control\
    *Example:* (Insert Example Repo when completed)
-   **Missing Value Imputation** --- robust handling of incomplete data\
    *Example:* (Insert Example Repo when completed)
-   **Data Validation (Great Expectations)** --- data quality gates\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Experimentation & Model Evaluation

-   **MLflow (or W&B)** --- experiment tracking and artifact logging\
    *Example:* (Insert Example Repo when completed)
-   **scikit-learn Metrics** --- precision/recall, ROC-AUC, calibration\
    *Example:* [NBA Points Scoring](notebooks/jupyter/nba-points-prediction/nba-points-prediction_rf.ipynb)
    *Example:* [Text Sender Identification](case-studies/text-sender-identification-nb/scripts/text_sender_identification_nb_LIVE.py)
-   **Slice-Based Evaluation** --- performance by segment (fairness +
    reliability)\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## MLOps & Production Systems

-   **Docker** --- containerized training and inference\
    *Example:* (Insert Example Repo when completed)
-   **FastAPI** --- real-time model serving APIs\
    *Example:* (Insert Example Repo when completed)
-   **GitHub Actions (CI/CD for ML)** --- automated tests and
    deployments\
    *Example:* (Insert Example Repo when completed)
-   **Model Registry (MLflow)** --- versioning, approvals, and rollback\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Feature Stores & Data Platforms

-   **Databricks Feature Store (conceptual + examples)**\
    *Example:* (Insert Example Repo when completed)
-   **Delta Lake / Iceberg** --- reliable, versioned data lakes\
    *Example:* (Insert Example Repo when completed)
-   **Feast (open-source feature store)** --- portable feature
    management\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Deployment & Reliability

-   **Batch / Streaming / Real-Time Inference Patterns**\
    *Example:* (Insert Example Repo when completed)
-   **Canary Releases & Shadow Testing** --- risk-controlled
    deployments\
    *Example:* (Insert Example Repo when completed)
-   **Monitoring (Drift, Latency, Errors, Fairness)**\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Responsible AI (Bias & Fairness)

-   **Fairness Metrics (Demographic Parity, Equal Opportunity, Equalized
    Odds)**\
    *Example:* (Insert Example Repo when completed)
-   **Bias-Aware Training & Calibration**\
    *Example:* (Insert Example Repo when completed)
-   **Model Cards & Documentation**\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Mathematics & Optimization

-   **NumPy Linear Algebra** --- vectors, matrices, norms, distances\
    *Example:* (Insert Example Repo when completed)
-   **Gradient Descent (from scratch)** --- optimization intuition\
    *Example:* (Insert Example Repo when completed)
-   **Probability & Statistics in ML** --- Bayes, distributions,
    uncertainty\
    *Example:* (Insert Example Repo when completed)
