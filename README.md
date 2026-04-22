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

### Repo Topology Diagrams

-   [Full Repository Overview](docs/diagrams/repo_overview.mmd) — all model families, case studies, notebooks, and shared infrastructure
-   [ML Lifecycle Diagram](docs/diagrams/ml_lifecycle.mmd) — data → config → training → artifacts → serving → governance

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

## Running the Test Suite

The repository includes a pytest-based unit test suite covering all Python
scripts and Jupyter notebooks.

### Prerequisites

Install the core dependencies plus the test runner:

```bash
pip install -r requirements.txt
pip install pytest nbformat
```

For API endpoint tests, also install:

```bash
pip install httpx
```

For CNN tests (optional — skipped automatically if not present):

```bash
pip install tensorflow gradio
```

### Run all tests

```bash
pytest
```

### Common options

| Command | Description |
|---|---|
| `pytest` | Run the full suite |
| `pytest -v` | Verbose output (test name per line) |
| `pytest tests/test_weather_markov.py` | Run a single test file |
| `pytest tests/test_weather_markov.py::TestSimulateMarkovChain` | Run a single test class |
| `pytest -k "smape"` | Run tests whose name contains a keyword |
| `pytest --ignore=tests/test_handwriting_cnn.py` | Skip a file (e.g. if TensorFlow is not installed) |
| `pytest -q` | Quiet mode — summary only |

### Test file map

| Test file | Script(s) covered |
|---|---|
| `tests/test_basketball_lr.py` | `basketball-points-prediction/scripts/points_prediction_linear_reg.py` |
| `tests/test_football_lr.py` | `football-points-prediction/scripts/points_prediction_linear_reg.py` |
| `tests/test_text_sender_nb.py` | `text-sender-identification/scripts/text_sender_identification_nb.py` |
| `tests/test_text_sender_live.py` | `text-sender-identification/scripts/text_sender_identification_nb_LIVE.py` |
| `tests/test_loan_default_rf.py` | `loan-default-prediction/scripts/loan_default_random_forest.py` |
| `tests/test_nfl_rf.py` | `nfl-game-prediction/scripts/nfl_game_prediction_random_forest.py` |
| `tests/test_breast_cancer_xgb.py` | `breast-cancer/scripts/train_eval.py` |
| `tests/test_loan_approval_xgb.py` | `loan-approval/scripts/loan_approval_model.py` |
| `tests/test_crypto_mc.py` | `crypto-market-prediction/scripts/crypto_market_prediction_mc.py` |
| `tests/test_stock_mc.py` | `stock-market-prediction/scripts/stock_market_monte_carlo.py` |
| `tests/test_handwriting_cnn.py` | `handwriting-recognition/scripts/handwriting_recognition_cnn.py` |
| `tests/test_weather_markov.py` | `weather-prediction/weather_markov.py` |
| `tests/test_loan_api.py` | `loan-default-prediction/api/api.py` |
| `tests/test_notebooks.py` | All three Jupyter notebooks (structural integrity) |

### Notes

- Tests use synthetic in-memory data and do not require any external data
  downloads or trained model artifacts on disk.
- The CNN test file (`test_handwriting_cnn.py`) is skipped automatically
  when TensorFlow or Gradio are not installed.
- The API tests mock `joblib.load` at import time, so no trained model
  file needs to exist before running them.
- Notebook tests verify structure and content (imports, keywords, cell
  counts) without executing the notebooks.

------------------------------------------------------------------------

# Featured Highlights

-   [Crypto Prediction Monte Carlo (Python)](case-studies/monte-carlo-models/crypto-market-prediction/README.md)
-   [Handwriting Prediction Nueral Network and App (Python)](case-studies/convolutional-models/handwriting-recognition/README.md)
-   [Superbowl Prediction 2026 Monte Carlo (Python)](case-studies\monte-carlo-models\superbowl-2026-prediction\README.md)
-   [NFL Passing Yards Prediction Linear Regression (Jupyter)](notebooks/jupyter/nfl-passing-yards-prediction/README.md)
-   [Loan Default Prediction Random Forest Model (Python)](case-studies/random-forest-models/loan-default-prediction/README.md)
-   [NBA Points Prediction Random Forest (Jupyter)](notebooks/jupyter/nba-points-prediction/README.md)
-   [Breast Cancer XGBoost Model (Python)](case-studies/xgb-models/breast-cancer/README.md)
-   [Text Send Identification Naives Bayes (Python)](case-studies/naives-bayes-models/text-sender-identification/README.md)

------------------------------------------------------------------------

# Featured Model Results

-   [Handwriting Prediction Algorithm and App](case-studies/convolutional-models/handwriting-recognition/results/metrics.json)
-   [Superbowl 2026 - Results](case-studies/monte-carlo-models/superbowl-2026-prediction/results/prediction.json)
-   [Ravens & Steelers Game Prediction (NFL) - Results](case-studies/random-forest-models/nfl-game-prediction/results/baseline_results.md)
-   [Loan Default Prediction - Results)](case-studies/random-forest-models/loan-default-prediction/README.md)
-   [Breast Cancer XGBoost - Results](case-studies/xgb-models/breast-cancer/results/baseline_results.md)
-   [Football Points Prediction (NFL) - Results](case-studies/linear-regression-models/football-points-prediction/results/baseline_results_nfl.md)
-   [Basketball Points Prediction - Results](case-studies/linear-regression-models/basketball-points-prediction/results/baseline_results.md)
-   [Text Send Identification NB - Results](case-studies/naives-bayes-models/text-sender-identification/results/baseline_results.md)

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
    *Example:* [Ravens & Steelers Game Prediction (NFL) Random Forest Model](case-studies/random-forest-models/nfl-game-prediction/scripts/nfl_game_prediction_random_forest.py)\
    *Example:* [Breast Cancer Identification](case-studies/xgb-models/breast-cancer/scripts/train_eval.py)
-   **NumPy** --- numerical computing, linear algebra, and optimization\
    *Example:* [Loan Approval](case-studies/xgb-models/breast-cancer/scripts/train_eval.py)\
    *Example:* [NFL Passing Yards](notebooks/jupyter/nfl-passing-yards-prediction/nfl-passing-yards-prediction_linear_reg.ipynb)
-   **Pandas** --- data manipulation, feature engineering, and analysis\
    *Example:* [Football Points Prediction](case-studies/linear-regression-models/football-points-prediction/scripts/points_prediction_linear_reg.py)\
    *Example:* [Text Sender Prediction](case-studies/naives-bayes-models/text-sender-identification/scripts/text_sender_identification_nb.py)

------------------------------------------------------------------------

## Modeling & Algorithms

-   **scikit-learn** --- baseline models, pipelines, evaluation, and CV\
    *Example:* [Breast Cancer Identification](case-studies/xgb-models/breast-cancer/scripts/train_eval.py)\
    *Example:* [Ravens & Steelers Game Prediction (NFL) Random Forest Model](case-studies/random-forest-models/nfl-game-prediction/scripts/nfl_game_prediction_random_forest.py)
-   **XGBoost / LightGBM** --- high-performance tabular modeling\
    *Example:* [Loan Approval](case-studies/xgb-models/loan-approval/scripts/loan_approval_model.py)\
    *Example:* [Breast Cancer XGBoost Model](case-studies/xgb-models/breast-cancer/scripts/train_eval.py)
-   **PyTorch** --- representation learning and deep models (when
    appropriate)\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Feature Engineering & Data Quality

-   **scikit-learn Pipelines** --- reproducible preprocessing +
    training\
    *Example:* [Text Sender Prediction](case-studies/naives-bayes-models/text-sender-identification/scripts/text_sender_identification_nb.py)\
    *Example:* [NBA Points Scoring](notebooks/jupyter/nba-points-prediction/nba-points-prediction_rf.ipynb)
-   **Feature Selection (Filter/Wrapper/Embedded)** --- dimensionality
    control\
    *Example:* [Loan Default Prediction — RF feature importance (embedded)](case-studies/random-forest-models/loan-default-prediction/scripts/loan_default_random_forest.py)
-   **Missing Value Imputation** --- robust handling of incomplete data\
    *Example:* [Loan Default Prediction — SimpleImputer median + most_frequent](case-studies/random-forest-models/loan-default-prediction/scripts/loan_default_random_forest.py)
-   **Data Validation (Great Expectations)** --- data quality gates\
    *Example:* (Insert Example Repo when completed)

-----------------------------------------------------------------------

## Experimentation & Model Evaluation

-   **MLflow (or W&B)** --- experiment tracking and artifact logging\
    *Example:* (Insert Example Repo when completed)
-   **scikit-learn Metrics** --- precision/recall, ROC-AUC, calibration\
    *Example:* [NBA Points Scoring](notebooks/jupyter/nba-points-prediction/nba-points-prediction_rf.ipynb)\
    *Example:* [Text Sender Identification](case-studies/naives-bayes-models/text-sender-identification/scripts/text_sender_identification_nb_LIVE.py)
-   **Slice-Based Evaluation** --- performance by segment (fairness +
    reliability)\
    *Example:* [Loan Default Prediction — AUC evaluated by sex segment](case-studies/random-forest-models/loan-default-prediction/scripts/loan_default_random_forest.py)

------------------------------------------------------------------------

## MLOps & Production Systems

-   **Docker** --- containerized training and inference\
    *Example:* (Insert Example Repo when completed)
-   **FastAPI** --- real-time model serving APIs\
    *Example:* [Loan Default Scoring API](case-studies/random-forest-models/loan-default-prediction/api/api.py)
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
    *Example:* [Text Sender ID — batch training script](case-studies/naives-bayes-models/text-sender-identification/scripts/text_sender_identification_nb.py)\
    *Example:* [Text Sender ID — live interactive inference](case-studies/naives-bayes-models/text-sender-identification/scripts/text_sender_identification_nb_LIVE.py)
-   **Canary Releases & Shadow Testing** --- risk-controlled
    deployments\
    *Example:* (Insert Example Repo when completed)
-   **Monitoring (Drift, Latency, Errors, Fairness)**\
    *Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Responsible AI (Bias & Fairness)

-   **Fairness Metrics (Demographic Parity, Equal Opportunity, Equalized
    Odds)**\
    *Example:* [Loan Default Prediction — ROC AUC sliced by sex](case-studies/random-forest-models/loan-default-prediction/scripts/loan_default_random_forest.py)
-   **Bias-Aware Training & Calibration**\
    *Example:* [Loan Default Prediction — balanced_subsample class weight + isotonic calibration](case-studies/random-forest-models/loan-default-prediction/scripts/loan_default_random_forest.py)
-   **Model Cards & Documentation**\
    *Example:* [Loan Default Prediction — model card](case-studies/random-forest-models/loan-default-prediction/supporting-documentation/model_card.md)\
    *Example:* [Handwriting Recognition CNN — model card](case-studies/convolutional-models/handwriting-recognition/supporting-documentation/model_card.md)

------------------------------------------------------------------------

## Mathematics & Optimization

-   **NumPy Linear Algebra** --- vectors, matrices, norms, distances\
    *Example:* [Crypto Monte Carlo — Cholesky decomposition for correlated asset paths](case-studies/monte-carlo-models/crypto-market-prediction/scripts/crypto_market_prediction_mc.py)
-   **Gradient Descent (from scratch)** --- optimization intuition\
    *Example:* (Insert Example Repo when completed)
-   **Probability & Statistics in ML** --- Bayes, distributions,
    uncertainty\
    *Example:* [Monte Carlo Stock Market Simulation — GBM stochastic paths](case-studies/monte-carlo-models/stock-market-prediction/scripts/stock_market_monte_carlo.py)\
    *Example:* [Weather Prediction — Markov Chain stationary distribution](notebooks/jupyter/weather-prediction/weather_markov.py)
