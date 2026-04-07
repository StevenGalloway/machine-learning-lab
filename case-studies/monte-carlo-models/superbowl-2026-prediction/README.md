# Super Bowl 2026 Winner Prediction — HistGradientBoosting Classifier (Enterprise Case Study)

This case study predicts a **neutral-site winner probability** for a Super Bowl matchup using a **HistGradientBoostingClassifier** with **isotonic calibration**, team-level rolling form metrics, and Elo-style strength features derived from historical NFL play-by-play data.

## Important modeling note

This project **does not use Geometric Brownian Motion (GBM)**. Geometric Brownian Motion is a stochastic process commonly used in quantitative finance to model continuously compounded asset-price paths. It is appropriate for stock-path simulation and option-pricing style workflows, but it is **not the right model family for NFL game-outcome classification**.

This case study instead uses **gradient boosting** for tabular sports prediction. If you see “GBM” in sports-analytics conversations, it often refers to **Gradient Boosting Machines**, not Geometric Brownian Motion.

## Dependencies

Install the required Python packages before running the model:

```bash
pip install pandas numpy polars scikit-learn nflreadpy pyarrow
```

## Quick Start

Run the default Super Bowl-style neutral-site matchup prediction:

```bash
python scripts/superbowl_2026_mc.py --refresh-cache
```

This command will:

1. Pull historical NFL play-by-play data from `nflreadpy`
2. Cache the raw play-by-play snapshot in `/data`
3. Engineer rolling offensive, defensive, and Elo-based features
4. Train a calibrated gradient-boosting classifier
5. Predict a neutral-site winner probability for the configured matchup
6. Save only runtime artifacts to `/results`

## Recommended command

```bash
python scripts/superbowl_2026_mc.py --team-a KC --team-b PHI --refresh-cache
```

## One-line execution example

```bash
python scripts/superbowl_2026_mc.py --team-a BUF --team-b SF --target-season 2025 --roll-window 8 --cache-ttl-hours 24 --refresh-cache
```

## Outputs

Generated runtime artifacts include:

- `results/metrics.json` — train/validation/test metrics and split information
- `results/prediction.json` — matchup probability output for the requested teams
- `results/feature_columns.json` — model feature list used in training and inference
- `results/summary.md` — recruiter-friendly execution summary

## Project Structure

```text
superbowl-2026-winner-prediction-hgb/
│
├── data/                         # Cached NFL play-by-play extracts
├── results/                      # Runtime prediction artifacts
├── scripts/
│   └── superbowl_2026_mc.py
│
├── supporting-documentation/
│   ├── data_description.md
│   ├── deployment_plan.md
│   ├── eda_summary.md
│   ├── error_analysis.md
│   ├── experiment_log.md
│   ├── feature_dictionary.md
│   ├── model_card.md
│   ├── monitoring_plan.md
│   ├── problem_statement.md
│   ├── risk_analysis.md
│   └── stakeholders.md
│
└── README.md
```

## CLI Reference

```
python scripts/superbowl_2026_mc.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--team-a` | `str` | `NE` | First team abbreviation (e.g. `KC`, `PHI`, `BUF`) |
| `--team-b` | `str` | `SEA` | Second team abbreviation (e.g. `SF`, `DAL`, `BAL`) |
| `--target-season` | `int` | latest available | Season year used for feature snapshots (e.g. `2025`) |
| `--roll-window` | `int` | `8` | Rolling window (games) for team form features |
| `--cache-ttl-hours` | `int` | `24` | Hours before cached play-by-play data is considered stale |
| `--refresh-cache` | flag | `False` | Force re-download of play-by-play data |
| `--prediction-week` | `int` | `20` | Synthetic week value used for the Super Bowl matchup row |
| `--random-state` | `int` | `42` | Random seed for the gradient boosting classifier |

## Why this model works well here

- **Gradient boosting** handles non-linear interactions among efficiency, explosiveness, turnover, and Elo features.
- **Isotonic calibration** improves probability quality, which matters because the deliverable is a win probability, not just a class label.
- **Rolling and season-to-date features** help approximate recent form while reducing leakage.
- **Neutral-site probability logic** averages both home/away orientations to reduce generic home-field bias.

## Enterprise-ready design choices

- Cached data is stored in `/data` and reused until the 24-hour TTL expires.
- Only runtime outputs are written during execution; documentation is static and lives separately.
- Configuration values are centralized in a dataclass and exposed through CLI arguments.
- Model metrics and prediction artifacts are saved as structured JSON for downstream review or automation.

## Recruiter-facing takeaway

This project demonstrates practical sports-analytics modeling with:

- feature engineering from raw event data
- probability calibration
- train/validation/test time-based splitting
- reproducible configuration patterns
- separation of runtime artifacts from static governance documentation
