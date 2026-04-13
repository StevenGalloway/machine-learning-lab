"""
Shared fixtures and utilities for the machine-learning-lab test suite.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file: tests/ → repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, rel_path: str) -> Any:
    """
    Load a Python source file as a module using importlib.
    rel_path is relative to REPO_ROOT.
    Returns the loaded module object.
    """
    full_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic DataFrames used across multiple test files
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_basketball_df() -> pd.DataFrame:
    """Minimal synthetic basketball points DataFrame."""
    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    players = ["Alice"] * 30 + ["Bob"] * 30
    return pd.DataFrame({
        "Player": players,
        "Date": dates.tolist(),
        "MP": [f"{int(m)}.{int(s):02d}" for m, s in zip(rng.integers(10, 40, n), rng.integers(0, 60, n))],
        "PTS": rng.integers(5, 40, n).astype(float),
        "FGA": rng.integers(5, 20, n).astype(float),
        "3PA": rng.integers(0, 10, n).astype(float),
        "FTA": rng.integers(0, 8, n).astype(float),
        "ORB": rng.integers(0, 5, n).astype(float),
        "DRB": rng.integers(0, 10, n).astype(float),
        "AST": rng.integers(0, 10, n).astype(float),
        "STL": rng.integers(0, 4, n).astype(float),
        "BLK": rng.integers(0, 3, n).astype(float),
        "TOV": rng.integers(0, 6, n).astype(float),
        "PF": rng.integers(0, 5, n).astype(float),
    })


@pytest.fixture()
def synthetic_football_df() -> pd.DataFrame:
    """Minimal synthetic football first-downs / points DataFrame."""
    rng = np.random.default_rng(1)
    n = 100
    return pd.DataFrame({
        "FirstDowns": rng.integers(10, 30, n).astype(float),
        "Points": rng.integers(0, 45, n).astype(float),
    })


@pytest.fixture()
def synthetic_sms_df() -> pd.DataFrame:
    """Tiny balanced SMS sender dataset with 4 senders."""
    rows = []
    senders = ["Alice", "Bob", "Charlie", "Dana"]
    times = ["morning", "afternoon", "evening", "night"]
    rng = np.random.default_rng(2)
    for _ in range(200):
        sender = senders[rng.integers(0, 4)]
        time = times[rng.integers(0, 4)]
        text = f"Hello this is a {sender.lower()} message at {time}"
        rows.append({"text": text, "time_of_day": time, "label": sender})
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_loan_default_df() -> pd.DataFrame:
    """Synthetic personal loan default DataFrame (numeric + categorical cols)."""
    rng = np.random.default_rng(3)
    n = 300
    return pd.DataFrame({
        "loan_amount": rng.uniform(5000, 50000, n),
        "term_months": rng.choice([12, 24, 36, 60], n),
        "interest_rate": rng.uniform(4.0, 22.0, n),
        "annual_income": rng.uniform(25000, 150000, n),
        "credit_score": rng.integers(550, 850, n),
        "dti": rng.uniform(0.05, 0.55, n),
        "utilization": rng.uniform(0.0, 1.0, n),
        "employment_years": rng.uniform(0, 20, n),
        "delinquencies": rng.integers(0, 5, n),
        "prior_defaults": rng.integers(0, 2, n),
        "payment_to_income": rng.uniform(0.02, 0.5, n),
        "purpose": rng.choice(["debt_consolidation", "home", "car"], n),
        "state": rng.choice(["CA", "TX", "FL", "NY"], n),
        "age": rng.integers(22, 70, n),
        "sex": rng.choice(["M", "F"], n),
        "default": rng.integers(0, 2, n),
    })


@pytest.fixture()
def synthetic_nfl_df() -> pd.DataFrame:
    """Minimal NFL game DataFrame with two seasons."""
    rng = np.random.default_rng(4)
    n = 64
    teams = ["Ravens", "Steelers", "Browns", "Bengals"]
    rows = []
    for season in [2023, 2024]:
        for week in range(1, (n // 2) // 2 + 1):
            for _ in range(2):
                sh = int(rng.integers(0, 45))
                sa = int(rng.integers(0, 45))
                rows.append({
                    "schedule_season": season,
                    "schedule_week": str(week),
                    "schedule_date": f"{season}-09-{(week % 28) + 1:02d}",
                    "team_home": rng.choice(teams),
                    "team_away": rng.choice(teams),
                    "score_home": sh,
                    "score_away": sa,
                })
    return pd.DataFrame(rows)
