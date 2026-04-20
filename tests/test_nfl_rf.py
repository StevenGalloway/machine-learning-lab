"""
Tests for case-studies/random-forest-models/nfl-game-prediction/
         scripts/nfl_game_prediction_random_forest.py
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from tests.conftest import load_module

_MOD = load_module(
    "nfl_rf",
    "case-studies/random-forest-models/nfl-game-prediction/"
    "scripts/nfl_game_prediction_random_forest.py",
)

_week_to_num = _MOD._week_to_num
load_and_prepare = _MOD.load_and_prepare
season_aware_split = _MOD.season_aware_split
predict_single_game = _MOD.predict_single_game
PredictionQuery = _MOD.PredictionQuery
_load_config = _MOD._load_config


# ---------------------------------------------------------------------------
# _week_to_num
# ---------------------------------------------------------------------------

class TestWeekToNum:
    @pytest.mark.parametrize("inp,expected", [
        ("1", 1),
        ("17", 17),
        ("0", 0),
    ])
    def test_numeric_strings(self, inp, expected):
        assert _week_to_num(inp) == expected

    @pytest.mark.parametrize("inp,expected", [
        ("wildcard", 19),
        ("Wildcard", 19),    # case insensitive
        ("division", 20),
        ("conference", 21),
        ("superbowl", 22),
        ("super bowl", 22),
    ])
    def test_keyword_strings(self, inp, expected):
        assert _week_to_num(inp) == expected

    def test_unknown_string_returns_default(self):
        # Default is 18
        assert _week_to_num("playoff") == 18

    def test_integer_input(self):
        assert _week_to_num(5) == 5


# ---------------------------------------------------------------------------
# load_and_prepare
# ---------------------------------------------------------------------------

class TestLoadAndPrepare:
    def test_raises_on_missing_required_columns(self, tmp_path, synthetic_nfl_df):
        bad = synthetic_nfl_df.drop(columns=["score_home"])
        csv = tmp_path / "bad.csv"
        bad.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_and_prepare(csv)

    def test_creates_home_win_column(self, tmp_path, synthetic_nfl_df):
        csv = tmp_path / "nfl.csv"
        synthetic_nfl_df.to_csv(csv, index=False)
        df = load_and_prepare(csv)
        assert "home_win" in df.columns

    def test_home_win_is_binary(self, tmp_path, synthetic_nfl_df):
        csv = tmp_path / "nfl.csv"
        synthetic_nfl_df.to_csv(csv, index=False)
        df = load_and_prepare(csv)
        assert set(df["home_win"].unique()).issubset({0, 1})

    def test_creates_week_num_column(self, tmp_path, synthetic_nfl_df):
        csv = tmp_path / "nfl.csv"
        synthetic_nfl_df.to_csv(csv, index=False)
        df = load_and_prepare(csv)
        assert "week_num" in df.columns

    def test_creates_month_column(self, tmp_path, synthetic_nfl_df):
        csv = tmp_path / "nfl.csv"
        synthetic_nfl_df.to_csv(csv, index=False)
        df = load_and_prepare(csv)
        assert "month" in df.columns

    def test_home_win_logic(self, tmp_path):
        df = pd.DataFrame({
            "schedule_season": [2024],
            "schedule_week": ["1"],
            "schedule_date": ["2024-09-01"],
            "team_home": ["Ravens"],
            "team_away": ["Steelers"],
            "score_home": [28],
            "score_away": [14],
        })
        csv = tmp_path / "one.csv"
        df.to_csv(csv, index=False)
        out = load_and_prepare(csv)
        assert out.iloc[0]["home_win"] == 1

    def test_home_loss_logic(self, tmp_path):
        df = pd.DataFrame({
            "schedule_season": [2024],
            "schedule_week": ["1"],
            "schedule_date": ["2024-09-01"],
            "team_home": ["Ravens"],
            "team_away": ["Steelers"],
            "score_home": [7],
            "score_away": [21],
        })
        csv = tmp_path / "loss.csv"
        df.to_csv(csv, index=False)
        out = load_and_prepare(csv)
        assert out.iloc[0]["home_win"] == 0


# ---------------------------------------------------------------------------
# season_aware_split
# ---------------------------------------------------------------------------

class TestSeasonAwareSplit:
    def test_multi_season_holds_out_last(self, tmp_path, synthetic_nfl_df):
        csv = tmp_path / "nfl.csv"
        synthetic_nfl_df.to_csv(csv, index=False)
        df = load_and_prepare(csv)
        last_season = df["schedule_season"].max()
        train, test = season_aware_split(df)
        # Train must not contain the last season
        assert (train["schedule_season"] < last_season).all()
        # Test must only contain the last season
        assert (test["schedule_season"] == last_season).all()

    def test_no_data_lost(self, tmp_path, synthetic_nfl_df):
        csv = tmp_path / "nfl.csv"
        synthetic_nfl_df.to_csv(csv, index=False)
        df = load_and_prepare(csv)
        train, test = season_aware_split(df)
        assert len(train) + len(test) == len(df)

    def test_single_season_falls_back_to_random_split(self, tmp_path):
        df = pd.DataFrame({
            "schedule_season": [2024] * 20,
            "schedule_week": ["1"] * 20,
            "schedule_date": ["2024-09-01"] * 20,
            "team_home": ["Ravens"] * 20,
            "team_away": ["Steelers"] * 20,
            "score_home": [28] * 20,
            "score_away": [14] * 20,
            "home_win": [1] * 20,
            "week_num": [1] * 20,
            "month": [9] * 20,
        })
        train, test = season_aware_split(df)
        assert len(train) + len(test) == 20


# ---------------------------------------------------------------------------
# predict_single_game
# ---------------------------------------------------------------------------

class TestPredictSingleGame:
    def _make_mock_model(self):
        m = MagicMock()
        m.predict_proba.return_value = np.array([[0.35, 0.65]])
        return m

    def _make_template_row(self):
        return pd.Series({
            "schedule_season": 2024,
            "week_num": 5,
            "month": 10,
            "team_home": "Ravens",
            "team_away": "Steelers",
        })

    def test_returns_p_home_win(self):
        model = self._make_mock_model()
        query = PredictionQuery(season=2025, week=3, home_team="Ravens", away_team="Steelers")
        result = predict_single_game(model, query, self._make_template_row())
        assert "p_home_win" in result

    def test_p_home_win_is_float(self):
        model = self._make_mock_model()
        query = PredictionQuery(season=2025, week=3, home_team="Ravens", away_team="Steelers")
        result = predict_single_game(model, query, self._make_template_row())
        assert isinstance(result["p_home_win"], float)

    def test_p_home_win_between_zero_and_one(self):
        model = self._make_mock_model()
        query = PredictionQuery(season=2025, week=3, home_team="Ravens", away_team="Steelers")
        result = predict_single_game(model, query, self._make_template_row())
        assert 0.0 <= result["p_home_win"] <= 1.0

    def test_predicted_label_is_zero_or_one(self):
        model = self._make_mock_model()
        query = PredictionQuery(season=2025, week=3, home_team="Ravens", away_team="Steelers")
        result = predict_single_game(model, query, self._make_template_row())
        assert result["predicted_label"] in (0, 1)

    def test_model_fields_updated_in_row(self):
        calls = []
        def mock_predict_proba(X):
            calls.append(X.iloc[0].to_dict())
            return np.array([[0.4, 0.6]])
        model = MagicMock()
        model.predict_proba = mock_predict_proba
        query = PredictionQuery(season=2026, week=7, home_team="KC", away_team="BUF")
        predict_single_game(model, query, self._make_template_row())
        assert calls[0]["schedule_season"] == 2026
        assert calls[0]["week_num"] == 7
        assert calls[0]["team_home"] == "KC"
        assert calls[0]["team_away"] == "BUF"


# ---------------------------------------------------------------------------
# PredictionQuery dataclass
# ---------------------------------------------------------------------------

class TestPredictionQuery:
    def test_fields_accessible(self):
        q = PredictionQuery(season=2025, week=3, home_team="X", away_team="Y")
        assert q.season == 2025
        assert q.week == 3
        assert q.home_team == "X"
        assert q.away_team == "Y"

    def test_is_frozen(self):
        q = PredictionQuery(season=2025, week=3, home_team="X", away_team="Y")
        with pytest.raises((AttributeError, TypeError)):
            q.season = 9999


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_dict(self):
        assert isinstance(_load_config(), dict)

    def test_has_default_prediction_key(self):
        cfg = _load_config()
        assert "default_prediction" in cfg
        dp = cfg["default_prediction"]
        assert "home_team" in dp
        assert "away_team" in dp
        assert "season" in dp
