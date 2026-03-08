from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nflreadpy as nfl
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.frozen import FrozenEstimator  # add this import


# =============================================================================
# Configuration
# =============================================================================
BASE_METRICS: List[str] = [
    "off_epa", "off_success_rate", "off_explosive_rate", "off_turnover_rate",
    "off_sack_rate_allowed", "off_pass_rate", "off_pass_epa", "off_rush_epa",
    "def_epa_allowed", "def_success_allowed", "def_explosive_allowed",
    "def_takeaway_rate", "def_sack_rate", "net_epa", "point_diff",
]

TEAM_ALIASES: Dict[str, List[str]] = {
    "ARI": ["ARI", "ARZ"],
    "ATL": ["ATL"],
    "BAL": ["BAL"],
    "BUF": ["BUF"],
    "CAR": ["CAR"],
    "CHI": ["CHI"],
    "CIN": ["CIN"],
    "CLE": ["CLE"],
    "DAL": ["DAL"],
    "DEN": ["DEN"],
    "DET": ["DET"],
    "GB": ["GB", "GNB"],
    "HOU": ["HOU"],
    "IND": ["IND"],
    "JAX": ["JAX", "JAC"],
    "KC": ["KC"],
    "LV": ["LV", "LVR", "OAK"],
    "LAC": ["LAC", "SD", "SDG"],
    "LAR": ["LAR", "STL"],
    "MIA": ["MIA"],
    "MIN": ["MIN"],
    "NE": ["NE", "NWE", "NEW"],
    "NO": ["NO", "NOR"],
    "NYG": ["NYG"],
    "NYJ": ["NYJ"],
    "PHI": ["PHI"],
    "PIT": ["PIT"],
    "SEA": ["SEA"],
    "SF": ["SF", "SFO"],
    "TB": ["TB", "TAM"],
    "TEN": ["TEN", "OTI"],
    "WAS": ["WAS", "WSH"],
}


@dataclass(frozen=True)
class EloParams:
    k_base: float = 20.0
    home_field: float = 55.0
    offseason_regress: float = 0.33


@dataclass(frozen=True)
class ModelConfig:
    data_dir: str = "data"
    results_dir: str = "results"
    cache_ttl_hours: int = 24
    roll_window: int = 8
    random_state: int = 42
    target_season: Optional[int] = None
    team_a: str = "NE"
    team_b: str = "SEA"
    neutral_site: bool = True
    season_filter_start: int = 1999
    prediction_week: int = 20
    pbp_cache_file: str = "nfl_pbp_all_seasons.parquet"
    games_cache_file: str = "games_table.parquet"
    team_games_cache_file: str = "team_games.parquet"
    metrics_output_file: str = "metrics.json"
    prediction_output_file: str = "prediction.json"
    feature_columns_output_file: str = "feature_columns.json"
    summary_output_file: str = "summary.md"


# =============================================================================
# Utility helpers
# =============================================================================
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def cache_is_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return utc_now() - modified <= timedelta(hours=ttl_hours)


import json
import numpy as np
import pandas as pd
from pathlib import Path


def make_json_safe(obj):
    """
    Recursively convert numpy/pandas objects into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, payload: dict) -> None:
    """
    Save a dictionary as formatted JSON after converting unsupported scalar types.
    """
    safe_payload = make_json_safe(payload)
    path.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# =============================================================================
# Data loading and caching
# =============================================================================
def load_pbp_all_seasons(config: ModelConfig, refresh_cache: bool = False) -> pl.DataFrame:
    """
    Load NFL play-by-play data and cache it locally.

    This function uses nflreadpy as the upstream source of truth, then writes a parquet
    snapshot to the local /data directory. The cache is reused until it expires.
    """
    project_root = resolve_project_root()
    data_dir = project_root / config.data_dir
    ensure_directory(data_dir)
    cache_path = data_dir / config.pbp_cache_file

    if not refresh_cache and cache_is_fresh(cache_path, config.cache_ttl_hours):
        return pl.read_parquet(cache_path)


    pbp = nfl.load_pbp(seasons=True)

    keep = [
        "game_id", "season", "week", "season_type", "game_date",
        "home_team", "away_team", "posteam", "defteam", "play_type",
        "epa", "yards_gained", "pass", "rush", "sack", "success",
        "interception", "fumble_lost", "home_score", "away_score",
        "total_home_score", "total_away_score",
    ]
    cols = set(pbp.columns)
    keep = [column for column in keep if column in cols]
    pbp = pbp.select(keep)

    if "game_date" in pbp.columns:
        pbp = pbp.with_columns(pl.col("game_date").cast(pl.Utf8))
        pbp = pbp.with_columns(pl.col("game_date").str.strptime(pl.Date, strict=False))

    # Schema fallbacks for environments where a few convenience columns do not exist.
    if "pass" not in pbp.columns and "play_type" in pbp.columns:
        pbp = pbp.with_columns((pl.col("play_type") == "pass").cast(pl.Int8).alias("pass"))
    if "rush" not in pbp.columns and "play_type" in pbp.columns:
        pbp = pbp.with_columns((pl.col("play_type") == "run").cast(pl.Int8).alias("rush"))
    if "sack" not in pbp.columns:
        pbp = pbp.with_columns(pl.lit(0).cast(pl.Int8).alias("sack"))
    if "success" not in pbp.columns:
        pbp = pbp.with_columns((pl.col("epa") > 0).cast(pl.Int8).alias("success"))
    for column in ["interception", "fumble_lost"]:
        if column not in pbp.columns:
            pbp = pbp.with_columns(pl.lit(0).cast(pl.Int8).alias(column))

    if "season" in pbp.columns:
        pbp = pbp.filter(pl.col("season") >= config.season_filter_start)

    pbp.write_parquet(cache_path)
    return pbp


# =============================================================================
# Feature engineering
# =============================================================================
def games_from_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    cols = set(pbp.columns)
    if "total_home_score" in cols and "total_away_score" in cols:
        home_score_col, away_score_col = "total_home_score", "total_away_score"
    elif "home_score" in cols and "away_score" in cols:
        home_score_col, away_score_col = "home_score", "away_score"
    else:
        raise ValueError("Score columns were not found in play-by-play data.")

    meta_cols = [
        c for c in ["game_id", "season", "week", "season_type", "game_date", "home_team", "away_team"]
        if c in cols
    ]
    game_meta = pbp.select(meta_cols).unique(subset=["game_id"])

    finals = pbp.group_by("game_id").agg([
        pl.col(home_score_col).max().alias("home_score_final"),
        pl.col(away_score_col).max().alias("away_score_final"),
    ])

    games = game_meta.join(finals, on="game_id", how="left")
    if "season_type" in games.columns:
        games = games.filter(pl.col("season_type").is_in(["REG", "POST"]))
    return games


def team_games_from_pbp(pbp: pl.DataFrame, games: pl.DataFrame) -> pd.DataFrame:
    plays = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("epa").is_not_null()
        & ((pl.col("pass") == 1) | (pl.col("rush") == 1) | (pl.col("sack") == 1))
    )

    explosive = (
        ((pl.col("pass") == 1) & (pl.col("yards_gained") >= 20))
        | ((pl.col("rush") == 1) & (pl.col("yards_gained") >= 10))
    ).cast(pl.Int8)

    turnover = ((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1)).cast(pl.Int8)
    sack = (pl.col("sack") == 1).cast(pl.Int8)

    off = plays.group_by(["game_id", "posteam"]).agg([
        pl.len().alias("off_nplays"),
        pl.col("epa").mean().alias("off_epa"),
        pl.col("success").mean().alias("off_success_rate"),
        explosive.mean().alias("off_explosive_rate"),
        turnover.mean().alias("off_turnover_rate"),
        sack.mean().alias("off_sack_rate_allowed"),
        pl.col("pass").mean().alias("off_pass_rate"),
        pl.when(pl.col("pass") == 1).then(pl.col("epa")).otherwise(None).mean().alias("off_pass_epa"),
        pl.when(pl.col("rush") == 1).then(pl.col("epa")).otherwise(None).mean().alias("off_rush_epa"),
    ]).rename({"posteam": "team"})

    defense = plays.group_by(["game_id", "defteam"]).agg([
        pl.len().alias("def_nplays"),
        pl.col("epa").mean().alias("def_epa_allowed"),
        pl.col("success").mean().alias("def_success_allowed"),
        explosive.mean().alias("def_explosive_allowed"),
        turnover.mean().alias("def_takeaway_rate"),
        sack.mean().alias("def_sack_rate"),
    ]).rename({"defteam": "team"})

    team_games = off.join(defense, on=["game_id", "team"], how="full")
    team_games = team_games.join(games, on="game_id", how="left")

    team_games = team_games.with_columns([
        pl.when(pl.col("team") == pl.col("home_team"))
        .then(pl.col("home_score_final"))
        .otherwise(pl.col("away_score_final"))
        .alias("points_for"),
        pl.when(pl.col("team") == pl.col("home_team"))
        .then(pl.col("away_score_final"))
        .otherwise(pl.col("home_score_final"))
        .alias("points_against"),
    ])

    team_games = team_games.with_columns([
        (pl.col("points_for") - pl.col("points_against")).alias("point_diff"),
        (pl.col("points_for") > pl.col("points_against")).cast(pl.Int8).alias("win"),
        (pl.col("off_epa") - pl.col("def_epa_allowed")).alias("net_epa"),
    ])

    df = team_games.to_pandas()
    df["game_date"] = pd.to_datetime(df["game_date"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    return df


def add_pregame_rolling_features(team_games: pd.DataFrame, roll_window: int = 8) -> pd.DataFrame:
    df = team_games.copy()
    df = df.sort_values(["season", "team", "game_date", "game_id"]).reset_index(drop=True)

    grouped = df.groupby(["season", "team"], sort=False)
    df["rest_days"] = grouped["game_date"].diff().dt.days
    df["rest_days"] = df["rest_days"].fillna(7).clip(lower=3, upper=21)
    df["games_played"] = grouped.cumcount()

    for metric in BASE_METRICS:
        df[f"{metric}_roll{roll_window}"] = (
            grouped[metric]
            .apply(lambda series: series.shift(1).rolling(roll_window, min_periods=3).mean())
            .reset_index(level=[0, 1], drop=True)
        )
        df[f"{metric}_szn"] = (
            grouped[metric]
            .apply(lambda series: series.shift(1).expanding(min_periods=3).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    roll_cols = [col for col in df.columns if col.endswith(f"_roll{roll_window}") or col.endswith("_szn")]
    df[roll_cols] = df.groupby("season")[roll_cols].transform(lambda values: values.fillna(values.mean()))
    return df


def compute_elo_table(games: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    df = games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    elos: Dict[str, float] = {}
    current_season: Optional[int] = None
    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        season = int(row["season"])
        if current_season is None:
            current_season = season
        elif season != current_season:
            for team in list(elos.keys()):
                elos[team] = 1500.0 * params.offseason_regress + elos[team] * (1.0 - params.offseason_regress)
            current_season = season

        home = row["home_team"]
        away = row["away_team"]
        home_score = float(row["home_score_final"])
        away_score = float(row["away_score_final"])

        if home not in elos:
            elos[home] = 1500.0
        if away not in elos:
            elos[away] = 1500.0

        home_pre = elos[home]
        away_pre = elos[away]
        elo_diff = (home_pre + params.home_field) - away_pre
        expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        if home_score > away_score:
            result_home = 1.0
        elif home_score < away_score:
            result_home = 0.0
        else:
            result_home = 0.5

        margin = abs(home_score - away_score)
        multiplier = math.log(margin + 1.0) * (2.2 / ((abs(elo_diff) * 0.001) + 2.2))
        k_value = params.k_base * multiplier

        delta = k_value * (result_home - expected_home)
        elos[home] = home_pre + delta
        elos[away] = away_pre - delta

        rows.append({
            "game_id": row["game_id"],
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "home_elo_post": elos[home],
            "away_elo_post": elos[away],
        })

    return pd.DataFrame(rows)


def make_matchups(team_games_feat: pd.DataFrame, elo_games: pd.DataFrame, roll_window: int = 8) -> Tuple[pd.DataFrame, List[str]]:
    df = team_games_feat.copy()
    home_rows = df[df["team"] == df["home_team"]].copy()
    away_rows = df[df["team"] == df["away_team"]].copy()

    carry = [c for c in ["game_id", "season", "week", "season_type", "game_date", "home_team", "away_team"] if c in home_rows.columns]
    home_rows = home_rows[carry + [c for c in home_rows.columns if c not in carry]].add_prefix("home_")
    away_rows = away_rows[["game_id"] + [c for c in away_rows.columns if c != "game_id"]].add_prefix("away_")

    matchups = home_rows.merge(away_rows, left_on="home_game_id", right_on="away_game_id", how="inner")
    matchups = matchups.merge(
        elo_games[["game_id", "home_elo_pre", "away_elo_pre"]],
        left_on="home_game_id",
        right_on="game_id",
        how="left",
    )
    matchups["elo_diff"] = matchups["home_elo_pre"] - matchups["away_elo_pre"]
    matchups["home_win"] = (matchups["home_points_for"] > matchups["home_points_against"]).astype(int)

    feature_cols: List[str] = []
    for metric in BASE_METRICS:
        home_roll = f"home_{metric}_roll{roll_window}"
        away_roll = f"away_{metric}_roll{roll_window}"
        if home_roll in matchups.columns and away_roll in matchups.columns:
            col = f"diff_{metric}_roll{roll_window}"
            matchups[col] = matchups[home_roll] - matchups[away_roll]
            feature_cols.append(col)

        home_szn = f"home_{metric}_szn"
        away_szn = f"away_{metric}_szn"
        if home_szn in matchups.columns and away_szn in matchups.columns:
            col = f"diff_{metric}_szn"
            matchups[col] = matchups[home_szn] - matchups[away_szn]
            feature_cols.append(col)

    if "home_rest_days" in matchups.columns and "away_rest_days" in matchups.columns:
        matchups["diff_rest_days"] = matchups["home_rest_days"] - matchups["away_rest_days"]
        feature_cols.append("diff_rest_days")
    if "home_games_played" in matchups.columns and "away_games_played" in matchups.columns:
        matchups["diff_games_played"] = matchups["home_games_played"] - matchups["away_games_played"]
        feature_cols.append("diff_games_played")

    feature_cols.append("elo_diff")
    if "home_week" in matchups.columns:
        matchups["week"] = matchups["home_week"]
        feature_cols.append("week")

    return matchups, feature_cols


# =============================================================================
# Training and prediction
# =============================================================================
def train_model(matchups: pd.DataFrame, feature_cols: List[str], random_state=42) -> Tuple[CalibratedClassifierCV, Dict]:
    data = matchups[matchups["home_season_type"] == "REG"].copy()

    seasons = sorted(data["home_season"].unique())
    if len(seasons) < 8:
        raise ValueError("Not enough seasons found for a stable train/val/test split.")

    test_season = seasons[-1]
    val_season = seasons[-2]
    train_seasons = seasons[:-2]

    train = data[data["home_season"].isin(train_seasons)]
    val = data[data["home_season"] == val_season]
    test = data[data["home_season"] == test_season]

    X_train, y_train = train[feature_cols], train["home_win"]
    X_val, y_val = val[feature_cols], val["home_win"]
    X_test, y_test = test[feature_cols], test["home_win"]

    base = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=4,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=0.2,
        early_stopping=True,
        random_state=random_state,
    )
    base.fit(X_train, y_train)

    # New sklearn-compatible replacement for cv="prefit"
    frozen_base = FrozenEstimator(base)
    cal = CalibratedClassifierCV(frozen_base, method="isotonic")
    cal.fit(X_val, y_val)

    def eval_split(name, X, y):
        p = cal.predict_proba(X)[:, 1]
        out = {
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)),
            "accuracy": float(((p >= 0.5).astype(int) == y.values).mean()),
        }
        print(f"{name}: {out}")
        return out

    metrics = {
        "train": eval_split("TRAIN", X_train, y_train),
        "val": eval_split("VAL", X_val, y_val),
        "test": eval_split("TEST", X_test, y_test),
        "split": {
            "train_seasons": train_seasons,
            "val_season": val_season,
            "test_season": test_season,
        },
    }

    return cal, metrics


def build_team_snapshot(team_games: pd.DataFrame, elo_games: pd.DataFrame, season: int, roll_window: int = 8) -> pd.DataFrame:
    regular_season = team_games[(team_games["season"] == season) & (team_games["season_type"] == "REG")].copy()
    regular_season = regular_season.sort_values(["team", "game_date", "game_id"])

    def weighted_avg(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
        weights = df[weight_col].astype(float).values
        values = df[value_col].astype(float).values
        if weights.sum() <= 0:
            return float(np.nanmean(values))
        return float(np.sum(weights * values) / np.sum(weights))

    rows: List[Dict[str, Any]] = []
    for team, team_df in regular_season.groupby("team", sort=False):
        recent = team_df.tail(roll_window)
        row: Dict[str, Any] = {"team": team}
        for metric in BASE_METRICS:
            weight_col = "off_nplays" if metric.startswith("off_") or metric in {"net_epa", "point_diff"} else "def_nplays"
            row[f"{metric}_szn"] = weighted_avg(team_df, metric, weight_col)
            row[f"{metric}_roll{roll_window}"] = weighted_avg(recent, metric, weight_col)
        rows.append(row)

    snapshot = pd.DataFrame(rows).set_index("team")

    regular_game_ids = set(regular_season["game_id"].unique())
    elo_regular = elo_games[elo_games["game_id"].isin(regular_game_ids)].copy()
    last_game_by_team = set(
        regular_season.sort_values(["game_date", "game_id"]).groupby("team")["game_id"].tail(1).values
    )

    game_map = regular_season.drop_duplicates("game_id")[["game_id", "home_team", "away_team"]].set_index("game_id").to_dict("index")
    elo_map = elo_regular.set_index("game_id").to_dict("index")

    team_elo: Dict[str, float] = {}
    for game_id in last_game_by_team:
        if game_id not in elo_map or game_id not in game_map:
            continue
        home_team = game_map[game_id]["home_team"]
        away_team = game_map[game_id]["away_team"]
        team_elo[home_team] = float(elo_map[game_id]["home_elo_post"])
        team_elo[away_team] = float(elo_map[game_id]["away_elo_post"])

    snapshot["elo"] = pd.Series(team_elo).reindex(snapshot.index).fillna(1500.0)
    return snapshot


def resolve_team_code(snapshot: pd.DataFrame, team: str) -> str:
    candidate = team.strip().upper()
    if candidate in snapshot.index:
        return candidate

    aliases = TEAM_ALIASES.get(candidate, []) + [candidate]
    for alias in aliases:
        if alias in snapshot.index:
            return alias

    contains_matches = [idx for idx in snapshot.index if candidate in idx]
    if len(contains_matches) == 1:
        return contains_matches[0]

    raise KeyError(f"Team '{team}' was not found in the snapshot index.")


def build_matchup_row_from_snapshot(
    snapshot: pd.DataFrame,
    feature_cols: List[str],
    team_a: str,
    team_b: str,
    roll_window: int,
    week: int,
    team_a_elo: Optional[float] = None,
    team_b_elo: Optional[float] = None,
) -> pd.DataFrame:
    home = resolve_team_code(snapshot, team_a)
    away = resolve_team_code(snapshot, team_b)
    home_row = snapshot.loc[home]
    away_row = snapshot.loc[away]

    row: Dict[str, float] = {}
    for metric in BASE_METRICS:
        roll_col = f"diff_{metric}_roll{roll_window}"
        szn_col = f"diff_{metric}_szn"
        if roll_col in feature_cols:
            row[roll_col] = float(home_row[f"{metric}_roll{roll_window}"] - away_row[f"{metric}_roll{roll_window}"])
        if szn_col in feature_cols:
            row[szn_col] = float(home_row[f"{metric}_szn"] - away_row[f"{metric}_szn"])

    home_elo = float(team_a_elo if team_a_elo is not None else home_row["elo"])
    away_elo = float(team_b_elo if team_b_elo is not None else away_row["elo"])
    if "elo_diff" in feature_cols:
        row["elo_diff"] = home_elo - away_elo
    if "diff_rest_days" in feature_cols:
        row["diff_rest_days"] = 0.0
    if "diff_games_played" in feature_cols:
        row["diff_games_played"] = 0.0
    if "week" in feature_cols:
        row["week"] = float(week)

    return pd.DataFrame([row])[feature_cols]


def neutral_site_probability(
    model: CalibratedClassifierCV,
    snapshot: pd.DataFrame,
    feature_cols: List[str],
    team_a: str,
    team_b: str,
    roll_window: int,
    week: int,
) -> float:
    matchup_a_home = build_matchup_row_from_snapshot(snapshot, feature_cols, team_a, team_b, roll_window, week)
    matchup_b_home = build_matchup_row_from_snapshot(snapshot, feature_cols, team_b, team_a, roll_window, week)
    p_a_home = float(model.predict_proba(matchup_a_home)[:, 1][0])
    p_b_home = float(model.predict_proba(matchup_b_home)[:, 1][0])
    return 0.5 * (p_a_home + (1.0 - p_b_home))


# =============================================================================
# Persistence
# =============================================================================
def save_results(
    config: ModelConfig,
    metrics: Dict[str, Any],
    prediction: Dict[str, Any],
    feature_cols: List[str],
) -> None:
    project_root = resolve_project_root()
    results_dir = project_root / config.results_dir
    ensure_directory(results_dir)

    save_json(results_dir / config.metrics_output_file, metrics)
    save_json(results_dir / config.prediction_output_file, prediction)
    (results_dir / config.feature_columns_output_file).write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )

    summary = f"""# Super Bowl Winner Prediction Summary

- Model family: HistGradientBoostingClassifier + isotonic calibration
- Prediction target: Neutral-site win probability
- Team A: {prediction['team_a']}
- Team B: {prediction['team_b']}
- Team A win probability: {prediction['team_a_win_probability']:.4f}
- Team B win probability: {prediction['team_b_win_probability']:.4f}
- Target season snapshot: {prediction['target_season']}
- Cache TTL: {config.cache_ttl_hours} hours

## Evaluation snapshot
- Validation log loss: {metrics['val']['log_loss']:.4f}
- Validation Brier score: {metrics['val']['brier']:.4f}
- Test AUC: {metrics['test']['auc']:.4f}
- Test accuracy: {metrics['test']['accuracy']:.4f}
"""
    (results_dir / config.summary_output_file).write_text(summary, encoding="utf-8")


# =============================================================================
# CLI and orchestration
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a Super Bowl matchup winner using historical NFL play-by-play data.")
    parser.add_argument("--team-a", default="NE", help="First team in the matchup, e.g. NE")
    parser.add_argument("--team-b", default="SEA", help="Second team in the matchup, e.g. SEA")
    parser.add_argument("--target-season", type=int, default=None, help="Season used for the end-of-regular-season snapshot")
    parser.add_argument("--roll-window", type=int, default=8, help="Rolling window size for team form features")
    parser.add_argument("--cache-ttl-hours", type=int, default=24, help="Hours before cached play-by-play data is refreshed")
    parser.add_argument("--refresh-cache", action="store_true", help="Force a refresh of cached play-by-play data")
    parser.add_argument("--prediction-week", type=int, default=20, help="Synthetic week value used for the Super Bowl matchup row")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for the model")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        cache_ttl_hours=args.cache_ttl_hours,
        roll_window=args.roll_window,
        target_season=args.target_season,
        team_a=args.team_a,
        team_b=args.team_b,
        prediction_week=args.prediction_week,
        random_state=args.random_state,
    )


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)

    pbp = load_pbp_all_seasons(config=config, refresh_cache=args.refresh_cache)
    games_pl = games_from_pbp(pbp)
    games = games_pl.to_pandas()
    games["game_date"] = pd.to_datetime(games["game_date"])

    team_games = team_games_from_pbp(pbp, games_pl)
    team_games_feat = add_pregame_rolling_features(team_games, roll_window=config.roll_window)
    elo_games = compute_elo_table(games)
    matchups, feature_cols = make_matchups(team_games_feat, elo_games, roll_window=config.roll_window)
    model, metrics = train_model(matchups, feature_cols, random_state=config.random_state)

    target_season = config.target_season
    if target_season is None:
        target_season = int(pd.Series(team_games["season"]).max())

    snapshot = build_team_snapshot(team_games, elo_games, season=target_season, roll_window=config.roll_window)
    team_a = resolve_team_code(snapshot, config.team_a)
    team_b = resolve_team_code(snapshot, config.team_b)

    if config.neutral_site:
        p_team_a = neutral_site_probability(
            model=model,
            snapshot=snapshot,
            feature_cols=feature_cols,
            team_a=team_a,
            team_b=team_b,
            roll_window=config.roll_window,
            week=config.prediction_week,
        )
    else:
        matchup = build_matchup_row_from_snapshot(snapshot, feature_cols, team_a, team_b, config.roll_window, config.prediction_week)
        p_team_a = float(model.predict_proba(matchup)[:, 1][0])

    prediction = {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win_probability": float(p_team_a),
        "team_b_win_probability": float(1.0 - p_team_a),
        "target_season": int(target_season),
        "neutral_site": bool(config.neutral_site),
        "prediction_week": int(config.prediction_week),
        "generated_at_utc": utc_now().isoformat(),
        "model_family": "HistGradientBoostingClassifier with isotonic calibration",
        "notes": "This case study uses gradient boosting for classification. It does not use Geometric Brownian Motion.",
    }

    save_results(config, metrics, prediction, feature_cols)

    print(f"P({team_a} beats {team_b}) = {p_team_a:.4f}")
    print(f"P({team_b} beats {team_a}) = {1.0 - p_team_a:.4f}")


if __name__ == "__main__":
    main()