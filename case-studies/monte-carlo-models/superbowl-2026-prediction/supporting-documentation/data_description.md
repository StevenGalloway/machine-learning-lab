# Data Description

## Source
This project uses historical NFL play-by-play data loaded through `nflreadpy`, which provides game-level and play-level information across multiple seasons.

## Grain
The raw source is play-by-play event data. The model pipeline aggregates those rows into:
- game-level final score tables
- team-game level offensive and defensive summaries
- game-level matchup rows for supervised learning

## Core fields used
- identifiers: `game_id`, `season`, `week`, `season_type`, `game_date`
- teams: `home_team`, `away_team`, `posteam`, `defteam`
- performance: `epa`, `yards_gained`, `success`, `pass`, `rush`, `sack`
- turnover indicators: `interception`, `fumble_lost`
- scoring: `home_score`, `away_score`, `total_home_score`, `total_away_score`

## Derived data assets
- cached raw play-by-play parquet in `/data`
- game table with final scores
- team-game feature table
- matchup-level training table

## Note on Geometric Brownian Motion
Geometric Brownian Motion is not used in this project. GBM is a stochastic process more appropriate for asset-price evolution than for discrete NFL game classification.
