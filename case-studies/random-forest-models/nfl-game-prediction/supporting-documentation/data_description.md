# Data Description

**Source (expected):** historical NFL game logs (regular + postseason), one row per game.  
**This repo includes:** `data/spreadspoke_scores_sample.csv` (small synthetic sample so the pipeline runs end-to-end).

## Schema (minimum required)
- `schedule_season` (int)
- `schedule_week` (int or string)
- `schedule_date` (YYYY-MM-DD)
- `team_home` (string)
- `team_away` (string)
- `score_home` (int)
- `score_away` (int)

## Optional fields used if present
- `neutral_site` (0/1)
- `home_rest_days`, `away_rest_days` (int)
- `over_under_line` (float)
- `spread_favorite` (float; negative means home favored)

## Target
`home_win` = 1 if `score_home > score_away`, else 0.

## Size + coverage (current run)
- Rows: **6,503**
- Seasons: **2002–2025**
- Class balance: **0.526** home-win rate

## Data fingerprint
- File: `/mnt/data/ml-lab_repo/case-studies/nfl-game-prediction/data/spreadspoke_scores_sample.csv`
- SHA256: `e4d108fae14776876eaa9b9e17991c9576ac0c523a35156a8ee3f0941475d38c`
