# Data Description

## Dataset
Synthetic player game logs (1000 records) designed to resemble real box-score distributions.

**File:** `data/basketball_player_game_logs_synth_1000.csv`

## Target engineering
- `PTS_next`: computed by sorting by `Player` + `Date` and shifting `PTS` by -1.

## Size
- Raw rows: 1000
- Modeling rows (after drop last game per player): 950
- Players: 50

## Summary stats (raw)
|       |       PTS |       FGA |        3PA |        FTA |        ORB |        DRB |        AST |        STL |         BLK |        TOV |        PF |
|:------|----------:|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|------------:|-----------:|----------:|
| count | 1000      | 1000      | 1000       | 1000       | 1000       | 1000       | 1000       | 1000       | 1000        | 1000       | 1000      |
| mean  |   25.1866 |   16.4871 |    5.561   |    3.0822  |    1.469   |    4.601   |    3.13    |    1.459   |    0.995    |    1.898   |    2.138  |
| std   |   12.3148 |    6.9982 |    3.64745 |    2.38304 |    1.39103 |    2.94355 |    2.41349 |    1.21152 |    0.926193 |    1.67344 |    1.5535 |
| min   |    0      |    0      |    0       |    0       |    0       |    0       |    0       |    0       |    0        |    0       |    0      |
| 25%   |   15.4    |   11.7    |    2.6     |    1.1     |    0       |    2       |    1       |    0       |    0        |    0       |    1      |
| 50%   |   25.8    |   16.6    |    5.5     |    3       |    1       |    4       |    3       |    1       |    1        |    2       |    2      |
| 75%   |   34.3    |   21.6    |    8.2     |    4.7     |    2       |    7       |    5       |    2       |    2        |    3       |    3      |
| max   |   57.2    |   34.6    |   16.4     |   11.8     |    7       |   14       |   14       |    6       |    4        |    9       |    6      |
