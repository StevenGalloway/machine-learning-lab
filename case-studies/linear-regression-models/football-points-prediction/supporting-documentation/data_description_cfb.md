# Data Description (CFB)

## Dataset intent
Synthetic dataset designed to mimic **college football** scoring behavior.

## Schema
- `FirstDowns` (int): number of first downs
- `Points` (float): points scored (target)

## Size
- Rows: 1000
- Train/Test split: 800/200
- Seed: 42

## Distribution summary
|       |   FirstDowns |    Points |
|:------|-------------:|----------:|
| count |   1000       | 1000      |
| mean  |     23.424   |   40.0952 |
| std   |      6.20473 |   13.6848 |
| min   |      9       |    0      |
| 25%   |     19       |   31.0825 |
| 50%   |     23       |   40.12   |
| 75%   |     28       |   49.1125 |
| max   |     39       |   80      |

## Range snapshot
- FirstDowns: 9 to 39
- Points: 0.00 to 80.00
