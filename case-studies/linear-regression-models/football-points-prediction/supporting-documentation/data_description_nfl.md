# Data Description (NFL)

## Dataset intent
Synthetic dataset designed to mimic **NFL** scoring behavior.

## Schema
- `FirstDowns` (int): number of first downs
- `Points` (float): points scored (target)

## Size
- Rows: 1000
- Train/Test split: 800/200
- Seed: 42

## Distribution summary
|       |   FirstDowns |     Points |
|:------|-------------:|-----------:|
| count |    1000      | 1000       |
| mean  |      19.985  |   28.7659  |
| std   |       5.0934 |    9.80568 |
| min   |       8      |    0       |
| 25%   |      16      |   21.665   |
| 50%   |      20      |   28.955   |
| 75%   |      23      |   35.2925  |
| max   |      34      |   60       |

## Range snapshot
- FirstDowns: 8 to 34
- Points: 0.00 to 60.00
