# Data Description

This project uses historical daily crypto prices retrieved from `yfinance` for a configurable list of digital assets. The runtime script prefers adjusted close prices when available and falls back to close prices when needed.

## Core Input Data

- **Source:** `yfinance`
- **Granularity:** Daily market data
- **Primary fields used:** `Adj Close` or `Close`
- **Default lookback window:** 2 years
- **Cached artifact:** `data/cached_prices.csv`

## Data Shape

Each row represents one day and each column represents one selected coin. The dataset is converted into a clean wide-format price frame, and complete rows are retained for return estimation.
