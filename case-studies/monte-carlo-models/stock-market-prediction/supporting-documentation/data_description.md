# Data Description

This project uses historical daily equity prices retrieved from `yfinance` for a configurable list of public tickers. The runtime script prefers adjusted close prices when available so historical returns better reflect splits and dividend adjustments.

## Core Input Data

- **Source:** `yfinance`
- **Granularity:** Daily market data
- **Primary fields used:** `Adj Close` or `Close`
- **Default lookback window:** 3 years
- **Cached artifact:** `data/cached_prices.csv`

## Data Shape

Each row represents one trading day and each column represents one selected ticker. The dataset is converted into a clean wide-format price frame, and complete rows are retained for return estimation.
