# FEATURE STORE

## What is a Feature Store?

A centralized system that manages: - Feature definitions and semantics -
Versioning and lineage - Point-in-time correctness - Access control and
governance

## Offline vs Online Features

    Batch ETL → Offline Feature Store → Model Training
    Real-time Stream → Online Feature Store → Low-latency Serving

## Training/Serving Consistency

A core guarantee of a good feature store is that the **same feature
logic and values are used in training and production**. Violations lead
to training/serving skew.

## Typical Stack

-   **Feature store:** Feast, Tecton, Databricks Feature Store
-   **Storage:** Delta Lake or Apache Iceberg
-   **Streaming:** Kafka, Kinesis, or Pub/Sub

## Common Failure Modes

-   **Training/Serving Skew:** Features differ between train and serve
-   **Stale Features:** Features not refreshed on time
-   **Label Leakage:** Labels computed with future information
-   **Backfill Bugs:** Inconsistent historical data

Example: (Insert Example Repo when completed)