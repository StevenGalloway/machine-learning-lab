# Data Leakage (Critical ML Risk)

**Definition:** When information from outside the training dataset is
improperly used to create the model.

### Types of Leakage

-   **Target Leakage:** Features include information from the future.
-   **Train/Test Contamination:** Data overlaps between train and test
    sets.
-   **Cross-Validation Leakage:** Preprocessing done before splitting
    data.

### How to Prevent Leakage

-   Split data before preprocessing
-   Use pipelines (e.g., `sklearn.pipeline`)
-   Time-based splits for temporal data
-   Strict feature governance

Example: https://github.com/example/data-leakage-lab

------------------------------------------------------------------------

