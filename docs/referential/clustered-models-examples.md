# Clustered Models (With Math, Use Cases, and Tradeoffs)

## K-Means Clustering

**Idea:** Partition data into `k` clusters by minimizing within-cluster
variance.

**Diagram (conceptual):**

    Raw Data → Assign to Nearest Centroid → Recompute Centroids → Repeat → Final Clusters

**Use cases** - Customer segmentation - Image color quantization -
Document clustering

**Pros** - Fast and scalable - Easy to interpret

**Cons** - Requires choosing `k` - Struggles with non-spherical
clusters - Sensitive to outliers

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## K-Means++

Improves K-Means initialization to avoid poor local minima.

**Benefit:** Better and more stable clusters with fewer iterations.

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## DBSCAN (Density-Based Spatial Clustering)

Clusters based on density rather than distance to centroids.

**Key parameters** - `eps`: neighborhood radius - `min_samples`: minimum
points to form a cluster

**Diagram:**

    High-density regions → Clusters
    Low-density points → Noise / Outliers

**Use cases** - Anomaly detection - Geospatial clustering -
Noise-resistant clustering

**Pros** - Finds arbitrary-shaped clusters - Automatically detects
outliers

**Cons** - Sensitive to `eps` - Struggles with varying density

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## Spectral Clustering

Uses eigenvectors of similarity matrices to find clusters in complex
structures.

**Use cases** - Graph clustering - Image segmentation - Social network
analysis

**Pros** - Works well on non-convex clusters

**Cons** - Computationally expensive

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## Gaussian Mixture Model (GMM)

Assumes data is generated from a mixture of Gaussian distributions.

**Key idea:** Soft clustering via probabilities.

**Use cases** - Density estimation - Anomaly detection - Speaker
diarization

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## Deep Clustering

Uses neural networks to learn embeddings before clustering (e.g.,
autoencoders + K-Means).

**Diagram:**

    Raw Data → Autoencoder → Latent Embeddings → K-Means → Clusters

**Use cases** - Image clustering - Representation learning - Large-scale
segmentation

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## Linear Regression (as a Clustered/Similarity Model)

Can be used in clustering-style settings (e.g., grouping by similar
regression behavior).

**Equation:** $$ y = w^T x + b $$

**Use cases** - Trend-based grouping - Behavioral segmentation

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

