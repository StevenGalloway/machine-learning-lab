# Naive Bayes Models

Naive Bayes is a family of **probabilistic supervised learning algorithms**
based on **Bayes’ Theorem**, primarily used for **classification tasks**.

Despite its simplicity, Naive Bayes is widely used in real-world systems
for problems involving **high-dimensional data**, **text classification**,
and **spam filtering**.

It is called *“naive”* because it makes a strong assumption that all input
features are **conditionally independent** given the target class.

------------------------------------------------------------------------

## Core Intuition

Naive Bayes answers the question:

> *“Given the features I observe, what is the most likely class?”*

It computes the probability of each class using:

\[
P(Class \mid Features) \propto P(Class) \times P(Features \mid Class)
\]

And chooses the class with the **highest posterior probability**.

------------------------------------------------------------------------

## Why It’s Called "Naive"

The model assumes:

> **All features are independent of each other, given the class.**

In practice, this is almost never true.

Example:

If you're classifying emails:
- Feature A: Contains the word "free"
- Feature B: Contains the word "money"

These are clearly correlated but Naive Bayes treats them as independent.

Despite this unrealistic assumption, Naive Bayes performs surprisingly well
in many domains.

------------------------------------------------------------------------

## Key Concepts & Keywords

### Bayes’ Theorem

The foundation of the model:

\[
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
\]

In ML terms:
- **Prior** → `P(Class)`
- **Likelihood** → `P(Features | Class)`
- **Evidence** → `P(Features)`
- **Posterior** → `P(Class | Features)`

------------------------------------------------------------------------

### Prior Probability

The probability of a class before seeing any data.

Example:
- 80% of emails are not spam
- 20% are spam

\[
P(Spam) = 0.2
\]

------------------------------------------------------------------------

### Likelihood

The probability of observing the features **given the class**.

Example:
\[
P("free" \mid Spam)
\]

This is learned from training data.

------------------------------------------------------------------------

### Posterior Probability

The final probability used for prediction.

\[
P(Class \mid Features)
\]

This is what the model compares across classes.

------------------------------------------------------------------------

### Conditional Independence

The **naive assumption**:

\[
P(x_1, x_2, x_3 \mid y) =
P(x_1 \mid y) \cdot P(x_2 \mid y) \cdot P(x_3 \mid y)
\]

This makes the math tractable and extremely fast.

------------------------------------------------------------------------

## Types of Naive Bayes Models

### Gaussian Naive Bayes

Used when features are **continuous** and assumed to follow a **normal distribution**.

**Examples**
- Medical measurements
- Sensor data
- Financial indicators

Example repo: (Insert Example Repo when completed)

------------------------------------------------------------------------

### Multinomial Naive Bayes

Used for **count-based features**.

**Examples**
- Word counts in text
- Bag-of-words models
- TF-IDF vectors

*Example:* [Text Sender Prediction](case-studies/text-sender-identification-nb/scripts/text_sender_identification_nb.py)

------------------------------------------------------------------------

### Bernoulli Naive Bayes

Used for **binary features**.

**Examples**
- Word present / not present
- Click / no click
- Yes / No attributes

Example repo: (Insert Example Repo when completed)

------------------------------------------------------------------------

## How Naive Bayes Works (Step-by-Step)

1. Compute class priors from training data
2. Compute likelihood distributions per feature per class
3. For a new sample:
   - Multiply all likelihoods together
   - Multiply by the class prior
4. Choose class with highest posterior probability

------------------------------------------------------------------------

## Example (Spam Classification)

Training data:
- 100 emails
- 40 spam
- 60 not spam

New email contains:
- "free"
- "offer"

Compute:
\[
P(Spam) \times P("free" \mid Spam) \times P("offer" \mid Spam)
\]

Do the same for Not Spam and compare.

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## Use Cases

### Text Classification
- Spam detection
- Sentiment analysis
- Topic classification
- News categorization

*Example:* [Text Sender Prediction](case-studies/text-sender-identification-nb/scripts/text_sender_identification_nb.py)

### Document Tagging
- Legal documents
- Medical records
- Support tickets


Example: (Insert Example Repo when completed)

### Recommendation Systems
- Content filtering
- User interest classification

Example: (Insert Example Repo when completed)

### Real-Time Systems
- Fraud detection
- Alert classification
- Log categorization

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## Pros

- Extremely fast training
- Very fast inference
- Works well with high-dimensional data
- Requires small amounts of training data
- Interpretable probabilities
- Strong baseline model

------------------------------------------------------------------------

## Cons

- Unrealistic independence assumption
- Poor performance with correlated features
- Cannot model complex relationships
- Sensitive to zero probabilities (needs smoothing)
- Usually worse than modern ensemble / neural models

------------------------------------------------------------------------

## When Naive Bayes Works Surprisingly Well

- NLP / text problems
- Bag-of-words models
- Small datasets
- High-dimensional sparse data
- Real-time classification systems

------------------------------------------------------------------------

## When It Performs Poorly

- Image classification
- Time series
- Highly correlated features
- Complex feature interactions
- Nonlinear decision boundaries

------------------------------------------------------------------------

## Laplace Smoothing

Used to avoid **zero probabilities**:

\[
P(x) = \frac{count(x) + 1}{N + k}
\]

Prevents a single unseen word from collapsing the entire probability.

------------------------------------------------------------------------

## Relationship to Other Models

| Model | Comparison |
|------|------------|
| Logistic Regression | Similar output, but discriminative |
| Decision Trees | Handles feature interactions |
| Neural Networks | Learns complex nonlinear patterns |
| Random Forest | Better accuracy, less interpretable |
| LLM embeddings | Replaces NB in modern NLP |

------------------------------------------------------------------------

## Mental Model for Data Scientists

Naive Bayes is best thought of as:

> **A probabilistic rule engine learned from data.**

It builds **probability tables** and multiplies them together.

------------------------------------------------------------------------

## Production Usage Pattern

Naive Bayes is often used as:

- First baseline model
- Fallback classifier
- Real-time edge classifier
- Explainability reference model

------------------------------------------------------------------------

## Example Repositories

Basic Feedforward Network

Image Classifier (CNN)
(Insert Example Repo when completed)

Transformer Model
(Insert Example Repo when completed)

------------------------------------------------------------------------