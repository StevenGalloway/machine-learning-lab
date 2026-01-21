# Neural Networks -- A Practical Overview

Neural networks are the foundation of modern machine learning systems
used in computer vision, natural language processing, recommendation
engines, speech recognition, and large language models.

------------------------------------------------------------------------

## What is a Neural Network? (Definition)

A **neural network** is a computational model inspired by the human
brain that learns patterns by transforming input data through layers of
interconnected units called **neurons**.

At a high level, a neural network learns a function:

f(x) → y

Where: - **x** = input data (features) - **y** = predicted output -
**f** = learned mapping through layers of neurons

Neural networks are especially powerful because they can learn
**non-linear relationships** and complex representations automatically
from data.

------------------------------------------------------------------------

## How Neural Networks Work

At a system level, neural networks operate in two main phases:

1.  **Forward Pass** -- compute predictions
2.  **Backward Pass (Backpropagation)** -- update weights based on error

### High-Level Flow

``` mermaid
flowchart LR
    A[Input Data] --> B[Hidden Layers]
    B --> C[Output Layer]
    C --> D[Loss Function]
    D --> E[Backpropagation]
    E --> B
```

------------------------------------------------------------------------

## Core Components Explained

### Neurons

A **neuron** is a mathematical unit that: - Receives input signals -
Applies weights - Adds bias - Passes result through an activation
function

Mathematically: z = wᵀx + b
a = f(z)

------------------------------------------------------------------------

### Inputs & Outputs

-   **Inputs:** Feature values (images, text embeddings, sensor
    readings)
-   **Outputs:** Predictions (class labels, probabilities, continuous
    values)

------------------------------------------------------------------------

### Weights

Weights represent **learned importance** of each input. - High weight →
strong influence - Low weight → weak influence

------------------------------------------------------------------------

### Bias

Bias allows the model to shift predictions even when inputs are zero. -
Acts as a baseline offset

------------------------------------------------------------------------

### Signals

Signals are the numerical values flowing through the network between
layers.

------------------------------------------------------------------------

### Layers

Neural networks are organized into layers:

``` mermaid
flowchart LR
    A[Input Layer] --> B[Hidden Layer 1]
    B --> C[Hidden Layer 2]
    C --> D[Output Layer]
```

#### Input Layer

-   Receives raw features
-   No computation

#### Hidden Layers

-   Perform transformations
-   Learn representations
-   Depth = number of hidden layers

#### Output Layer

-   Produces final prediction
-   Uses task-specific activation (sigmoid, softmax, linear)

#### Layer Width & Depth

-   **Width:** number of neurons in a layer
-   **Depth:** number of layers

More depth → more representational power
More width → more feature capacity

------------------------------------------------------------------------

## Loss Functions

The **loss function** measures how wrong the prediction is.

Examples: - Mean Squared Error (regression) - Cross-Entropy
(classification) - Binary Log Loss

Loss drives learning by providing an optimization signal.

------------------------------------------------------------------------

## Backpropagation

Backpropagation computes gradients of loss with respect to weights.

``` mermaid
flowchart LR
    A[Prediction] --> B[Loss]
    B --> C[Gradient]
    C --> D[Weight Update]
```

This allows the model to: - Reduce error - Adjust internal
representations - Learn from mistakes

------------------------------------------------------------------------

## Sensitivity & Influence

### Sensitivity

How much output changes when input changes.

High sensitivity → model reacts strongly to small changes.

### Influence

How much a specific feature contributes to prediction. Measured via: -
Gradients - SHAP values - Feature attribution methods

------------------------------------------------------------------------

## Use Cases

Neural networks are used when patterns are complex and high-dimensional.

### Common Applications

-   Image classification (CNNs)
-   Language translation (Transformers)
-   Recommendation systems
-   Speech recognition
-   Autonomous systems
-   Time-series forecasting

------------------------------------------------------------------------

## Pros & Cons

### Pros

-   Learns complex nonlinear patterns
-   Works on unstructured data
-   End-to-end learning
-   State-of-the-art performance

### Cons

-   Requires large datasets
-   Computationally expensive
-   Hard to interpret
-   Sensitive to hyperparameters

------------------------------------------------------------------------

## When Should You Use Neural Networks?

Use neural networks when: - You have large amounts of data - Features
are complex or unstructured - Traditional models underperform -
Performance is more important than interpretability

Avoid when: - Data is small - Interpretability is critical - Simpler
models already perform well

------------------------------------------------------------------------

## Example Repositories

Basic Feedforward Network
(Insert Example Repo when completed)

Image Classifier (CNN)
(Insert Example Repo when completed)

Transformer Model
(Insert Example Repo when completed)

------------------------------------------------------------------------


# Activation Functions in Neural Networks

Activation functions are a core component of neural networks. They
introduce **non-linearity** into the model, enabling neural networks to
learn complex patterns beyond simple linear relationships.

Without activation functions, a neural network would effectively
collapse into a single linear transformation, regardless of how many
layers it has.

------------------------------------------------------------------------

## What is an Activation Function?

An **activation function** transforms the weighted sum of inputs to a
neuron into an output signal that is passed to the next layer.

Mathematically:

z = w\^T x + b\
a = f(z)

Where: - **x** = input features
- **w** = weights
- **b** = bias
- **f(z)** = activation function
- **a** = activated output

Activation functions: - Enable **non-linear decision boundaries** -
Control **gradient flow during training** - Influence **model
expressiveness and convergence**

------------------------------------------------------------------------

## Where Activation Functions Fit in a Neural Network

Input → Linear Layer → Activation → Linear Layer → Activation → Output

Each hidden layer applies an activation function to determine what
information is propagated forward.

------------------------------------------------------------------------

# Core Activation Functions

------------------------------------------------------------------------

## 1. Rectified Linear Unit (ReLU)

**Definition:**
Outputs the input directly if positive, otherwise outputs zero.

**Formula:** f(x) = max(0, x)

**How it works:** - Suppresses negative values - Passes positive values
unchanged

**Diagram:** x \< 0 → 0\
x ≥ 0 → x

**Use Cases:** - Default activation for hidden layers - CNNs for
computer vision - Deep feedforward networks

**Pros:** - Very fast computation - Reduces vanishing gradient problem -
Sparse activation (many zeros)

**Cons:** - Dying ReLU problem (neurons stuck at 0) - Unbounded output -
Not differentiable at 0

**Example Repo:**
(Insert Example Repo when completed)

------------------------------------------------------------------------

## 2. Gaussian Error Linear Unit (GELU)

**Definition:**
Smooth probabilistic version of ReLU used heavily in Transformer models.

**How it works:** - Weighs inputs by probability of being useful -
Allows small negative values

**Use Cases:** - Transformers (BERT, GPT) - Large language models -
Vision Transformers

**Pros:** - Better gradient flow than ReLU - Improves convergence in
deep models - State-of-the-art in NLP

**Cons:** - Slower than ReLU - More complex computation - Less
interpretable

**Example Repo:**
(Insert Example Repo when completed)

------------------------------------------------------------------------

## 3. Sigmoid Linear Unit (SiLU / Swish)

**Definition:**
A smooth, self-gated activation function.

**Formula:** f(x) = x \* sigmoid(x)

**How it works:** - Combines identity mapping with sigmoid gate -
Preserves small negative values

**Use Cases:** - Deep CNNs - EfficientNet - Reinforcement learning
models

**Pros:** - Smooth gradient - Outperforms ReLU in some deep models - No
hard cutoff

**Cons:** - Computationally heavier - Risk of vanishing gradients - Less
interpretable

**Example Repo:**
(Insert Example Repo when completed)

------------------------------------------------------------------------

## 4. Sigmoid

**Definition:**
Squashes input into range (0, 1).

**Formula:** f(x) = 1 / (1 + e\^-x)

**How it works:** - Converts values into probabilities - Historically
used in early neural networks

**Use Cases:** - Binary classification output layers - Logistic
regression - Probabilistic models

**Pros:** - Output interpretable as probability - Smooth and
differentiable - Simple

**Cons:** - Severe vanishing gradient - Saturates quickly - Poor for
deep hidden layers

**Example Repo:**
(Insert Example Repo when completed)r

------------------------------------------------------------------------

## 5. Softmax

**Definition:**
Normalizes a vector into a probability distribution.

**Formula:** f(x_i) = e\^(x_i) / Σ e\^(x_j)

**How it works:** - Converts logits into probabilities - Ensures all
outputs sum to 1

**Use Cases:** - Multi-class classification - Language models - Image
classification

**Pros:** - Probabilistic interpretation - Works naturally with
cross-entropy loss - Stable training

**Cons:** - Sensitive to large values - Only suitable for output layer -
Cannot represent independent probabilities

**Example Repo:**
(Insert Example Repo when completed)

------------------------------------------------------------------------

# Additional Activation Functions

## Leaky ReLU

-   Allows small negative slope
-   Fixes dying ReLU problem

## Tanh

-   Outputs in (-1, 1)
-   Zero-centered but saturates

## ELU / SELU

-   Self-normalizing networks
-   Faster convergence

------------------------------------------------------------------------

# Choosing the Right Activation Function

  Scenario                  Recommended
  ------------------------- -------------
  Hidden layers (general)   ReLU / GELU
  Transformers / LLMs       GELU
  Deep CNNs                 SiLU / ReLU
  Binary output             Sigmoid
  Multi-class output        Softmax
  Self-normalizing nets     SELU

------------------------------------------------------------------------

# Interview-Level Takeaways

-   Why **ReLU dominates hidden layers**
-   Why **Softmax pairs with cross-entropy**
-   Why **Sigmoid causes vanishing gradients**
-   Why **GELU is used in Transformers**
-   How activation functions affect:
    -   Gradient flow
    -   Model expressiveness
    -   Training stability