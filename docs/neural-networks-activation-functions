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