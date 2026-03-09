# Markov Chain Model Reference

## Definition

A **Markov Chain** is a stochastic process where the probability of the
next state depends only on the current state. This is known as the
**Markov Property**.

## The Markov Property

P(X\_{t+1} \| X_t, X\_{t-1}, ..., X_0) = P(X\_{t+1} \| X_t)

The future state depends only on the present state.

## States

A **state** represents a condition of the system.

Examples: - Weather: Sunny, Rainy - Finance: Bull Market, Bear Market -
Customer Behavior: Browse → Cart → Purchase

## Transition Probabilities

Probabilities describing movement from one state to another.

  Current   Next    Probability
  --------- ------- -------------
  Sunny     Sunny   0.7
  Sunny     Rainy   0.3
  Rainy     Sunny   0.4
  Rainy     Rainy   0.6

## Transition Matrix

          Sunny   Rainy
  ------- ------- -------
  Sunny   0.7     0.3
  Rainy   0.4     0.6

Each row sums to **1**.

## Stationary Distribution

The long‑term probability distribution across states where:

πP = π

## Example Python

``` python
import numpy as np

states = ["Sunny", "Rainy"]

transition_matrix = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

current_state = 0

for _ in range(10):
    current_state = np.random.choice([0,1], p=transition_matrix[current_state])
    print(states[current_state])
```
