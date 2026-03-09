# Linear Regression Model Reference

## Definition

**Linear Regression** is a supervised learning algorithm that models the
relationship between a dependent variable and one or more independent
variables.

## Mathematical Model

y = β0 + β1x + ε

Where:

  Term   Meaning
  ------ -----------------
  y      target variable
  x      feature
  β0     intercept
  β1     coefficient
  ε      error term

## Multiple Linear Regression

y = β0 + β1x1 + β2x2 + ... + βnxn

## Key Terminology

**Feature** -- input variable\
**Coefficient** -- weight applied to a feature\
**Residual** -- difference between predicted and actual value

Residual = y − ŷ

**R² Score** measures how much variance the model explains.

## Advantages

-   Simple and interpretable
-   Fast to train
-   Useful baseline model

## Limitations

-   Assumes linear relationships
-   Sensitive to outliers
-   Multicollinearity issues

## Python Example

``` python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1],[2],[3],[4]])
y = np.array([2,4,6,8])

model = LinearRegression()
model.fit(X,y)

print(model.predict([[5]]))
```
