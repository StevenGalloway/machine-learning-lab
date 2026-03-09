# Convolutional Neural Network (CNN) Reference

## Definition

A **Convolutional Neural Network (CNN)** is a deep learning model
designed to process image‑like data using convolution operations.

## CNN Pipeline

Input Image → Convolution → Activation → Pooling → Flatten → Dense →
Output

## Convolution

A convolution applies a **filter (kernel)** across an image to detect
patterns.

## Key Terminology

**Kernel (Filter)** -- small matrix detecting patterns\
**Feature Map** -- output of convolution\
**Stride** -- step size of filter movement\
**Padding** -- borders added to maintain dimensions\
**Pooling** -- reduces spatial size while preserving key features

## Activation Functions

  Activation   Purpose
  ------------ ----------------------------
  ReLU         Introduces non‑linearity
  Sigmoid      Binary classification
  Softmax      Multi‑class classification

## Example CNN Architecture

Input (28x28) → Conv2D(32 filters) → ReLU → MaxPooling → Conv2D(64
filters) → Flatten → Dense → Softmax

## Example Python

``` python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation="relu"),
    Flatten(),
    Dense(128,activation="relu"),
    Dense(26,activation="softmax")
])
```
