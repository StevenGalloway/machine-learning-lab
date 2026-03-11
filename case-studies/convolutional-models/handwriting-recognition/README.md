# Handwriting Recognition with a Convolutional Neural Network

This case study demonstrates an enterprise-style convolutional neural network workflow for classifying hand-drawn lowercase letters using the provided EMNIST letters dataset. The project supports both batch evaluation and an interactive Gradio app where a user can draw a character and receive live predictions.

## Quick Start

Install dependencies:

```bash
pip install tensorflow keras pandas numpy pillow gradio matplotlib
```

Run the project:

```bash
python case-studies/convolutional-models/handwriting-recognition/scripts/handwriting_recognition_cnn.py
```

Run without launching the app:

```bash
python case-studies/convolutional-models/handwriting-recognition/scripts/handwriting_recognition_cnn.py --skip-demo
```

Force a cache rebuild and retrain:

```bash
python case-studies/convolutional-models/handwriting-recognition/scripts/handwriting_recognition_cnn.py --force-rebuild-cache --force-retrain
```

## What the Script Does

1. Loads the provided EMNIST CSV data from `/data`
2. Caches preprocessed arrays to `/data/cache`
3. Trains a CNN or loads an existing saved Keras model
4. Writes runtime artifacts to `/results`
5. Optionally launches a Gradio sketchpad app for live inference

## Outputs

Runtime artifacts written to `/results`:

- `metrics.json`
- `sample_predictions.json`
- `confusion_matrix.csv`
- `model_architecture.json`
- `summary.md`

Persistent model/data artifacts:

- `data/cache/emnist_letters_cache.npz`
- `data/models/handwriting_cnn.keras`

## Project Structure

```text
case-studies/convolutional-models/handwriting-recognition/
├── data/
│   ├── emnist-letters-train.csv
│   ├── emnist-letters-test.csv
│   ├── cache/
│   │   └── emnist_letters_cache.npz
│   └── models/
│       └── handwriting_cnn.keras
├── results/
│   ├── metrics.json
│   ├── sample_predictions.json
│   ├── confusion_matrix.csv
│   ├── model_architecture.json
│   └── summary.md
├── scripts/
│   ├── handwriting_recognition_cnn.py
│   └── handwriting-recognition-cm.py
└── supporting-documentation/
    ├── cnn_specific_considerations.md
    ├── convolution_and_pooling_primer.md
    ├── data_description.md
    ├── deployment_plan.md
    ├── eda_summary.md
    ├── error_analysis.md
    ├── experiment_log.md
    ├── feature_dictionary.md
    ├── model_card.md
    ├── monitoring_plan.md
    ├── problem_statement.md
    ├── receptive_field_notes.md
    ├── risk_analysis.md
    └── stakeholders.md
```

## Architecture Summary

The model is a CNN with two convolution + pooling blocks followed by dense classification layers.

## Why the Original Gradio Error Happened

The original script used `gr.interface(...)`. That fails because `Interface` is a Gradio class and must be instantiated with a capital `I`: `gr.Interface(...)`.
