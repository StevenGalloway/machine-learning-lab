from __future__ import annotations

import argparse
import json
import logging
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from keras import layers

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Config:
    project_root: Path = Path("case-studies/convolutional-models/handwriting-recognition")
    train_file_name: str = "emnist-letters-train.csv"
    test_file_name: str = "emnist-letters-test.csv"
    cache_file_name: str = "emnist_letters_cache.npz"
    model_file_name: str = "handwriting_cnn.keras"
    metrics_output_file: str = "metrics.json"
    predictions_output_file: str = "sample_predictions.json"
    summary_output_file: str = "summary.md"
    confusion_output_file: str = "confusion_matrix.csv"
    architecture_output_file: str = "model_architecture.json"
    image_size: int = 28
    num_classes: int = 26
    epochs: int = 5
    batch_size: int = 128
    validation_split: float = 0.10
    random_seed: int = 42
    launch_demo: bool = True

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file_name

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file_name

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / self.cache_file_name

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_file_name


LETTERS = string.ascii_lowercase


def ensure_directories(config: Config) -> None:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and serve a CNN letter recognizer.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    base = Config()
    data = asdict(base)
    if args.epochs is not None:
        data["epochs"] = args.epochs
    if args.batch_size is not None:
        data["batch_size"] = args.batch_size
    if args.skip_demo:
        data["launch_demo"] = False
    return Config(**data)


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    def make_json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    path.write_text(json.dumps(make_json_safe(payload), indent=2), encoding="utf-8")


def load_raw_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not config.train_path.exists():
        raise FileNotFoundError(f"Missing training data: {config.train_path}")
    if not config.test_path.exists():
        raise FileNotFoundError(f"Missing test data: {config.test_path}")
    return pd.read_csv(config.train_path, header=None), pd.read_csv(config.test_path, header=None)


def preprocess_and_cache(config: Config, force_rebuild_cache: bool = False) -> Dict[str, np.ndarray]:
    if config.cache_path.exists() and not force_rebuild_cache:
        cached = np.load(config.cache_path, allow_pickle=False)
        return {
            k: cached[k]
            for k in ["X_train", "y_train", "y_train_cat", "X_test", "y_test", "y_test_cat"]
        }

    train_df, test_df = load_raw_data(config)
    y_train = train_df.iloc[:, 0].to_numpy(dtype=np.int64) - 1
    X_train = train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    y_test = test_df.iloc[:, 0].to_numpy(dtype=np.int64) - 1
    X_test = test_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    X_train = X_train.reshape(-1, config.image_size, config.image_size).transpose(0, 2, 1)
    X_test = X_test.reshape(-1, config.image_size, config.image_size).transpose(0, 2, 1)
    X_train = (X_train / 255.0).reshape(-1, config.image_size, config.image_size, 1)
    X_test = (X_test / 255.0).reshape(-1, config.image_size, config.image_size, 1)

    y_train_cat = keras.utils.to_categorical(y_train, config.num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, config.num_classes)

    np.savez_compressed(
        config.cache_path,
        X_train=X_train,
        y_train=y_train,
        y_train_cat=y_train_cat,
        X_test=X_test,
        y_test=y_test,
        y_test_cat=y_test_cat,
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_cat": y_train_cat,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_cat": y_test_cat,
    }


def build_model(config: Config) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(config.image_size, config.image_size, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(config.num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_or_load_model(config: Config, arrays: Dict[str, np.ndarray], force_retrain: bool = False):
    if config.model_path.exists() and not force_retrain:
        return keras.models.load_model(config.model_path), {"loaded_from_disk": True}

    model = build_model(config)
    history_obj = model.fit(
        arrays["X_train"],
        arrays["y_train_cat"],
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        verbose=1,
    )
    model.save(config.model_path)
    return model, history_obj.history


def compute_metrics(model: keras.Model, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    test_loss, test_accuracy = model.evaluate(arrays["X_test"], arrays["y_test_cat"], verbose=0)
    pred_probs = model.predict(arrays["X_test"], verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)

    samples = []
    for i in range(10):
        true_idx = int(arrays["y_test"][i])
        pred_idx = int(pred_labels[i])
        samples.append(
            {
                "sample_index": i,
                "true_letter": LETTERS[true_idx],
                "predicted_letter": LETTERS[pred_idx],
                "predicted_probability": float(pred_probs[i][pred_idx]),
            }
        )

    confusion = np.zeros((26, 26), dtype=np.int32)
    for truth, pred in zip(arrays["y_test"], pred_labels):
        confusion[int(truth), int(pred)] += 1

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "sample_predictions": samples,
        "confusion_matrix": confusion,
    }


def save_results(config: Config, history: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    save_json(
        config.results_dir / config.metrics_output_file,
        {
            "history": history,
            "test_loss": metrics["test_loss"],
            "test_accuracy": metrics["test_accuracy"],
        },
    )
    save_json(
        config.results_dir / config.predictions_output_file,
        {"sample_predictions": metrics["sample_predictions"]},
    )
    pd.DataFrame(metrics["confusion_matrix"], index=list(LETTERS), columns=list(LETTERS)).to_csv(
        config.results_dir / config.confusion_output_file
    )
    save_json(
        config.results_dir / config.architecture_output_file,
        {
            "model_type": "Convolutional Neural Network",
            "layers": [
                "Input(28, 28, 1)",
                "Conv2D(32, 3x3, relu)",
                "MaxPooling2D(2x2)",
                "Conv2D(64, 3x3, relu)",
                "MaxPooling2D(2x2)",
                "Flatten",
                "Dense(128, relu)",
                "Dropout(0.25)",
                "Dense(26, softmax)",
            ],
        },
    )
    summary = f"""# Handwriting Recognition CNN Run Summary

## Test Performance
- Test loss: {metrics['test_loss']:.4f}
- Test accuracy: {metrics['test_accuracy']:.4f}

## Runtime Outputs
- Metrics JSON: `{config.metrics_output_file}`
- Sample predictions JSON: `{config.predictions_output_file}`
- Confusion matrix CSV: `{config.confusion_output_file}`
- Architecture JSON: `{config.architecture_output_file}`

## Notes
- Cached arrays are stored in `{config.cache_path.as_posix()}`
- Saved Keras model is stored in `{config.model_path.as_posix()}`
"""
    (config.results_dir / config.summary_output_file).write_text(summary, encoding="utf-8")


def prepare_image_for_inference(img: Any, config: Config) -> np.ndarray:
    """
    Convert Gradio sketch input into a model-ready tensor.
    This version avoids OpenCV and works with older Gradio payload styles.
    """
    if img is None:
        raise ValueError("No image was provided.")

    if isinstance(img, dict):
        composite = img.get("composite")
        background = img.get("background")
        if composite is not None:
            img = composite
        elif background is not None:
            img = background
        else:
            raise ValueError("No usable image data found in sketch input.")

    img = np.array(img)

    if img.ndim == 3:
        img = img[..., 0]

    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = ImageOps.grayscale(pil_img)
    pil_img = pil_img.resize((config.image_size, config.image_size))

    img_arr = np.array(pil_img, dtype=np.float32) / 255.0
    img_arr = 1.0 - img_arr
    img_arr = img_arr.reshape(1, config.image_size, config.image_size, 1)
    return img_arr


def build_predict_fn(model: keras.Model, config: Config):
    def predict_letter(img: Any) -> Dict[str, float]:
        img_np = prepare_image_for_inference(img, config)
        pred = model.predict(img_np, verbose=0)[0]
        return {LETTERS[i]: float(pred[i]) for i in range(config.num_classes)}

    return predict_letter


def build_demo(model: keras.Model, config: Config) -> gr.Interface:
    return gr.Interface(
        fn=build_predict_fn(model, config),
        inputs=gr.Sketchpad(label="Draw a lowercase letter here"),
        outputs="label",
        title="Handwriting Recognition CNN Demo",
        description="Draw a lowercase English letter. The CNN returns predicted class probabilities.",
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    ensure_directories(config)
    set_reproducibility(config.random_seed)
    arrays = preprocess_and_cache(config, force_rebuild_cache=args.force_rebuild_cache)
    model, history = train_or_load_model(config, arrays, force_retrain=args.force_retrain)
    metrics = compute_metrics(model, arrays)
    save_results(config, history, metrics)
    logger.info("Test accuracy: %s", f"{metrics['test_accuracy']:.4f}")
    logger.info("Results written to: %s", config.results_dir.as_posix())
    if config.launch_demo:
        build_demo(model=model, config=config).launch(share=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    main()
