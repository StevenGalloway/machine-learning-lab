"""
Tests for case-studies/convolutional-models/handwriting-recognition/
         scripts/handwriting_recognition_cnn.py

Heavy dependencies (tensorflow, keras, gradio) are optional.
All tests that require them are skipped if the packages are not installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if tensorflow or gradio is unavailable
# ---------------------------------------------------------------------------
tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")
gr = pytest.importorskip("gradio", reason="gradio not installed")

from tests.conftest import load_module

_MOD = load_module(
    "handwriting_cnn",
    "case-studies/convolutional-models/handwriting-recognition/"
    "scripts/handwriting_recognition_cnn.py",
)

Config = _MOD.Config
build_config = _MOD.build_config
save_json = _MOD.save_json
prepare_image_for_inference = _MOD.prepare_image_for_inference
ensure_directories = _MOD.ensure_directories
LETTERS = _MOD.LETTERS


# ---------------------------------------------------------------------------
# Config defaults and properties
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_image_size(self):
        assert Config().image_size == 28

    def test_default_num_classes(self):
        assert Config().num_classes == 26

    def test_default_launch_demo_is_true(self):
        assert Config().launch_demo is True

    def test_data_dir_property(self):
        cfg = Config()
        assert cfg.data_dir == cfg.project_root / "data"

    def test_cache_dir_property(self):
        cfg = Config()
        assert cfg.cache_dir == cfg.data_dir / "cache"

    def test_models_dir_property(self):
        cfg = Config()
        assert cfg.models_dir == cfg.data_dir / "models"

    def test_results_dir_property(self):
        cfg = Config()
        assert cfg.results_dir == cfg.project_root / "results"

    def test_train_path_property(self):
        cfg = Config()
        assert cfg.train_path.name == cfg.train_file_name

    def test_cache_path_property(self):
        cfg = Config()
        assert cfg.cache_path.name == cfg.cache_file_name


# ---------------------------------------------------------------------------
# build_config from argparse namespace
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def _ns(self, **kwargs):
        defaults = {
            "epochs": None,
            "batch_size": None,
            "skip_demo": False,
            "force_rebuild_cache": False,
            "force_retrain": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_returns_config(self):
        assert isinstance(build_config(self._ns()), Config)

    def test_skip_demo_disables_launch(self):
        cfg = build_config(self._ns(skip_demo=True))
        assert cfg.launch_demo is False

    def test_override_epochs(self):
        cfg = build_config(self._ns(epochs=10))
        assert cfg.epochs == 10

    def test_override_batch_size(self):
        cfg = build_config(self._ns(batch_size=64))
        assert cfg.batch_size == 64

    def test_no_overrides_preserves_defaults(self):
        base = Config()
        cfg = build_config(self._ns())
        assert cfg.epochs == base.epochs
        assert cfg.batch_size == base.batch_size


# ---------------------------------------------------------------------------
# save_json  (handles numpy types)
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "out.json"
        save_json(p, {"a": 1})
        assert p.exists()

    def test_numpy_integer_serialised(self, tmp_path):
        p = tmp_path / "np_int.json"
        save_json(p, {"val": np.int64(42)})
        loaded = json.loads(p.read_text())
        assert loaded["val"] == 42
        assert isinstance(loaded["val"], int)

    def test_numpy_float_serialised(self, tmp_path):
        p = tmp_path / "np_float.json"
        save_json(p, {"val": np.float32(3.14)})
        loaded = json.loads(p.read_text())
        assert loaded["val"] == pytest.approx(3.14, abs=1e-3)

    def test_numpy_array_serialised(self, tmp_path):
        p = tmp_path / "arr.json"
        save_json(p, {"arr": np.array([1, 2, 3])})
        loaded = json.loads(p.read_text())
        assert loaded["arr"] == [1, 2, 3]

    def test_numpy_bool_serialised(self, tmp_path):
        p = tmp_path / "bool.json"
        save_json(p, {"flag": np.bool_(True)})
        loaded = json.loads(p.read_text())
        assert loaded["flag"] is True

    def test_nested_dict_serialised(self, tmp_path):
        p = tmp_path / "nested.json"
        payload = {"outer": {"inner": np.int64(7), "arr": np.array([1.0, 2.0])}}
        save_json(p, payload)
        loaded = json.loads(p.read_text())
        assert loaded["outer"]["inner"] == 7
        assert loaded["outer"]["arr"] == [1.0, 2.0]


# ---------------------------------------------------------------------------
# prepare_image_for_inference
# ---------------------------------------------------------------------------

class TestPrepareImageForInference:
    def _cfg(self):
        return Config()

    def test_raises_on_none_input(self):
        with pytest.raises(ValueError, match="No image"):
            prepare_image_for_inference(None, self._cfg())

    def test_accepts_numpy_array(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (28, 28, 1), dtype=np.uint8)
        cfg = self._cfg()
        result = prepare_image_for_inference(img, cfg)
        assert result.shape == (1, cfg.image_size, cfg.image_size, 1)

    def test_output_is_float32(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (28, 28), dtype=np.uint8)
        result = prepare_image_for_inference(img, self._cfg())
        assert result.dtype == np.float32

    def test_output_values_in_zero_one(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 255, (28, 28), dtype=np.uint8)
        result = prepare_image_for_inference(img, self._cfg())
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dict_input_with_composite(self):
        rng = np.random.default_rng(3)
        composite = rng.integers(0, 255, (28, 28), dtype=np.uint8)
        img = {"composite": composite}
        cfg = self._cfg()
        result = prepare_image_for_inference(img, cfg)
        assert result.shape == (1, cfg.image_size, cfg.image_size, 1)

    def test_dict_input_with_background(self):
        rng = np.random.default_rng(4)
        bg = rng.integers(0, 255, (28, 28), dtype=np.uint8)
        img = {"background": bg}
        cfg = self._cfg()
        result = prepare_image_for_inference(img, cfg)
        assert result.shape == (1, cfg.image_size, cfg.image_size, 1)

    def test_dict_input_with_no_usable_data_raises(self):
        with pytest.raises(ValueError, match="No usable image data"):
            prepare_image_for_inference({"other": None}, self._cfg())


# ---------------------------------------------------------------------------
# LETTERS constant
# ---------------------------------------------------------------------------

class TestLetters:
    def test_has_26_letters(self):
        assert len(LETTERS) == 26

    def test_is_lowercase(self):
        assert LETTERS == LETTERS.lower()

    def test_starts_with_a(self):
        assert LETTERS[0] == "a"

    def test_ends_with_z(self):
        assert LETTERS[-1] == "z"


# ---------------------------------------------------------------------------
# ensure_directories
# ---------------------------------------------------------------------------

class TestEnsureDirectories:
    def test_creates_required_dirs(self, tmp_path):
        from dataclasses import asdict
        base = Config()
        data = asdict(base)
        data["project_root"] = tmp_path
        cfg = Config(**data)
        ensure_directories(cfg)
        assert cfg.cache_dir.exists()
        assert cfg.models_dir.exists()
        assert cfg.results_dir.exists()
