"""
Gender-specific face-shape models (training v3): which model file answers a
request, the ResNet50 preprocessing, and the loaded-model limit. No TensorFlow
needed - keras is replaced by a stub.
Run: python -m unittest discover -s tests
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from app.services import face_shape_predictor as fsp

V3_META = {"class_names": ["heart", "oblong", "oval", "round", "square"], "img_size": [224, 224],
           "preprocess": "resnet50", "models": {"female": "face_shape_female.keras", "male": "face_shape_male.keras"}}


def touch(d: Path, *names):
    for n in names:
        (d / n).write_bytes(b"x")


class ResolveSpecs(unittest.TestCase):
    def test_legacy_meta_uses_single_model(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, f"{fsp.MODEL_FILE}.part00")
            specs = fsp.resolve_model_specs({"class_names": [], "img_size": [240, 240]}, d)
            self.assertEqual(list(specs), ["default"])
            self.assertEqual(specs["default"], {"file": fsp.MODEL_FILE, "img_size": (240, 240), "preprocess": "none"})
            self.assertEqual(fsp.pick_spec_key(specs, "male"), "default")
            self.assertEqual(fsp.pick_spec_key(specs, None), "default")

    def test_gender_models_with_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, "face_shape_female.keras", f"{fsp.MODEL_FILE}.part00")   # male not trained / not copied
            specs = fsp.resolve_model_specs(V3_META, d)
            self.assertEqual(specs["female"], {"file": "face_shape_female.keras", "img_size": (224, 224), "preprocess": "resnet50"})
            self.assertNotIn("male", specs)
            # the old EfficientNet keeps its own input size / preprocessing
            self.assertEqual(specs["default"], {"file": fsp.MODEL_FILE, "img_size": (240, 240), "preprocess": "none"})
            self.assertEqual(fsp.pick_spec_key(specs, "female"), "female")
            self.assertEqual(fsp.pick_spec_key(specs, "male"), "default")

    def test_gender_models_without_legacy(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, "face_shape_female.keras.part00", "face_shape_male.keras")
            specs = fsp.resolve_model_specs(V3_META, d)
            self.assertEqual(set(specs), {"female", "male", "default"})
            self.assertEqual(specs["default"]["file"], "face_shape_female.keras")
            self.assertEqual(fsp.pick_spec_key(specs, "male"), "male")
            self.assertEqual(fsp.pick_spec_key(specs, "other"), "default")

    def test_nothing_on_disk(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(fsp.ModelUnavailableError):
                fsp.resolve_model_specs(V3_META, Path(t))


class Preprocess(unittest.TestCase):
    def test_resnet50_matches_keras_caffe_mode(self):
        arr = np.array([[[[10.0, 20.0, 30.0]]]], dtype="float32")     # RGB
        out = fsp.apply_preprocess(arr, "resnet50")
        np.testing.assert_allclose(out[0, 0, 0], [30 - 103.939, 20 - 116.779, 10 - 123.68], rtol=1e-6)
        self.assertEqual(arr[0, 0, 0, 0], 10.0)        # input not modified

    def test_none_is_identity_and_unknown_fails(self):
        arr = np.ones((1, 2, 2, 3), dtype="float32")
        self.assertIs(fsp.apply_preprocess(arr, "none"), arr)
        with self.assertRaises(fsp.ModelUnavailableError):
            fsp.apply_preprocess(arr, "vgg")


class LoadedModelLimit(unittest.TestCase):
    def setUp(self):
        self.loaded = []
        stub = types.ModuleType("keras")
        stub.models = types.SimpleNamespace(load_model=self._load)
        self._old = sys.modules.get("keras")
        sys.modules["keras"] = stub

    def tearDown(self):
        if self._old is None:
            sys.modules.pop("keras", None)
        else:
            sys.modules["keras"] = self._old

    def _load(self, path, compile=False):
        self.loaded.append(Path(path).name)
        return types.SimpleNamespace(predict=lambda x, verbose=0: np.zeros((1, 5)), name=Path(path).name)

    def test_lru_and_shared_default(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, "face_shape_female.keras", "face_shape_male.keras")
            p = fsp.KerasFaceShapePredictor(d, max_loaded=1)
            p._specs = fsp.resolve_model_specs(V3_META, d)
            self.assertEqual(p._get_model("female").name, "face_shape_female.keras")
            self.assertIs(p._get_model("default"), p._get_model("female"))   # same file, loaded once
            self.assertEqual(self.loaded, ["face_shape_female.keras"])
            p._get_model("male")                                            # limit 1 -> female unloaded
            self.assertEqual(set(p._models), {"male"})
            p._get_model("female")
            self.assertEqual(self.loaded, ["face_shape_female.keras", "face_shape_male.keras", "face_shape_female.keras"])


if __name__ == "__main__":
    unittest.main()
