"""
Gender-specific face-shape models (training v3): which model file answers a
request, the ResNet50 preprocessing, and the loaded-model limit. No TensorFlow
needed - keras is replaced by a stub.
Run: python -m unittest discover -s tests
"""
import json
import os
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
            self.assertEqual(specs["default"], {"files": (fsp.MODEL_FILE,), "img_size": (240, 240),
                                                "preprocess": "none", "tta": False})
            self.assertEqual(fsp.pick_spec_key(specs, "male"), "default")
            self.assertEqual(fsp.pick_spec_key(specs, None), "default")

    def test_gender_models_with_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, "face_shape_female.keras", f"{fsp.MODEL_FILE}.part00")   # male not trained / not copied
            specs = fsp.resolve_model_specs(V3_META, d)
            self.assertEqual(specs["female"], {"files": ("face_shape_female.keras",), "img_size": (224, 224),
                                               "preprocess": "resnet50", "tta": False})
            self.assertNotIn("male", specs)
            # the old EfficientNet keeps its own input size / preprocessing
            self.assertEqual(specs["default"], {"files": (fsp.MODEL_FILE,), "img_size": (240, 240),
                                                "preprocess": "none", "tta": False})
            self.assertEqual(fsp.pick_spec_key(specs, "female"), "female")
            self.assertEqual(fsp.pick_spec_key(specs, "male"), "default")

    def test_gender_models_without_legacy(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, "face_shape_female.keras.part00", "face_shape_male.keras")
            specs = fsp.resolve_model_specs(V3_META, d)
            self.assertEqual(set(specs), {"female", "male", "default"})
            self.assertEqual(specs["default"]["files"], ("face_shape_female.keras",))
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
        stub.models = types.SimpleNamespace(load_model=self._load)   # used by fsp.KerasModel
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
            self.assertEqual([m.name for m in p._get_model("female")], ["face_shape_female.keras"])
            self.assertIs(p._get_model("default"), p._get_model("female"))   # same file, loaded once
            self.assertEqual(self.loaded, ["face_shape_female.keras"])
            p._get_model("male")                                            # limit 1 -> female unloaded
            self.assertEqual(set(p._models), {"male"})
            p._get_model("female")
            self.assertEqual(self.loaded, ["face_shape_female.keras", "face_shape_male.keras", "face_shape_female.keras"])


ENSEMBLE_META = {**V3_META, "tta": "flip",
                 "models": {g: [f"face_shape_{g}_fold{k}.tflite" for k in (1, 2, 3)] for g in ("female", "male")}}


class FakeTFLite:
    """Stands in for fsp.TFLiteModel: returns fixed probabilities per row, records the batch it saw."""
    def __init__(self, path, rows):
        self.name, self.rows, self.seen = Path(path).name, rows, []

    def predict(self, batch):
        self.seen.append(batch.shape)
        return np.array(self.rows[: len(batch)], dtype="float32")


class Ensemble(unittest.TestCase):
    def test_lists_become_ensembles_with_tta(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, *[f for fs in ENSEMBLE_META["models"].values() for f in fs])
            specs = fsp.resolve_model_specs(ENSEMBLE_META, d)
            self.assertEqual(specs["male"]["files"], tuple(f"face_shape_male_fold{k}.tflite" for k in (1, 2, 3)))
            self.assertTrue(specs["male"]["tta"])
            self.assertEqual(specs["default"], specs["female"])        # both genders present: no legacy model

    def test_env_limits_ensemble_and_tta(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, *[f for fs in ENSEMBLE_META["models"].values() for f in fs])
            old = {k: os.environ.get(k) for k in ("FACE_SHAPE_ENSEMBLE", "FACE_SHAPE_TTA")}
            os.environ.update(FACE_SHAPE_ENSEMBLE="1", FACE_SHAPE_TTA="0")
            try:
                specs = fsp.resolve_model_specs(ENSEMBLE_META, d)
            finally:
                for k, v in old.items():
                    os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
            self.assertEqual(specs["female"]["files"], ("face_shape_female_fold1.tflite",))
            self.assertFalse(specs["female"]["tta"])

    def test_missing_fold_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, "face_shape_female_fold1.tflite", "face_shape_female_fold3.tflite")
            specs = fsp.resolve_model_specs(ENSEMBLE_META, d)
            self.assertEqual(specs["female"]["files"], ("face_shape_female_fold1.tflite", "face_shape_female_fold3.tflite"))
            self.assertNotIn("male", specs)

    def test_average_over_models_and_mirrored_rows(self):
        a = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.2, 0.5, 0.1, 0.1, 0.1]])   # one model: original + mirror
        b = np.array([[0.2, 0.2, 0.4, 0.1, 0.1], [0.2, 0.2, 0.4, 0.1, 0.1]])
        np.testing.assert_allclose(fsp.average_probs([a, b]), [0.3, 0.25, 0.25, 0.1, 0.1])

    def test_predict_returns_ranking_from_ensemble(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            touch(d, *ENSEMBLE_META["models"]["male"])
            rows = {1: [[.1, .2, .5, .1, .1]] * 2, 2: [[.1, .5, .2, .1, .1]] * 2, 3: [[.1, .3, .4, .1, .1]] * 2}
            made = []

            def fake_loader(path):
                k = int(Path(path).stem[-1])
                made.append(FakeTFLite(path, rows[k]))
                return made[-1]

            p = fsp.KerasFaceShapePredictor(d, max_loaded=2)
            p._specs = fsp.resolve_model_specs(ENSEMBLE_META, d)
            p._class_names = ENSEMBLE_META["class_names"]
            p._ready = True
            p._preprocess = lambda image, spec: np.zeros((2 if spec["tta"] else 1, 224, 224, 3), "float32")
            old_loader, old_decode = fsp.load_model_file, fsp.decode_data_url
            fsp.load_model_file, fsp.decode_data_url = fake_loader, lambda url: object()
            try:
                r = p.predict("data:image/png;base64,xxxxxxxxxxxx", "male")
            finally:
                fsp.load_model_file, fsp.decode_data_url = old_loader, old_decode
            self.assertEqual(len(made), 3)
            self.assertTrue(all(s == (2, 224, 224, 3) for m in made for s in m.seen))   # original + mirror
            self.assertEqual((r.face_shape, r.confidence, r.model), ("oval", 36.7, "male"))
            self.assertEqual([x["face_shape"] for x in r.ranking[:2]], ["oval", "oblong"])
            self.assertEqual(r.ranking[1]["confidence"], 33.3)
            self.assertAlmostEqual(sum(x["confidence"] for x in r.ranking), 100, delta=0.3)


if __name__ == "__main__":
    unittest.main()
