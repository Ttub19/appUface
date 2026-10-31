"""
Tests for the face-shape predictor plumbing that don't need TensorFlow:
model-file assembly + checksum, image decoding, and the shipped model files.
Run: python -m unittest discover -s tests
"""
import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from app.services import face_shape_predictor as fsp


class ModelFileAssembly(unittest.TestCase):
    def _parts(self, d: Path, data: bytes, checksum: str):
        for i in range(0, len(data), 7):
            (d / f"{fsp.MODEL_FILE}.part{i // 7:02d}").write_bytes(data[i:i + 7])
        (d / f"{fsp.MODEL_FILE}.sha256").write_text(checksum + "\n")

    def test_parts_are_joined_once_and_verified(self):
        data = bytes(range(50))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._parts(d, data, hashlib.sha256(data).hexdigest())
            path = fsp.ensure_model_file(d)
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(fsp.ensure_model_file(d), path)  # reused

    def test_corrupt_parts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._parts(d, b"not the model", "0" * 64)
            with self.assertRaises(fsp.ModelUnavailableError):
                fsp.ensure_model_file(d)
            self.assertFalse((d / fsp.MODEL_FILE).exists())

    def test_missing_model_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(fsp.ModelUnavailableError):
                fsp.ensure_model_file(Path(tmp))


class ShippedModelFiles(unittest.TestCase):
    def test_listed_models_match_checksum_and_metadata(self):
        d = fsp.MODELS_DIR
        meta = json.loads((d / fsp.CLASS_FILE).read_text(encoding="utf-8"))
        self.assertEqual(meta["class_names"], ["heart", "oblong", "oval", "round", "square"])
        self.assertEqual(set(meta["class_names"]), set(fsp.FACE_SHAPES))
        self.assertEqual(meta["face_crop"]["version"], 2)
        files = [f for v in (meta.get("models") or {}).values() for f in ([v] if isinstance(v, str) else v)]
        self.assertTrue(files, "class_names.json lists the models")
        for name in files:
            parts = sorted(d.glob(f"{name}.part*"))
            self.assertTrue(parts, f"{name} parts are in backend/models")
            h = hashlib.sha256()
            for p in parts:
                h.update(p.read_bytes())
            self.assertEqual(h.hexdigest(), (d / f"{name}.sha256").read_text().strip(), name)


class DecodeImage(unittest.TestCase):
    def test_data_url_and_bare_base64(self):
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (30, 20), (200, 10, 10)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        for value in (f"data:image/png;base64,{b64}", b64):
            img = fsp.decode_data_url(value)
            self.assertEqual(img.size, (30, 20))
            self.assertEqual(img.mode, "RGB")

    def test_garbage_is_a_value_error(self):
        for bad in ("", "short", "data:image/png;base64," + base64.b64encode(b"not an image at all").decode()):
            with self.assertRaises(ValueError):
                fsp.decode_data_url(bad)


class MockPredictor(unittest.TestCase):
    def test_mock_returns_known_shape(self):
        r = fsp.MockFaceShapePredictor().predict("data:image/png;base64,xxxxxxxxxxxx")
        self.assertIn(r.face_shape, fsp.FACE_SHAPES)


if __name__ == "__main__":
    unittest.main()
