"""
Face-shape prediction service.

`FaceShapePredictor` is the interface the rest of the app depends on.

`KerasFaceShapePredictor` runs the trained EfficientNetB1 classifier
(models/face_shape_effnetb1.keras, 80.4% on the 1,000-image test set) with
exactly the preprocessing it was trained with:

  1. decode the data-URL image, apply EXIF rotation;
  2. MediaPipe face landmarks -> upright square face crop (face_crop.py, the
     same module the Kaggle training script used);
  3. JPEG round-trip (training read the crops back from JPEG files);
  4. tf.image.resize_with_pad to 240x240, float 0-255 (EfficientNet's
     preprocess_input is the identity);
  5. softmax over [heart, oblong, oval, round, square].

`MockFaceShapePredictor` (random result) is kept for development without
TensorFlow: set FACE_SHAPE_PREDICTOR=mock.
"""
import base64
import binascii
import hashlib
import io
import json
import logging
import os
import random
import threading
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

FACE_SHAPES: dict[str, str] = {
    "oval": "หน้ารูปไข่ (Oval)",
    "round": "หน้ากลม (Round)",
    "square": "หน้าเหลี่ยม (Square)",
    "heart": "หน้ารูปหัวใจ (Heart)",
    "oblong": "หน้ายาว (Oblong)",
}

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_FILE = "face_shape_effnetb1.keras"
CLASS_FILE = "class_names.json"
LANDMARKER_FILE = "face_landmarker.task"
MAX_IMAGE_BYTES = 12 * 1024 * 1024

NO_FACE_MESSAGE = "ไม่พบใบหน้าในรูป กรุณาใช้รูปหน้าตรงที่เห็นใบหน้าชัดเจน ไม่มีอะไรบัง"


class PredictionResult(NamedTuple):
    face_shape: str
    face_shape_name: str
    confidence: float


class ModelUnavailableError(RuntimeError):
    """The real model can't run on this server (missing files or packages)."""


class FaceShapePredictor(ABC):
    @abstractmethod
    def predict(self, image_data_url: str) -> PredictionResult:
        """Runs face-shape classification on a base64 data-URL image."""
        raise NotImplementedError


class MockFaceShapePredictor(FaceShapePredictor):
    """Development stand-in: doesn't look at the image, returns a random shape."""

    def predict(self, image_data_url: str) -> PredictionResult:
        if not image_data_url or len(image_data_url) < 10:
            raise ValueError("ไม่มีรูปภาพสำหรับส่งวิเคราะห์")
        shape_id = random.choice(list(FACE_SHAPES.keys()))
        return PredictionResult(shape_id, FACE_SHAPES[shape_id], round(random.uniform(80, 95), 1))


def decode_data_url(image_data_url: str):
    """data:image/...;base64,.... (or bare base64) -> upright RGB PIL image."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not image_data_url or len(image_data_url) < 10:
        raise ValueError("ไม่มีรูปภาพสำหรับส่งวิเคราะห์")
    payload = image_data_url.split(",", 1)[1] if image_data_url.startswith("data:") else image_data_url
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("รูปภาพเสียหาย อ่านไม่ได้") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("รูปใหญ่เกินไป (เกิน 12 MB)")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("ไฟล์นี้ไม่ใช่รูปภาพที่รองรับ") from exc


def ensure_model_file(models_dir: Path = MODELS_DIR) -> Path:
    """
    The .keras file is stored as parts (<= 25 MB each, so it fits through
    file-size limits); join them once, check the SHA-256, then reuse the file.
    """
    target = models_dir / MODEL_FILE
    checksum_file = models_dir / f"{MODEL_FILE}.sha256"
    expected = checksum_file.read_text().strip() if checksum_file.exists() else None
    if target.exists() and (expected is None or _sha256(target) == expected):
        return target
    parts = sorted(models_dir.glob(f"{MODEL_FILE}.part*"))
    if not parts:
        raise ModelUnavailableError(f"ไม่พบไฟล์โมเดล {target}")
    tmp = target.with_suffix(".keras.tmp")
    with open(tmp, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    if expected and _sha256(tmp) != expected:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailableError("ไฟล์โมเดลไม่สมบูรณ์ (checksum ไม่ตรง) กรุณาคัดลอกไฟล์ใน backend/models ใหม่")
    os.replace(tmp, target)
    logger.info("Assembled %s from %d parts", target.name, len(parts))
    return target


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_landmarker_file(models_dir: Path = MODELS_DIR) -> Optional[Path]:
    """MediaPipe FaceLandmarker model (~3.6 MB): downloaded once on first use."""
    from app.services.face_crop import FACE_LANDMARKER_URL

    path = models_dir / LANDMARKER_FILE
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    try:
        tmp = path.with_suffix(".task.part")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)
        os.replace(tmp, path)
        logger.info("Downloaded %s", path.name)
        return path
    except Exception as exc:  # offline: the legacy face_mesh (older mediapipe) may still work
        logger.warning("Could not download %s (%s)", LANDMARKER_FILE, exc)
        return None


class KerasFaceShapePredictor(FaceShapePredictor):
    """The trained classifier. Loads lazily (first request or warm_up()) and is thread-safe."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._detector = None
        self._class_names = None
        self._img_size = (240, 240)
        self._crop_params = None

    def warm_up(self) -> None:
        with self._lock:
            self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import keras
            import numpy as np
            import tensorflow as tf  # noqa: F401  (backend for keras)
        except ImportError as exc:
            raise ModelUnavailableError(
                "เซิร์ฟเวอร์ยังไม่ได้ติดตั้ง TensorFlow — รัน pip install -r requirements.txt") from exc
        from app.services import face_crop

        meta = json.loads((self.models_dir / CLASS_FILE).read_text(encoding="utf-8"))
        self._class_names = meta["class_names"]
        self._img_size = tuple(meta.get("img_size", (240, 240)))
        params = (meta.get("face_crop") or {}).get("params") or {}
        self._crop_params = face_crop.CropParams(**params)

        # compile=False: the file stores a custom training loss we don't need for inference
        self._model = keras.models.load_model(ensure_model_file(self.models_dir), compile=False)
        try:
            self._detector = face_crop.FaceLandmarkDetector(
                str(ensure_landmarker_file(self.models_dir) or ""), max_faces=3)
        except ImportError as exc:
            raise ModelUnavailableError("เซิร์ฟเวอร์ยังไม่ได้ติดตั้ง mediapipe — รัน pip install -r requirements.txt") from exc
        except RuntimeError as exc:
            raise ModelUnavailableError(
                f"โหลดตัวหาใบหน้าไม่สำเร็จ: ต่ออินเทอร์เน็ตครั้งแรกเพื่อดาวน์โหลด {LANDMARKER_FILE}") from exc
        # first call builds the graph; do it now so the first user doesn't wait
        self._model.predict(np.zeros((1, *self._img_size, 3), dtype="float32"), verbose=0)
        self._loaded = True
        logger.info("Face-shape model ready (%s)", ", ".join(self._class_names))

    def _preprocess(self, image):
        """PIL RGB -> (1, H, W, 3) float32, same steps as training. Raises ValueError when there's no face."""
        import numpy as np
        import tensorflow as tf
        from PIL import Image
        from app.services import face_crop

        crop, info = face_crop.crop_face(image, self._detector, self._crop_params)
        if info["status"] == "fallback":
            raise ValueError(NO_FACE_MESSAGE)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=95)  # training read its crops back from JPEG
        buf.seek(0)
        with Image.open(buf) as im:
            arr = np.asarray(im.convert("RGB"), dtype="float32")
        arr = tf.image.resize_with_pad(arr, self._img_size[0], self._img_size[1]).numpy()
        return arr[None, ...]

    def predict(self, image_data_url: str) -> PredictionResult:
        import numpy as np

        image = decode_data_url(image_data_url)
        with self._lock:
            self._load()
            batch = self._preprocess(image)
            probs = np.asarray(self._model.predict(batch, verbose=0))[0]
        idx = int(np.argmax(probs))
        shape_id = self._class_names[idx]
        return PredictionResult(
            face_shape=shape_id,
            face_shape_name=FACE_SHAPES.get(shape_id, shape_id),
            confidence=round(float(probs[idx]) * 100, 1),
        )


def _make_predictor() -> FaceShapePredictor:
    if os.getenv("FACE_SHAPE_PREDICTOR", "model").strip().lower() == "mock":
        logger.warning("FACE_SHAPE_PREDICTOR=mock — face-shape results are RANDOM (development only)")
        return MockFaceShapePredictor()
    return KerasFaceShapePredictor()


_predictor_instance = _make_predictor()


def get_predictor() -> FaceShapePredictor:
    """FastAPI dependency-friendly accessor for the active predictor implementation."""
    return _predictor_instance
