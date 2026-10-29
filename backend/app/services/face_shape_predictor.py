"""
Face-shape prediction service.

`FaceShapePredictor` is the interface the rest of the app depends on.

`KerasFaceShapePredictor` runs the trained classifier(s) with exactly the
preprocessing they were trained with:

  1. decode the data-URL image, apply EXIF rotation;
  2. MediaPipe face landmarks -> upright square face crop (face_crop.py, the
     same module the Kaggle training script used);
  3. JPEG round-trip (training read the crops back from JPEG files);
  4. tf.image.resize_with_pad to the model's input size, float 0-255, then the
     backbone's own preprocessing ("none" for EfficientNet, "resnet50" =
     RGB->BGR + ImageNet mean subtraction);
  5. softmax over [heart, oblong, oval, round, square].

Models per gender (training v3): class_names.json lists
{"models": {"female": "face_shape_female.keras", "male": "face_shape_male.keras"}}
and the request's gender picks one. A gender without its own model - or an
old class_names.json without "models" - falls back to the original
EfficientNetB1 model (face_shape_effnetb1.keras, 80.4% on the 1,000-image
test set) when its files are present.

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
MODEL_FILE = "face_shape_effnetb1.keras"          # legacy single model (fallback)
LEGACY_SPEC = {"img_size": (240, 240), "preprocess": "none"}
GENDERS = ("female", "male")
IMAGENET_BGR_MEAN = (103.939, 116.779, 123.68)
CLASS_FILE = "class_names.json"
LANDMARKER_FILE = "face_landmarker.task"
MAX_IMAGE_BYTES = 12 * 1024 * 1024

NO_FACE_MESSAGE = "ไม่พบใบหน้าในรูป กรุณาใช้รูปหน้าตรงที่เห็นใบหน้าชัดเจน ไม่มีอะไรบัง"


class PredictionResult(NamedTuple):
    face_shape: str
    face_shape_name: str
    confidence: float
    model: str = "default"   # which model answered: "female" / "male" / "default"


class ModelUnavailableError(RuntimeError):
    """The real model can't run on this server (missing files or packages)."""


class FaceShapePredictor(ABC):
    @abstractmethod
    def predict(self, image_data_url: str, gender: Optional[str] = None) -> PredictionResult:
        """Runs face-shape classification on a base64 data-URL image (gender picks the model when there are several)."""
        raise NotImplementedError


class MockFaceShapePredictor(FaceShapePredictor):
    """Development stand-in: doesn't look at the image, returns a random shape."""

    def predict(self, image_data_url: str, gender: Optional[str] = None) -> PredictionResult:
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


def ensure_model_file(models_dir: Path = MODELS_DIR, filename: str = MODEL_FILE) -> Path:
    """
    The .keras file is stored as parts (<= 25 MB each, so it fits through
    file-size limits); join them once, check the SHA-256, then reuse the file.
    """
    target = models_dir / filename
    checksum_file = models_dir / f"{filename}.sha256"
    expected = checksum_file.read_text().strip() if checksum_file.exists() else None
    if target.exists() and (expected is None or _sha256(target) == expected):
        return target
    parts = sorted(models_dir.glob(f"{filename}.part*"))
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


def model_available(models_dir: Path, filename: str) -> bool:
    return (models_dir / filename).exists() or any(models_dir.glob(f"{filename}.part*"))


def resolve_model_specs(meta: dict, models_dir: Path = MODELS_DIR) -> dict:
    """
    {"female": spec, "male": spec, "default": spec} for the models actually on disk.
    spec = {"file", "img_size", "preprocess"}. Genders without their own model use "default".
    """
    specs = {}
    img_size = tuple(meta.get("img_size", LEGACY_SPEC["img_size"]))
    preprocess = meta.get("preprocess", "none")
    for key, filename in (meta.get("models") or {}).items():
        if model_available(models_dir, filename):
            specs[key] = {"file": filename, "img_size": img_size, "preprocess": preprocess}
    if "default" not in specs:
        if not meta.get("models") and model_available(models_dir, MODEL_FILE):
            specs["default"] = {"file": MODEL_FILE, "img_size": img_size, "preprocess": preprocess}
        elif model_available(models_dir, MODEL_FILE):          # new meta, keep the old model as fallback
            specs["default"] = {"file": MODEL_FILE, **LEGACY_SPEC}
        elif specs:                                            # no legacy model: any trained gender answers
            specs["default"] = specs.get("female") or next(iter(specs.values()))
    if not specs:
        raise ModelUnavailableError(f"ไม่พบไฟล์โมเดลใน {models_dir}")
    return specs


def pick_spec_key(specs: dict, gender: Optional[str]) -> str:
    return gender if gender in specs and gender in GENDERS else "default"


def apply_preprocess(arr, name: str):
    """arr: (..., H, W, 3) float32 RGB 0-255 -> what the backbone was trained on."""
    if name == "resnet50":   # keras.applications.resnet50.preprocess_input ("caffe" mode)
        out = arr[..., ::-1].copy()
        out -= IMAGENET_BGR_MEAN
        return out
    if name in ("none", "", None):
        return arr
    raise ModelUnavailableError(f"ไม่รู้จัก preprocess แบบ {name!r} ใน {CLASS_FILE}")


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
    """The trained classifier(s). Load lazily (first request or warm_up()) and are thread-safe."""

    def __init__(self, models_dir: Path = MODELS_DIR, max_loaded: Optional[int] = None):
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._ready = False
        self._detector = None
        self._class_names = None
        self._crop_params = None
        self._specs = {}
        self._models = {}           # spec key -> loaded keras model (most recently used last)
        # each extra model costs RAM (ResNet50 ~ 100 MB+); Render free has 512 MB
        self._max_loaded = max_loaded or int(os.getenv("FACE_SHAPE_MAX_MODELS", "2"))

    def warm_up(self) -> None:
        with self._lock:
            self._prepare()
            for key in list(self._specs)[: self._max_loaded]:
                self._get_model(key)

    def _prepare(self) -> None:
        if self._ready:
            return
        try:
            import keras  # noqa: F401
            import tensorflow as tf  # noqa: F401  (backend for keras)
        except ImportError as exc:
            raise ModelUnavailableError(
                "เซิร์ฟเวอร์ยังไม่ได้ติดตั้ง TensorFlow — รัน pip install -r requirements.txt") from exc
        from app.services import face_crop

        meta = json.loads((self.models_dir / CLASS_FILE).read_text(encoding="utf-8"))
        self._class_names = meta["class_names"]
        self._specs = resolve_model_specs(meta, self.models_dir)
        params = (meta.get("face_crop") or {}).get("params") or {}
        self._crop_params = face_crop.CropParams(**params)
        try:
            self._detector = face_crop.FaceLandmarkDetector(
                str(ensure_landmarker_file(self.models_dir) or ""), max_faces=3)
        except ImportError as exc:
            raise ModelUnavailableError("เซิร์ฟเวอร์ยังไม่ได้ติดตั้ง mediapipe — รัน pip install -r requirements.txt") from exc
        except RuntimeError as exc:
            raise ModelUnavailableError(
                f"โหลดตัวหาใบหน้าไม่สำเร็จ: ต่ออินเทอร์เน็ตครั้งแรกเพื่อดาวน์โหลด {LANDMARKER_FILE}") from exc
        self._ready = True
        logger.info("Face-shape models available: %s", {k: v["file"] for k, v in self._specs.items()})

    def _get_model(self, key: str):
        import keras
        import numpy as np

        if key in self._models:
            self._models[key] = self._models.pop(key)          # mark as most recently used
            return self._models[key]
        spec = self._specs[key]
        same_file = [k for k, m in self._models.items() if self._specs[k]["file"] == spec["file"]]
        if same_file:                                          # e.g. "default" is the female model
            self._models[key] = self._models[same_file[0]]
            return self._models[key]
        while len({self._specs[k]["file"] for k in self._models}) >= self._max_loaded:
            evicted = next(iter(self._models))
            self._models.pop(evicted)
            logger.info("Unloaded face-shape model %s (FACE_SHAPE_MAX_MODELS=%d)", evicted, self._max_loaded)
        # compile=False: the file stores a custom training loss we don't need for inference
        model = keras.models.load_model(ensure_model_file(self.models_dir, spec["file"]), compile=False)
        # first call builds the graph; do it now so the user doesn't wait twice
        model.predict(np.zeros((1, *spec["img_size"], 3), dtype="float32"), verbose=0)
        self._models[key] = model
        logger.info("Face-shape model ready: %s -> %s", key, spec["file"])
        return model

    def _preprocess(self, image, spec):
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
        h, w = spec["img_size"]
        arr = tf.image.resize_with_pad(arr, h, w).numpy()
        return apply_preprocess(arr, spec["preprocess"])[None, ...]

    def predict(self, image_data_url: str, gender: Optional[str] = None) -> PredictionResult:
        import numpy as np

        image = decode_data_url(image_data_url)
        with self._lock:
            self._prepare()
            key = pick_spec_key(self._specs, gender)
            spec = self._specs[key]
            model = self._get_model(key)
            batch = self._preprocess(image, spec)
            probs = np.asarray(model.predict(batch, verbose=0))[0]
        idx = int(np.argmax(probs))
        shape_id = self._class_names[idx]
        return PredictionResult(
            face_shape=shape_id,
            face_shape_name=FACE_SHAPES.get(shape_id, shape_id),
            confidence=round(float(probs[idx]) * 100, 1),
            model=key,
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
