"""
Face-shape prediction service.

`FaceShapePredictor` is the interface the rest of the app depends on.

`KerasFaceShapePredictor` runs the trained classifier(s) with exactly the
preprocessing they were trained with:

  1. decode the data-URL image, apply EXIF rotation;
  2. MediaPipe face landmarks -> upright square face crop (face_crop.py, the
     same module the Kaggle training script used);
  3. JPEG round-trip (training read the crops back from JPEG files);
  4. resize_with_pad to the model's input size + round to 0-255, then the
     backbone's own preprocessing ("none" for EfficientNet, "resnet50" =
     RGB->BGR + ImageNet mean subtraction) - image_ops.py, plain NumPy;
  5. softmax over [heart, oblong, oval, round, square].

Models per gender (training v3): class_names.json lists, per gender, one model
file or a list of them, e.g.
{"models": {"female": ["face_shape_female_fold1.tflite", ...3 folds], ...}, "tta": "flip"}.
With a list the server averages the probabilities of every model (ensemble of
the K-fold models); with "tta": "flip" each model also sees the mirrored crop
(test-time augmentation). .tflite files run on LiteRT (ai-edge-litert, ~10 MB,
no TensorFlow); .keras files still work when TensorFlow is installed.

A gender without its own model - or an old class_names.json without "models" -
falls back to the original EfficientNetB1 model (face_shape_effnetb1.keras,
80.4% on the 1,000-image test set, needs TensorFlow) when its files are present.

Env:
  FACE_SHAPE_MAX_MODELS  how many gender models (each = its whole ensemble) stay in RAM (default 2)
  FACE_SHAPE_ENSEMBLE    use only the first N models of each ensemble (default: all) - faster on a slow CPU
  FACE_SHAPE_TTA         "0" turns the mirrored second pass off (default: as class_names.json says)

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
from typing import NamedTuple, Optional, Sequence

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
CLASS_FILE = "class_names.json"
LANDMARKER_FILE = "face_landmarker.task"
MAX_IMAGE_BYTES = 12 * 1024 * 1024

NO_FACE_MESSAGE = "ไม่พบใบหน้าในรูป กรุณาใช้รูปหน้าตรงที่เห็นใบหน้าชัดเจน ไม่มีอะไรบัง"


class PredictionResult(NamedTuple):
    face_shape: str
    face_shape_name: str
    confidence: float
    model: str = "default"   # which model answered: "female" / "male" / "default"
    ranking: tuple = ()      # every shape, most likely first: ({"face_shape", "face_shape_name", "confidence"}, ...)


def build_ranking(class_names: Sequence[str], probs) -> tuple:
    order = sorted(range(len(class_names)), key=lambda i: -float(probs[i]))
    return tuple({"face_shape": class_names[i],
                  "face_shape_name": FACE_SHAPES.get(class_names[i], class_names[i]),
                  "confidence": round(float(probs[i]) * 100, 1)} for i in order)


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
        names = list(FACE_SHAPES.keys())
        probs = [random.random() for _ in names]
        probs = [p / sum(probs) for p in probs]
        ranking = build_ranking(names, probs)
        top = ranking[0]
        return PredictionResult(top["face_shape"], top["face_shape_name"], top["confidence"], ranking=ranking)


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
    tmp = target.parent / (target.name + ".tmp")
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
    spec = {"files": (file, ...), "img_size", "preprocess", "tta"}: more than one file = an ensemble
    (probabilities averaged). Genders without their own model use "default".
    """
    specs = {}
    img_size = tuple(meta.get("img_size", LEGACY_SPEC["img_size"]))
    preprocess = meta.get("preprocess", "none")
    tta = str(meta.get("tta", "")).lower() == "flip"
    env_tta = os.getenv("FACE_SHAPE_TTA", "").strip()
    if env_tta:
        tta = tta and env_tta not in ("0", "false", "no", "off")
    limit = int(os.getenv("FACE_SHAPE_ENSEMBLE", "0") or 0)
    for key, files in (meta.get("models") or {}).items():
        files = [files] if isinstance(files, str) else list(files)
        files = [f for f in files if model_available(models_dir, f)]
        if limit > 0:
            files = files[:limit]
        if files:
            specs[key] = {"files": tuple(files), "img_size": img_size, "preprocess": preprocess, "tta": tta}
    legacy = {"files": (MODEL_FILE,), **LEGACY_SPEC, "tta": False}
    if "default" not in specs:
        complete = all(g in specs for g in GENDERS)
        if not meta.get("models") and model_available(models_dir, MODEL_FILE):
            specs["default"] = {"files": (MODEL_FILE,), "img_size": img_size, "preprocess": preprocess, "tta": tta}
        elif not complete and model_available(models_dir, MODEL_FILE):   # a gender is missing: old model covers it
            specs["default"] = legacy
        elif specs:                                            # every gender has a model: female answers "no gender"
            specs["default"] = specs.get("female") or next(iter(specs.values()))
    if not specs:
        raise ModelUnavailableError(f"ไม่พบไฟล์โมเดลใน {models_dir}")
    return specs


def pick_spec_key(specs: dict, gender: Optional[str]) -> str:
    return gender if gender in specs and gender in GENDERS else "default"


def apply_preprocess(arr, name: str):
    """arr: (..., H, W, 3) float32 RGB 0-255 -> what the backbone was trained on."""
    from app.services import image_ops

    try:
        return image_ops.apply_preprocess(arr, name)
    except ValueError as exc:
        raise ModelUnavailableError(f"ไม่รู้จัก preprocess แบบ {name!r} ใน {CLASS_FILE}") from exc


class TFLiteModel:
    """One .tflite classifier on LiteRT. predict(batch (n, h, w, 3) float32) -> (n, classes)."""

    def __init__(self, path: Path, threads: Optional[int] = None):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                try:
                    from tensorflow.lite import Interpreter
                except ImportError as exc:
                    raise ModelUnavailableError(
                        "เซิร์ฟเวอร์ยังไม่ได้ติดตั้ง LiteRT — รัน pip install -r requirements.txt") from exc
        self.name = path.name
        self._interp = Interpreter(model_path=str(path), num_threads=threads or int(os.getenv("FACE_SHAPE_THREADS", "1")))
        self._in = self._interp.get_input_details()[0]["index"]
        self._out = self._interp.get_output_details()[0]["index"]
        self._batch = None

    def predict(self, batch):
        import numpy as np

        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if self._batch != batch.shape:
            self._interp.resize_tensor_input(self._in, list(batch.shape))
            self._interp.allocate_tensors()
            self._batch = batch.shape
        self._interp.set_tensor(self._in, batch)
        self._interp.invoke()
        return np.array(self._interp.get_tensor(self._out))


class KerasModel:
    """A .keras classifier (needs TensorFlow) with the same predict() as TFLiteModel."""

    def __init__(self, path: Path):
        try:
            import keras
        except ImportError as exc:
            raise ModelUnavailableError(
                f"{path.name} ต้องใช้ TensorFlow ซึ่งเซิร์ฟเวอร์นี้ไม่ได้ติดตั้ง (ใช้ไฟล์ .tflite แทน)") from exc
        self.name = path.name
        # compile=False: the file stores a custom training loss we don't need for inference
        self._model = keras.models.load_model(path, compile=False)

    def predict(self, batch):
        import numpy as np

        return np.asarray(self._model.predict(batch, verbose=0))


def load_model_file(path: Path):
    return TFLiteModel(path) if path.suffix == ".tflite" else KerasModel(path)


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
    """The trained classifier(s) (.tflite or .keras). Load lazily (first request or warm_up()); thread-safe."""

    def __init__(self, models_dir: Path = MODELS_DIR, max_loaded: Optional[int] = None):
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._ready = False
        self._detector = None
        self._class_names = None
        self._crop_params = None
        self._specs = {}
        self._models = {}           # spec key -> list of loaded models (most recently used last)
        # each gender model (its whole ensemble) costs RAM; Render free has 512 MB
        self._max_loaded = max_loaded or int(os.getenv("FACE_SHAPE_MAX_MODELS", "2"))

    def warm_up(self) -> None:
        with self._lock:
            self._prepare()
            for key in list(self._specs)[: self._max_loaded]:
                self._get_model(key)

    def _prepare(self) -> None:
        if self._ready:
            return
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
        logger.info("Face-shape models available: %s",
                    {k: (list(v["files"]), "tta" if v["tta"] else "") for k, v in self._specs.items()})

    def _get_model(self, key: str):
        """The loaded model(s) for a spec key: a list (one entry per ensemble member)."""
        import numpy as np

        if key in self._models:
            self._models[key] = self._models.pop(key)          # mark as most recently used
            return self._models[key]
        spec = self._specs[key]
        same_files = [k for k in self._models if self._specs[k]["files"] == spec["files"]]
        if same_files:                                         # e.g. "default" is the female model
            self._models[key] = self._models[same_files[0]]
            return self._models[key]
        while len({self._specs[k]["files"] for k in self._models}) >= self._max_loaded:
            evicted = next(iter(self._models))
            dropped = self._models.pop(evicted)
            if not any(m is dropped for m in self._models.values()):
                logger.info("Unloaded face-shape model %s (FACE_SHAPE_MAX_MODELS=%d)", evicted, self._max_loaded)
        models = []
        for filename in spec["files"]:
            model = load_model_file(ensure_model_file(self.models_dir, filename))
            # first call builds the graph / allocates tensors; do it now so the user doesn't wait twice
            model.predict(np.zeros((2 if spec["tta"] else 1, *spec["img_size"], 3), dtype="float32"))
            models.append(model)
        self._models[key] = models
        logger.info("Face-shape model ready: %s -> %s%s", key, list(spec["files"]), " + flip TTA" if spec["tta"] else "")
        return models

    def _preprocess(self, image, spec):
        """PIL RGB -> (1 or 2, H, W, 3) float32, same steps as training. Raises ValueError when there's no face."""
        import numpy as np
        from PIL import Image
        from app.services import face_crop, image_ops

        crop, info = face_crop.crop_face(image, self._detector, self._crop_params)
        if info["status"] == "fallback":
            raise ValueError(NO_FACE_MESSAGE)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=95)  # training read its crops back from JPEG
        buf.seek(0)
        with Image.open(buf) as im:
            arr = np.asarray(im.convert("RGB"), dtype="float32")
        if spec["preprocess"] not in ("resnet50", "none", "", None):
            apply_preprocess(arr, spec["preprocess"])          # raises ModelUnavailableError
        return image_ops.prepare_batch(arr, spec["img_size"], spec["preprocess"], spec["tta"])

    def predict(self, image_data_url: str, gender: Optional[str] = None) -> PredictionResult:
        import numpy as np

        image = decode_data_url(image_data_url)
        with self._lock:
            self._prepare()
            key = pick_spec_key(self._specs, gender)
            spec = self._specs[key]
            models = self._get_model(key)
            batch = self._preprocess(image, spec)
            probs = average_probs([m.predict(batch) for m in models])
        ranking = build_ranking(self._class_names, probs)
        top = ranking[0]
        return PredictionResult(
            face_shape=top["face_shape"],
            face_shape_name=top["face_shape_name"],
            confidence=top["confidence"],
            model=key,
            ranking=ranking,
        )


def average_probs(outputs) -> "list[float]":
    """Mean over every model and every batch row (original + mirrored crop) -> one probability vector."""
    import numpy as np

    stacked = np.concatenate([np.asarray(o, dtype="float64").reshape(-1, np.shape(o)[-1]) for o in outputs])
    return stacked.mean(axis=0).tolist()


def _make_predictor() -> FaceShapePredictor:
    if os.getenv("FACE_SHAPE_PREDICTOR", "model").strip().lower() == "mock":
        logger.warning("FACE_SHAPE_PREDICTOR=mock — face-shape results are RANDOM (development only)")
        return MockFaceShapePredictor()
    return KerasFaceShapePredictor()


_predictor_instance = _make_predictor()


def get_predictor() -> FaceShapePredictor:
    """FastAPI dependency-friendly accessor for the active predictor implementation."""
    return _predictor_instance
