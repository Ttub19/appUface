# =============================================================================
# uShape — face-shape training v3 (ตามคำแนะนำอาจารย์ 29 ต.ค.)
#   1. นับรูปแยกตามแหล่งข้อมูล (Kaggle / Roboflow) แล้วเลือกใช้ "แหล่งเดียว" ไม่รวมกัน
#   2. แยกรูปชาย / หญิง (ติดป้ายเพศอัตโนมัติด้วยโมเดล FairFace ViT + แก้มือได้ด้วย CSV)
#      แล้วเทรนโมเดลแยกเพศ: face_shape_female.keras / face_shape_male.keras
#   3. Data augmentation เพิ่มขึ้น รวม "เปลี่ยนสี" (hue / saturation / ขาวดำ)
#   4. Stratified Group K-Fold 3 รอบ -> รายงาน mean ± SD
#   5. Backbone เปลี่ยนเป็น ResNet50 (สลับกลับเป็น EfficientNetB1 ได้ที่ BACKBONE เพื่อเทียบผล)
#   6. ขั้นตอนนิ่ง (reproducible): seed ตายตัว, บันทึก split ทุก fold เป็น CSV, cache ทุกขั้น
#
# Kaggle: Settings -> Accelerator = GPU, Internet = On, แนบ dataset ที่มีโฟลเดอร์ datasetformodelai
# รันทั้งไฟล์ในเซลล์เดียว (หรือแบ่งเซลล์ตามหัวข้อ ### ก็ได้)
# =============================================================================
import os
import re
import sys
import csv
import json
import time
import subprocess
from collections import defaultdict, Counter
import numpy as np
import tensorflow as tf

# ให้ TensorFlow จอง GPU ทีละน้อย — ขั้นติดป้ายเพศ (PyTorch) ต้องใช้ GPU ก่อนเทรน
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

try:
    import imagehash
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=True)
    import imagehash
from PIL import Image, ImageDraw
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))

# =============================================================================
# ### CONFIG
# =============================================================================
# path ของ dataset บน Kaggle (ดูได้จากแผงขวา Input -> คลิกชื่อ dataset -> Copy path)
DATASET_PATH = "/kaggle/input/datasets/ttttt33t/faceshape1"


def find_data_root(starts=(DATASET_PATH, "/kaggle/input"), max_depth=6):
    """หาโฟลเดอร์ที่มี train1_cropped + testing_set อยู่ข้างใน (ชื่อ slug ของ Kaggle เปลี่ยนได้)"""
    for base in starts:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            if root[len(base):].count(os.sep) > max_depth:
                dirs[:] = []
                continue
            if os.path.isdir(os.path.join(root, "train1_cropped")) and os.path.isdir(os.path.join(root, "testing_set")):
                return root
    raise FileNotFoundError(
        f"หาโฟลเดอร์ที่มี train1_cropped และ testing_set ไม่เจอใต้ {starts} — "
        "เช็กว่าแนบ dataset แล้ว และแก้ DATASET_PATH ให้ตรงกับ path ที่ Kaggle แสดง")


DATA_ROOT = find_data_root()
print("DATA_ROOT:", DATA_ROOT)
TRAIN_DIRS = [os.path.join(DATA_ROOT, f"train{i}_cropped") for i in range(1, 6)]
TEST_DIR = os.path.join(DATA_ROOT, "testing_set")
OUTPUT_DIR = "/kaggle/working"

# --- แหล่งข้อมูล ---------------------------------------------------------------
# "auto" = นับแล้วเลือกแหล่งที่มีรูปใช้ได้มากที่สุด, หรือบังคับ "roboflow" / "kaggle"
# รูปจาก Roboflow แยกได้จากชื่อไฟล์ "<ต้นฉบับ>_jpg.rf.<hash>.jpg" ที่ Roboflow ตั้งให้ทุกไฟล์
SOURCE = "auto"

# --- เพศ -----------------------------------------------------------------------
GENDERS = ["female", "male"]            # เทรนโมเดลละเพศ ลบเพศที่ไม่ต้องการออกได้
GENDER_MODEL = "dima806/fairface_gender_image_detection"   # ViT จาก FairFace (~93% บนชุดทดสอบของผู้สร้าง)
GENDER_MIN_CONF = 0.80                  # ต่ำกว่านี้ = "unknown" ไม่ใช้เทรน (กันป้ายเพศผิด)
# CSV แก้ป้ายเพศเอง (คอลัมน์: filename,gender) — ชื่อไฟล์เทียบแค่ basename, แนบเป็น dataset หรือวางใน /kaggle/working
GENDER_OVERRIDES_CSV = None

# --- ชุดทดสอบ -------------------------------------------------------------------
# testing_set ใช้เป็นชุดทดสอบได้เมื่อเป็นแหล่งเดียวกับที่เลือก และมีรูปเพศนั้น >= MIN_TEST_PER_CLASS ทุกคลาส
# ไม่อย่างนั้นแบ่ง TEST_HOLDOUT ของแหล่งที่เลือกออกมาเป็นชุดทดสอบ (แบ่งทั้งกลุ่มภาพซ้ำ กันรั่ว)
MIN_TEST_PER_CLASS = 20
TEST_HOLDOUT = 0.15

# --- โมเดล / การเทรน ---------------------------------------------------------------
BACKBONE = "resnet50"                   # "resnet50" หรือ "effnetb1"
K_FOLDS = 3
SEED = 42
BATCH_SIZE = 32
EPOCHS_HEAD = 8
EPOCHS_FINETUNE = 30
FINE_TUNE_FREEZE_FRACTION = 0.65        # ล็อก 65% ล่าง ปรับ 35% บน
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
WEIGHT_DECAY_FINETUNE = 1e-4
LABEL_SMOOTHING = 0.1
QUICK_TEST = False                      # True = ลองรันให้จบเร็ว ๆ (epoch น้อย) เช็กว่าโค้ดไม่พังก่อนรันจริง

# --- Face crop ------------------------------------------------------------------
FACE_CROP_DIR = "/kaggle/working/face_crops"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

if QUICK_TEST:
    EPOCHS_HEAD, EPOCHS_FINETUNE = 1, 1

IMG_SIZE = (224, 224) if BACKBONE == "resnet50" else (240, 240)
os.makedirs(OUTPUT_DIR, exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE
# seed เดียวคุมทั้ง Python / NumPy / TensorFlow -> รันซ้ำได้ split เดิม ผลใกล้เดิม
tf.keras.utils.set_random_seed(SEED)

# =============================================================================
# ### (ไม่ต้องแก้) ซอร์สของ face_crop.py — ตัวเดียวกับไฟล์ face_crop.py ที่ส่งให้
# =============================================================================
# ฝังไว้ในสคริปต์เพื่อให้ copy ไฟล์นี้ไฟล์เดียวไปรันใน Kaggle ได้เลย
# (backend จะใช้ไฟล์ face_crop.py ตัวเดียวกันนี้ครอปรูปจากแอปตอนทำนายจริง)
FACE_CROP_SOURCE = r'''"""
Face crop for face-shape classification (training on Kaggle + inference in the backend).

Why: the classifier was learning hair, background and framing instead of the
face outline, and `tf.image.resize` squashed every non-square photo (a
squashed oval looks round, a stretched round looks oblong). Every image -
train, validation, test and later the live app - goes through the same crop:

  1. MediaPipe face mesh -> 468 landmarks (largest face in the image).
  2. Rotate so the eyes are level (head tilt removed).
  3. Box = face-oval landmarks, with the top lifted from landmark 10
     (mid-forehead) up to about the hairline, so forehead width is visible.
  4. Square box + margin, cut with one affine warp (no stretching); anything
     outside the photo is filled with neutral grey.
  5. No face found -> centre square crop, marked status="fallback".

Pure geometry (`crop_box_from_landmarks`, `render_crop`) needs only numpy +
Pillow, so the backend can reuse it with landmarks from any MediaPipe build.

Kaggle use (the training script does this for you):
    import face_crop
    manifest = face_crop.ensure_face_crops(image_paths, data_root, "/kaggle/working/face_crops")
MediaPipe is installed into its own folder and run in a separate process, so
it cannot change the numpy/protobuf versions TensorFlow is using.

Command line:
    python face_crop.py --data-root /kaggle/input/.../datasetformodelai --out /kaggle/working/face_crops
"""
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageOps

CROP_VERSION = 2  # 2: upright from forehead->chin (fixes upside-down), rotated-photo retries

# MediaPipe face-mesh indices (same topology for the legacy FaceMesh solution and
# the Tasks FaceLandmarker; the latter adds iris points 468-477, which we ignore).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
LM_FOREHEAD = 10      # top of the mesh oval: mid-forehead, below the real hairline
LM_CHIN = 152
LM_NOSE_TIP = 1
LM_EYE_A = 33         # outer eye corners (subject's right / left)
LM_EYE_B = 263
LM_CHEEK_A = 234
LM_CHEEK_B = 454

FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                       "face_landmarker/float16/1/face_landmarker.task")
MEDIAPIPE_PIP_SPECS = ["mediapipe==0.10.21", "mediapipe"]  # pinned first, newest as fallback
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class CropParams:
    out_size: int = 320            # saved crop size; training resizes to its own input size
    forehead_lift: float = 0.20    # face widths above landmark 10 (~ the hairline)
    margin: float = 0.08           # extra border on each side, fraction of the box's long side
    align_roll: bool = True        # rotate so the face is upright with level eyes
    fill: int = 128               # grey for areas outside the photo
    detect_max_side: int = 1280    # downscale big photos before detection (speed)
    min_face_frac: float = 0.08    # faces narrower than this fraction of the photo are ignored

    def key(self):
        return json.dumps({"v": CROP_VERSION, **asdict(self)}, sort_keys=True)


# -----------------------------------------------------------------------------
# Geometry (numpy + Pillow only)
# -----------------------------------------------------------------------------
def _rot(points, angle, origin):
    """Rotate Nx2 points by -angle around origin (into the eye-aligned frame)."""
    c, s = math.cos(angle), math.sin(angle)
    d = points - origin
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], axis=1)


def crop_box_from_landmarks(pts, params=CropParams()):
    """
    pts: (N>=468, 2) landmark pixel coordinates in the image.
    Returns dict(center=(x, y) in image px, side=px, angle=radians, roll_deg, yaw, face_width).
    `angle` is the rotation of the crop's x-axis in the image (0 = upright).
    """
    pts = np.asarray(pts, dtype=np.float64)[:, :2]
    if pts.shape[0] < 468 or not np.all(np.isfinite(pts[:468])):
        raise ValueError("need 468 finite landmarks")
    eye_a, eye_b = pts[LM_EYE_A], pts[LM_EYE_B]
    # Eye line gives the tilt precisely; forehead->chin decides which way is "down",
    # so upside-down (rotated 180) and mirrored photos both come out upright.
    e = eye_b - eye_a
    down = pts[LM_CHIN] - pts[LM_FOREHEAD]
    if -e[1] * down[0] + e[0] * down[1] < 0:
        e = -e
    roll = math.atan2(e[1], e[0])
    angle = roll if params.align_roll else 0.0
    origin = (eye_a + eye_b) / 2

    oval = _rot(pts[FACE_OVAL], angle, origin)
    top10 = _rot(pts[[LM_FOREHEAD]], angle, origin)[0]
    x0, x1 = oval[:, 0].min(), oval[:, 0].max()
    y0, y1 = oval[:, 1].min(), oval[:, 1].max()
    face_width = x1 - x0
    if face_width <= 1:
        raise ValueError("degenerate face")
    y0 = min(y0, top10[1] - params.forehead_lift * face_width)

    long_side = max(x1 - x0, y1 - y0)
    side = long_side * (1 + 2 * params.margin)
    cx_a, cy_a = (x0 + x1) / 2, (y0 + y1) / 2
    c, s = math.cos(angle), math.sin(angle)
    center = (origin[0] + cx_a * c - cy_a * s, origin[1] + cx_a * s + cy_a * c)

    cheeks = _rot(pts[[LM_CHEEK_A, LM_CHEEK_B, LM_NOSE_TIP]], angle, origin)
    mid = (cheeks[0, 0] + cheeks[1, 0]) / 2
    half = abs(cheeks[1, 0] - cheeks[0, 0]) / 2 or 1.0
    yaw = float(np.clip((cheeks[2, 0] - mid) / half, -1, 1))   # -1..1, 0 = frontal
    return {"center": (float(center[0]), float(center[1])), "side": float(side), "angle": float(angle),
            "roll_deg": round(math.degrees(roll), 2), "yaw": round(yaw, 3), "face_width": float(face_width)}


def render_crop(img, center, side, angle, out_size, fill=128):
    """Cut a rotated square (centre, side, angle) out of a PIL image -> out_size x out_size RGB."""
    s = side / out_size
    c, n = math.cos(angle), math.sin(angle)
    h = out_size / 2
    # output (u, v) -> input (x, y) = center + R(angle) @ ((u - h) * s, (v - h) * s)
    a, b = s * c, -s * n
    d, e = s * n, s * c
    cx, cy = center
    coeffs = (a, b, cx - h * a - h * b, d, e, cy - h * d - h * e)
    return img.convert("RGB").transform((out_size, out_size), Image.AFFINE, coeffs,
                                        resample=Image.BICUBIC, fillcolor=(fill, fill, fill))


def center_square_crop(img, out_size, fill=128):
    w, h = img.size
    return render_crop(img, (w / 2, h / 2), min(w, h), 0.0, out_size, fill)


def load_rgb(path):
    """Open an image upright (EXIF rotation applied - tf.io.decode_image ignores it) as RGB."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


# -----------------------------------------------------------------------------
# Detection (MediaPipe imported lazily - only the crop worker / backend needs it)
# -----------------------------------------------------------------------------
class FaceLandmarkDetector:
    """
    Wraps MediaPipe face landmarks for still images. Uses the Tasks
    FaceLandmarker when a .task model file is given (all recent versions),
    otherwise the legacy `mp.solutions.face_mesh` (bundled model, older versions).
    detect(rgb uint8 HxWx3) -> list of (468, 2) pixel arrays.
    """

    def __init__(self, model_path=None, max_faces=3, min_confidence=0.5):
        import mediapipe as mp
        self._mp = mp
        self._tasks = None
        self._legacy = None
        if model_path and os.path.exists(model_path):
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=max_faces,
                min_face_detection_confidence=min_confidence,
                min_face_presence_confidence=min_confidence,
            )
            self._tasks = vision.FaceLandmarker.create_from_options(options)
            self.backend = "tasks"
        elif hasattr(mp, "solutions"):
            self._legacy = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=max_faces,
                refine_landmarks=False, min_detection_confidence=min_confidence)
            self.backend = "legacy"
        else:
            raise RuntimeError("This MediaPipe build has no face_mesh solution; pass model_path "
                               f"to a face_landmarker.task file ({FACE_LANDMARKER_URL})")

    def detect(self, rgb):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        if self._tasks is not None:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            faces = self._tasks.detect(image).face_landmarks or []
            lists = [[(p.x, p.y) for p in face] for face in faces]
        else:
            res = self._legacy.process(rgb)
            lists = [[(p.x, p.y) for p in face.landmark] for face in (res.multi_face_landmarks or [])]
        return [np.array(face, dtype=np.float64)[:468] * [w, h] for face in lists if len(face) >= 468]

    def close(self):
        for obj in (self._tasks, self._legacy):
            if obj is not None:
                obj.close()


def _largest(faces, min_width):
    best, best_w = None, 0.0
    for f in faces:
        width = np.ptp(f[FACE_OVAL, 0])
        if width >= min_width and width > best_w:
            best, best_w = f, width
    return best


# PIL transpose op, and how to map a point in the rotated image back to the original (w, h)
_ROTATIONS = {
    "rot180": (Image.Transpose.ROTATE_180, lambda p, w, h: np.stack([w - p[:, 0], h - p[:, 1]], 1)),
    "rot90": (Image.Transpose.ROTATE_90, lambda p, w, h: np.stack([w - p[:, 1], p[:, 0]], 1)),    # CCW
    "rot270": (Image.Transpose.ROTATE_270, lambda p, w, h: np.stack([p[:, 1], h - p[:, 0]], 1)),  # CW
}


def _detect(detector, im, pad, up, fill):
    """Faces in `im` (pixel coords of `im`), after optional grey border + upscale."""
    if pad:
        canvas = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad), (fill,) * 3)
        canvas.paste(im, (pad, pad))
        im = canvas
    if up != 1:
        im = im.resize((round(im.width * up), round(im.height * up)), Image.BICUBIC)
    return [f / up - pad for f in detector.detect(np.asarray(im))]


def find_landmarks(detector, img, params=CropParams()):
    """
    Landmarks of the largest face in a PIL RGB image, retrying the cases
    MediaPipe misses: tightly pre-cropped faces (pad the borders so the face
    detector sees some context), small images (upscale) and photos rotated by
    90/180/270 degrees (common in augmented datasets).
    Returns (pts (468,2) in original pixel coords, method) or (None, "none").
    """
    w, h = img.size
    scale = min(1.0, params.detect_max_side / max(w, h))
    base = img if scale == 1.0 else img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    bw, bh = base.size
    pad = int(round(0.35 * max(bw, bh)))
    up = 2.0 if max(bw, bh) < 400 else 1.0
    min_width = params.min_face_frac * min(bw, bh)

    attempts = [("mesh", None, 0, 1.0), ("mesh_padded", None, pad, 1.0)]
    if up != 1:
        attempts.append(("mesh_upscaled", None, pad, up))
    attempts += [(f"mesh_{name}", name, pad, up) for name in _ROTATIONS]

    for method, rot, p, u in attempts:
        im = base if rot is None else base.transpose(_ROTATIONS[rot][0])
        faces = _detect(detector, im, p, u, params.fill)
        if rot is not None:
            faces = [_ROTATIONS[rot][1](f, bw, bh) for f in faces]
        pts = _largest(faces, min_width)
        if pts is not None:
            return pts / scale, method
    return None, "none"


def crop_face(img, detector, params=CropParams()):
    """PIL RGB image -> (square crop, info). info["status"]: mesh / mesh_padded / mesh_upscaled / fallback."""
    pts, method = find_landmarks(detector, img, params)
    if pts is None:
        return center_square_crop(img, params.out_size, params.fill), {"status": "fallback"}
    box = crop_box_from_landmarks(pts, params)
    crop = render_crop(img, box["center"], box["side"], box["angle"], params.out_size, params.fill)
    return crop, {"status": method, "roll_deg": box["roll_deg"], "yaw": box["yaw"]}


# -----------------------------------------------------------------------------
# Batch worker (runs in the isolated MediaPipe process)
# -----------------------------------------------------------------------------
_WORKER = {}


def _pool_init(model_path, params_dict):
    _WORKER["params"] = CropParams(**params_dict)
    _WORKER["detector"] = FaceLandmarkDetector(model_path)


def _pool_crop(job):
    src, dst = job
    params = _WORKER["params"]
    try:
        img = load_rgb(src)
        crop, info = crop_face(img, _WORKER["detector"], params)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        crop.save(dst, quality=95)
        info["crop"] = dst
    except Exception as exc:  # unreadable file etc. - reported, never stops the batch
        info = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return src, info


def _worker_main(job_file, result_file, workers):
    import multiprocessing
    with open(job_file, encoding="utf-8") as f:
        spec = json.load(f)
    jobs = [tuple(j) for j in spec["jobs"]]
    workers = max(1, min(int(workers), len(jobs)))
    init_args = (spec.get("model_path"), spec["params"])
    results, t0 = {}, time.time()
    probe = FaceLandmarkDetector(spec.get("model_path"))
    print(f"[face_crop] MediaPipe backend: {probe.backend}, {len(jobs)} images, {workers} processes", flush=True)
    probe.close()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers, initializer=_pool_init, initargs=init_args) as pool:
        for i, (src, info) in enumerate(pool.imap_unordered(_pool_crop, jobs, chunksize=8), 1):
            results[src] = info
            if i % 500 == 0 or i == len(jobs):
                ok = sum(1 for r in results.values() if r["status"].startswith("mesh"))
                print(f"[face_crop] {i}/{len(jobs)}  face found so far {ok / i:.1%} (running total)  ({time.time() - t0:.0f}s)", flush=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f)


# -----------------------------------------------------------------------------
# Orchestration (runs in the training kernel: stdlib + numpy + Pillow only)
# -----------------------------------------------------------------------------
def _env_python_ok(env_dir):
    code = f"import sys; sys.path.insert(0, {env_dir!r}); import mediapipe"
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True).returncode == 0


def ensure_mediapipe(env_dir="/tmp/mediapipe_env"):
    """Install MediaPipe into its own folder (not the TF environment). Needs internet."""
    if _env_python_ok(env_dir):
        return env_dir
    errors = []
    for spec in MEDIAPIPE_PIP_SPECS:
        print(f"[face_crop] installing {spec} into {env_dir} (separate from TensorFlow) ...", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location",
                            "--upgrade", "--target", env_dir, spec], capture_output=True, text=True)
        if r.returncode == 0 and _env_python_ok(env_dir):
            return env_dir
        errors.append(f"{spec}: {(r.stderr or r.stdout)[-800:]}")
    raise RuntimeError("Could not install MediaPipe. On Kaggle turn on Settings -> Internet, "
                       "then run again.\n" + "\n".join(errors))


def ensure_model(path):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    import urllib.request
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, path + ".part")
        os.replace(path + ".part", path)
        return path
    except Exception as exc:
        print(f"[face_crop] could not download face_landmarker.task ({exc}); "
              "falling back to the legacy face_mesh model if available", flush=True)
        return None


def crop_destination(src, data_root, out_dir):
    rel = os.path.relpath(src, data_root)
    if rel.startswith(".."):
        rel = os.path.join("_external", src.lstrip(os.sep))
    stem, ext = os.path.splitext(rel)
    return os.path.join(out_dir, stem + ("" if ext.lower() == ".jpg" else "_" + ext.lstrip(".").lower()) + ".jpg")


def _stream(cmd, env=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    tail = []
    for line in proc.stdout:   # echo into the notebook cell (child output doesn't show there otherwise)
        print(line, end="", flush=True)
        tail = (tail + [line])[-40:]
    if proc.wait() != 0:
        raise RuntimeError("face crop worker failed:\n" + "".join(tail))


def ensure_face_crops(image_paths, data_root, out_dir, params=CropParams(), workers=None,
                      env_dir="/tmp/mediapipe_env", run_worker=None):
    """
    Crop every image once (cached in out_dir, reused on later runs with the same
    params). Returns {original_path: {"crop": path, "status": ..., ...}}.
    `run_worker(job_file, result_file, workers)` overrides how the worker is
    launched (used by tests); by default an isolated MediaPipe process.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    items = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("params_key") == params.key():
            items = {k: v for k, v in old["items"].items()
                     if v.get("status") == "error" or os.path.exists(v.get("crop", ""))}
    unique = list(dict.fromkeys(image_paths))
    todo = [p for p in unique if p not in items]
    print(f"[face_crop] {len(unique)} images: {len(unique) - len(todo)} cached, {len(todo)} to crop", flush=True)

    if todo:
        dests, seen = [], set()
        for p in todo:
            d = crop_destination(p, data_root, out_dir)
            while d in seen:
                d = d[:-4] + "_dup.jpg"
            seen.add(d)
            dests.append(d)
        job_file = os.path.join(out_dir, "_jobs.json")
        result_file = os.path.join(out_dir, "_results.json")
        workers = workers or os.cpu_count() or 2
        if run_worker is None:
            env = ensure_mediapipe(env_dir)
            model_path = ensure_model(os.path.join(env, "face_landmarker.task"))
        else:
            env, model_path = None, None
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump({"params": asdict(params), "model_path": model_path,
                       "jobs": [[s, d] for s, d in zip(todo, dests)]}, f)
        if run_worker is not None:
            run_worker(job_file, result_file, workers)
        else:
            here = os.path.dirname(os.path.abspath(__file__))
            boot = (f"import sys; sys.path[:0] = [{env!r}, {here!r}]; import face_crop; "
                    f"face_crop._worker_main({job_file!r}, {result_file!r}, {workers})")
            _stream([sys.executable, "-I", "-c", boot])
        with open(result_file, encoding="utf-8") as f:
            items.update(json.load(f))
        for tmp in (job_file, result_file):
            os.remove(tmp)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"params_key": params.key(), "params": asdict(params), "items": items}, f)
    return {p: items[p] for p in unique if p in items}


def summarize(manifest, groups):
    """groups: {name: [original paths]} -> prints face-found rate per group, returns rates."""
    rates = {}
    print(f"\n[face_crop] {'group':28s} {'images':>7s} {'face found':>11s} {'fallback':>9s} {'error':>6s}")
    for name, paths in groups.items():
        st = [manifest.get(p, {}).get("status", "missing") for p in paths]
        n = max(1, len(st))
        found = sum(s.startswith("mesh") for s in st)
        rates[name] = found / n
        print(f"[face_crop] {name:28s} {len(st):7d} {found / n:10.1%} {st.count('fallback'):9d} "
              f"{st.count('error') + st.count('missing'):6d}")
    return rates


def save_sample_grid(manifest, labeled_paths, class_names, out_path, per_class=8, thumb=120, seed=0):
    """Contact sheet: one row per class of found faces, plus one row of fallbacks, to eyeball the crops."""
    rng = np.random.default_rng(seed)
    rows = []
    for idx, name in enumerate(class_names):
        cand = [p for p, l in labeled_paths if l == idx and manifest.get(p, {}).get("status", "").startswith("mesh")]
        rows.append((name, [cand[i] for i in rng.permutation(len(cand))[:per_class]]))
    fb = [p for p, _ in labeled_paths if manifest.get(p, {}).get("status") == "fallback"]
    rows.append(("no face", [fb[i] for i in rng.permutation(len(fb))[:per_class]]))
    label_w = 80
    sheet = Image.new("RGB", (label_w + per_class * thumb, len(rows) * thumb), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, (name, paths) in enumerate(rows):
        draw.text((6, r * thumb + thumb // 2 - 6), name, fill=(0, 0, 0))
        for c, p in enumerate(paths):
            with Image.open(manifest[p]["crop"]) as im:
                sheet.paste(im.convert("RGB").resize((thumb, thumb)), (label_w + c * thumb, r * thumb))
    sheet.save(out_path, quality=90)
    return out_path


def list_images(root):
    out = []
    for dirpath, _, files in os.walk(root):
        out += [os.path.join(dirpath, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXTENSIONS)]
    return sorted(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Crop faces for face-shape training")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    paths = list_images(args.data_root)
    m = ensure_face_crops(paths, args.data_root, args.out, workers=args.workers)
    summarize(m, {"all": paths})
'''
# =============================================================================
# ### 0) Normalize ชื่อคลาส — จุดสำคัญที่สุดของสคริปต์นี้
# =============================================================================
# dataset นี้ตั้งชื่อโฟลเดอร์คลาสไม่ตรงกันเอง (ยืนยันจากการเช็กจริงแล้ว):
#   - บางโฟลเดอร์ใช้ตัวพิมพ์ใหญ่ ('Oval'), บางอันใช้พิมพ์เล็ก ('oval')
#   - บางโฟลเดอร์ใช้ 'long' แทน 'Oblong' (ยืนยันแล้วว่าหมายถึงคลาสเดียวกัน)
# ปล่อยให้ image_dataset_from_directory เดา label จากชื่อโฟลเดอร์ตรงๆ จะทำให้
# ลำดับคลาส (integer label) ของแต่ละโฟลเดอร์ไม่ตรงกัน แล้ว concatenate รวมกัน
# จะได้ label ผิดแบบไม่มี error เตือน จึงต้อง map ชื่อดิบ -> ชื่อมาตรฐานเองที่นี่ก่อน

CANONICAL_CLASSES = ["heart", "oblong", "oval", "round", "square"]  # ลำดับ index คงที่ ใช้ตลอดทั้งสคริปต์
CLASS_TO_IDX = {name: i for i, name in enumerate(CANONICAL_CLASSES)}

ALIAS_MAP = {
    "oval": "oval",
    "round": "round",
    "square": "square",
    "heart": "heart",
    "oblong": "oblong",
    "long": "oblong",  # ยืนยันกับผู้ใช้แล้วว่า long = oblong
}


def normalize_class_name(raw_folder_name):
    """แปลงชื่อโฟลเดอร์ดิบ (ไม่สนตัวพิมพ์เล็ก-ใหญ่/ช่องว่างหัวท้าย) ให้เป็นชื่อคลาสมาตรฐาน"""
    key = raw_folder_name.strip().lower()
    if key not in ALIAS_MAP:
        raise ValueError(
            f"พบชื่อโฟลเดอร์คลาสที่ไม่รู้จัก: {raw_folder_name!r} "
            f"(รู้จักแค่ {sorted(ALIAS_MAP.keys())}) — เช็กว่าเป็นโฟลเดอร์ขยะ/สะกดผิดหรือเปล่า"
        )
    return ALIAS_MAP[key]


def list_image_label_pairs(directory):
    """
    เดินดูทุกโฟลเดอร์ย่อยใน directory, normalize ชื่อคลาสของแต่ละโฟลเดอร์,
    แล้วคืนลิสต์ (path, class_idx) ของรูปทั้งหมดข้างใน
    """
    pairs = []
    per_class_count = {name: 0 for name in CANONICAL_CLASSES}
    for raw_name in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, raw_name)
        if not os.path.isdir(folder_path):
            continue
        canonical = normalize_class_name(raw_name)
        idx = CLASS_TO_IDX[canonical]
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(VALID_EXTENSIONS):
                pairs.append((os.path.join(folder_path, fname), idx))
                per_class_count[canonical] += 1
    return pairs, per_class_count


def report_counts(label, per_class_count):
    print(f"\n{label}:")
    total = sum(per_class_count.values())
    for name in CANONICAL_CLASSES:
        c = per_class_count[name]
        flag = "  <-- [WARNING] 0 รูปเลย ตรวจสอบก่อนเทรน" if c == 0 else ""
        print(f"   {name:8s}: {c:5d} รูป{flag}")
    print(f"   รวม     : {total:5d} รูป")


# =============================================================================
# ### กลุ่มภาพซ้ำ (กัน data leakage) — ตัวเดียวกับสคริปต์เดิม
# =============================================================================
ROBOFLOW_PREFIX_RE = re.compile(r"^(.+?)_jpg\.rf\.", re.IGNORECASE)
HASH_THRESHOLD = 6  # hamming distance <= นี้ถือว่าเกือบซ้ำกัน (phash 64 บิต)


def _bucket_near_duplicates(items, hash_threshold):
    """items: list of (path, imagehash). จัดกลุ่มภาพที่ hash ใกล้กันมาก
    โดย bucket ด้วยบิตสูงของ hash ก่อน กันเทียบคู่ทุกคู่ (O(n^2)) ตรงๆ ซึ่งช้าเกินไป"""
    buckets = defaultdict(list)
    for path, h in items:
        key = int(str(h), 16) >> 54  # ใช้ ~10 บิตบนสุดเป็น bucket key
        buckets[key].append((path, h))
    visited = set()
    groups = []
    for bucket_items in buckets.values():
        for i, (p1, h1) in enumerate(bucket_items):
            if p1 in visited:
                continue
            group = [p1]
            visited.add(p1)
            for p2, h2 in bucket_items[i + 1:]:
                if p2 in visited:
                    continue
                if h1 - h2 <= hash_threshold:
                    group.append(p2)
                    visited.add(p2)
            groups.append(group)
    return groups


def compute_duplicate_groups(pairs, hash_threshold=HASH_THRESHOLD):
    """จัดกลุ่ม (path, label) ที่น่าจะมาจากภาพต้นฉบับเดียวกัน แยกคำนวณทีละคลาส
    (ข้ามคลาสไม่มีทางเป็นต้นฉบับเดียวกันอยู่แล้ว) คืนค่า list ของ group,
    แต่ละ group คือ list ของ (path, label)"""
    by_label = defaultdict(list)
    for path, label in pairs:
        by_label[label].append(path)

    all_groups = []
    for label, paths in by_label.items():
        roboflow_groups = defaultdict(list)
        remaining = []
        for p in paths:
            m = ROBOFLOW_PREFIX_RE.match(os.path.basename(p))
            if m:
                roboflow_groups[m.group(1)].append(p)
            else:
                remaining.append(p)

        all_groups.extend([[(p, label) for p in g] for g in roboflow_groups.values()])

        hashed = []
        for p in remaining:
            try:
                hashed.append((p, imagehash.phash(Image.open(p).convert("RGB"))))
            except Exception:
                all_groups.append([(p, label)])  # อ่านไม่ได้ -> ถือเป็นกลุ่มเดี่ยว ไม่บล็อก pipeline
        for g in _bucket_near_duplicates(hashed, hash_threshold):
            all_groups.append([(p, label) for p in g])

    return all_groups


# =============================================================================
# ### 1) รายชื่อรูปทั้งหมด + แหล่งที่มา (Kaggle / Roboflow)
# =============================================================================
def source_of(path):
    """Roboflow ตั้งชื่อไฟล์ส่งออกเป็น <ต้นฉบับ>_jpg.rf.<hash>.jpg เสมอ — ที่เหลือมาจาก Kaggle"""
    name = os.path.basename(path)
    return "roboflow" if (ROBOFLOW_PREFIX_RE.match(name) or ".rf." in name.lower()) else "kaggle"


def folder_of(path):
    rel = os.path.relpath(path, DATA_ROOT)
    return rel.split(os.sep)[0]


train_pool = []
for folder in TRAIN_DIRS:
    pairs, counts = list_image_label_pairs(folder)
    train_pool.extend(pairs)
    print(f"  {os.path.basename(folder)}: {sum(counts.values())} รูป")
test_pairs_all, _ = list_image_label_pairs(TEST_DIR)

print("\n--- แหล่งที่มาของรูป (นับจากชื่อไฟล์) ---")
audit = Counter()
for p, l in train_pool + test_pairs_all:
    audit[(folder_of(p), source_of(p), CANONICAL_CLASSES[l])] += 1
folders = sorted({k[0] for k in audit})
print(f"{'โฟลเดอร์':18s} {'แหล่ง':9s} " + " ".join(f"{c:>7s}" for c in CANONICAL_CLASSES) + f" {'รวม':>7s}")
for fo in folders:
    for so in ("kaggle", "roboflow"):
        row = [audit[(fo, so, c)] for c in CANONICAL_CLASSES]
        if sum(row):
            print(f"{fo:18s} {so:9s} " + " ".join(f"{v:7d}" for v in row) + f" {sum(row):7d}")
for so in ("kaggle", "roboflow"):
    tot = sum(v for (fo, s, c), v in audit.items() if s == so and fo != os.path.basename(TEST_DIR))
    print(f"รวม train จาก {so:9s}: {tot} รูป")
with open(os.path.join(OUTPUT_DIR, "source_audit.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["folder", "source"] + CANONICAL_CLASSES + ["total"])
    for fo in folders:
        for so in ("kaggle", "roboflow"):
            row = [audit[(fo, so, c)] for c in CANONICAL_CLASSES]
            if sum(row):
                w.writerow([fo, so] + row + [sum(row)])

# =============================================================================
# ### 2) Face crop ทุกภาพ (cache ไว้ รันซ้ำไม่ครอปใหม่)
# =============================================================================
_face_crop_path = os.path.join(OUTPUT_DIR, "face_crop.py")
with open(_face_crop_path, "w", encoding="utf-8") as f:
    f.write(FACE_CROP_SOURCE)
if OUTPUT_DIR not in sys.path:
    sys.path.insert(0, OUTPUT_DIR)
import importlib
import face_crop
face_crop = importlib.reload(face_crop)

print("\n--- Face crop ---")
crop_manifest = face_crop.ensure_face_crops([p for p, _ in train_pool + test_pairs_all], DATA_ROOT, FACE_CROP_DIR)


def face_found(p):
    return crop_manifest.get(p, {}).get("status", "").startswith("mesh")


face_crop.summarize(crop_manifest, {
    **{f"{fo} / {so}": [p for p, _ in train_pool + test_pairs_all if folder_of(p) == fo and source_of(p) == so]
       for fo in folders for so in ("kaggle", "roboflow")
       if any(folder_of(p) == fo and source_of(p) == so for p, _ in train_pool + test_pairs_all)},
})

# =============================================================================
# ### 3) ติดป้ายเพศอัตโนมัติ (รันใน process แยกด้วย PyTorch, cache เป็น CSV)
# =============================================================================
GENDER_SCRIPT = r'''
import sys, json, csv
import torch
from PIL import Image
from transformers import pipeline
jobs, out, model = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
clf = pipeline("image-classification", model=model, device=0 if torch.cuda.is_available() else -1)
rows, B = [], 64
for i in range(0, len(jobs), B):
    batch = jobs[i:i + B]
    ims = [Image.open(p).convert("RGB") for p in batch]
    for p, res in zip(batch, clf(ims, batch_size=B, top_k=2)):
        sc = {d["label"].strip().lower(): float(d["score"]) for d in res}
        if "female" not in sc and "male" not in sc:
            raise SystemExit(f"unexpected labels from {model}: {list(sc)}")
        fe, ma = sc.get("female", 0.0), sc.get("male", 0.0)
        rows.append((p, "female" if fe >= ma else "male", max(fe, ma)))
    for im in ims:
        im.close()
    if (i // B) % 25 == 0:
        print(f"[gender] {min(i + B, len(jobs))}/{len(jobs)}", flush=True)
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["crop", "gender", "conf"]); w.writerows(rows)
'''

GENDER_CSV = os.path.join(OUTPUT_DIR, "gender_labels.csv")


def load_gender_cache():
    if not os.path.exists(GENDER_CSV):
        return {}
    with open(GENDER_CSV, encoding="utf-8") as f:
        return {r["crop"]: (r["gender"], float(r["conf"])) for r in csv.DictReader(f)}


gender_cache = load_gender_cache()
todo = sorted({crop_manifest[p]["crop"] for p, _ in train_pool + test_pairs_all if face_found(p)} - set(gender_cache))
print(f"\n--- ป้ายเพศ: cache {len(gender_cache)} รูป, ต้องทำใหม่ {len(todo)} รูป ---")
if todo:
    script = os.path.join(OUTPUT_DIR, "_gender_label.py")
    jobs_file, out_file = os.path.join(OUTPUT_DIR, "_gender_jobs.json"), os.path.join(OUTPUT_DIR, "_gender_new.csv")
    with open(script, "w", encoding="utf-8") as f:
        f.write(GENDER_SCRIPT)
    with open(jobs_file, "w", encoding="utf-8") as f:
        json.dump(todo, f)
    try:   # _stream พิมพ์ผลของ process ลูกลงเซลล์ notebook ด้วย
        face_crop._stream([sys.executable, script, jobs_file, out_file, GENDER_MODEL])
    except RuntimeError as exc:
        raise RuntimeError("ติดป้ายเพศไม่สำเร็จ — เปิด Internet ใน Settings แล้วรันใหม่ "
                           "(ต้องโหลดโมเดลจาก Hugging Face)") from exc
    with open(out_file, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gender_cache[r["crop"]] = (r["gender"], float(r["conf"]))
    with open(GENDER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop", "gender", "conf"])
        for k, (g, c) in sorted(gender_cache.items()):
            w.writerow([k, g, f"{c:.4f}"])
    for tmp in (jobs_file, out_file):
        os.remove(tmp)

overrides = {}
if GENDER_OVERRIDES_CSV and os.path.exists(GENDER_OVERRIDES_CSV):
    with open(GENDER_OVERRIDES_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            overrides[os.path.basename(r["filename"]).strip()] = r["gender"].strip().lower()
    print(f"ใช้ป้ายเพศที่แก้เอง {len(overrides)} รูป จาก {GENDER_OVERRIDES_CSV}")


def gender_of(p):
    """'female' / 'male' / 'unknown' (หาหน้าไม่เจอ หรือโมเดลไม่มั่นใจพอ)"""
    if os.path.basename(p) in overrides:
        return overrides[os.path.basename(p)]
    if not face_found(p):
        return "unknown"
    g, c = gender_cache.get(crop_manifest[p]["crop"], ("unknown", 0.0))
    return g if c >= GENDER_MIN_CONF else "unknown"


# รูปตัวอย่างให้ตรวจด้วยตาว่าป้ายเพศถูกไหม (แถวละ 10 รูป สุ่ม)
def save_gender_sheet(paths, out_path, n=40, thumb=110):
    rng = np.random.default_rng(SEED)
    pick = [paths[i] for i in rng.permutation(len(paths))[:n]]
    cols = 10
    sheet = Image.new("RGB", (cols * thumb, max(1, (len(pick) + cols - 1) // cols) * thumb), (255, 255, 255))
    for i, p in enumerate(pick):
        with Image.open(crop_manifest[p]["crop"]) as im:
            sheet.paste(im.convert("RGB").resize((thumb, thumb)), ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_path, quality=88)


print("\n--- จำนวนรูปที่ใช้ได้ (หาหน้าเจอ + รู้เพศ) แยกแหล่ง / เพศ / คลาส ---")
usable = Counter()
for p, l in train_pool:
    usable[(source_of(p), gender_of(p), CANONICAL_CLASSES[l])] += 1
for so in ("kaggle", "roboflow"):
    for g in ("female", "male", "unknown"):
        row = [usable[(so, g, c)] for c in CANONICAL_CLASSES]
        print(f"  {so:9s} {g:8s} " + " ".join(f"{c}={v:<5d}" for c, v in zip(CANONICAL_CLASSES, row)) + f" รวม={sum(row)}")
for g in ("female", "male"):
    ps = [p for p, _ in train_pool if gender_of(p) == g]
    if ps:
        save_gender_sheet(ps, os.path.join(OUTPUT_DIR, f"gender_check_{g}.jpg"))
        print(f"[SAVED] gender_check_{g}.jpg  <- เปิดดูว่ามีรูปเพศผิดปนไหม (ถ้ามีเยอะ เพิ่ม GENDER_MIN_CONF หรือแก้ใน CSV)")

# =============================================================================
# ### 4) เลือกแหล่งข้อมูลเดียว
# =============================================================================
if SOURCE == "auto":
    totals = {so: sum(v for (s, g, c), v in usable.items() if s == so and g in GENDERS) for so in ("kaggle", "roboflow")}
    CHOSEN_SOURCE = max(totals, key=totals.get)
    print(f"\nSOURCE=auto -> เลือก '{CHOSEN_SOURCE}' (รูปที่ใช้ได้: {totals})")
else:
    CHOSEN_SOURCE = SOURCE
    print(f"\nSOURCE -> '{CHOSEN_SOURCE}' (กำหนดเอง)")

# =============================================================================
# ### 5) แบ่งชุดทดสอบ + Stratified Group K-Fold (ต่อเพศ)
# =============================================================================
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix


def group_ids(pairs):
    groups = compute_duplicate_groups(pairs)
    gid = {}
    for i, g in enumerate(groups):
        for p, _ in g:
            gid[p] = i
    return [gid[p] for p, _ in pairs]


def split_holdout(pairs, fraction, seed):
    """แบ่งชุดทดสอบออกทั้งกลุ่มภาพซ้ำ ให้สัดส่วนคลาสใกล้กัน (StratifiedGroupKFold fold แรก)"""
    n_splits = max(2, int(round(1 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = [l for _, l in pairs]
    rest_idx, test_idx = next(sgkf.split(np.zeros(len(pairs)), y, group_ids(pairs)))
    return [pairs[i] for i in rest_idx], [pairs[i] for i in test_idx]


def counts_of(pairs):
    c = Counter(l for _, l in pairs)
    return {n: c[i] for i, n in enumerate(CANONICAL_CLASSES)}


def write_split(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original", "crop", "label", "split"])
        w.writerows(rows)


plan = {}
for g in GENDERS:
    pool = [(p, l) for p, l in train_pool if source_of(p) == CHOSEN_SOURCE and gender_of(p) == g]
    ext_test = [(p, l) for p, l in test_pairs_all if source_of(p) == CHOSEN_SOURCE and gender_of(p) == g]
    if min(counts_of(ext_test).values()) >= MIN_TEST_PER_CLASS:
        test, how = ext_test, "testing_set"
    else:
        pool, test = split_holdout(pool, TEST_HOLDOUT, SEED)
        how = f"แบ่ง {TEST_HOLDOUT:.0%} จากแหล่งที่เลือก"
    if min(counts_of(pool).values()) < K_FOLDS * 2:
        print(f"[ข้าม] {g}: บางคลาสมีรูปน้อยเกินไปสำหรับ {K_FOLDS}-fold -> {counts_of(pool)}")
        continue
    plan[g] = {"pool": pool, "test": test, "test_from": how}
    print(f"\n{g}: train+val {len(pool)} รูป {counts_of(pool)}")
    print(f"{g}: test ({how}) {len(test)} รูป {counts_of(test)}")

# =============================================================================
# ### 6) tf.data + augmentation (เพิ่มเรื่องสี)
# =============================================================================
if BACKBONE == "resnet50":
    from tensorflow.keras.applications import ResNet50 as Backbone
    from tensorflow.keras.applications.resnet50 import preprocess_input
    PREPROCESS_NAME = "resnet50"      # backend ต้องทำ preprocess นี้เหมือนกัน (RGB->BGR + ลบค่าเฉลี่ย)
else:
    from tensorflow.keras.applications import EfficientNetB1 as Backbone
    from tensorflow.keras.applications.efficientnet import preprocess_input
    PREPROCESS_NAME = "none"          # EfficientNet มี rescale อยู่ในโมเดลแล้ว

geometric_aug = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),            # ±11° — หน้าถูกหมุนตรงแล้ว หมุนแรงจะทำลายการจัดหน้า
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
], name="geometric_aug")
light_aug = models.Sequential([
    layers.RandomContrast(0.20),
    layers.RandomBrightness(0.15, value_range=(0, 255)),
], name="light_aug")


def color_aug(img):
    """สุ่มเปลี่ยนสีต่อรูป: hue, saturation, ขาวดำ 10%, คุณภาพ JPEG (จำลองกล้องมือถือ)"""
    x = img / 255.0
    x = tf.image.random_hue(x, 0.04)
    x = tf.image.random_saturation(x, 0.6, 1.4)
    gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))
    x = tf.where(tf.random.uniform([]) < 0.10, gray, x)
    x = tf.image.random_jpeg_quality(tf.clip_by_value(x, 0.0, 1.0), 60, 100)
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0


def load_image(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize_with_pad(tf.cast(img, tf.float32), IMG_SIZE[0], IMG_SIZE[1])
    return img, label


def augment(img, label):
    img = geometric_aug(img, training=True)
    img = light_aug(img, training=True)
    img = color_aug(img)
    img.set_shape(IMG_SIZE + (3,))
    return img, label


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE) if training else ds.cache().prefetch(AUTOTUNE)


# ตัวอย่างรูปหลัง augment (ดูว่าแรงไปไหม)
def save_aug_preview(paths, out_path, n=8):
    ims = []
    for p in paths[:n]:
        img, _ = load_image(p, 0)
        ims.append(tf.cast(augment(img, 0)[0], tf.uint8).numpy())
        ims.append(tf.cast(img, tf.uint8).numpy())
    h, w = IMG_SIZE
    sheet = Image.new("RGB", (w * n, h * 2))
    for i in range(n):
        sheet.paste(Image.fromarray(ims[2 * i + 1]), (i * w, 0))
        sheet.paste(Image.fromarray(ims[2 * i]), (i * w, h))
    sheet.save(out_path, quality=88)

# =============================================================================
# ### 7) โมเดล + เทรน 2 เฟส
# =============================================================================
num_classes = len(CANONICAL_CLASSES)


def loss_fn(y_true, y_pred):
    y = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=num_classes)
    return tf.keras.losses.categorical_crossentropy(y, y_pred, label_smoothing=LABEL_SMOOTHING)


def build_model():
    base = Backbone(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)           # BatchNorm ใช้ค่าสถิติเดิมตลอด (สำคัญตอน fine-tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs), base


def train_fold(train_p, train_l, val_p, val_l, ckpt):
    counts = Counter(train_l)
    class_weight = {i: len(train_l) / (num_classes * counts[i]) if counts[i] else 0.0 for i in range(num_classes)}
    tr, va = make_ds(train_p, train_l, True), make_ds(val_p, val_l, False)
    model, base = build_model()
    cbs = [callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
           callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
           callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)]
    # เฟส 1: ล็อก backbone เทรนเฉพาะ head
    model.compile(optimizer=optimizers.Adam(LR_HEAD), loss=loss_fn, metrics=["accuracy"])
    h1 = model.fit(tr, validation_data=va, epochs=EPOCHS_HEAD, class_weight=class_weight, callbacks=cbs, verbose=2)
    # เฟส 2: ปลด 35% บนของ backbone, LR ต่ำมาก
    base.trainable = True
    for layer in base.layers[:int(len(base.layers) * FINE_TUNE_FREEZE_FRACTION)]:
        layer.trainable = False
    model.compile(optimizer=optimizers.AdamW(learning_rate=LR_FINETUNE, weight_decay=WEIGHT_DECAY_FINETUNE),
                  loss=loss_fn, metrics=["accuracy"])
    h2 = model.fit(tr, validation_data=va, epochs=EPOCHS_FINETUNE, class_weight=class_weight, callbacks=cbs, verbose=2)
    hist = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in ("accuracy", "val_accuracy", "loss", "val_loss")}
    return model, hist, len(h1.history["accuracy"])


def evaluate(model, paths, labels):
    probs = model.predict(make_ds(paths, labels, False), verbose=0)
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == np.array(labels))), pred


def crops_of(pairs):
    return [crop_manifest[p]["crop"] for p, _ in pairs], [l for _, l in pairs]


results = {}
for g, info in plan.items():
    pool, test = info["pool"], info["test"]
    test_p, test_l = crops_of(test)
    y = np.array([l for _, l in pool])
    sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    split_rows = [(p, crop_manifest[p]["crop"], CANONICAL_CLASSES[l], "test") for p, l in test]
    folds, hists = [], []
    print(f"\n{'=' * 70}\n=== {g.upper()} — {K_FOLDS}-fold, backbone={BACKBONE}, source={CHOSEN_SOURCE} ===\n{'=' * 70}")
    for k, (tr_idx, va_idx) in enumerate(sgkf.split(np.zeros(len(pool)), y, group_ids(pool)), 1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + k)
        tr_pairs, va_pairs = [pool[i] for i in tr_idx], [pool[i] for i in va_idx]
        split_rows += [(p, crop_manifest[p]["crop"], CANONICAL_CLASSES[l], f"fold{k}_val") for p, l in va_pairs]
        print(f"\n--- {g} fold {k}/{K_FOLDS}: train {len(tr_pairs)} {counts_of(tr_pairs)} | val {len(va_pairs)} ---")
        tr_p, tr_l = crops_of(tr_pairs)
        va_p, va_l = crops_of(va_pairs)
        if k == 1:
            save_aug_preview(tr_p, os.path.join(OUTPUT_DIR, f"aug_preview_{g}.jpg"))
        ckpt = os.path.join(OUTPUT_DIR, f"face_shape_{g}_fold{k}.keras")
        model, hist, phase1_len = train_fold(tr_p, tr_l, va_p, va_l, ckpt)
        model.save(ckpt)        # restore_best_weights แล้ว = น้ำหนักที่ดีที่สุดของ fold นี้
        val_acc, _ = evaluate(model, va_p, va_l)
        test_acc, test_pred = evaluate(model, test_p, test_l)
        print(f"{g} fold {k}: val={val_acc:.4f}  test={test_acc:.4f}")
        folds.append({"fold": k, "val_acc": val_acc, "test_acc": test_acc, "model": ckpt,
                      "confusion": confusion_matrix(test_l, test_pred, labels=range(num_classes)).tolist(),
                      "report": classification_report(test_l, test_pred, labels=range(num_classes),
                                                      target_names=CANONICAL_CLASSES, output_dict=True, zero_division=0)})
        hists.append((hist, phase1_len))
        del model
    write_split(split_rows, os.path.join(OUTPUT_DIR, f"splits_{g}.csv"))
    va = np.array([f["val_acc"] for f in folds]); te = np.array([f["test_acc"] for f in folds])
    best = max(folds, key=lambda f: f["val_acc"])          # เลือกด้วย val เท่านั้น (ไม่แอบดู test)
    final_path = os.path.join(OUTPUT_DIR, f"face_shape_{g}.keras")
    tf.keras.models.load_model(best["model"], compile=False).save(final_path)
    results[g] = {"folds": folds, "val_mean": float(va.mean()), "val_std": float(va.std(ddof=1)) if len(va) > 1 else 0.0,
                  "test_mean": float(te.mean()), "test_std": float(te.std(ddof=1)) if len(te) > 1 else 0.0,
                  "best_fold": best["fold"], "best_test_acc": best["test_acc"], "final_model": final_path,
                  "test_from": info["test_from"], "n_trainval": len(pool), "n_test": len(test)}
    print(f"\n### {g}: val {va.mean():.2%} ± {results[g]['val_std']:.2%} | test {te.mean():.2%} ± {results[g]['test_std']:.2%}"
          f" | ใช้ fold {best['fold']} (test {best['test_acc']:.2%}) -> {final_path}")
    print("Confusion matrix (fold ที่เลือก, แถว=จริง คอลัมน์=ทาย):", CANONICAL_CLASSES)
    for name, row in zip(CANONICAL_CLASSES, best["confusion"]):
        print(f"  {name:7s} {row}")

    fig, axes = plt.subplots(1, len(hists), figsize=(5 * len(hists), 3.6), squeeze=False)
    for ax, (k, (hist, p1)) in zip(axes[0], enumerate(hists, 1)):
        ax.plot(hist["accuracy"], label="train"); ax.plot(hist["val_accuracy"], label="val")
        ax.axvline(p1 - 0.5, color="gray", ls="--", lw=1)
        ax.set_title(f"{g} fold {k}"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, f"history_{g}.png"), dpi=120); plt.close(fig)

# =============================================================================
# ### 8) บันทึกผล + ไฟล์ที่ backend ต้องใช้
# =============================================================================
with open(os.path.join(OUTPUT_DIR, "cv_results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gender", "fold", "val_acc", "test_acc"])
    for g, r in results.items():
        for fo in r["folds"]:
            w.writerow([g, fo["fold"], f"{fo['val_acc']:.4f}", f"{fo['test_acc']:.4f}"])
        w.writerow([g, "mean", f"{r['val_mean']:.4f}", f"{r['test_mean']:.4f}"])
        w.writerow([g, "std", f"{r['val_std']:.4f}", f"{r['test_std']:.4f}"])
with open(os.path.join(OUTPUT_DIR, "cv_results.json"), "w", encoding="utf-8") as f:
    json.dump({"backbone": BACKBONE, "source": CHOSEN_SOURCE, "k_folds": K_FOLDS, "seed": SEED, "results": results},
              f, ensure_ascii=False, indent=2)

from dataclasses import asdict as _asdict
meta = {
    "class_names": CANONICAL_CLASSES,
    "img_size": list(IMG_SIZE),
    "resize": "resize_with_pad",
    "preprocess": PREPROCESS_NAME,
    "backbone": BACKBONE,
    "source": CHOSEN_SOURCE,
    "models": {g: os.path.basename(r["final_model"]) for g, r in results.items()},
    "accuracy": {g: {"cv_test_mean": round(r["test_mean"], 4), "cv_test_std": round(r["test_std"], 4),
                     "final_test": round(r["best_test_acc"], 4)} for g, r in results.items()},
    "face_crop": {"module": "face_crop.py", "version": face_crop.CROP_VERSION, "params": _asdict(face_crop.CropParams())},
}
with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print(f"สรุป: backbone={BACKBONE}, แหล่งข้อมูล={CHOSEN_SOURCE}, {K_FOLDS}-fold")
for g, r in results.items():
    print(f"  {g:7s} val {r['val_mean']:.2%} ± {r['val_std']:.2%} | test {r['test_mean']:.2%} ± {r['test_std']:.2%}"
          f" | โมเดลที่ใช้: fold {r['best_fold']} (test {r['best_test_acc']:.2%})")
print("\nไฟล์ที่ต้องดาวน์โหลดไปใส่ backend/models/: " + ", ".join(meta["models"].values()) + ", class_names.json")
print("ไฟล์ผลการทดลองสำหรับเล่ม: source_audit.csv, cv_results.csv/json, history_*.png, gender_check_*.jpg, aug_preview_*.jpg")
print("DONE.")
