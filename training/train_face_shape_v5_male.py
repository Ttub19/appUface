# =============================================================================
# uShape — face-shape training v5 (เฉพาะเพศชาย)
#   ปัญหา v4: ชายทาย heart ไม่ได้เลย (ข้อมูลเทรน heart ชายมีแค่ 9 รูป) แม้ top-1 รวมจะสูงขึ้น
#   แนวคิด: เริ่มจากโมเดลหญิง v4 (ConvNeXt-Tiny ที่เห็น heart มากว่า 3,600 รูป) แล้วปรับต่อด้วยข้อมูลชาย
#   ลอง 2 แบบในรอบเดียว แล้วเลือกด้วย "ค่าเฉลี่ย val" (กติกาตั้งไว้ก่อน ไม่ดูชุดทดสอบ):
#     A "male_only"  : ปรับต่อด้วยข้อมูลชายอย่างเดียว (LR ต่ำ)
#     B "mix_female" : ข้อมูลชาย + รูปหญิงที่โมเดลตั้งต้นเคยเห็น คลาสละ MIX_PER_CLASS รูป (กันลืม heart)
#   split / ชุดทดสอบ / การทำความสะอาดป้าย เหมือน v3/v4 ทุกอย่าง -> เทียบกันได้ตรง ๆ
#   จบด้วย .tflite + วัดผลแบบเดียวกับเซิร์ฟเวอร์ (ชายชุดทดสอบ 396 รูป ใช้เวลาไม่กี่นาที)
#   + (รวมไว้ในรอบเดียว) วัดผล .tflite หญิง v4 ที่แปลงไว้แล้ว -> tflite_v4_report.json
#     ทำก่อนเริ่มเทรน (ใช้ CPU ~30-60 นาที) ถ้าเทรนพังทีหลัง ผลหญิงก็ยังได้
#
# Kaggle: Settings -> Accelerator = GPU T4 x2, Internet = On
#   Add Input: faceshape1 + facetrain-v1-output (เหมือน v4)
#            + output ของ notebook v4 (ที่มี face_shape_female_v4_fold1..3.keras)
# รันทั้งไฟล์ในเซลล์เดียว (Save & Run All) — ประมาณ 2–3 ชม.
# =============================================================================
import os
import sys
import csv
import json
import math
import time
import shutil
import zipfile
from collections import Counter, defaultdict

import numpy as np

# =============================================================================
# ### CONFIG
# =============================================================================
GENDERS = ["male"]                      # เทรน/วัดผลเฉพาะชาย
LOAD_GENDERS = ["male", "female"]       # หญิงใช้เป็นข้อมูลเสริมของแบบ B
K_FOLDS = 3
SEED = 42
BACKBONE = "convnext_tiny"              # "convnext_tiny" | "effnetv2s" | "resnet50"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32                         # ต่อ GPU
EPOCHS_HEAD = 3                         # เฟส 1: ล็อก backbone เทรนเฉพาะหัว
EPOCHS_FINETUNE = 30                    # เฟส 2: ปลดล็อกทั้งโมเดล
PATIENCE = 6                            # val ไม่ดีขึ้นกี่ epoch แล้วหยุด (คืนน้ำหนักที่ดีที่สุด)
LR_HEAD = 1e-3
LR_FINETUNE = 6e-5
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3
CLASS_WEIGHT_MAX = 5.0

# --- v5: ปรับต่อจากโมเดลหญิง v4 ---
VARIANTS = ["male_only", "mix_female"]
MIX_PER_CLASS = 150                     # แบบ B: รูปหญิงต่อคลาสต่อ fold (จาก fold ที่โมเดลตั้งต้นเคยเทรน)
EPOCHS_V5 = 20
LR_V5 = 2e-5
PATIENCE_V5 = 6

EVAL_FEMALE_V4_TFLITE = True            # วัดผล face_shape_female_v4_fold*.tflite จาก input

# --- ทำความสะอาดป้าย (เฉพาะ train/val, ไม่แตะชุดทดสอบ) ---
CLEAN = True
CLEAN_MAX_P_LABEL = 0.10                # โมเดล v3 ให้ป้ายเดิมต่ำกว่านี้
CLEAN_MIN_P_OTHER = 0.70                # และให้ทรงอื่นสูงกว่านี้ -> ตัดออก
CLEAN_MAX_FRACTION = 0.15               # ตัดไม่เกินกี่ % ต่อคลาส (กันตัดคลาสยาก ๆ ทิ้งหมด)

# --- ความเร็ว ---
USE_MULTI_GPU = True
USE_MIXED_PRECISION = False             # True เร็วขึ้น ~2 เท่าบน T4 แต่เสี่ยง loss เป็น nan
CACHE_IMAGES_IN_RAM = True
QUICK_TEST = False                      # True = ลองให้จบเร็ว: รูปน้อย + epoch น้อย (เช็กว่าโค้ดไม่พัง)
QUICK_TEST_PER_CLASS = 40

QUANT = "dynamic"                       # .tflite: "dynamic" (int8 weights) | "float16" | "none"
OUTPUT_DIR = "/kaggle/working"
FACE_CROP_DIR = os.path.join(OUTPUT_DIR, "face_crops")
WORK = os.path.join(OUTPUT_DIR, "v5_work")
TFLITE_DIR = os.path.join(OUTPUT_DIR, "tflite_v5_male")
CLASS_NAMES = ["heart", "oblong", "oval", "round", "square"]
C = len(CLASS_NAMES)

# ผล v3 (ensemble 3 fold + TTA, ชุดทดสอบเดียวกัน) ไว้เทียบ
V3_REFERENCE = {"female": {"top1": 0.8084, "top2": 0.9123, "macro_f1": 0.8071, "n_test": 2986},
                "male": {"top1": 0.8586, "top2": 0.9697, "macro_f1": 0.8374, "n_test": 396}}

if QUICK_TEST:
    EPOCHS_HEAD, EPOCHS_FINETUNE, PATIENCE = 1, 2, 2
    EPOCHS_V5, PATIENCE_V5, MIX_PER_CLASS = 2, 2, 10
for d in (WORK, TFLITE_DIR):
    os.makedirs(d, exist_ok=True)

# =============================================================================
# ### (ไม่ต้องแก้) face_crop.py + image_ops.py ตัวเดียวกับ backend
# =============================================================================
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
IMAGE_OPS_SOURCE = r'''"""
Image preprocessing for the face-shape models in plain NumPy (no TensorFlow).

The models were trained on Kaggle with TensorFlow:

    img = tf.image.resize_with_pad(img, 224, 224)      # bilinear, half-pixel centres, no antialias
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = keras.applications.resnet50.preprocess_input(img)

`resize_with_pad` and `apply_preprocess` below reproduce those steps exactly,
so the server needs only NumPy + LiteRT at inference time. The Kaggle script
training/convert_tflite.py checks this file against TensorFlow on the real
test images before the models are shipped.
"""
import math

import numpy as np

IMAGENET_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """tf.image.resize(method="bilinear", antialias=False): half-pixel centres, edge clamping."""
    in_h, in_w = img.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return img.astype(np.float32, copy=True)

    def axis(out_size, in_size):
        scale = in_size / out_size
        src = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
        lo_f = np.floor(src)
        frac = (src - lo_f).astype(np.float32)
        lo = np.clip(lo_f, 0, in_size - 1).astype(np.int64)
        hi = np.clip(lo_f + 1, 0, in_size - 1).astype(np.int64)
        return lo, hi, frac

    y0, y1, fy = axis(out_h, in_h)
    x0, x1, fx = axis(out_w, in_w)
    img = img.astype(np.float32, copy=False)
    top = img[y0][:, x0] * (1 - fx)[None, :, None] + img[y0][:, x1] * fx[None, :, None]
    bottom = img[y1][:, x0] * (1 - fx)[None, :, None] + img[y1][:, x1] * fx[None, :, None]
    return top * (1 - fy)[:, None, None] + bottom * fy[:, None, None]


def resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """tf.image.resize_with_pad: keep the aspect ratio, resize, pad the rest with zeros (centred)."""
    h, w = img.shape[:2]
    ratio = max(w / target_w, h / target_h)
    new_h = int(math.floor(h / ratio))
    new_w = int(math.floor(w / ratio))
    resized = _resize_bilinear(img, new_h, new_w)
    out = np.zeros((target_h, target_w, img.shape[2]), dtype=np.float32)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


def to_uint8_range(img: np.ndarray) -> np.ndarray:
    """Training stored the resized image as uint8 (round + clip) before preprocessing."""
    return np.clip(np.round(img), 0, 255).astype(np.float32)


def apply_preprocess(arr, name):
    """arr: (..., H, W, 3) float32 RGB 0-255 -> what the backbone was trained on."""
    if name == "resnet50":   # keras.applications.resnet50.preprocess_input ("caffe" mode)
        out = arr[..., ::-1].astype(np.float32, copy=True)
        out -= IMAGENET_BGR_MEAN
        return out
    if name in ("none", "", None):
        return arr
    raise ValueError(f"unknown preprocess {name!r}")


def prepare_batch(rgb: np.ndarray, img_size, preprocess: str, tta: bool) -> np.ndarray:
    """(H, W, 3) RGB 0-255 -> (1 or 2, h, w, 3) model input; with tta the second row is the mirror image."""
    x = to_uint8_range(resize_with_pad(rgb, img_size[0], img_size[1]))
    batch = np.stack([x, x[:, ::-1]]) if tta else x[None]
    return apply_preprocess(batch, preprocess)
'''
for name, src in (("face_crop.py", FACE_CROP_SOURCE), ("image_ops.py", IMAGE_OPS_SOURCE)):
    with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as f:
        f.write(src)
if OUTPUT_DIR not in sys.path:
    sys.path.insert(0, OUTPUT_DIR)
import importlib
import face_crop
import image_ops
face_crop, image_ops = importlib.reload(face_crop), importlib.reload(image_ops)


# =============================================================================
# ### 1) หา dataset + split/โมเดลของ v3
# =============================================================================
def walk_files(base="/kaggle/input"):
    yield from os.walk(base)


def find_data_root(base="/kaggle/input"):
    for root, dirs, _ in walk_files(base):
        if "train1_cropped" in dirs and "testing_set" in dirs:
            return root
    raise FileNotFoundError("ไม่พบ dataset faceshape1 (โฟลเดอร์ที่มี train1_cropped + testing_set) — Add Input ก่อน")


def as_keras_file(path):
    """notebook/dataset อาจแตก .keras เป็นโฟลเดอร์ -> รวมกลับเป็นไฟล์"""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.weights.h5")):
        out = os.path.join(WORK, os.path.basename(path))
        if not os.path.exists(out):
            with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
                for f in sorted(os.listdir(path)):
                    z.write(os.path.join(path, f), f)
        return out
    return None


def splits_name(files, gender):
    for n in (f"splits_{gender}.csv", f"splits_{gender}.csv.txt", f"splits_{gender}.txt"):
        if n in files:
            return n
    return None


def find_v3_run(gender, base="/kaggle/input"):
    for root, _, files in walk_files(base):
        name = splits_name(files, gender)
        if not name:
            continue
        if "male_extra" in open(os.path.join(root, name), encoding="utf-8").read():
            continue
        folds = [as_keras_file(os.path.join(root, f"face_shape_{gender}_fold{k}.keras")) for k in range(1, K_FOLDS + 1)]
        if all(folds):
            return os.path.join(root, name), folds
    return None


DATA_ROOT = find_data_root()
print("DATA_ROOT:", DATA_ROOT)


# ตรวจ input ของ v5 ตั้งแต่ต้น (ไม่ต้องรอครอปหน้าเสร็จแล้วค่อยพัง)
def _dir_to_keras(d, out_name):
    """โฟลเดอร์ที่ Kaggle แตกจาก .keras/.zip (มี model.weights.h5 อยู่ข้างในไม่เกิน 2 ชั้น) -> ไฟล์ .keras"""
    for root, _, files in os.walk(d):
        if "model.weights.h5" in files and "config.json" in files:
            out = os.path.join(WORK, out_name)
            if not os.path.exists(out):
                with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
                    for f in sorted(files):
                        z.write(os.path.join(root, f), f)
            return out
        if os.path.relpath(root, d).count(os.sep) >= 2:
            break
    return None


def find_female_v4(k, base="/kaggle/input"):
    """face_shape_female_v4_fold{k} ในรูปแบบไหนก็ได้: .keras / .zip (ดาวน์โหลดจาก Kaggle) / โฟลเดอร์ที่ถูกแตกไฟล์"""
    stem = f"face_shape_female_v4_fold{k}"
    out_name = stem + ".keras"
    for root, dirs, files in walk_files(base):
        for f in files:
            if f in (stem + ".keras", stem + ".zip", stem + ".keras.zip"):
                src = os.path.join(root, f)
                if f.endswith(".keras"):
                    return src
                out = os.path.join(WORK, out_name)          # .keras ก็คือ zip อยู่แล้ว แค่เปลี่ยนนามสกุล
                if not os.path.exists(out):
                    with zipfile.ZipFile(src) as z:
                        names = z.namelist()
                    if "model.weights.h5" in names:
                        shutil.copyfile(src, out)
                    else:                                    # zip ที่ห่อโฟลเดอร์ไว้อีกชั้น
                        tmp = os.path.join(WORK, stem + "_unzip")
                        with zipfile.ZipFile(src) as z:
                            z.extractall(tmp)
                        return _dir_to_keras(tmp, out_name)
                return out
        for d in dirs:
            if d in (stem, stem + ".keras", stem + ".zip"):
                got = _dir_to_keras(os.path.join(root, d), out_name)
                if got:
                    return got
    return None


def _check_v5_inputs(base="/kaggle/input"):
    found = {k: find_female_v4(k, base) for k in range(1, K_FOLDS + 1)}
    tfl = sum(1 for _, _, files in walk_files(base) for f in files
              if f.startswith("face_shape_female_v4_fold") and f.endswith(".tflite"))
    print("[input] พบโมเดลหญิง v4:", {f"fold{k}": bool(v) for k, v in found.items()}, f"| .tflite หญิง v4 {tfl} ไฟล์")
    if not all(found.values()):
        print("[input] ไฟล์/โฟลเดอร์ที่แนบอยู่ตอนนี้ (2 ชั้นแรก):")
        for root, dirs, files in walk_files(base):
            depth = os.path.relpath(root, base).count(os.sep)
            if depth <= 1:
                print("   ", os.path.relpath(root, base), "|", sorted(dirs)[:8], sorted(files)[:8])
        raise FileNotFoundError("ไม่พบ face_shape_female_v4_fold1..3 (.keras / .zip) — Add Input เป็น output ของ notebook v4 "
                                "หรือ Dataset ที่อัปโหลดไฟล์เหล่านี้")
    return found


FEMALE_V4_INIT = _check_v5_inputs()


def local_original(path):
    if os.path.exists(path):
        return path
    key = "datasetformodelai" + os.sep
    if key in path:
        cand = os.path.join(DATA_ROOT, path.split(key, 1)[1])
        if os.path.exists(cand):
            return cand
    return None


def read_splits(path):
    """-> {"test": [(orig, label)], 1: [...fold1 val], 2: [...], 3: [...]}"""
    parts, missing = defaultdict(list), 0
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = local_original(r["original"])
            if not p:
                missing += 1
                continue
            sp = r["split"]
            key = "test" if sp == "test" else int(sp.replace("fold", "").split("_")[0])
            parts[key].append((p, CLASS_NAMES.index(r["label"])))
    return parts, missing


def subsample(pairs, per_class, seed):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for p, l in pairs:
        by[l].append((p, l))
    out = []
    for l in sorted(by):
        g = by[l]
        out += [g[i] for i in sorted(rng.permutation(len(g))[:per_class])]
    return out


data, v3_models = {}, {}
for g in LOAD_GENDERS:
    run = find_v3_run(g)
    if not run:
        raise FileNotFoundError(f"ไม่พบ splits_{g} + face_shape_{g}_fold1..3 ของ v3 — แนบ dataset facetrain-v1-output")
    parts, missing = read_splits(run[0])
    if QUICK_TEST:
        parts = {k: subsample(v, QUICK_TEST_PER_CLASS if k != "test" else QUICK_TEST_PER_CLASS // 2, SEED) for k, v in parts.items()}
    data[g], v3_models[g] = parts, run[1]
    print(f"{g}: test {len(parts['test'])} | " + " | ".join(f"fold{k} {len(parts[k])}" for k in range(1, K_FOLDS + 1))
          + (f" | หาไฟล์ไม่เจอ {missing}" if missing else ""))

# =============================================================================
# ### 2) ครอปหน้า (face_crop ตัวเดียวกับตอนเทรน v3 และ backend)
# =============================================================================
all_paths = sorted({p for g in LOAD_GENDERS for v in data[g].values() for p, _ in v})
manifest = face_crop.ensure_face_crops(all_paths, DATA_ROOT, FACE_CROP_DIR)
for g in LOAD_GENDERS:
    for k in list(data[g]):
        before = len(data[g][k])
        data[g][k] = [(p, l) for p, l in data[g][k] if manifest.get(p, {}).get("status", "").startswith("mesh")]
        if len(data[g][k]) != before:
            print(f"  {g} {k}: ตัดรูปที่หาหน้าไม่เจอ {before - len(data[g][k])} รูป")
crop = lambda p: manifest[p]["crop"]

# =============================================================================
# ### 3) tf.data + augmentation (เหมือน v3)
# =============================================================================
import tensorflow as tf
import keras
from keras import layers, models, optimizers, callbacks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

keras.utils.set_random_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE
_gpus = tf.config.list_physical_devices("GPU")
STRATEGY = tf.distribute.MirroredStrategy() if (USE_MULTI_GPU and len(_gpus) > 1) else tf.distribute.get_strategy()
keras.mixed_precision.set_global_policy("mixed_float16" if (USE_MIXED_PRECISION and _gpus) else "float32")
GLOBAL_BATCH = BATCH_SIZE * STRATEGY.num_replicas_in_sync
print(f"TensorFlow {tf.__version__}, GPU {len(_gpus)}, batch รวม {GLOBAL_BATCH}, backbone {BACKBONE}")

# preprocessing ของ backbone v4: ConvNeXt / EfficientNetV2 มี normalization อยู่ในโมเดลแล้ว (รับ 0-255)
PREPROCESS_NAME = "resnet50" if BACKBONE == "resnet50" else "none"
v4_preprocess = resnet_preprocess if BACKBONE == "resnet50" else (lambda x: x)

geometric_aug = models.Sequential([
    layers.RandomFlip("horizontal", dtype="float32"),
    layers.RandomRotation(0.03, dtype="float32"),
    layers.RandomZoom(0.10, dtype="float32"),
    layers.RandomTranslation(0.05, 0.05, dtype="float32"),
], name="geometric_aug")
light_aug = models.Sequential([
    layers.RandomContrast(0.20, dtype="float32"),
    layers.RandomBrightness(0.15, value_range=(0, 255), dtype="float32"),
], name="light_aug")


def color_aug(img):
    x = img / 255.0
    x = tf.image.random_hue(x, 0.04)
    x = tf.image.random_saturation(x, 0.6, 1.4)
    gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))
    x = tf.where(tf.random.uniform([]) < 0.10, gray, x)
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0


def load_image(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize_with_pad(tf.cast(img, tf.float32), IMG_SIZE[0], IMG_SIZE[1])
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8), label


def augment(img, label):
    img = tf.cast(img, tf.float32)
    img = geometric_aug(img, training=True)
    img = light_aug(img, training=True)
    img = color_aug(img)
    img.set_shape(IMG_SIZE + (3,))
    return img, label


def make_ds(paths, labels, training, preprocess, flip=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load_image, num_parallel_calls=AUTOTUNE)
    if training and CACHE_IMAGES_IN_RAM:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(min(len(paths), 10000), seed=SEED, reshuffle_each_iteration=True).map(augment, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.image.flip_left_right(tf.cast(x, tf.float32)) if flip else tf.cast(x, tf.float32), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.batch(GLOBAL_BATCH).map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def predict_probs(model, pairs, preprocess, tta=True):
    paths, labels = [crop(p) for p, _ in pairs], [l for _, l in pairs]
    probs = model.predict(make_ds(paths, labels, False, preprocess), verbose=0)
    if tta:
        probs = (probs + model.predict(make_ds(paths, labels, False, preprocess, flip=True), verbose=0)) / 2
    return probs


def scores(probs, y):
    order = np.argsort(-probs, axis=1)
    pred = order[:, 0]
    return {"top1": round(float(np.mean(pred == y)), 4),
            "top2": round(float(np.mean((order[:, 0] == y) | (order[:, 1] == y))), 4),
            "macro_f1": round(float(f1_score(y, pred, labels=range(C), average="macro", zero_division=0)), 4),
            "per_class_f1": dict(zip(CLASS_NAMES, [round(float(v), 4) for v in
                                                   f1_score(y, pred, labels=range(C), average=None, zero_division=0)])),
            "confusion": confusion_matrix(y, pred, labels=range(C)).tolist()}


# =============================================================================
# ### 4) ทำความสะอาดป้าย: โมเดล v3 fold k ทาย val ของ fold k (out-of-fold) — ไม่แตะชุดทดสอบ
# =============================================================================
def clean_flags(labels, probs, max_p_label, min_p_other, max_fraction):
    """True = ป้ายน่าจะผิด (ตัดออก). เรียงจากมั่นใจที่สุด และไม่เกิน max_fraction ต่อคลาส"""
    labels = np.asarray(labels)
    p_label = probs[np.arange(len(labels)), labels]
    other = probs.copy()
    other[np.arange(len(labels)), labels] = -1
    p_other = other.max(axis=1)
    cand = (p_label < max_p_label) & (p_other >= min_p_other)
    flags = np.zeros(len(labels), bool)
    for c in np.unique(labels):
        idx = np.where((labels == c) & cand)[0]
        cap = int(math.floor(max_fraction * np.sum(labels == c)))
        keep_order = idx[np.argsort(-p_other[idx])][:cap]
        flags[keep_order] = True
    return flags, p_label, other.argmax(axis=1), p_other


def save_examples(rows, out_path, per_class=8, thumb=120):
    pick = []
    for c in range(C):
        pick += [r for r in rows if r[1] == c][:per_class]
    if not pick:
        return
    cols = per_class
    sheet = Image.new("RGB", (cols * thumb, C * (thumb + 18)), (255, 255, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    counts = Counter()
    for orig, lab, pred, pp in pick:
        i = counts[lab]
        counts[lab] += 1
        x, y = i * thumb, lab * (thumb + 18)
        with Image.open(crop(orig)) as im:
            sheet.paste(im.convert("RGB").resize((thumb, thumb)), (x, y + 18))
        d.text((x + 3, y + 3), f"{CLASS_NAMES[lab]}->{CLASS_NAMES[pred]} {pp:.0%}", fill=(0, 0, 0))
    sheet.save(out_path, quality=88)


cleaning = {}
for g in LOAD_GENDERS:
    dropped_all, report_rows = set(), []
    if CLEAN:
        for k in range(1, K_FOLDS + 1):
            model = keras.models.load_model(v3_models[g][k - 1], compile=False)
            pairs = data[g][k]
            probs = predict_probs(model, pairs, resnet_preprocess, tta=True)
            del model
            keras.backend.clear_session()
            y = np.array([l for _, l in pairs])
            flags, p_label, pred, p_other = clean_flags(y, probs, CLEAN_MAX_P_LABEL, CLEAN_MIN_P_OTHER, CLEAN_MAX_FRACTION)
            for (p, l), f, pl, pr, po in zip(pairs, flags, p_label, pred, p_other):
                report_rows.append((p, l, int(pr), float(po), float(pl), bool(f), k))
                if f:
                    dropped_all.add(p)
        with open(os.path.join(OUTPUT_DIR, f"cleaning_{g}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["original", "label", "v3_pred", "v3_p_pred", "v3_p_label", "dropped", "fold"])
            for p, l, pr, po, pl, fl, k in report_rows:
                w.writerow([p, CLASS_NAMES[l], CLASS_NAMES[pr], f"{po:.4f}", f"{pl:.4f}", int(fl), k])
        dropped_rows = sorted([(p, l, pr, po) for p, l, pr, po, pl, fl, k in report_rows if fl], key=lambda r: -r[3])
        save_examples(dropped_rows, os.path.join(OUTPUT_DIR, f"cleaning_examples_{g}.jpg"))
    by_class = Counter(l for k in range(1, K_FOLDS + 1) for p, l in data[g][k] if p in dropped_all)
    total = Counter(l for k in range(1, K_FOLDS + 1) for _, l in data[g][k])
    cleaning[g] = {"dropped": len(dropped_all), "trainval": sum(total.values()),
                   "per_class": {CLASS_NAMES[c]: f"{by_class[c]}/{total[c]}" for c in range(C)}}
    print(f"\n[clean] {g}: ตัด {len(dropped_all)}/{sum(total.values())} รูป "
          f"({len(dropped_all) / max(1, sum(total.values())):.1%}) {cleaning[g]['per_class']}  (ชุดทดสอบไม่แตะ)")
    for k in range(1, K_FOLDS + 1):
        data[g][k] = [(p, l) for p, l in data[g][k] if p not in dropped_all]


# =============================================================================
# ### 4.5) วัดผล .tflite หญิง v4 (ไฟล์ที่ v4 แปลงไว้แล้ว) แบบเดียวกับเซิร์ฟเวอร์: image_ops + ensemble 3 fold + TTA
# =============================================================================
def find_input_file(name, base="/kaggle/input"):
    for root, dirs, files in walk_files(base):
        for cand in (name, name + ".txt", os.path.splitext(name)[0] + ".txt"):
            if cand in files:
                return os.path.join(root, cand)
    return None


def litert_interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=path, num_threads=os.cpu_count())


def run_interp(it, x, batch=16):
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    out, shape = [], None
    for s0 in range(0, len(x), batch):
        xb = np.ascontiguousarray(x[s0:s0 + batch], dtype=np.float32)
        if xb.shape != shape:
            it.resize_tensor_input(i_in, list(xb.shape))
            it.allocate_tensors()
            shape = xb.shape
        it.set_tensor(i_in, xb)
        it.invoke()
        out.append(it.get_tensor(i_out).copy())
    return np.concatenate(out)


female_v4_tflite = None
if EVAL_FEMALE_V4_TFLITE:
    f_files = [find_input_file(f"face_shape_female_v4_fold{k}.tflite") for k in range(1, K_FOLDS + 1)]
    if not all(f_files):
        print("[tflite หญิง v4] ไม่พบไฟล์ face_shape_female_v4_fold*.tflite ใน input — ข้ามขั้นนี้")
    else:
        t0 = time.time()
        f_test = data["female"]["test"]
        y_f = np.array([l for _, l in f_test])
        interps = [litert_interpreter(f) for f in f_files]
        plain = [[] for _ in f_files]
        flipped = [[] for _ in f_files]
        CHUNK = 256
        print(f"\n[tflite หญิง v4] ชุดทดสอบ {len(f_test)} รูป x 3 โมเดล x 2 (TTA) บน CPU {os.cpu_count()} core ...")
        for c0 in range(0, len(f_test), CHUNK):
            xs = []
            for p, _ in f_test[c0:c0 + CHUNK]:
                with Image.open(crop(p)) as im:
                    xs.append(image_ops.prepare_batch(np.asarray(im.convert("RGB"), dtype="float32"), IMG_SIZE, "none", True))
            xs = np.stack(xs)
            for i, it in enumerate(interps):
                plain[i].append(run_interp(it, xs[:, 0]))
                flipped[i].append(run_interp(it, xs[:, 1]))
            done = min(c0 + CHUNK, len(f_test))
            print(f"  {done}/{len(f_test)} ({(time.time() - t0) / 60:.0f} นาที)", flush=True)
        plain = [np.concatenate(v) for v in plain]
        flipped = [np.concatenate(v) for v in flipped]
        female_v4_tflite = {
            "folds": [{"fold": k, "tta": scores((a + b) / 2, y_f)} for k, (a, b) in enumerate(zip(plain, flipped), 1)],
            "ensemble": scores(np.mean(plain, axis=0), y_f),
            "ensemble+TTA": scores(np.mean(plain + flipped, axis=0), y_f),
            "n_test": int(len(f_test)), "minutes": round((time.time() - t0) / 60, 1),
        }
        keras_ref = None
        cv4 = find_input_file("cv_results_v4.json")
        if cv4:
            with open(cv4, encoding="utf-8") as f:
                keras_ref = json.load(f)["results"]["female"]["ensemble+TTA"]
        r = female_v4_tflite["ensemble+TTA"]
        print(f"### หญิง v4 .tflite ensemble+TTA: top-1 {r['top1']:.2%} top-2 {r['top2']:.2%} macro F1 {r['macro_f1']:.3f}"
              + (f" (keras {keras_ref['top1']:.2%} / {keras_ref['top2']:.2%})" if keras_ref else "")
              + " | v3 บนเซิร์ฟเวอร์ 80.38% / 91.39%")
        print("  F1 ต่อคลาส:", r["per_class_f1"])
        with open(os.path.join(OUTPUT_DIR, "tflite_v4_report.json"), "w", encoding="utf-8") as f:
            json.dump({"female_tflite": female_v4_tflite, "female_keras_ensemble_tta": keras_ref,
                       "v3_tflite_reference": {"top1": 0.8038, "top2": 0.9139, "macro_f1": 0.802},
                       "preprocess": "none", "img_size": list(IMG_SIZE), "files": [os.path.basename(f) for f in f_files]},
                      f, ensure_ascii=False, indent=2)
        del interps, plain, flipped


# =============================================================================
# ### 5) โมเดล (สถาปัตยกรรมเดียวกับ v4) + เทรน v5 ต่อจากโมเดลหญิง
# =============================================================================
def build_model():
    shape = IMG_SIZE + (3,)
    if BACKBONE == "convnext_tiny":
        base = keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet", input_shape=shape,
                                               pooling="avg", include_preprocessing=True)
    elif BACKBONE == "effnetv2s":
        base = keras.applications.EfficientNetV2S(include_top=False, weights="imagenet", input_shape=shape,
                                                  pooling="avg", include_preprocessing=True)
    else:
        base = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    inputs = layers.Input(shape=shape)
    x = base(inputs, training=False)        # BatchNorm (ถ้ามี) ใช้ค่าสถิติเดิม; ConvNeXt ใช้ LayerNorm
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(C, activation="softmax", dtype="float32")(x)
    return models.Model(inputs, outputs), base


def loss_fn(y_true, y_pred):
    y = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=C)
    return keras.losses.categorical_crossentropy(y, y_pred, label_smoothing=LABEL_SMOOTHING)


def train_fold(tr, va, ckpt):
    tr_p, tr_l = [crop(p) for p, _ in tr], [l for _, l in tr]
    va_p, va_l = [crop(p) for p, _ in va], [l for _, l in va]
    counts = Counter(tr_l)
    class_weight = {i: min(CLASS_WEIGHT_MAX, len(tr_l) / (C * counts[i])) if counts[i] else 0.0 for i in range(C)}
    print("class weight:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weight.items()})
    tr_ds, va_ds = make_ds(tr_p, tr_l, True, v4_preprocess), make_ds(va_p, va_l, False, v4_preprocess)
    steps = math.ceil(len(tr_p) / GLOBAL_BATCH)
    with STRATEGY.scope():
        model, base = build_model()
        base.trainable = False
        model.compile(optimizer=optimizers.Adam(LR_HEAD), loss=loss_fn, metrics=["accuracy"])
    h1 = model.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS_HEAD, class_weight=class_weight, verbose=2)
    with STRATEGY.scope():
        base.trainable = True
        warm = steps * WARMUP_EPOCHS
        schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=LR_FINETUNE * 0.01, decay_steps=max(1, steps * EPOCHS_FINETUNE - warm),
            alpha=0.01, warmup_target=LR_FINETUNE, warmup_steps=warm)
        opt = optimizers.AdamW(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
        opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "layer_scale", "normalization"])
        model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    cbs = [callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True)]
    h2 = model.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS_FINETUNE, class_weight=class_weight, callbacks=cbs, verbose=2)
    model.save(ckpt)
    hist = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in ("accuracy", "val_accuracy")}
    return model, hist, len(h1.history["accuracy"])


def convert(keras_path, out_path):
    model = keras.models.load_model(keras_path, compile=False)
    sm = os.path.join(WORK, os.path.basename(out_path) + "_sm")
    shutil.rmtree(sm, ignore_errors=True)
    try:
        model.export(sm, format="tf_saved_model")
        conv = tf.lite.TFLiteConverter.from_saved_model(sm)
    except Exception as exc:
        print(f"  model.export failed ({exc}); using from_keras_model")
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]     # เซิร์ฟเวอร์ไม่มี Flex ops
    if QUANT in ("dynamic", "float16"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if QUANT == "float16":
        conv.target_spec.supported_types = [tf.float16]
    with open(out_path, "wb") as f:
        f.write(conv.convert())
    shutil.rmtree(sm, ignore_errors=True)


# ลองแปลงโมเดลเปล่าเป็น .tflite ก่อนเทรน — ถ้า backbone นี้แปลงไม่ได้ จะรู้ตั้งแต่นาทีแรก ไม่ใช่หลังเทรนหลายชั่วโมง
_smoke_k, _smoke_t = os.path.join(WORK, "smoke.keras"), os.path.join(WORK, "smoke.tflite")
build_model()[0].save(_smoke_k)
convert(_smoke_k, _smoke_t)
print(f"[smoke] {BACKBONE} -> .tflite ได้ ({os.path.getsize(_smoke_t) / 1e6:.1f} MB)")
keras.backend.clear_session()

# ---- โมเดลหญิง v4 (ตั้งต้น) + กันรูปซ้ำ
female_init = [FEMALE_V4_INIT[k] for k in range(1, K_FOLDS + 1)]
if not all(female_init):
    raise FileNotFoundError("ไม่พบ face_shape_female_v4_fold1..3.keras — Add Input เป็น output ของ notebook v4")
print("โมเดลตั้งต้น:", female_init)

import hashlib


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# รูปชายทุกรูป (เทรน/val/ทดสอบ) -> ห้ามมีรูปหญิงที่เนื้อไฟล์เหมือนกันปนเข้าไปในชุดเสริม
male_md5 = {md5(p) for v in data["male"].values() for p, _ in v}


def female_mix(k, seed):
    """รูปหญิงจาก fold ที่โมเดลหญิง v4 fold k เคยเทรน (ไม่ใช่ val k / ไม่ใช่ชุดทดสอบ) คลาสละ MIX_PER_CLASS รูป"""
    pool = [pl for j in range(1, K_FOLDS + 1) if j != k for pl in data["female"][j]]
    rng = np.random.default_rng(seed)
    out = []
    for c in range(C):
        cand = [pool[i] for i in rng.permutation(len(pool)) if pool[i][1] == c]
        picked = []
        for p, l in cand:
            if len(picked) >= MIX_PER_CLASS:
                break
            if md5(p) not in male_md5:
                picked.append((p, l))
        out += picked
    return out


def train_fold_v5(tr, va, ckpt, init_path):
    tr_p, tr_l = [crop(p) for p, _ in tr], [l for _, l in tr]
    va_p, va_l = [crop(p) for p, _ in va], [l for _, l in va]
    counts = Counter(tr_l)
    class_weight = {i: min(CLASS_WEIGHT_MAX, len(tr_l) / (C * counts[i])) if counts[i] else 0.0 for i in range(C)}
    print("class weight:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weight.items()}, "| train", dict(sorted(counts.items())))
    tr_ds, va_ds = make_ds(tr_p, tr_l, True, v4_preprocess), make_ds(va_p, va_l, False, v4_preprocess)
    steps = math.ceil(len(tr_p) / GLOBAL_BATCH)
    with STRATEGY.scope():
        model = keras.models.load_model(init_path, compile=False)
        model.trainable = True
        warm = steps * WARMUP_EPOCHS
        schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=LR_V5 * 0.01, decay_steps=max(1, steps * EPOCHS_V5 - warm),
            alpha=0.01, warmup_target=LR_V5, warmup_steps=warm)
        opt = optimizers.AdamW(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
        opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "layer_scale", "normalization"])
        model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    h0 = model.evaluate(va_ds, verbose=0)          # จุดเริ่ม: โมเดลหญิงล้วน ทายรูปชาย
    cbs = [callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE_V5, restore_best_weights=True)]
    h = model.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS_V5, class_weight=class_weight, callbacks=cbs, verbose=2)
    model.save(ckpt)
    hist = {"accuracy": [float("nan")] + h.history["accuracy"], "val_accuracy": [float(h0[1])] + h.history["val_accuracy"]}
    return model, hist


results, keras_files, variant_results = {}, {}, {}
t_start = time.time()
g = "male"
test = data[g]["test"]
y_test = np.array([l for _, l in test])
split_rows = [(p, crop(p), CLASS_NAMES[l], "test") for p, l in test]
for k in range(1, K_FOLDS + 1):
    split_rows += [(p, crop(p), CLASS_NAMES[l], f"fold{k}_val") for p, l in data[g][k]]

# จุดอ้างอิง: โมเดลหญิง v4 ทายรูปชายตรง ๆ (ยังไม่ปรับ)
init_probs = []
for k in range(1, K_FOLDS + 1):
    keras.backend.clear_session()
    m = keras.models.load_model(female_init[k - 1], compile=False)
    init_probs.append(predict_probs(m, test, v4_preprocess, tta=True))
    del m
female_only = scores(np.mean(init_probs, axis=0), y_test)
print(f"\n[อ้างอิง] โมเดลหญิง v4 ทายชายตรง ๆ (ensemble+TTA): top-1 {female_only['top1']:.2%} top-2 {female_only['top2']:.2%} "
      f"heart F1 {female_only['per_class_f1']['heart']:.2f}")

for variant in VARIANTS:
    print(f"\n{'=' * 70}\n=== {g.upper()} v5 — แบบ {variant} (เริ่มจากโมเดลหญิง v4) ===\n{'=' * 70}")
    fold_probs, folds, hists, ckpts = {}, [], [], []
    for k in range(1, K_FOLDS + 1):
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + k)
        va = data[g][k]
        tr = [pl for j in range(1, K_FOLDS + 1) if j != k for pl in data[g][j]]
        extra = female_mix(k, SEED + k) if variant == "mix_female" else []
        print(f"\n--- {variant} fold {k}: train ชาย {len(tr)} + หญิง {len(extra)} | val {len(va)} | test {len(test)} ---")
        t0 = time.time()
        ckpt = os.path.join(OUTPUT_DIR, f"face_shape_male_v5_{variant}_fold{k}.keras")
        model, hist = train_fold_v5(tr + extra, va, ckpt, female_init[k - 1])
        ckpts.append(ckpt)
        val_s = scores(predict_probs(model, va, v4_preprocess, tta=False), np.array([l for _, l in va]))
        plain = predict_probs(model, test, v4_preprocess, tta=False)
        both = predict_probs(model, test, v4_preprocess, tta=True)
        fold_probs[k] = (plain, both)
        fs = {"fold": k, "val": val_s["top1"], "val_macro_f1": val_s["macro_f1"], "start_val": hist["val_accuracy"][0],
              "test": scores(plain, y_test), "test_tta": scores(both, y_test), "minutes": round((time.time() - t0) / 60, 1)}
        folds.append(fs)
        hists.append(hist)
        print(f"{variant} fold {k}: val {fs['start_val']:.2%} -> {fs['val']:.2%} | test {fs['test']['top1']:.2%} | "
              f"test+TTA {fs['test_tta']['top1']:.2%} | heart F1 {fs['test_tta']['per_class_f1']['heart']:.2f} | {fs['minutes']} นาที")
        del model
    ens = scores(np.mean([fold_probs[k][0] for k in fold_probs], axis=0), y_test)
    ens_tta = scores(np.mean([fold_probs[k][1] for k in fold_probs], axis=0), y_test)
    mean_val = float(np.mean([f["val"] for f in folds]))
    variant_results[variant] = {"folds": folds, "ensemble": ens, "ensemble+TTA": ens_tta, "mean_val": round(mean_val, 4),
                                "keras_files": ckpts, "n_test": int(len(test))}
    fig, axes = plt.subplots(1, len(hists), figsize=(5 * len(hists), 3.6), squeeze=False)
    for ax, (k, hh) in zip(axes[0], enumerate(hists, 1)):
        ax.plot(hh["accuracy"], label="train")
        ax.plot(hh["val_accuracy"], label="val")
        ax.set_title(f"male v5 {variant} fold {k}")
        ax.set_xlabel("epoch (0 = female v4, before fine-tune)")
        ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"history_v5_male_{variant}.png"), dpi=120)
    plt.close(fig)
    ref = V3_REFERENCE[g]
    print(f"\n### {variant}: mean val {mean_val:.2%} | ensemble+TTA top-1 {ens_tta['top1']:.2%} (v3 {ref['top1']:.2%}) | "
          f"top-2 {ens_tta['top2']:.2%} (v3 {ref['top2']:.2%}) | macroF1 {ens_tta['macro_f1']:.3f} (v3 {ref['macro_f1']:.3f})")
    print("  F1 ต่อคลาส:", ens_tta["per_class_f1"])
    print("  confusion (แถว=จริง คอลัมน์=ทาย)", CLASS_NAMES)
    for name, row in zip(CLASS_NAMES, ens_tta["confusion"]):
        print(f"    {name:7s} {row}")

# เลือกแบบด้วยค่าเฉลี่ย val (ตั้งกติกาไว้ก่อน ไม่ใช้ชุดทดสอบตัดสิน)
CHOSEN = max(VARIANTS, key=lambda v: variant_results[v]["mean_val"])
print(f"\nเลือกแบบ {CHOSEN} (mean val สูงสุด: " + ", ".join(f"{v} {variant_results[v]['mean_val']:.2%}" for v in VARIANTS) + ")")
results[g] = {**variant_results[CHOSEN], "variant": CHOSEN, "cleaning": cleaning[g], "v3_reference": V3_REFERENCE[g]}
keras_files[g] = variant_results[CHOSEN]["keras_files"]
with open(os.path.join(OUTPUT_DIR, "splits_v5_male.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["original", "crop", "label", "split"])
    w.writerows(split_rows)

with open(os.path.join(OUTPUT_DIR, "cv_results_v5_male.json"), "w", encoding="utf-8") as f:
    json.dump({"backbone": BACKBONE, "preprocess": PREPROCESS_NAME, "seed": SEED, "quick_test": QUICK_TEST,
               "init": "female v4 fold k", "lr": LR_V5, "epochs": EPOCHS_V5, "mix_per_class": MIX_PER_CLASS,
               "chosen": CHOSEN, "female_v4_only": female_only, "variants": variant_results,
               "clean": {"enabled": CLEAN, "max_p_label": CLEAN_MAX_P_LABEL, "min_p_other": CLEAN_MIN_P_OTHER,
                         "max_fraction": CLEAN_MAX_FRACTION},
               "results": results, "hours": round((time.time() - t_start) / 3600, 2)}, f, ensure_ascii=False, indent=2)

# =============================================================================
# ### 6) .tflite + class_names.json สำหรับ backend (ตรวจด้วย image_ops แบบเดียวกับเซิร์ฟเวอร์)
# =============================================================================
def tflite_probs(path, x, batch=32):
    it = tf.lite.Interpreter(model_path=path, num_threads=os.cpu_count())
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    out = []
    for s in range(0, len(x), batch):
        xb = np.ascontiguousarray(x[s:s + batch], dtype=np.float32)
        it.resize_tensor_input(i_in, list(xb.shape))
        it.allocate_tensors()
        it.set_tensor(i_in, xb)
        it.invoke()
        out.append(it.get_tensor(i_out).copy())
    return np.concatenate(out)


tfl_files, tfl_results = {}, {}
for g in GENDERS:
    tfl_files[g] = []
    for k, kp in enumerate(keras_files[g], 1):
        out = os.path.join(TFLITE_DIR, f"face_shape_{g}_v5_fold{k}.tflite")
        convert(kp, out)
        tfl_files[g].append(out)
        print(f"[tflite] {os.path.basename(out)}: {os.path.getsize(out) / 1e6:.1f} MB")
    test = data[g]["test"]
    y = np.array([l for _, l in test])
    xs = []
    for p, _ in test:
        with Image.open(crop(p)) as im:
            xs.append(image_ops.prepare_batch(np.asarray(im.convert("RGB"), dtype="float32"), IMG_SIZE, PREPROCESS_NAME, True))
    xs = np.stack(xs)
    probs = np.mean([tflite_probs(f, xs[:, j]) for f in tfl_files[g] for j in (0, 1)], axis=0)
    tfl_results[g] = scores(probs, y)
    print(f"[tflite] {g} ensemble+TTA (pipeline เดียวกับเซิร์ฟเวอร์): top-1 {tfl_results[g]['top1']:.2%} "
          f"top-2 {tfl_results[g]['top2']:.2%} (keras {results[g]['ensemble+TTA']['top1']:.2%})")

crop_params = dict(vars(face_crop.CropParams())) if not hasattr(face_crop, "asdict") else face_crop.asdict(face_crop.CropParams())
meta = {
    "class_names": CLASS_NAMES, "img_size": list(IMG_SIZE), "resize": "resize_with_pad", "preprocess": PREPROCESS_NAME,
    "backbone": BACKBONE, "source": "roboflow", "format": f"tflite ({QUANT})", "tta": "flip", "version": f"v5 ({CHOSEN}, init female v4)",
    "models": {g: [os.path.basename(f) for f in tfl_files[g]] for g in GENDERS},
    "accuracy": {g: {"ensemble_tta_top1": tfl_results[g]["top1"], "ensemble_tta_top2": tfl_results[g]["top2"],
                     "ensemble_tta_macro_f1": tfl_results[g]["macro_f1"], "n_test": results[g]["n_test"]} for g in GENDERS},
    "face_crop": {"module": "face_crop.py", "version": face_crop.CROP_VERSION, "params": crop_params},
}
with open(os.path.join(TFLITE_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, "tflite_v5_male_report.json"), "w", encoding="utf-8") as f:
    json.dump({"tflite_ensemble_tta": tfl_results, "keras_ensemble_tta": {g: results[g]["ensemble+TTA"] for g in GENDERS},
               "v3_reference": V3_REFERENCE}, f, ensure_ascii=False, indent=2)
zip_path = os.path.join(OUTPUT_DIR, "tflite_models_v5_male.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(TFLITE_DIR)):
        z.write(os.path.join(TFLITE_DIR, name), name)
shutil.rmtree(WORK, ignore_errors=True)

print("\n" + "=" * 70)
print(f"สรุป v5 ชาย (เริ่มจากหญิง v4, เลือกแบบ {CHOSEN}{', QUICK_TEST' if QUICK_TEST else ''}) — "
      f"{(time.time() - t_start) / 3600:.1f} ชม.")
for g in GENDERS:
    r, ref = results[g]["ensemble+TTA"], V3_REFERENCE[g]
    print(f"  {g:7s} top-1 {r['top1']:.2%} (v3 {ref['top1']:.2%}, {100 * (r['top1'] - ref['top1']):+.1f}) | "
          f"top-2 {r['top2']:.2%} | tflite {tfl_results[g]['top1']:.2%} | ตัดป้าย {cleaning[g]['dropped']} รูป")
if female_v4_tflite:
    r = female_v4_tflite["ensemble+TTA"]
    print(f"  female  v4 .tflite top-1 {r['top1']:.2%} | top-2 {r['top2']:.2%} | macro F1 {r['macro_f1']:.3f} (v3 80.38% / 91.39%)")
print("ไฟล์ที่ต้องส่ง: cv_results_v5_male.json, tflite_v5_male_report.json, tflite_v4_report.json,"
      " tflite_models_v5_male.zip, history_v5_male_*.png")
print("DONE.")
