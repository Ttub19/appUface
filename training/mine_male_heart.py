# =============================================================================
# uShape — คัดรูป "หน้า heart เพศชาย" จาก CelebA ให้คนตรวจทาน (pseudo-label + human review)
#
#   1. สุ่มรูปผู้ชายจาก CelebA (ป้าย Male=1, ไม่ใส่แว่น/หมวก)
#   2. ครอปหน้าด้วย face_crop.py ตัวเดียวกับที่ใช้เทรน
#   3. เกณฑ์ที่ 1 (โมเดล): ให้โมเดลที่เทรนแล้วทายความน่าจะเป็นของ heart
#   4. เกณฑ์ที่ 2 (เรขาคณิต ไม่พึ่งโมเดล): จุด landmark -> หน้าผาก/กราม และ คาง/โหนกแก้ม
#   5. เลือก ~400 รูปที่ผ่านทั้งสองเกณฑ์ + สุ่มรูปทั่วไป ~100 รูป (กลุ่มควบคุม วัดว่าเครื่องคัดแม่นแค่ไหน)
#   6. สร้าง review.html (เปิดในเบราว์เซอร์ กดคีย์บอร์ดตรวจทีละรูป) + heart_review.zip
#
# Kaggle: Settings -> Accelerator = GPU, Internet = On
#   Add Input 1: Datasets -> "CelebFaces Attributes (CelebA) Dataset" (jessicali9530/celeba-dataset)
#   Add Input 2: Your Work -> Notebooks -> notebook ที่เทรนโมเดล v3 (ต้องมี face_shape_female.keras)
# รันทั้งไฟล์ในเซลล์เดียว ~20-30 นาที -> ดาวน์โหลด /kaggle/working/heart_review.zip
# =============================================================================
import os
import sys
import csv
import json
import time
import base64
import random
import shutil
import subprocess
from io import BytesIO

import numpy as np
from PIL import Image

# =============================================================================
# ### CONFIG
# =============================================================================
N_SAMPLE = 20000            # สุ่มรูปผู้ชายมากี่รูปเพื่อคัด (มาก = ได้ตัวเลือกดีขึ้น แต่ครอปนานขึ้น)
N_CANDIDATES = 400          # รูปที่ผ่านทั้งสองเกณฑ์
N_RANDOM = 100              # กลุ่มควบคุม: สุ่มจากรูปที่เหลือ (ไม่บอกคนตรวจว่ารูปไหนเป็นกลุ่มไหน)
MIN_PERCENTILE = 0.70       # ต้องอยู่ใน 30% บนของ "ทั้งสองเกณฑ์"
MAX_ABS_YAW = 0.25          # หันข้างมากกว่านี้ไม่เอา (วัดความกว้างหน้าไม่แม่น)
SEED = 42
OUTPUT_DIR = "/kaggle/working"
WORK = os.path.join(OUTPUT_DIR, "heart_mining")
REVIEW = os.path.join(OUTPUT_DIR, "heart_review")
CLASS_NAMES = ["heart", "oblong", "oval", "round", "square"]   # ลำดับเดียวกับตอนเทรน (อ่านจาก class_names.json ถ้ามี)
IMG_SIZE = (224, 224)

# จุด landmark (MediaPipe 468 จุด) ที่ใช้วัด
LM = {"top": 10, "chin": 152, "fore_l": 54, "fore_r": 284, "cheek_l": 234, "cheek_r": 454,
      "jaw_l": 172, "jaw_r": 397, "chin_l": 148, "chin_r": 377}

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(WORK, exist_ok=True)


def find_file(name, base="/kaggle/input", max_depth=7):
    for root, dirs, files in os.walk(base):
        if root[len(base):].count(os.sep) > max_depth:
            dirs[:] = []
            continue
        if name in files:
            return os.path.join(root, name)
    return None


# =============================================================================
# ### (ไม่ต้องแก้) face_crop.py ตัวเดียวกับที่ใช้เทรน
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

LANDMARK_WORKER = r'''
import sys, json, multiprocessing
env, here, jobs_file, out_file, model_path, workers = sys.argv[1:7]
sys.path[:0] = [env, here]
import face_crop
from PIL import Image
IDX = json.loads(sys.argv[7])
_D = {}
def init():
    _D["det"] = face_crop.FaceLandmarkDetector(model_path if model_path != "-" else None)
def run(path):
    try:
        im = face_crop.load_rgb(path)
        pts, method = face_crop.find_landmarks(_D["det"], im)
        if pts is None:
            return path, None
        w, h = im.size
        return path, {k: [float(pts[i][0]) / w, float(pts[i][1]) / h] for k, i in IDX.items()}
    except Exception as exc:
        return path, None
if __name__ == "__main__":
    jobs = json.load(open(jobs_file))
    out = {}
    with multiprocessing.get_context("fork").Pool(int(workers), initializer=init) as pool:
        for i, (p, r) in enumerate(pool.imap_unordered(run, jobs, chunksize=16), 1):
            out[p] = r
            if i % 2000 == 0 or i == len(jobs):
                print(f"[landmarks] {i}/{len(jobs)}", flush=True)
    json.dump(out, open(out_file, "w"))
'''

# =============================================================================
# ### 1) รูปผู้ชายจาก CelebA
# =============================================================================
ATTR_CSV = find_file("list_attr_celeba.csv")
SAMPLE_IMG = find_file("000001.jpg")
if not ATTR_CSV or not SAMPLE_IMG:
    raise FileNotFoundError("ไม่พบ CelebA — Add Input -> ค้น 'celeba-dataset' (jessicali9530) แล้วรันใหม่")
IMG_DIR = os.path.dirname(SAMPLE_IMG)
print("CelebA:", IMG_DIR)

with open(ATTR_CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
males = [r["image_id"] for r in rows
         if r["Male"] == "1" and r.get("Eyeglasses") != "1" and r.get("Wearing_Hat") != "1"]
print(f"ผู้ชาย (ไม่ใส่แว่น/หมวก): {len(males)} จากทั้งหมด {len(rows)} รูป")
sample = sorted(random.sample(males, min(N_SAMPLE, len(males))))
sample_paths = [os.path.join(IMG_DIR, n) for n in sample]

# =============================================================================
# ### 2) ครอปหน้า (cache ไว้ใน WORK)
# =============================================================================
with open(os.path.join(OUTPUT_DIR, "face_crop.py"), "w", encoding="utf-8") as f:
    f.write(FACE_CROP_SOURCE)
if OUTPUT_DIR not in sys.path:
    sys.path.insert(0, OUTPUT_DIR)
import importlib
import face_crop
face_crop = importlib.reload(face_crop)

manifest = face_crop.ensure_face_crops(sample_paths, IMG_DIR, os.path.join(WORK, "crops"))
ok = [p for p in sample_paths
      if manifest.get(p, {}).get("status", "").startswith("mesh") and abs(manifest[p].get("yaw", 1)) <= MAX_ABS_YAW]
print(f"หาหน้าเจอ + หน้าตรง (|yaw| <= {MAX_ABS_YAW}): {len(ok)} รูป")

# =============================================================================
# ### 3) Landmark บนรูปที่ครอปแล้ว -> สัดส่วนหน้า (เกณฑ์ที่ไม่พึ่งโมเดล)
# =============================================================================
LM_JSON = os.path.join(WORK, "landmarks.json")
landmarks = json.load(open(LM_JSON)) if os.path.exists(LM_JSON) else {}
todo = [manifest[p]["crop"] for p in ok if manifest[p]["crop"] not in landmarks]
if todo:
    env = face_crop.ensure_mediapipe("/tmp/mediapipe_env")
    model_path = face_crop.ensure_model(os.path.join(env, "face_landmarker.task")) or "-"
    script = os.path.join(WORK, "_lm_worker.py")
    jobs_file, out_file = os.path.join(WORK, "_lm_jobs.json"), os.path.join(WORK, "_lm_out.json")
    open(script, "w").write(LANDMARK_WORKER)
    json.dump(todo, open(jobs_file, "w"))
    face_crop._stream([sys.executable, "-I", script, env, OUTPUT_DIR, jobs_file, out_file, model_path,
                       str(os.cpu_count() or 2), json.dumps(LM)])
    landmarks.update(json.load(open(out_file)))
    json.dump(landmarks, open(LM_JSON, "w"))


def dist(pts, a, b):
    return float(np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))


def geometry(pts):
    fore, cheek = dist(pts, "fore_l", "fore_r"), dist(pts, "cheek_l", "cheek_r")
    jaw, chin = dist(pts, "jaw_l", "jaw_r"), dist(pts, "chin_l", "chin_r")
    length = dist(pts, "top", "chin")
    return {"forehead_jaw": fore / jaw, "chin_cheek": chin / cheek, "length_cheek": length / cheek}


def percentile_rank(values):
    """0..1 (1 = ค่ามากที่สุด)"""
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order / max(1, len(values) - 1)


items = []
for p in ok:
    pts = landmarks.get(manifest[p]["crop"])
    if pts:
        items.append({"src": p, "crop": manifest[p]["crop"], "pts": pts, **geometry(pts)})
print(f"วัดสัดส่วนหน้าได้: {len(items)} รูป")

# =============================================================================
# ### 4) ความน่าจะเป็น heart จากโมเดลที่เทรนแล้ว
# =============================================================================
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

meta_path = find_file("class_names.json")
if meta_path:
    meta = json.load(open(meta_path, encoding="utf-8"))
    CLASS_NAMES = meta.get("class_names", CLASS_NAMES)
    IMG_SIZE = tuple(meta.get("img_size", IMG_SIZE))
HEART = CLASS_NAMES.index("heart")
def find_model(name, base="/kaggle/input"):
    """หาไฟล์ .keras — ถ้า Kaggle แตก zip เป็นโฟลเดอร์ (config.json + model.weights.h5) ให้รวมกลับเป็นไฟล์"""
    path = find_file(name)
    if path:
        return path
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            folder = os.path.join(root, d)
            if d == name and os.path.exists(os.path.join(folder, "config.json")) \
                    and os.path.exists(os.path.join(folder, "model.weights.h5")):
                import zipfile
                out = os.path.join(WORK, name)
                with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
                    for f in sorted(os.listdir(folder)):
                        z.write(os.path.join(folder, f), f)
                print(f"รวมโฟลเดอร์ {folder} -> {out}")
                return out
    return None


model_files = {g: find_model(f"face_shape_{g}.keras") for g in ("female", "male")}
if not model_files["female"]:
    raise FileNotFoundError("ไม่พบ face_shape_female.keras — Add Input -> Your Work -> Notebooks -> notebook ที่เทรน v3")
print("โมเดล:", model_files)


def load_for_model(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize_with_pad(tf.cast(img, tf.float32), IMG_SIZE[0], IMG_SIZE[1])
    return preprocess_input(img)


def heart_probs(model_path, paths):
    model = tf.keras.models.load_model(model_path, compile=False)
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load_for_model, num_parallel_calls=tf.data.AUTOTUNE).batch(128)
    probs = model.predict(ds, verbose=0)
    del model
    tf.keras.backend.clear_session()
    return probs[:, HEART]


crop_paths = [it["crop"] for it in items]
p_f = heart_probs(model_files["female"], crop_paths)
p_m = heart_probs(model_files["male"], crop_paths) if model_files["male"] else np.zeros(len(items))
for it, a, b in zip(items, p_f, p_m):
    it["p_heart_female_model"], it["p_heart_male_model"] = float(a), float(b)

# =============================================================================
# ### 5) เลือก: ผ่านทั้งสองเกณฑ์ + กลุ่มควบคุมสุ่ม
# =============================================================================
def select(items, n_candidates, n_random, min_pct, seed):
    model_score = np.array([max(it["p_heart_female_model"], it["p_heart_male_model"]) for it in items])
    pr_model = percentile_rank(model_score)
    pr_geo = (percentile_rank(np.array([it["forehead_jaw"] for it in items]))
              + percentile_rank(-np.array([it["chin_cheek"] for it in items]))) / 2
    pr_geo = percentile_rank(pr_geo)
    combined = (pr_model + pr_geo) / 2
    for it, a, b, c in zip(items, pr_model, pr_geo, combined):
        it["pct_model"], it["pct_geometry"], it["score"] = float(a), float(b), float(c)
    both = [i for i in np.argsort(-combined, kind="stable") if pr_model[i] >= min_pct and pr_geo[i] >= min_pct]
    chosen = both[:n_candidates]
    if len(chosen) < n_candidates:   # ผ่านทั้งสองเกณฑ์ไม่พอ -> เติมจากคะแนนรวมสูงสุด
        taken = set(chosen)
        extra = [i for i in np.argsort(-combined, kind="stable") if i not in taken]
        chosen += extra[:n_candidates - len(chosen)]
    rest = sorted(set(range(len(items))) - set(chosen))
    rng = random.Random(seed)
    control = rng.sample(rest, min(n_random, len(rest)))
    print(f"ผ่านทั้งสองเกณฑ์: {len(both)} รูป -> เลือก {len(chosen)} + กลุ่มควบคุม {len(control)}")
    return [(i, "candidate") for i in chosen] + [(i, "random") for i in control]


picked = select(items, N_CANDIDATES, N_RANDOM, MIN_PERCENTILE, SEED)
random.Random(SEED + 1).shuffle(picked)   # สลับลำดับ คนตรวจจะไม่รู้ว่ารูปไหนมาจากกลุ่มไหน

# =============================================================================
# ### 6) review.html + heart_review.zip
# =============================================================================
REVIEW_HTML = r'''<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ตรวจทานหน้า Heart</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--fg:#1d1c1a;--muted:#6b6862;--line:#e3e0da;--yes:#1f7a4d;--no:#b3261e;--unsure:#8a6d1d;--accent:#2d5bd7}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1f1e1c;--fg:#ecebe8;--muted:#a19e97;--line:#34322e;--yes:#4cc28a;--no:#f07068;--unsure:#e0bb52;--accent:#7d9cff;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Noto Sans Thai","Leelawadee UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:20px}
@media (max-width:760px){main{grid-template-columns:1fr}}
h1{font-size:18px;margin:0 0 4px}
.muted{color:var(--muted);font-size:13px}
.stage{position:relative;background:var(--card);border:1px solid var(--line);border-radius:10px;aspect-ratio:1;max-width:560px;width:100%;overflow:hidden}
.stage img,.stage svg{position:absolute;inset:0;width:100%;height:100%}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:10px 0}
.bar i{display:block;height:100%;background:var(--accent);width:0}
.btns{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px;max-width:560px}
button{font:inherit;padding:12px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.b-no{border-color:var(--no);color:var(--no)}.b-yes{border-color:var(--yes);color:var(--yes)}.b-un{border-color:var(--unsure);color:var(--unsure)}
.on kbd{background:transparent;color:inherit;border-color:currentColor}
.b-no.on{background:var(--no);color:#fff}.b-yes.on{background:var(--yes);color:#fff}.b-un.on{background:var(--unsure);color:#fff}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:var(--card)}
aside{display:flex;flex-direction:column;gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:14px;margin:0 0 6px}
.card ul{margin:0;padding-left:18px}
label{display:block;font-size:13px;margin-bottom:4px}
input[type=text]{width:100%;font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;text-align:center;font-size:13px}
.stats b{display:block;font-size:18px}
.legend span{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:4px}
textarea{width:100%;height:120px;font:12px ui-monospace,monospace}
.row{display:flex;gap:8px;flex-wrap:wrap}
</style>
</head>
<body>
<main>
<section>
  <h1>หน้านี้เป็นทรง heart ไหม?</h1>
  <div class="muted" id="pos">–</div>
  <div class="bar"><i id="prog"></i></div>
  <div class="stage"><img id="img" alt="ใบหน้าที่ต้องตรวจ"><svg id="ov" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
  <div class="btns">
    <button class="b-no" id="bno">ไม่ใช่ <kbd>←</kbd></button>
    <button class="b-un" id="bun">ไม่แน่ใจ <kbd>↓</kbd></button>
    <button class="b-yes" id="byes">ใช่ heart <kbd>→</kbd></button>
  </div>
  <div class="row" style="margin-top:8px">
    <button id="bprev">ย้อนกลับ <kbd>Z</kbd></button>
    <button id="bnext">ข้าม <kbd>Space</kbd></button>
    <button id="blines">เส้นวัด เปิด/ปิด <kbd>L</kbd></button>
  </div>
</section>
<aside>
  <div class="card">
    <h2>หน้า heart คือ</h2>
    <ul>
      <li>หน้าผาก<b>กว้าง</b> ใกล้เคียงหรือกว้างกว่าโหนกแก้ม</li>
      <li>กราม<b>แคบ</b>กว่าหน้าผากชัดเจน</li>
      <li>คาง<b>แหลม/แคบ</b> รูปหน้าเรียวลงเป็นรูปหัวใจคว่ำ</li>
    </ul>
    <p class="muted" style="margin:8px 0 0">ดูที่โครงหน้า ไม่ดูทรงผม หนวดเครา หรือหน้าม้า ถ้าผม/เคราบังโครงหน้าจนตัดสินไม่ได้ ให้กด "ไม่แน่ใจ"</p>
    <p class="muted legend" style="margin:8px 0 0"><span style="background:#2d9bf0"></span>หน้าผาก <span style="background:#f0a72d"></span>โหนกแก้ม <span style="background:#e0466b"></span>กราม <span style="background:#8a5cf6"></span>คาง</p>
  </div>
  <div class="card">
    <label for="who">ชื่อผู้ตรวจ</label>
    <input type="text" id="who" placeholder="เช่น teerapon" autocomplete="off">
    <div class="muted">พิมพ์ชื่อแล้วกด Enter ก่อนใช้ปุ่มลูกศร</div>
    <div class="stats" style="margin-top:10px">
      <div><b id="sy">0</b>ใช่</div><div><b id="sn">0</b>ไม่ใช่</div><div><b id="su">0</b>ไม่แน่ใจ</div>
    </div>
  </div>
  <div class="card">
    <h2>ส่งผล</h2>
    <p class="muted" style="margin:0 0 8px">ผลบันทึกในเบราว์เซอร์นี้อัตโนมัติ ปิดแล้วเปิดใหม่ทำต่อได้ ตรวจครบแล้วกดดาวน์โหลด</p>
    <div class="row"><button id="bdl">ดาวน์โหลดผล (CSV)</button><button id="bcopy">แสดง CSV</button></div>
    <textarea id="csv" hidden readonly></textarea>
  </div>
</aside>
</main>
<script>
const DATA = __DATA__;
const SET_ID = "__SET_ID__";
const KEY = "heart-review:" + SET_ID;
let labels = {}, idx = 0, lines = true;
try { const s = JSON.parse(localStorage.getItem(KEY) || "{}"); labels = s.labels || {}; idx = s.idx || 0; lines = s.lines !== false;
      document.getElementById("who").value = s.who || ""; } catch (e) {}
const $ = id => document.getElementById(id);
function save(){ try{ localStorage.setItem(KEY, JSON.stringify({labels, idx, lines, who: $("who").value})); }catch(e){} }
const PAIRS = [["fore_l","fore_r","#2d9bf0"],["cheek_l","cheek_r","#f0a72d"],["jaw_l","jaw_r","#e0466b"],["chin_l","chin_r","#8a5cf6"]];
function render(){
  idx = Math.max(0, Math.min(DATA.length - 1, idx));
  const d = DATA[idx];
  $("img").src = "data:image/jpeg;base64," + d.img;
  $("ov").innerHTML = !lines ? "" : PAIRS.map(([a,b,c]) => d.pts[a] && d.pts[b] ?
    `<line x1="${d.pts[a][0]}" y1="${d.pts[a][1]}" x2="${d.pts[b][0]}" y2="${d.pts[b][1]}" stroke="${c}" stroke-width="0.006" stroke-linecap="round"/>` : "").join("");
  const done = Object.keys(labels).length;
  $("pos").textContent = `รูปที่ ${idx + 1} / ${DATA.length} · ตรวจแล้ว ${done} รูป` + (labels[d.id] ? ` · รูปนี้: ${{yes:"ใช่",no:"ไม่ใช่",unsure:"ไม่แน่ใจ"}[labels[d.id]]}` : "");
  $("prog").style.width = (100 * done / DATA.length) + "%";
  $("byes").classList.toggle("on", labels[d.id] === "yes");
  $("bno").classList.toggle("on", labels[d.id] === "no");
  $("bun").classList.toggle("on", labels[d.id] === "unsure");
  const c = {yes:0,no:0,unsure:0}; Object.values(labels).forEach(v => c[v]++);
  $("sy").textContent = c.yes; $("sn").textContent = c.no; $("su").textContent = c.unsure;
  if (DATA[idx + 1]) new Image().src = "data:image/jpeg;base64," + DATA[idx + 1].img;
  save();
}
function mark(v){ labels[DATA[idx].id] = v; if (idx < DATA.length - 1) idx++; render(); }
function toCSV(){
  const who = ($("who").value || "reviewer").replace(/[^\w฀-๿-]/g, "_");
  return "id,label,reviewer\n" + DATA.filter(d => labels[d.id]).map(d => `${d.id},${labels[d.id]},${who}`).join("\n") + "\n";
}
$("byes").onclick = () => mark("yes"); $("bno").onclick = () => mark("no"); $("bun").onclick = () => mark("unsure");
$("bprev").onclick = () => { idx--; render(); }; $("bnext").onclick = () => { idx++; render(); };
$("blines").onclick = () => { lines = !lines; render(); };
$("who").oninput = save;
$("who").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === "Escape") e.target.blur(); });
$("who").addEventListener("change", e => e.target.blur());
$("bdl").onclick = () => {
  const who = ($("who").value || "reviewer").replace(/[^\w-]/g, "_");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([toCSV()], {type: "text/csv"}));
  a.download = `heart_labels_${who}.csv`; document.body.appendChild(a); a.click(); a.remove();
};
$("bcopy").onclick = () => { const t = $("csv"); t.hidden = !t.hidden; t.value = toCSV(); t.select(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  const k = e.key;
  if (k === "ArrowRight" || k === "d" || k === "1") mark("yes");
  else if (k === "ArrowLeft" || k === "a" || k === "2") mark("no");
  else if (k === "ArrowDown" || k === "s" || k === "3") mark("unsure");
  else if (k === "z" || k === "Backspace" || k === "ArrowUp") { idx--; render(); }
  else if (k === " ") { idx++; render(); }
  else if (k === "l" || k === "L") { lines = !lines; render(); }
  else return;
  e.preventDefault();
});
render();
</script>
</body>
</html>
'''


def thumb_b64(path, size=256):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def build_review(picked, items, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(os.path.join(out_dir, "images"))
    records, rows_csv = [], []
    for n, (i, group) in enumerate(picked, 1):
        it = items[i]
        rid = f"c{n:04d}"
        name = os.path.basename(it["src"])
        shutil.copy(it["src"], os.path.join(out_dir, "images", f"{rid}_{name}"))   # รูปต้นฉบับ (ใช้เทรน)
        records.append({"id": rid, "img": thumb_b64(it["crop"]),
                        "pts": {k: [round(v[0], 4), round(v[1], 4)] for k, v in it["pts"].items()}})
        rows_csv.append([rid, f"images/{rid}_{name}", name, group,
                         f"{it['p_heart_female_model']:.4f}", f"{it['p_heart_male_model']:.4f}",
                         f"{it['forehead_jaw']:.4f}", f"{it['chin_cheek']:.4f}", f"{it['length_cheek']:.4f}",
                         f"{it['pct_model']:.3f}", f"{it['pct_geometry']:.3f}"])
    with open(os.path.join(out_dir, "candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "file", "celeba_id", "group", "p_heart_female_model", "p_heart_male_model",
                    "forehead_jaw", "chin_cheek", "length_cheek", "pct_model", "pct_geometry"])
        w.writerows(rows_csv)
    html = REVIEW_HTML.replace("__DATA__", json.dumps(records, separators=(",", ":"))) \
                      .replace("__SET_ID__", f"celeba-male-heart-{SEED}-{len(records)}")
    with open(os.path.join(out_dir, "review.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return len(records)


n = build_review(picked, items, REVIEW)
shutil.copy(os.path.join(REVIEW, "review.html"), os.path.join(OUTPUT_DIR, "review.html"))
shutil.make_archive(os.path.join(OUTPUT_DIR, "heart_review"), "zip", REVIEW)
print(f"\nเสร็จ: {n} รูป")
print("ดาวน์โหลด /kaggle/working/heart_review.zip -> แตกไฟล์ -> เปิด review.html ในเบราว์เซอร์")
print("ตรวจครบแล้วกด 'ดาวน์โหลดผล' จะได้ heart_labels_<ชื่อ>.csv -> ส่ง CSV + heart_review.zip กลับมา")
print("DONE.")
