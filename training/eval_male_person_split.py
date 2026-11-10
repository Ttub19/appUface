# =============================================================================
# uShape — วัดผลโมเดลชาย บนชุดทดสอบ "แบ่งตามคน" (ไม่ต้องเทรน, ใช้ CPU ได้)
#   ก. เลือก fold ของ Vision (v6 ConvNeXt) ด้วยค่า val ที่ได้ตอนเทรน — กฎตั้งไว้ก่อนดูผลชุดทดสอบ:
#      ตัด fold ที่ val ต่ำกว่า fold ที่ดีที่สุดเกิน DROP_MARGIN (รอบนี้: fold2 val 0.49 vs 0.64)
#   ข. วัด Core (v3 ResNet50) กับชุดทดสอบเดียวกัน -> เทียบกันได้อย่างยุติธรรม (ถ้าแนบไฟล์ Core มา)
#
# Kaggle: Accelerator = None (CPU) ก็ได้, Internet = On (ติดตั้ง MediaPipe สำหรับครอปหน้า)
#   Add Input: 1) dataset faceshape1
#              2) dataset ที่มี splits_person_male.csv
#              3) (ไม่บังคับ) output ของ notebook ที่เทรน v6 ชาย — ถ้าไม่มี ใช้ Vision fold 1+3 จาก GitHub
#              4) (ไม่บังคับ) ไฟล์ .tflite ของ Core / Vision — ถ้าไม่แนบ สคริปต์ดาวน์โหลดจาก GitHub
#                 (Ttub19/appUface branch main1, backend/models/*.part*) ให้เอง ต้องเปิด Internet
# วางทั้งไฟล์ในเซลล์เดียวแล้ว Run All (~10–20 นาที) — รันต่อใน notebook ที่เทรน v6 ชายก็ได้ (หาไฟล์ใน /kaggle/working ด้วย)
# ได้ไฟล์: eval_male_person_split.json + สรุปในหน้าจอ
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
GENDER = "male"
K_FOLDS = 3
CLASS_NAMES = ["heart", "oblong", "oval", "round", "square"]
C = len(CLASS_NAMES)
IMG_SIZE = (224, 224)
SPLITS_NAME = "splits_person"
DROP_MARGIN = 0.05                      # ตัด fold ที่ val ต่ำกว่า fold ที่ดีที่สุดเกินค่านี้
FALLBACK_VAL = {1: 0.6068, 2: 0.4921, 3: 0.6373}   # จาก cv_results_v6_male.json (ใช้ถ้าไม่ได้แนบไฟล์นั้น)
OUTPUT_DIR = "/kaggle/working"
FACE_CROP_DIR = os.path.join(OUTPUT_DIR, "face_crops")
WORK = os.path.join(OUTPUT_DIR, "eval_work")
os.makedirs(WORK, exist_ok=True)

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


# ### helpers (เหมือนในสคริปต์เทรน)
def walk_files(base="/kaggle/input"):
    yield from os.walk(base)


def find_data_root(base="/kaggle/input"):
    for root, dirs, _ in walk_files(base):
        if "train1_cropped" in dirs and "testing_set" in dirs:
            return root
    raise FileNotFoundError("ไม่พบ dataset faceshape1 (โฟลเดอร์ที่มี train1_cropped + testing_set) — Add Input ก่อน")


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


def find_person_splits(gender, base="/kaggle/input"):
    names = [f"{SPLITS_NAME}_{gender}.csv", f"{SPLITS_NAME}_{gender}.csv.txt", f"{SPLITS_NAME}_{gender}.txt"]
    for root, _, files in walk_files(base):
        for n in names:
            if n in files:
                return os.path.join(root, n)
    for root, _, files in walk_files(base):           # ชื่อที่ Kaggle/Windows เติมเลขต่อท้าย เช่น splits_person_female_1.csv
        for n in files:
            if n.startswith(f"{SPLITS_NAME}_{gender}") and n.split(".")[0].replace(f"{SPLITS_NAME}_{gender}", "").strip("_ ()0123456789") == "":
                return os.path.join(root, n)
    return None


# =============================================================================
# ### 1) ชุดทดสอบชาย (แบ่งตามคน) + ครอปหน้า
# =============================================================================
DATA_ROOT = find_data_root()
print("DATA_ROOT:", DATA_ROOT)
sp = find_person_splits(GENDER)
if not sp:
    raise FileNotFoundError(f"ไม่พบ {SPLITS_NAME}_{GENDER}.csv — Add Input dataset ที่มีไฟล์นี้")
parts, missing = read_splits(sp)
test = parts["test"]
print(f"ชุดทดสอบ {len(test)} รูป" + (f" (หาไฟล์ไม่เจอ {missing})" if missing else ""))
manifest = face_crop.ensure_face_crops(sorted({p for p, _ in test}), DATA_ROOT, FACE_CROP_DIR)
before = len(test)
test = [(p, l) for p, l in test if manifest.get(p, {}).get("status", "").startswith("mesh")]
print(f"หาหน้าเจอ {len(test)}/{before} รูป (ตอนเทรนก็ตัดรูปที่หาหน้าไม่เจอแบบเดียวกัน)")
y = np.array([l for _, l in test])
from PIL import Image
rgb = []
for p, _ in test:
    with Image.open(manifest[p]["crop"]) as im:
        rgb.append(np.asarray(im.convert("RGB"), dtype="float32"))


def inputs(preprocess):
    """(N, 2, H, W, 3): ภาพ + กลับซ้ายขวา (TTA) แบบเดียวกับ backend"""
    return np.stack([image_ops.prepare_batch(a, IMG_SIZE, preprocess, True) for a in rgb])


# =============================================================================
# ### 2) หาไฟล์ .tflite (รวม .part00/.part01 ให้ถ้าจำเป็น)
# =============================================================================
SEARCH_ROOTS = ["/kaggle/input", OUTPUT_DIR]          # รันใน notebook ที่เทรนก็ได้ (ไฟล์อยู่ใน /kaggle/working)


def scan(roots=SEARCH_ROOTS):
    files, dirs = [], []
    for base in roots:
        for r, ds, fs in os.walk(base):
            ds[:] = [d for d in ds if d not in ("face_crops", "eval_work")]   # ข้ามโฟลเดอร์รูปเป็นหมื่น
            files += [os.path.join(r, f) for f in fs]
            dirs += [os.path.join(r, d) for d in ds]
    return files, dirs


ALL_FILES, ALL_DIRS = scan()
# zip ของ tflite (ทั้งที่อยู่ใน input และใน working) -> แตกไว้ใน eval_work
for z in [f for f in ALL_FILES if os.path.basename(f).startswith("tflite_models") and f.endswith(".zip")]:
    dest = os.path.join(WORK, os.path.splitext(os.path.basename(z))[0])
    if not os.path.isdir(dest):
        with zipfile.ZipFile(z) as zz:
            zz.extractall(dest)
    print("แตกไฟล์", z, "->", dest)
    ALL_FILES += [os.path.join(dest, f) for f in os.listdir(dest)]


def find_tflite(name):
    hits = sorted(f for f in ALL_FILES if os.path.basename(f) == name)
    if hits:
        return hits[0]
    for dd in (d for d in ALL_DIRS if os.path.basename(d) == name):   # Kaggle แตกเป็นโฟลเดอร์
        inner = [os.path.join(dd, x) for x in os.listdir(dd)]
        if len(inner) == 1 and os.path.isfile(inner[0]):
            return inner[0]
    chunks = sorted(f for f in ALL_FILES if os.path.basename(f).startswith(name + ".part") and not f.endswith(".sha256"))
    if chunks:
        out = os.path.join(WORK, name)
        with open(out, "wb") as w:
            for c in chunks:
                with open(c, "rb") as r:
                    shutil.copyfileobj(r, w)
        print(f"  รวม {len(chunks)} ส่วน -> {name} ({os.path.getsize(out) / 1e6:.1f} MB)")
        return out
    return None


GITHUB_RAW = "https://raw.githubusercontent.com/Ttub19/appUface/main1/backend/models/"   # ต้องเปิด Internet


def from_github(name):
    """ดาวน์โหลด <name>.part00, .part01, ... จาก repo แล้วรวม + ตรวจ sha256 (ถ้าไม่ได้แนบไฟล์มาเป็น input)"""
    import urllib.request, urllib.error, hashlib
    out, i = os.path.join(WORK, name), 0
    try:
        with open(out, "wb") as w:
            while True:
                try:
                    with urllib.request.urlopen(f"{GITHUB_RAW}{name}.part{i:02d}", timeout=120) as r:
                        shutil.copyfileobj(r, w)
                except urllib.error.HTTPError as e:
                    if e.code == 404 and i > 0:
                        break
                    raise
                i += 1
        with urllib.request.urlopen(f"{GITHUB_RAW}{name}.sha256", timeout=60) as r:
            want = r.read().decode().split()[0]
        with open(out, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            raise ValueError(f"sha256 ไม่ตรง ({got[:12]} != {want[:12]})")
        print(f"  ดาวน์โหลดจาก GitHub {i} ส่วน -> {name} ({os.path.getsize(out) / 1e6:.1f} MB)")
        return out
    except Exception as e:
        print(f"  [GitHub] โหลด {name} ไม่ได้: {e}")
        if os.path.exists(out):
            os.remove(out)
        return None


def get_model(name):
    return find_tflite(name) or from_github(name)


vision = {k: p for k in range(1, K_FOLDS + 1) if (p := get_model(f"face_shape_male_v6_fold{k}.tflite"))}
core = {k: p for k in range(1, K_FOLDS + 1) if (p := get_model(f"face_shape_male_fold{k}.tflite"))}
print("Vision folds:", sorted(vision), "| Core folds:", sorted(core))
if not vision and not core:
    print("ไฟล์ .tflite ที่เห็น:", sorted({os.path.basename(f) for f in ALL_FILES if ".tflite" in f})[:20])
    raise FileNotFoundError("ไม่พบโมเดลทั้ง Vision และ Core — เปิด Internet (โหลดจาก GitHub) หรือ Add Input ไฟล์ .tflite")
HAVE_CORE = len(core) == K_FOLDS
if not HAVE_CORE:
    print("[ข้อ ข.] ไม่พบไฟล์ Core ครบ 3 fold — ข้ามการวัด Core")

val = dict(FALLBACK_VAL)
cvp = next((f for f in ALL_FILES if os.path.basename(f) in ("cv_results_v6_male.json", "cv_results_v6_male.json.txt")), None)
if cvp:
    with open(cvp, encoding="utf-8") as f:
        folds = json.load(f)["results"]["male"]["folds"]
    val = {int(r["fold"]): float(r["val"]) for r in folds}
    print("ค่า val จาก", cvp, val)
else:
    print("ไม่พบ cv_results_v6_male.json — ใช้ค่า val ที่จดไว้", val)


# =============================================================================
# ### 3) รันโมเดล + วัดผล
# =============================================================================
def load_interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=path, num_threads=os.cpu_count())


def run_model(path, x, batch=16):
    it = load_interpreter(path)
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    out, shape = [], None
    for s in range(0, len(x), batch):
        xb = np.ascontiguousarray(x[s:s + batch], dtype=np.float32)
        if xb.shape != shape:
            it.resize_tensor_input(i_in, list(xb.shape))
            it.allocate_tensors()
            shape = xb.shape
        it.set_tensor(i_in, xb)
        it.invoke()
        out.append(it.get_tensor(i_out).copy())
    return np.concatenate(out)


def probs_tta(path, xs):
    """เฉลี่ยภาพปกติ + กลับซ้ายขวา -> (N, C)"""
    return (run_model(path, xs[:, 0]) + run_model(path, xs[:, 1])) / 2


def scores(p):
    order = np.argsort(-p, axis=1)
    conf = np.zeros((C, C), dtype=int)
    for t, q in zip(y, order[:, 0]):
        conf[t, q] += 1
    f1 = {}
    for i, n in enumerate(CLASS_NAMES):
        tp, ps, rs = conf[i, i], conf[:, i].sum(), conf[i].sum()
        pr, rc = (tp / ps if ps else 0.0), (tp / rs if rs else 0.0)
        f1[n] = round(float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0, 4)
    return {"top1": round(float(np.mean(order[:, 0] == y)), 4),
            "top2": round(float(np.mean((order[:, 0] == y) | (order[:, 1] == y))), 4),
            "macro_f1": round(float(np.mean(list(f1.values()))), 4), "per_class_f1": f1,
            "confusion": conf.tolist(), "n": int(len(y))}


t0 = time.time()
P_v, P_c = {}, {}
if vision:
    xs = inputs("none")
    P_v = {k: probs_tta(p, xs) for k, p in vision.items()}
    del xs
    print(f"Vision เสร็จ ({(time.time() - t0) / 60:.1f} นาที)")
if HAVE_CORE:
    xs = inputs("resnet50")
    P_c = {k: probs_tta(core[k], xs) for k in range(1, K_FOLDS + 1)}
    del xs
    print(f"Core เสร็จ ({(time.time() - t0) / 60:.1f} นาที)")

best = max(val.values())
keep = [k for k in range(1, K_FOLDS + 1) if val[k] >= best - DROP_MARGIN]
w = np.array([max(val[k] - 1 / C, 1e-6) for k in range(1, K_FOLDS + 1)])   # น้ำหนัก = val ส่วนที่เกินการเดาสุ่ม
w = w / w.sum()
keep_have = [k for k in keep if k in P_v]          # fold ที่ผ่านกฎและมีไฟล์ (บน GitHub มีแค่ fold 1+3)

results = {}
if len(P_v) == K_FOLDS:
    results["vision_all3"] = scores(np.mean([P_v[k] for k in P_v], axis=0))
    results["vision_val_weighted"] = scores(sum(w[k - 1] * P_v[k] for k in P_v))
if keep_have:
    results["vision_val_selected"] = scores(np.mean([P_v[k] for k in keep_have], axis=0))
for k in P_v:
    results[f"vision_fold{k}"] = scores(P_v[k])
if HAVE_CORE:
    results["core_all3"] = scores(np.mean([P_c[k] for k in P_c], axis=0))
    for k in P_c:
        results[f"core_fold{k}"] = scores(P_c[k])
    if keep_have:
        results["core+vision_selected"] = scores((np.mean([P_c[k] for k in P_c], axis=0)
                                                  + np.mean([P_v[k] for k in keep_have], axis=0)) / 2)
PRIMARY = "vision_val_selected" if "vision_val_selected" in results else "core_all3"

report = {"rule": f"keep folds with val >= best_val - {DROP_MARGIN}", "val": val, "kept_folds": keep,
          "weights": {k: round(float(w[k - 1]), 4) for k in range(1, K_FOLDS + 1)},
          "primary": PRIMARY, "vision_folds_used": sorted(P_v), "n_test": int(len(y)), "results": results,
          "deploy_files": [f"face_shape_male_v6_fold{k}.tflite" for k in keep_have],
          "minutes": round((time.time() - t0) / 60, 1)}
with open(os.path.join(OUTPUT_DIR, "eval_male_person_split.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 72)
print(f"ชุดทดสอบชาย (แบ่งตามคน) {len(y)} รูป | กฎ: เก็บ fold ที่ val >= {best:.3f} - {DROP_MARGIN} -> fold {keep}")
print(f"{'แบบ':28s} {'top-1':>7s} {'top-2':>7s} {'macro F1':>9s}")
for name, r in results.items():
    mark = "  <- Vision ที่ใช้ในแอป" if name == "vision_val_selected" else ("  <- Core ที่ใช้ในแอป" if name == "core_all3" else "")
    print(f"{name:28s} {r['top1']:7.2%} {r['top2']:7.2%} {r['macro_f1']:9.3f}{mark}")
for name in ("vision_val_selected", "core_all3"):
    if name in results:
        print(f"F1 ต่อคลาส ({name}):", results[name]["per_class_f1"])
print("ไฟล์ที่ต้องส่ง: eval_male_person_split.json")
print("DONE.")
