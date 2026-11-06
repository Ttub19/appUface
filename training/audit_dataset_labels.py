# =============================================================================
# uShape — ตรวจคุณภาพป้ายของชุดข้อมูลแบบอัตโนมัติ (ไม่ต้องมีคนตัดสิน)
#   1) รูปซ้ำ/เกือบซ้ำที่มาจาก "คนละไฟล์ต้นฉบับ" (dHash) -> ป้ายขัดกันไหม / ข้ามชุดทดสอบ-ชุดเทรนไหม
#   2) สัดส่วนใบหน้าจาก landmark 468 จุด (ยาว/กว้าง, หน้าผาก/กราม, กราม/โหนกแก้ม, คาง/กราม)
#      เทียบระหว่างคลาส + ลองจำแนกด้วยสัดส่วนอย่างเดียว (ป้ายสอดคล้องกับนิยามเรขาคณิตแค่ไหน)
#   3) ขนาดจริงของชุดทดสอบ (นับรูปต้นฉบับไม่ซ้ำ ไม่นับสำเนาที่ Roboflow ทำเพิ่ม)
#
# Kaggle: Accelerator = None (CPU), Internet = On (ติดตั้ง MediaPipe + โหลดโมเดล landmark)
#   Add Input: faceshape1 + facetrain-v1-output
# วางทั้งไฟล์ในเซลล์เดียว แล้ว Save Version (~10-20 นาที)
# ได้ไฟล์: label_audit_report.json, geometry_female.png, geometry_male.png,
#          near_duplicates.csv, conflict_examples.jpg
# =============================================================================
import collections
import csv
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

INPUT_ROOT = "/kaggle/input"
OUTPUT_DIR = "/kaggle/working"
CLASS_NAMES = ["heart", "oblong", "oval", "round", "square"]
GENDERS = ["female", "male"]
HASH_MAX_DIST = 4             # dHash 64 บิต ต่างกันไม่เกินนี้ = รูปเดียวกัน/เกือบเหมือน
SEED = 42
QUICK_TEST = False            # True = ใช้รูปน้อย ๆ ลองว่าโค้ดไม่พัง
QUICK_N = 300

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
with open(os.path.join(OUTPUT_DIR, "face_crop.py"), "w", encoding="utf-8") as f:
    f.write(FACE_CROP_SOURCE)
sys.path.insert(0, OUTPUT_DIR)
import importlib
import face_crop
face_crop = importlib.reload(face_crop)


# ----------------------------------------------------------------------------- input
def walk(base=INPUT_ROOT):
    yield from os.walk(base)


def find_data_root():
    for root, dirs, _ in walk():
        if "train1_cropped" in dirs and "testing_set" in dirs:
            return root
    raise SystemExit("ไม่พบ dataset faceshape1 — Add Input ก่อน")


def find_splits(gender):
    names = (f"splits_{gender}.csv", f"splits_{gender}.csv.txt", f"splits_{gender}.txt")
    for root, _, files in walk():
        for n in names:
            if n in files:
                p = os.path.join(root, n)
                with open(p, encoding="utf-8") as f:
                    if "male_extra" in f.read():
                        continue
                return p
    raise SystemExit(f"ไม่พบ splits_{gender}.csv — Add Input facetrain-v1-output")


DATA_ROOT = find_data_root()


def local_original(path):
    if os.path.exists(path):
        return path
    norm = path.replace("\\", "/")
    if "datasetformodelai/" in norm:
        cand = os.path.join(DATA_ROOT, norm.split("datasetformodelai/", 1)[1])
        if os.path.exists(cand):
            return cand
    return None


def source_name(path):
    """ชื่อไฟล์ต้นฉบับก่อน Roboflow เติม _jpg.rf.<hash> (สำเนา augment ของรูปเดียวกันได้ชื่อเดียวกัน)"""
    n = os.path.basename(path)
    b = n.split(".rf.")[0] if ".rf." in n else os.path.splitext(n)[0]
    return re.sub(r"_(jpg|jpeg|png|webp)$", "", b, flags=re.I).lower().strip("-_ ")


def source_key(path, label):
    rel = os.path.relpath(path, DATA_ROOT).replace("\\", "/")
    name = source_name(path)
    # ชื่อที่เป็นตัวเลขล้วน (เช่น "17") ถูกนับใหม่ในทุกโฟลเดอร์คลาส -> คนละรูปกัน ต้องผูกกับโฟลเดอร์ด้วย
    return name if re.search(r"[a-z]", name) else f"{rel.split('/')[0]}/{label}/{name}"


rows = []
for g in GENDERS:
    with open(find_splits(g), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = local_original(r["original"])
            if p:
                rows.append({"path": p, "label": r["label"], "split": "test" if r["split"] == "test" else "trainval",
                             "gender": g, "key": source_key(p, r["label"])})
if QUICK_TEST:
    rng = np.random.default_rng(SEED)
    rows = [rows[i] for i in sorted(rng.permutation(len(rows))[:QUICK_N])]
print(f"รูปทั้งหมด {len(rows)}")

report = {"test_size": {}, "near_duplicates": {}, "geometry": {}}

# ----------------------------------------------------------------------------- 3) ขนาดจริงของชุดทดสอบ
for g in GENDERS:
    t = [r for r in rows if r["gender"] == g and r["split"] == "test"]
    allg = [r for r in rows if r["gender"] == g]
    report["test_size"][g] = {
        "images": len(t), "unique_sources": len({r["key"] for r in t}),
        "per_class_images": dict(collections.Counter(r["label"] for r in t)),
        "per_class_sources": {c: len({r["key"] for r in t if r["label"] == c}) for c in CLASS_NAMES},
        "all_images": len(allg), "all_unique_sources": len({r["key"] for r in allg}),
    }
    print(f"[test] {g}: {len(t)} รูป จากรูปต้นฉบับ {report['test_size'][g]['unique_sources']} รูป "
          f"{report['test_size'][g]['per_class_sources']}")


# ----------------------------------------------------------------------------- 1) dHash
def dhash_bits(im, size=8):
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hashes(path):
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return path, to_int(dhash_bits(im)), to_int(dhash_bits(ImageOps.mirror(im)))
    except Exception:
        return path, None, None


import multiprocessing

t0 = time.time()
with multiprocessing.get_context("fork").Pool(os.cpu_count() or 2) as pool:
    H = {p: (h, hm) for p, h, hm in pool.imap_unordered(hashes, [r["path"] for r in rows], chunksize=64) if h is not None}
print(f"[dhash] {len(H)} รูป ({time.time() - t0:.0f}s)")

by_path = {r["path"]: r for r in rows}
bands = collections.defaultdict(list)            # LSH: แบ่ง 64 บิตเป็น 8 ช่วง ช่วงละ 8 บิต
for p, (h, _) in H.items():
    for b in range(8):
        bands[(b, (h >> (8 * b)) & 0xFF)].append(p)


def ham(a, b):
    return bin(a ^ b).count("1")


pairs = {}
for p, (h, hm) in H.items():
    cands = set()
    for hh in (h, hm):                           # รูปกลับซ้าย-ขวาด้วย
        for b in range(8):
            cands.update(bands.get((b, (hh >> (8 * b)) & 0xFF), ()))
    for q in cands:
        if q <= p:
            continue
        rp, rq = by_path[p], by_path[q]
        if rp["key"] == rq["key"]:
            continue                             # สำเนา augment ของต้นฉบับเดียวกัน (รู้อยู่แล้ว)
        d = min(ham(h, H[q][0]), ham(hm, H[q][0]))
        if d <= HASH_MAX_DIST:
            pairs[(p, q)] = d
print(f"[dhash] คู่รูปเกือบซ้ำจากต่างต้นฉบับ {len(pairs)} คู่")

pair_rows = []
for (p, q), d in sorted(pairs.items(), key=lambda kv: kv[1]):
    a, b = by_path[p], by_path[q]
    pair_rows.append({"dist": d, "a": os.path.relpath(p, DATA_ROOT), "a_label": a["label"], "a_split": a["split"],
                      "a_gender": a["gender"], "b": os.path.relpath(q, DATA_ROOT), "b_label": b["label"],
                      "b_split": b["split"], "b_gender": b["gender"]})
with open(os.path.join(OUTPUT_DIR, "near_duplicates.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(pair_rows[0]) if pair_rows else ["dist"])
    w.writeheader()
    w.writerows(pair_rows)


def group_count(cond):
    """นับเป็นกลุ่มต้นฉบับ (key) ไม่นับซ้ำทุกสำเนา"""
    return len({tuple(sorted((by_path[p]["key"], by_path[q]["key"]))) for (p, q) in pairs if cond(by_path[p], by_path[q])})


report["near_duplicates"] = {
    "pairs_images": len(pairs),
    "source_pairs": group_count(lambda a, b: True),
    "source_pairs_conflicting_label": group_count(lambda a, b: a["label"] != b["label"]),
    "source_pairs_test_vs_trainval": group_count(lambda a, b: a["split"] != b["split"]),
    "source_pairs_cross_gender": group_count(lambda a, b: a["gender"] != b["gender"]),
    "conflict_label_pairs": dict(collections.Counter(
        "/".join(sorted((by_path[p]["label"], by_path[q]["label"]))) for (p, q) in pairs
        if by_path[p]["label"] != by_path[q]["label"])),
    "hash_max_dist": HASH_MAX_DIST,
}
print("[dhash]", json.dumps(report["near_duplicates"], ensure_ascii=False))

# ตัวอย่างคู่ที่ป้ายขัดกัน (1 คู่ต่อคู่ต้นฉบับ)
seen, ex = set(), []
for (p, q), d in sorted(pairs.items(), key=lambda kv: kv[1]):
    a, b = by_path[p], by_path[q]
    k = tuple(sorted((a["key"], b["key"])))
    if a["label"] != b["label"] and k not in seen:
        seen.add(k)
        ex.append((p, q))
    if len(ex) >= 16:
        break
if ex:
    T = 150
    sheet = Image.new("RGB", (4 * (2 * T + 10), math.ceil(len(ex) / 4) * (T + 20)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (p, q) in enumerate(ex):
        x, y = (i % 4) * (2 * T + 10), (i // 4) * (T + 20)
        for j, s in enumerate((p, q)):
            with Image.open(s) as im:
                sheet.paste(ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (T, T)), (x + j * T, y + 18))
            r = by_path[s]
            draw.text((x + j * T + 3, y + 3), f"{r['label']} ({r['split']})", fill=(200, 0, 0) if j else (0, 0, 160))
    sheet.save(os.path.join(OUTPUT_DIR, "conflict_examples.jpg"), quality=88)

# ----------------------------------------------------------------------------- 2) สัดส่วนใบหน้า
LM = {"top": 10, "chin": 152, "cheek_r": 234, "cheek_l": 454, "jaw_r": 172, "jaw_l": 397,
      "brow_r": 54, "brow_l": 284, "chin_r": 176, "chin_l": 400}
FEATURES = ["length_width", "forehead_jaw", "jaw_cheek", "chin_jaw", "jaw_angle"]
FEATURE_TH = {"length_width": "ยาว/กว้าง", "forehead_jaw": "หน้าผาก/กราม", "jaw_cheek": "กราม/โหนกแก้ม",
              "chin_jaw": "คาง/กราม", "jaw_angle": "มุมกราม (องศา)"}

reps = {}
for r in rows:                                   # ต้นฉบับละ 1 รูป (สำเนา augment ไม่ช่วยอะไร)
    reps.setdefault((r["gender"], r["key"]), r)
reps = list(reps.values())
print(f"[geometry] วัด landmark {len(reps)} รูป (ต้นฉบับละ 1 รูป)")

WORKER = r'''
import json, sys, multiprocessing, numpy as np
import face_crop
from face_crop import FaceLandmarkDetector, find_landmarks, load_rgb, CropParams
D = {}
def init(model):
    D["det"] = FaceLandmarkDetector(model)
def run(path):
    try:
        pts, method = find_landmarks(D["det"], load_rgb(path), CropParams())
        return path, (None if pts is None else np.asarray(pts)[:468].round(2).tolist()), method
    except Exception as e:
        return path, None, "error: " + str(e)[:100]
if __name__ == "__main__":
    job = json.load(open(sys.argv[1]))
    out = {}
    with multiprocessing.get_context("fork").Pool(job["workers"], initializer=init, initargs=(job["model"],)) as pool:
        for i, (p, pts, m) in enumerate(pool.imap_unordered(run, job["paths"], chunksize=8), 1):
            out[p] = {"pts": pts, "method": m}
            if i % 500 == 0 or i == len(job["paths"]):
                print(f"[landmark] {i}/{len(job['paths'])}", flush=True)
    json.dump(out, open(sys.argv[2], "w"))
'''
env = face_crop.ensure_mediapipe()
model = face_crop.ensure_model(os.path.join(env, "face_landmarker.task"))
with open(os.path.join(OUTPUT_DIR, "_lm_worker.py"), "w") as f:
    f.write(WORKER)
job, res = os.path.join(OUTPUT_DIR, "_lm_job.json"), os.path.join(OUTPUT_DIR, "_lm_out.json")
with open(job, "w") as f:
    json.dump({"paths": [r["path"] for r in reps], "model": model, "workers": os.cpu_count() or 2}, f)
code = (f"import sys, runpy; sys.path[:0] = [{env!r}, {OUTPUT_DIR!r}]; sys.argv = ['w', {job!r}, {res!r}]; "
        f"runpy.run_path({os.path.join(OUTPUT_DIR, '_lm_worker.py')!r}, run_name='__main__')")
face_crop._stream([sys.executable, "-I", "-c", code])
with open(res) as f:
    LMS = json.load(f)
for tmp in (job, res, os.path.join(OUTPUT_DIR, "_lm_worker.py")):
    os.remove(tmp)


def features(pts):
    P = np.asarray(pts, dtype=float)

    def dist(a, b):
        return float(np.linalg.norm(P[LM[a]] - P[LM[b]]))

    width = dist("cheek_r", "cheek_l")
    jaw = dist("jaw_r", "jaw_l")

    def angle(a, b, c):                          # มุมที่จุด b
        v1, v2 = P[LM[a]] - P[LM[b]], P[LM[c]] - P[LM[b]]
        return math.degrees(math.acos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))

    return {"length_width": dist("top", "chin") / width, "forehead_jaw": dist("brow_r", "brow_l") / jaw,
            "jaw_cheek": jaw / width, "chin_jaw": dist("chin_r", "chin_l") / jaw,
            "jaw_angle": (angle("cheek_r", "jaw_r", "chin") + angle("cheek_l", "jaw_l", "chin")) / 2}


geo = []
for r in reps:
    item = LMS.get(r["path"]) or {}
    if item.get("pts"):
        geo.append({**r, **features(item["pts"])})
with open(os.path.join(OUTPUT_DIR, "geometry.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["gender", "label", "split", "key", "path"] + FEATURES)
    w.writeheader()
    for gr in geo:
        w.writerow({k: (os.path.relpath(gr[k], DATA_ROOT) if k == "path" else
                        (round(gr[k], 4) if k in FEATURES else gr[k])) for k in w.fieldnames})
print(f"[geometry] หา landmark ได้ {len(geo)}/{len(reps)} รูป")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

for g in GENDERS:
    data = [x for x in geo if x["gender"] == g]
    if len(data) < 20:
        continue
    X = np.array([[x[k] for k in FEATURES] for x in data])
    y = np.array([CLASS_NAMES.index(x["label"]) for x in data])
    stats = {c: {k: [round(float(v), 3) for v in np.percentile([x[k] for x in data if x["label"] == c], [25, 50, 75])]
                 for k in FEATURES} for c in CLASS_NAMES if any(x["label"] == c for x in data)}
    present = sorted(set(y))
    n_splits = max(2, min(5, min(collections.Counter(y).values())))
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=SEED))
    report["geometry"][g] = {
        "n_sources": len(data), "quartiles_25_50_75": stats,
        "geometry_only_accuracy": round(float(np.mean(pred == y)), 4),
        "geometry_only_macro_f1": round(float(f1_score(y, pred, labels=present, average="macro")), 4),
        "confusion": confusion_matrix(y, pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "cv_folds": n_splits,
    }
    print(f"[geometry] {g}: จำแนกด้วยสัดส่วนอย่างเดียว acc {report['geometry'][g]['geometry_only_accuracy']:.2%} "
          f"macro F1 {report['geometry'][g]['geometry_only_macro_f1']:.3f} (ต้นฉบับ {len(data)} รูป)")
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 3.6))
    for ax, k in zip(axes, FEATURES):
        cls = [c for c in CLASS_NAMES if any(x["label"] == c for x in data)]
        ax.boxplot([[x[k] for x in data if x["label"] == c] for c in cls], showfliers=False)
        ax.set_xticks(range(1, len(cls) + 1), cls, rotation=30)
        ax.set_title(k)
    fig.suptitle(f"{g}: face geometry by dataset label (one image per source photo)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"geometry_{g}.png"), dpi=120)
    plt.close(fig)

report["feature_names_th"] = FEATURE_TH
with open(os.path.join(OUTPUT_DIR, "label_audit_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("ไฟล์ที่ต้องส่ง: label_audit_report.json, geometry_female.png, geometry_male.png, conflict_examples.jpg, near_duplicates.csv")
print("DONE.")
