# =============================================================================
# uShape — ตรวจ "ตัวตน" ในชุดข้อมูล รุ่น 2 (แก้การจัดกลุ่มคนที่โยงต่อกันเป็นก้อนเดียวในรุ่นแรก)
#   - จัดกลุ่มคนด้วย average linkage (ทั้งกลุ่มต้องคล้ายกันโดยเฉลี่ย ไม่ใช่โยงต่อกันทีละคู่)
#     และไม่เอารูปที่ MTCNN หาหน้าไม่เจอมาจัดกลุ่ม (embedding ของรูปพวกนี้เป็นตัวเชื่อมมั่ว)
#   - รูปซ้ำจริง / ขนาดภาพ เหมือนรุ่นแรก
#   - ใหม่: วัดความแม่นยำของโมเดลที่มีอยู่ แยก "คนที่โมเดลเคยเห็นตอนเทรน" กับ "คนที่ไม่เคยเห็น"
#     (ใช้ไฟล์ .tflite ที่แนบเป็น input: face_shape_female_v4_fold1-3.tflite และ/หรือ face_shape_<เพศ>_fold1-3.tflite ของ v3)
#   - split ใหม่แบ่งตามคน (splits_person_<เพศ>.csv)
#
# Kaggle: Accelerator = GPU T4, Internet = On
#   Add Input: faceshape1 + facetrain-v1-output + dataset ที่มี .tflite (เช่น ushape-v4-female)
# วางทั้งไฟล์ในเซลล์เดียว แล้ว Save Version (~30-45 นาที)
# ส่งกลับ: identity_report.json, identity_examples.jpg (+ splits_person_*.csv)
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
SAME_PERSON_COS = [0.5, 0.6, 0.7]      # วัดหลายเกณฑ์ (ผลไม่ควรพลิกเพราะเกณฑ์)
MAIN_COS = 0.6                         # เกณฑ์หลักสำหรับรายงาน + split ใหม่
SAME_PHOTO_COS = 0.9                   # รูปเดียวกัน: embedding ใกล้มาก
SAME_PHOTO_PIX = 0.9                   #   และพิกเซล (64x64 ขาวดำ) สัมพันธ์กันสูง (คิดแบบกลับซ้าย-ขวาด้วย)
TEST_FRACTION = 0.14                   # เท่ากับ split เดิม (~14%)
K_FOLDS = 3
SEED = 42
QUICK_TEST = False
QUICK_N = 400


print("[start] audit_identity เริ่มทำงาน", flush=True)


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
    return r.returncode == 0


try:
    import facenet_pytorch  # noqa: F401
except ImportError:
    print("installing facenet-pytorch (--no-deps: ใช้ torch ที่มีอยู่แล้ว) ...", flush=True)
    if not sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "facenet-pytorch"]):
        raise SystemExit("ติดตั้ง facenet-pytorch ไม่ได้ — เปิด Settings -> Internet แล้วรันใหม่")
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__, flush=True)


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
print("DATA_ROOT:", DATA_ROOT, flush=True)


def local_original(path):
    if os.path.exists(path):
        return path
    norm = path.replace("\\", "/")
    if "datasetformodelai/" in norm:
        cand = os.path.join(DATA_ROOT, norm.split("datasetformodelai/", 1)[1])
        if os.path.exists(cand):
            return cand
    return None


def source_key(path, label):
    n = os.path.basename(path)
    b = n.split(".rf.")[0] if ".rf." in n else os.path.splitext(n)[0]
    b = re.sub(r"_(jpg|jpeg|png|webp)$", "", b, flags=re.I).lower().strip("-_ ")
    folder = os.path.relpath(path, DATA_ROOT).replace("\\", "/").split("/")[0]
    return b if re.search(r"[a-z]", b) else f"{folder}/{label}/{b}"


rows = []
for g in GENDERS:
    with open(find_splits(g), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = local_original(r["original"])
            if p:
                rows.append({"orig_csv": r["original"], "path": p, "label": r["label"], "gender": g,
                             "split": "test" if r["split"] == "test" else "trainval", "split_raw": r["split"],
                             "key": source_key(p, r["label"])})
if QUICK_TEST:
    rng0 = np.random.default_rng(SEED)
    rows = [rows[i] for i in sorted(rng0.permutation(len(rows))[:QUICK_N])]
print(f"รูปทั้งหมด {len(rows)} — กำลังอ่านขนาดภาพ ...", flush=True)
report = {"settings": {"same_person_cos": SAME_PERSON_COS, "main_cos": MAIN_COS, "same_photo_cos": SAME_PHOTO_COS,
                       "same_photo_pix": SAME_PHOTO_PIX, "model": "FaceNet InceptionResnetV1 (VGGFace2)"}}

# ----------------------------------------------------------------------------- 3) ขนาดภาพ
sizes = collections.defaultdict(collections.Counter)
for n_done, r in enumerate(rows, 1):
    if n_done % 5000 == 0:
        print(f"  [size] {n_done}/{len(rows)}", flush=True)
    with Image.open(r["path"]) as im:
        r["wh"] = im.size
    folder = os.path.relpath(r["path"], DATA_ROOT).split(os.sep)[0]
    sizes[f"{r['gender']}/{folder}"][f"{im.size[0]}x{im.size[1]}"] += 1
report["image_sizes"] = {k: {"top": v.most_common(5), "n": sum(v.values()),
                             "square_fraction": round(sum(c for s, c in v.items() if s.split("x")[0] == s.split("x")[1]) / sum(v.values()), 3)}
                         for k, v in sorted(sizes.items())}
for k, v in report["image_sizes"].items():
    print(f"[size] {k}: {v['n']} รูป, จัตุรัส {v['square_fraction']:.0%}, ขนาดที่พบบ่อย {v['top'][:3]}")

# ----------------------------------------------------------------------------- embedding (ต้นฉบับละ 1 รูป)
reps = {}
for r in rows:
    reps.setdefault((r["gender"], r["key"]), r)
reps = list(reps.values())
print(f"[face] ต้นฉบับ {len(reps)} รูป -> FaceNet embedding", flush=True)

mtcnn = MTCNN(image_size=160, margin=24, post_process=True, select_largest=True, device=DEVICE)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def load(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def fallback_tensor(img):
    """รูปในชุดข้อมูลครอปหน้ามาแล้ว: ถ้า MTCNN หาไม่เจอ ใช้ทั้งรูป (ครอปกลางเป็นจัตุรัส)"""
    x = np.asarray(ImageOps.fit(img, (160, 160)), dtype=np.float32)
    return fixed_image_standardization(torch.from_numpy(x).permute(2, 0, 1))


def pix_vec(img):
    g = np.asarray(img.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)
    out = []
    for a in (g, g[:, ::-1]):
        v = a.flatten() - a.mean()
        out.append(v / (np.linalg.norm(v) + 1e-6))
    return out


t0 = time.time()
faces, detected = [], 0
PIX = {}
for i, r in enumerate(reps, 1):
    img = load(r["path"])
    PIX[r["path"]] = pix_vec(img)
    face = None
    try:
        face = mtcnn(img)
    except Exception:
        face = None
    r["detected"] = face is not None
    if face is None:
        face = fallback_tensor(img)
    else:
        detected += 1
    faces.append(face)
    if i % 500 == 0 or i == len(reps):
        print(f"  [mtcnn] {i}/{len(reps)} (เจอหน้า {detected / i:.0%}) {time.time() - t0:.0f}s", flush=True)
emb = []
with torch.no_grad():
    for s in range(0, len(faces), 128):
        e = facenet(torch.stack(faces[s:s + 128]).to(DEVICE))
        emb.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
emb = np.concatenate(emb).astype(np.float32)
for r, e in zip(reps, emb):
    r["emb"] = e
report["face_detection_rate"] = round(detected / max(1, len(reps)), 3)
del faces

# ----------------------------------------------------------------------------- 1) รูปซ้ำจริง
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

dup_pairs = []
by_gender = {g: [r for r in reps if r["gender"] == g] for g in GENDERS}
for g, R in by_gender.items():
    if len(R) < 2:
        continue
    E = np.stack([r["emb"] for r in R])
    S = E @ E.T
    ii, jj = np.where(np.triu(S, 1) >= SAME_PHOTO_COS)
    for i, j in zip(ii, jj):
        a, b = R[i], R[j]
        pa, pb = PIX[a["path"]], PIX[b["path"]]
        pix = max(float(pa[0] @ pb[0]), float(pa[0] @ pb[1]))
        if pix >= SAME_PHOTO_PIX:
            dup_pairs.append((a, b, float(S[i, j]), pix))
# ข้ามเพศ (ป้ายเพศผิด) ด้วย
if all(len(by_gender[g]) for g in GENDERS):
    F, M = by_gender["female"], by_gender["male"]
    S = np.stack([r["emb"] for r in F]) @ np.stack([r["emb"] for r in M]).T
    for i, j in zip(*np.where(S >= SAME_PHOTO_COS)):
        pa, pb = PIX[F[i]["path"]], PIX[M[j]["path"]]
        pix = max(float(pa[0] @ pb[0]), float(pa[0] @ pb[1]))
        if pix >= SAME_PHOTO_PIX:
            dup_pairs.append((F[i], M[j], float(S[i, j]), pix))
report["same_photo"] = {
    "pairs": len(dup_pairs),
    "label_conflict": sum(a["label"] != b["label"] for a, b, _, _ in dup_pairs),
    "test_vs_trainval": sum(a["split"] != b["split"] for a, b, _, _ in dup_pairs),
    "cross_gender": sum(a["gender"] != b["gender"] for a, b, _, _ in dup_pairs),
    "conflicts": dict(collections.Counter("/".join(sorted((a["label"], b["label"]))) for a, b, _, _ in dup_pairs
                                          if a["label"] != b["label"])),
    "by_pixel_threshold": {str(t): {"pairs": sum(p >= t for *_, p in dup_pairs),
                                    "label_conflict": sum(p >= t and a["label"] != b["label"] for a, b, _, p in dup_pairs),
                                    "test_vs_trainval": sum(p >= t and a["split"] != b["split"] for a, b, _, p in dup_pairs)}
                           for t in (0.9, 0.95, 0.98)},
}
print("[same photo]", json.dumps(report["same_photo"], ensure_ascii=False))


def thumb(path, T=110):
    return ImageOps.fit(load(path), (T, T))


def sheet_pairs(pairs, out, T=110, cols=4):
    if not pairs:
        return
    W, Hh = cols * (2 * T + 8), math.ceil(len(pairs) / cols) * (T + 30)
    sh_ = Image.new("RGB", (W, Hh), (255, 255, 255))
    d = ImageDraw.Draw(sh_)
    for k, (a, b, c, p) in enumerate(pairs):
        x, y = (k % cols) * (2 * T + 8), (k // cols) * (T + 30)
        for j, r in enumerate((a, b)):
            sh_.paste(thumb(r["path"], T), (x + j * T, y + 28))
            d.text((x + j * T + 2, y + 2), f"{r['label']}/{r['split'][:5]}", fill=(180, 0, 0) if j else (0, 0, 160))
        d.text((x + 2, y + 14), f"cos {c:.2f} pix {p:.2f}", fill=(80, 80, 80))
    sh_.save(out, quality=88)


conf = sorted([x for x in dup_pairs if x[0]["label"] != x[1]["label"]], key=lambda x: -x[3])
same = sorted([x for x in dup_pairs if x[0]["label"] == x[1]["label"]], key=lambda x: -x[3])
sheet_pairs(conf[:12] + same[:4], os.path.join(OUTPUT_DIR, "duplicate_examples.jpg"))

# ----------------------------------------------------------------------------- 2) จัดกลุ่มตามคน
from sklearn.cluster import AgglomerativeClustering


def clusters(R, cos):
    """average linkage: กลุ่มจะรวมกันได้ก็ต่อเมื่อ cos เฉลี่ยระหว่างทุกคู่ >= cos (กันการโยงต่อเป็นก้อนใหญ่)
    รูปที่หาหน้าไม่เจอ = คนละคนเสมอ (ไม่เอามาจัดกลุ่ม)"""
    idx = [i for i, r in enumerate(R) if r.get("detected")]
    lab = np.arange(len(R)) + 10 ** 6
    if len(idx) >= 2:
        E = np.stack([R[i]["emb"] for i in idx])
        ac = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - cos).fit(E)
        lab[idx] = ac.labels_
    return lab


identity, person_maps = {}, {}
for g, R in by_gender.items():
    if len(R) < 2:
        continue
    res = {}
    for cos in SAME_PERSON_COS:
        lab = clusters(R, cos)
        groups = collections.defaultdict(list)
        for r, l in zip(R, lab):
            groups[l].append(r)
        multi = {l: m for l, m in groups.items() if len(m) > 1}
        spans = {l for l, m in groups.items() if len({r["split"] for r in m}) > 1}
        mixed = {l for l, m in groups.items() if len({r["label"] for r in m}) > 1}
        test_idx = [i for i, r in enumerate(R) if r["split"] == "test"]
        test_src = test_idx
        leaked = [i for i in test_idx if lab[i] in spans]
        sizes_ = sorted((len(m) for m in groups.values()), reverse=True)
        res[str(cos)] = {
            "people": len(groups), "people_with_2plus_photos": len(multi), "largest_group": sizes_[:5],
            "people_in_both_test_and_trainval": len(spans),
            "test_sources": len(test_src), "test_sources_whose_person_is_in_trainval": len(leaked),
            "test_leak_fraction": round(len(leaked) / max(1, len(test_src)), 3),
            "people_with_conflicting_labels": len(mixed),
            "people_with_conflicting_labels_fraction_of_multi": round(len(mixed) / max(1, len(multi)), 3),
            "largest_group_fraction": round(sizes_[0] / len(R), 3),
        }
        if sizes_[0] > 0.05 * len(R):
            print(f"  [identity] {g} cos≥{cos}: กลุ่มใหญ่สุด {sizes_[0]} รูป ({sizes_[0] / len(R):.0%}) — เกณฑ์นี้หลวมเกินไป (คนละคนถูกโยงต่อกัน)")
        person_maps.setdefault(g, {})[str(cos)] = {r["key"]: f"{g[0]}{l}" for r, l in zip(R, lab)}
        if cos == MAIN_COS:
            for r, l in zip(R, lab):
                r["person"] = f"{g[0]}{l}"
            identity[g] = groups
    report.setdefault("identity", {})[g] = res
    print(f"[identity] {g}: " + " | ".join(
        f"cos≥{c}: คน {v['people']}, test รั่ว {v['test_leak_fraction']:.0%}, ป้ายขัดในคนเดียวกัน {v['people_with_conflicting_labels']}"
        for c, v in res.items()))

# ตัวอย่างคน (ที่อยู่ทั้งสองชุด) — แถวละคน
ex_rows = []
rng_ex = np.random.default_rng(SEED)
for g, groups in identity.items():
    multi = [m for m in groups.values() if len(m) > 1]
    big = sorted(multi, key=len, reverse=True)[:3]                                   # กลุ่มใหญ่สุด: เช็กว่าไม่ปนคน
    mixed = [m for m in multi if len({r["label"] for r in m}) > 1 and m not in big][:3]  # คนเดียวกันป้ายต่างกัน
    rest = [m for m in multi if not any(m is x for x in big + mixed)]
    rnd = [rest[i] for i in rng_ex.permutation(len(rest))[:3]]                       # สุ่ม
    ex_rows += big + mixed + rnd
if ex_rows:
    T, cols = 96, 8
    img = Image.new("RGB", (cols * T, len(ex_rows) * (T + 16)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, m in enumerate(ex_rows):
        m = sorted(m, key=lambda r: r["split"])[:cols]
        for j, r in enumerate(m):
            img.paste(thumb(r["path"], T), (j * T, i * (T + 16) + 16))
            d.text((j * T + 2, i * (T + 16) + 2), f"{r['label']}/{r['split'][:5]}",
                   fill=(180, 0, 0) if r["split"] == "test" else (0, 0, 160))
    img.save(os.path.join(OUTPUT_DIR, "identity_examples.jpg"), quality=88)

# ----------------------------------------------------------------------------- 4) split ใหม่แบ่งตามคน
from sklearn.model_selection import StratifiedGroupKFold

person_of = {(r["gender"], r["key"]): r.get("person") for r in reps}
for g in GENDERS:
    G = [r for r in rows if r["gender"] == g and person_of.get((g, r["key"]))]
    if len(G) < 20:
        continue
    y = [r["label"] for r in G]
    groups = [person_of[(g, r["key"])] for r in G]
    biggest = collections.Counter(groups).most_common(1)[0][1]
    if biggest > 0.05 * len(G):
        print(f"[split] {g}: กลุ่มคนใหญ่สุด {biggest} รูป ({biggest / len(G):.0%}) ใหญ่เกินไป — ไม่สร้าง split ใหม่")
        continue
    n_test = max(2, round(1 / TEST_FRACTION))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(sgkf.split(np.zeros(len(G)), y, groups))
    split = {}
    for i in te_idx:
        split[i] = "test"
    rest = list(tr_idx)
    sg3 = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(sg3.split(np.zeros(len(rest)), [y[i] for i in rest], [groups[i] for i in rest]), 1):
        for j in va:
            split[rest[j]] = f"fold{k}_val"
    with open(os.path.join(OUTPUT_DIR, f"splits_person_{g}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original", "crop", "label", "split", "person"])
        for i, r in enumerate(G):
            w.writerow([r["orig_csv"], "", r["label"], split[i], groups[i]])
    tp = {groups[i] for i in te_idx}
    assert not (tp & {groups[i] for i in tr_idx}), "person leak in new split"
    cnt = collections.Counter(split.values())
    report.setdefault("person_split", {})[g] = {"counts": dict(cnt),
                                                 "test_per_class": dict(collections.Counter(y[i] for i in te_idx))}
    print(f"[split] {g}: {dict(cnt)} (ไม่มีคนซ้ำระหว่าง test กับ train)")

# ----------------------------------------------------------------------------- 5) โมเดลที่มีอยู่: คนที่เคยเห็น vs ไม่เคยเห็น
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
sys.path.insert(0, OUTPUT_DIR)
import importlib
import face_crop
import image_ops
face_crop, image_ops = importlib.reload(face_crop), importlib.reload(image_ops)

TFL = re.compile(r"^face_shape_(female|male)(?:_(v\d+))?_fold(\d)\.tflite$")
sets = collections.defaultdict(dict)
for root, _, files in walk():
    for fn in files:
        m = TFL.match(fn)
        if m:
            sets[(m.group(1), m.group(2) or "v3")][int(m.group(3))] = os.path.join(root, fn)
sets = {k: [v[i] for i in sorted(v)] for k, v in sets.items() if len(v) >= 1}
print("[eval] ชุดโมเดลที่พบ:", {f"{g} {t}": len(v) for (g, t), v in sets.items()}, flush=True)


def interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    return it


def run_it(it, x):
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    it.resize_tensor_input(i_in, list(x.shape))
    it.allocate_tensors()
    it.set_tensor(i_in, np.ascontiguousarray(x, dtype=np.float32))
    it.invoke()
    return it.get_tensor(i_out).copy()


evals = {}
if sets:
    test_reps = [r for r in reps if r["split"] == "test" and r["gender"] in {g for g, _ in sets}]
    manifest = face_crop.ensure_face_crops([r["path"] for r in test_reps], DATA_ROOT, os.path.join(OUTPUT_DIR, "_crops"))
    seen_people = {}
    for g in GENDERS:
        for cos_key, lab_of in person_maps.get(g, {}).items():
            tv = {lab_of[r["key"]] for r in by_gender[g] if r["split"] == "trainval"}
            seen_people[(g, cos_key)] = tv
    for (g, tag), files in sets.items():
        pre = "resnet50" if tag == "v3" else "none"
        its = [interpreter(f) for f in files]
        R = [r for r in test_reps if r["gender"] == g and manifest.get(r["path"], {}).get("status", "").startswith("mesh")]
        t1 = time.time()
        correct = []
        for r in R:
            with Image.open(manifest[r["path"]]["crop"]) as im:
                x = image_ops.prepare_batch(np.asarray(im.convert("RGB"), dtype="float32"), (224, 224), pre, True)
            p = np.mean([run_it(it, x) for it in its], axis=(0, 1))
            r[f"pred_{tag}"] = CLASS_NAMES[int(np.argmax(p))]
            order = np.argsort(-p)
            correct.append((r, CLASS_NAMES[order[0]] == r["label"], r["label"] in (CLASS_NAMES[order[0]], CLASS_NAMES[order[1]])))
        res = {"models": [os.path.basename(f) for f in files], "preprocess": pre, "unit": "รูปต้นฉบับละ 1 รูป (ชุดทดสอบ)",
               "n": len(correct), "top1": round(float(np.mean([c for _, c, _ in correct])), 4) if correct else None,
               "top2": round(float(np.mean([c2 for _, _, c2 in correct])), 4) if correct else None,
               "minutes": round((time.time() - t1) / 60, 1)}
        for cos_key in person_maps.get(g, {}):
            lab_of, tv = person_maps[g][cos_key], seen_people[(g, cos_key)]
            seen = [(c, c2) for r, c, c2 in correct if lab_of[r["key"]] in tv]
            unseen = [(c, c2) for r, c, c2 in correct if lab_of[r["key"]] not in tv]
            res[f"cos{cos_key}"] = {
                "seen_person": {"n": len(seen), "top1": round(float(np.mean([a for a, _ in seen])), 4) if seen else None,
                                "top2": round(float(np.mean([b for _, b in seen])), 4) if seen else None},
                "unseen_person": {"n": len(unseen), "top1": round(float(np.mean([a for a, _ in unseen])), 4) if unseen else None,
                                  "top2": round(float(np.mean([b for _, b in unseen])), 4) if unseen else None}}
        evals[f"{g} {tag}"] = res
        print(f"[eval] {g} {tag}: " + json.dumps({k: v for k, v in res.items() if k not in ("models",)}, ensure_ascii=False), flush=True)
report["model_eval_seen_vs_unseen"] = evals

with open(os.path.join(OUTPUT_DIR, "identity_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("ส่งกลับ: identity_report.json, identity_examples.jpg, duplicate_examples.jpg (+ splits_person_*.csv)")
print("DONE.")
