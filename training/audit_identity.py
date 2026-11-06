# =============================================================================
# uShape — ตรวจ "ตัวตน" ในชุดข้อมูลด้วยโมเดลจดจำใบหน้า (FaceNet, VGGFace2)
#   1) รูปซ้ำจริง (คนเดียวกัน + ภาพเกือบเหมือนกันทุกพิกเซล) ที่มาจากคนละไฟล์ต้นฉบับ -> ป้ายขัดกัน / ข้ามชุดทดสอบ
#   2) จัดกลุ่มรูปตามคน -> มีกี่คน, คนเดียวกันอยู่ทั้งชุดเทรนและชุดทดสอบไหม (identity leakage),
#      คนเดียวกันถูกติดป้ายต่างกันไหม
#   3) ขนาดภาพ (ถูกยืดเป็นสี่เหลี่ยมจัตุรัสไหม)
#   4) สร้าง split ใหม่ที่แบ่ง "ตามคน" (splits_person_<เพศ>.csv) ไว้เทรนรอบถัดไป ถ้าพบว่าคนรั่วข้ามชุด
#
# Kaggle: Accelerator = GPU T4 (เร็วกว่า; ใช้ CPU ได้แต่ช้ากว่า), Internet = On (ติดตั้ง facenet-pytorch + โหลดน้ำหนัก)
#   Add Input: faceshape1 + facetrain-v1-output
# วางทั้งไฟล์ในเซลล์เดียว แล้ว Save Version (~15-30 นาที)
# ส่งกลับ: identity_report.json, identity_examples.jpg, duplicate_examples.jpg
#          (+ splits_person_female.csv, splits_person_male.csv ถ้าจะเทรนใหม่)
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
print("device:", DEVICE, "| torch", torch.__version__)


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
print(f"รูปทั้งหมด {len(rows)}")
report = {"settings": {"same_person_cos": SAME_PERSON_COS, "main_cos": MAIN_COS, "same_photo_cos": SAME_PHOTO_COS,
                       "same_photo_pix": SAME_PHOTO_PIX, "model": "FaceNet InceptionResnetV1 (VGGFace2)"}}

# ----------------------------------------------------------------------------- 3) ขนาดภาพ
sizes = collections.defaultdict(collections.Counter)
for r in rows:
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
def clusters(R, cos):
    E = np.stack([r["emb"] for r in R])
    S = E @ E.T
    # complete-ish: เชื่อมเฉพาะคู่ที่ใกล้กันมาก แล้ว connected components (ง่าย + เร็ว), รายงานหลายเกณฑ์
    A = csr_matrix(S >= cos)
    _, lab = connected_components(A, directed=False)
    return lab


identity = {}
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
for g, groups in identity.items():
    cand = [m for m in groups.values() if len({r["split"] for r in m}) > 1 and 3 <= len(m) <= 40]
    cand.sort(key=lambda m: (-len({r["label"] for r in m}), -len(m)))
    ex_rows += cand[:5 if g == "female" else 3]
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

with open(os.path.join(OUTPUT_DIR, "identity_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("ส่งกลับ: identity_report.json, identity_examples.jpg, duplicate_examples.jpg (+ splits_person_*.csv)")
print("DONE.")
