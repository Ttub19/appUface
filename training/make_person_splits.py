# =============================================================================
# uShape — สร้าง split ใหม่ที่ "แบ่งตามคน" (คนเดียวกันไม่อยู่ทั้งชุดเทรนและชุดทดสอบ) ทั้งหญิงและชาย
#   - หน้า -> FaceNet embedding (ต้นฉบับละ 1 รูป) -> จัดกลุ่มคน average linkage cos >= PERSON_COS
#   - ชุดทดสอบ: เลือกทีละคน (ทุกรูปของคนนั้น) จนได้ ~14% ของแต่ละคลาส (สมดุลคลาส ไม่เหมือนรุ่นก่อน)
#   - ที่เหลือแบ่ง 3 fold ตามคน เกลี่ยจำนวนแต่ละคลาสให้ใกล้กัน
# Kaggle: GPU T4 (หรือ CPU ก็ได้ ช้ากว่า), Internet = On
#   Add Input: faceshape1 + facetrain-v1-output
# ~10 นาที (GPU) | ได้ splits_person_female.csv, splits_person_male.csv, person_split_report.json
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
PERSON_COS = 0.6                       # average linkage; ต่ำ = รวมกลุ่มมากขึ้น = กันรั่วได้ดีขึ้น
SAME_PHOTO_COS = 0.9                   # รูปเดียวกัน: embedding ใกล้มาก
SAME_PHOTO_PIX = 0.9                   #   และพิกเซล (64x64 ขาวดำ) สัมพันธ์กันสูง (คิดแบบกลับซ้าย-ขวาด้วย)
TEST_FRACTION = 0.14                   # เท่ากับ split เดิม (~14%)
K_FOLDS = 3
SEED = 42
QUICK_TEST = False
QUICK_N = 400


print("[start] make_person_splits เริ่มทำงาน", flush=True)


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
print(f"รูปทั้งหมด {len(rows)} — กำลังหาหน้า ...", flush=True)
report = {}

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



out_report = {"person_cos": PERSON_COS, "test_fraction": TEST_FRACTION, "genders": {}}
rng = np.random.default_rng(SEED)
for g in GENDERS:
    R = [r for r in reps if r["gender"] == g]
    if len(R) < 10:
        continue
    lab = clusters(R, PERSON_COS)
    person_of = {r["key"]: f"{g[0]}{l}" for r, l in zip(R, lab)}
    G = [r for r in rows if r["gender"] == g]
    imgs = collections.defaultdict(list)
    for i, r in enumerate(G):
        imgs[person_of[r["key"]]].append(i)
    major = {p: collections.Counter(G[i]["label"] for i in ix).most_common(1)[0][0] for p, ix in imgs.items()}
    cls_total = collections.Counter(r["label"] for r in G)
    target = {c: TEST_FRACTION * cls_total[c] for c in CLASS_NAMES}

    # ---- ชุดทดสอบ: สุ่มลำดับคน แล้วเติมทีละคนจนแต่ละคลาสถึงเป้า (ไม่เกิน 1.3 เท่าของเป้า)
    split = {}
    test_cnt = collections.Counter()
    persons = list(imgs)
    rng.shuffle(persons)
    for c in CLASS_NAMES:                                 # คลาสที่คนน้อย (เช่น heart ชาย) ให้มีอย่างน้อย 1 คนถ้ามีคนพอ
        pc = [p for p in persons if major[p] == c]
        if len(pc) >= 2 and target[c] > 0:
            p = min(pc, key=lambda q: len(imgs[q]))
            for i in imgs[p]:
                split[i] = "test"
                test_cnt[G[i]["label"]] += 1
    for p in persons:
        if split.get(imgs[p][0]) == "test":
            continue
        c = major[p]
        n = len(imgs[p])
        if test_cnt[c] >= target[c] or test_cnt[c] + n > 1.3 * target[c] + 3:
            continue
        for i in imgs[p]:
            split[i] = "test"
            test_cnt[G[i]["label"]] += 1
    test_persons = {p for p in imgs if split.get(imgs[p][0]) == "test"}

    # ---- 3 fold: คนใหญ่สุดก่อน ใส่ fold ที่คลาสนั้นยังน้อยสุด
    fold_cnt = [collections.Counter() for _ in range(K_FOLDS)]
    for p in sorted((p for p in imgs if p not in test_persons), key=lambda q: -len(imgs[q])):
        c = major[p]
        k = min(range(K_FOLDS), key=lambda j: (fold_cnt[j][c], sum(fold_cnt[j].values())))
        for i in imgs[p]:
            split[i] = f"fold{k + 1}_val"
            fold_cnt[k][G[i]["label"]] += 1

    tr_persons = {person_of[G[i]["key"]] for i, s_ in split.items() if s_ != "test"}
    assert not (test_persons & tr_persons), "person leak"
    with open(os.path.join(OUTPUT_DIR, f"splits_person_{g}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original", "crop", "label", "split", "person"])
        for i, r in enumerate(G):
            w.writerow([r["orig_csv"], "", r["label"], split[i], person_of[r["key"]]])
    per = {s_: dict(collections.Counter(G[i]["label"] for i, x in split.items() if x == s_))
           for s_ in ["test"] + [f"fold{k}_val" for k in range(1, K_FOLDS + 1)]}
    out_report["genders"][g] = {
        "images": len(G), "people": len(imgs), "test_people": len(test_persons),
        "test_people_per_class": dict(collections.Counter(major[p] for p in test_persons)),
        "per_split_per_class": per,
        "largest_people": sorted((len(v) for v in imgs.values()), reverse=True)[:5],
        "people_with_mixed_labels": sum(1 for ix in imgs.values() if len({G[i]["label"] for i in ix}) > 1),
    }
    print(f"[split] {g}: คน {len(imgs)} | test {len(test_persons)} คน", json.dumps(per, ensure_ascii=False), flush=True)

with open(os.path.join(OUTPUT_DIR, "person_split_report.json"), "w", encoding="utf-8") as f:
    json.dump(out_report, f, ensure_ascii=False, indent=2)
print("ได้ไฟล์: splits_person_female.csv, splits_person_male.csv, person_split_report.json")
print("DONE.")
