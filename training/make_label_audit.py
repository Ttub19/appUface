# =============================================================================
# uShape — ตรวจคุณภาพป้ายของ "ชุดทดสอบ" แบบ blind (คนติดป้ายใหม่โดยไม่เห็นป้ายเดิม)
#
# Kaggle: Accelerator = None (CPU), Internet ไม่ต้อง
#   Add Input: faceshape1 (รูป) + facetrain-v1-output (splits_female.csv / splits_male.csv ของ v3 = ชุดทดสอบเดียวกับ v4/v5)
# วางทั้งไฟล์ในเซลล์เดียวแล้ว Save Version (~2-3 นาที)
#
# ได้ไฟล์:
#   label_audit.html   -> ส่งให้คนติดป้าย (เปิดในเบราว์เซอร์ได้เลย ไม่ต้องต่อเน็ต) ไม่มีป้ายเดิมอยู่ในไฟล์
#   answer_key.csv     -> ป้ายเดิมของชุดข้อมูล ห้ามให้คนติดป้ายเห็น (ส่งให้ผู้วิเคราะห์เท่านั้น)
# =============================================================================
import base64
import csv
import hashlib
import io
import json
import os
import random

from PIL import Image, ImageOps

INPUT_ROOT = "/kaggle/input"
OUTPUT_DIR = "/kaggle/working"
SEED = 2026
CLASS_NAMES = ["heart", "oblong", "oval", "round", "square"]
FEMALE_PER_CLASS = 30           # หญิง 5 x 30 = 150 รูป
MALE_TOTAL = 50                 # ชาย 50 รูป (heart ชายในชุดทดสอบมีน้อย -> เอาทั้งหมด แล้วเฉลี่ยที่เหลือให้คลาสอื่น)
THUMB = 360                     # ด้านยาวสุดของรูปใน HTML (px)


def walk(base=INPUT_ROOT):
    yield from os.walk(base)


def find_data_root():
    for root, dirs, _ in walk():
        if "train1_cropped" in dirs and "testing_set" in dirs:
            return root
    raise SystemExit("ไม่พบ dataset faceshape1 (โฟลเดอร์ที่มี train1_cropped + testing_set) — Add Input ก่อน")


def find_splits(gender):
    names = (f"splits_{gender}.csv", f"splits_{gender}.csv.txt", f"splits_{gender}.txt")
    for root, _, files in walk():
        for n in names:
            if n in files:
                path = os.path.join(root, n)
                with open(path, encoding="utf-8") as f:
                    if "male_extra" in f.read():       # รอบทดลอง male_extra ไม่ใช่ split หลัก
                        continue
                return path
    raise SystemExit(f"ไม่พบ splits_{gender}.csv — Add Input facetrain-v1-output")


DATA_ROOT = find_data_root()


def local_original(path):
    if os.path.exists(path):
        return path
    key = "datasetformodelai" + os.sep
    norm = path.replace("\\", os.sep).replace("/", os.sep)
    if key in norm:
        cand = os.path.join(DATA_ROOT, norm.split(key, 1)[1])
        if os.path.exists(cand):
            return cand
    return None


def test_rows(gender):
    rows, missing = [], 0
    with open(find_splits(gender), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            p = local_original(r["original"])
            if not p:
                missing += 1
                continue
            rows.append((p, r["label"]))
    print(f"{gender}: ชุดทดสอบ {len(rows)} รูป" + (f" (หาไฟล์ไม่เจอ {missing})" if missing else ""))
    return rows


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def sample(rows, quota, rng, seen):
    by = {c: [] for c in CLASS_NAMES}
    for p, l in rows:
        by[l].append(p)
    out = []
    for c in CLASS_NAMES:
        pool = by[c][:]
        rng.shuffle(pool)
        taken = 0
        for p in pool:
            if taken >= quota[c]:
                break
            h = md5(p)
            if h in seen:                               # รูปซ้ำ (เนื้อไฟล์เดียวกัน) ไม่เอา
                continue
            seen.add(h)
            out.append((p, c))
            taken += 1
    return out


rng = random.Random(SEED)
seen = set()
female = sample(test_rows("female"), {c: FEMALE_PER_CLASS for c in CLASS_NAMES}, rng, seen)
male_rows = test_rows("male")
n_heart = min(sum(1 for _, l in male_rows if l == "heart"), MALE_TOTAL // 5)
rest = (MALE_TOTAL - n_heart) // 4
male_quota = {c: (n_heart if c == "heart" else rest) for c in CLASS_NAMES}
male_quota["square"] += MALE_TOTAL - n_heart - 4 * rest
male = sample(male_rows, male_quota, rng, seen)

items = [("female", p, c) for p, c in female] + [("male", p, c) for p, c in male]
rng.shuffle(items)
print(f"สุ่มได้ หญิง {len(female)} + ชาย {len(male)} = {len(items)} รูป")


def thumb_b64(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((THUMB, THUMB))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


key_rows, page_items = [], []
for i, (g, p, c) in enumerate(items, 1):
    iid = f"img{i:03d}"
    key_rows.append({"id": iid, "gender": g, "dataset_label": c, "original": os.path.relpath(p, DATA_ROOT)})
    page_items.append({"id": iid, "img": thumb_b64(p)})

with open(os.path.join(OUTPUT_DIR, "answer_key.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "gender", "dataset_label", "original"])
    w.writeheader()
    w.writerows(key_rows)

HTML = r"""<!doctype html>
<html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>uShape label audit</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--fg:#1f2328;--mute:#656d76;--line:#d8dee4;--acc:#0b6bcb;--sel:#e7f0fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Segoe UI","Noto Sans Thai",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
.row{display:flex;gap:16px;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
#start label{display:block;margin:8px 0 4px}#start input{font:inherit;padding:8px;width:260px;border:1px solid var(--line);border-radius:6px}
button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--card);border-radius:8px;padding:10px 12px}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
#imgbox{flex:1 1 380px;display:flex;align-items:center;justify-content:center;min-height:380px}
#imgbox img{max-width:100%;max-height:420px;border-radius:6px}
#side{flex:1 1 320px;min-width:0}
.choices{display:grid;gap:8px}.choices button{text-align:left;display:flex;gap:10px;align-items:baseline}
.choices button.on{background:var(--sel);border-color:var(--acc)}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:#fafbfc}
.desc{color:var(--mute);font-size:13px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0}.bar>i{display:block;height:100%;background:var(--acc)}
.nav{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
#unsure.on{background:#fff4d6;border-color:#d4a72c}
.hidden{display:none!important}
</style></head><body><main>
<h2 style="margin:4px 0 12px">ติดป้ายทรงหน้า (ชุดตรวจสอบ)</h2>
<section id="start" class="card">
<p>ดูรูปแล้วเลือกทรงหน้าที่ตรงที่สุด <b>ไม่มีคำตอบที่ถูกให้ดู</b> ตอบตามที่เห็น ถ้าก้ำกึ่งให้เลือกที่ใกล้สุดแล้วกด "ไม่แน่ใจ" ได้</p>
<p class="desc">คีย์ลัด: <kbd>1</kbd>-<kbd>5</kbd> เลือกทรง (ไปรูปถัดไปเอง) · <kbd>U</kbd> ไม่แน่ใจ · <kbd>←</kbd> ย้อนกลับ · <kbd>→</kbd> ถัดไป · บันทึกอัตโนมัติในเบราว์เซอร์นี้</p>
<label for="rater">ชื่อผู้ติดป้าย</label><input id="rater" placeholder="เช่น ต้น">
<div class="nav"><button class="primary" id="go">เริ่ม / ทำต่อ</button></div>
</section>
<section id="work" class="hidden">
<div class="bar"><i id="prog"></i></div><div class="desc" id="count"></div>
<div class="row" style="margin-top:8px">
<div id="imgbox" class="card"><img id="pic" alt="face"></div>
<div id="side" class="card"><div class="choices" id="choices"></div>
<div class="nav"><button id="unsure">ไม่แน่ใจ (U)</button><button id="prev">← ก่อนหน้า</button><button id="next">ถัดไป →</button></div>
<div class="nav"><button class="primary" id="export">ดาวน์โหลดผล (.json)</button></div>
<p class="desc" id="done"></p></div></div>
</section>
</main>
<script>
const ITEMS = __ITEMS__;
const SHAPES = [
 ["heart","หัวใจ (Heart)","หน้าผากกว้าง โหนกแก้มกว้าง กรามแคบ คางเรียวแหลม"],
 ["oblong","ยาว (Oblong)","หน้ายาวกว่ากว้างชัดเจน หน้าผาก แก้ม กรามกว้างใกล้กัน"],
 ["oval","รูปไข่ (Oval)","ยาวกว่ากว้างเล็กน้อย หน้าผากกว้างกว่ากรามเล็กน้อย คางมน"],
 ["round","กลม (Round)","กว้างใกล้เคียงยาว แก้มอิ่ม กรามและคางมน"],
 ["square","เหลี่ยม (Square)","กว้างใกล้เคียงยาว กรามกว้างเป็นเหลี่ยมชัด"]];
let rater="", idx=0, ans={};
const $=id=>document.getElementById(id);
function key(){return "ushape_audit_"+rater}
function save(){try{localStorage.setItem(key(),JSON.stringify({idx,ans}))}catch(e){}}
function load(){try{const s=JSON.parse(localStorage.getItem(key())||"null");if(s){idx=s.idx||0;ans=s.ans||{}}}catch(e){}}
SHAPES.forEach((s,i)=>{const b=document.createElement("button");b.dataset.k=s[0];
 b.innerHTML=`<kbd>${i+1}</kbd><span><b>${s[1]}</b><br><span class="desc">${s[2]}</span></span>`;b.onclick=()=>pick(s[0]);$("choices").appendChild(b)});
function show(){idx=Math.max(0,Math.min(ITEMS.length-1,idx));const it=ITEMS[idx],a=ans[it.id]||{};
 $("pic").src="data:image/jpeg;base64,"+it.img;
 document.querySelectorAll("#choices button").forEach(b=>b.classList.toggle("on",b.dataset.k===a.label));
 $("unsure").classList.toggle("on",!!a.unsure);
 const n=Object.values(ans).filter(x=>x.label).length;
 $("count").textContent=`รูปที่ ${idx+1} / ${ITEMS.length} · ติดป้ายแล้ว ${n}`;
 $("prog").style.width=(100*n/ITEMS.length)+"%";
 $("done").textContent=n===ITEMS.length?"ครบทุกรูปแล้ว กดดาวน์โหลดผลแล้วส่งไฟล์ .json กลับมา":"";save()}
function pick(k){const id=ITEMS[idx].id;ans[id]={...(ans[id]||{}),label:k};
 const first=ITEMS.findIndex((it,j)=>j>idx&&!(ans[it.id]&&ans[it.id].label));
 idx=first>=0?first:Math.min(idx+1,ITEMS.length-1);show()}
function toggleUnsure(){const id=ITEMS[idx].id;ans[id]={...(ans[id]||{}),unsure:!(ans[id]&&ans[id].unsure)};show()}
$("unsure").onclick=toggleUnsure;$("prev").onclick=()=>{idx--;show()};$("next").onclick=()=>{idx++;show()};
$("go").onclick=()=>{rater=$("rater").value.trim()||"rater";load();$("start").classList.add("hidden");$("work").classList.remove("hidden");show()};
$("rater").addEventListener("keydown",e=>{if(e.key==="Enter")$("go").click()});
document.addEventListener("keydown",e=>{if($("work").classList.contains("hidden")||e.target.tagName==="INPUT")return;
 if(e.key>="1"&&e.key<="5")pick(SHAPES[+e.key-1][0]);else if(e.key==="u"||e.key==="U")toggleUnsure();
 else if(e.key==="ArrowLeft"){idx--;show()}else if(e.key==="ArrowRight"){idx++;show()}});
$("export").onclick=()=>{const out={rater,created:new Date().toISOString(),n:ITEMS.length,
  answers:ITEMS.map(it=>({id:it.id,label:(ans[it.id]||{}).label||null,unsure:!!(ans[it.id]||{}).unsure}))};
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
 a.download=`labels_${rater}.json`;document.body.appendChild(a);a.click();a.remove()};
</script></body></html>"""

with open(os.path.join(OUTPUT_DIR, "label_audit.html"), "w", encoding="utf-8") as f:
    f.write(HTML.replace("__ITEMS__", json.dumps(page_items)))
size = os.path.getsize(os.path.join(OUTPUT_DIR, "label_audit.html")) / 1e6
print(f"label_audit.html {size:.1f} MB ({len(page_items)} รูป) + answer_key.csv")
print("ส่ง label_audit.html ให้คนติดป้าย, เก็บ answer_key.csv ไว้ (อย่าให้คนติดป้ายเห็น)")
print("DONE.")
