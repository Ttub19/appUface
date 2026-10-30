"""
uShape — E2E walkthrough: เปิดเบราว์เซอร์จริง ล็อกอิน แล้วเดินทุกฟีเจอร์พร้อมถ่ายภาพหน้าจอเก็บไว้

ติดตั้งครั้งแรก:
    python -m pip install playwright pillow
    python -m playwright install chromium

รัน:
    python e2e/walkthrough.py --email you@example.com --photo D:\\path\\face.jpg
    (ถามรหัสผ่านตอนรัน — หรือตั้ง E2E_PASSWORD ไว้ก่อน)

ตัวเลือก:
    --viewport desktop|mobile|both   (ค่าเริ่มต้น both)
    --headed                         เปิดหน้าต่างให้ดูตอนรัน
    --skip-hair                      ข้ามหน้าลองทรงผมทั้งหมด
    --url                            ค่าเริ่มต้น https://ttub19.github.io/appUface/

สิ่งที่สคริปต์ "ไม่ทำ" (กันเปลืองโควตา / กันข้อมูลขยะในระบบจริง):
    - ไม่กด "สร้างภาพสมจริง" และบล็อก POST /hair/realistic ไว้ทุกกรณี (ไม่เสียโควตา AI)
    - ไม่ส่งคำขอจองคิว, ไม่ส่งข้อความติดต่อแอดมิน, ไม่บันทึกผล Golden Ratio, ไม่แก้โปรไฟล์
    - วิเคราะห์รูปหน้าแล้วกด "บันทึกผลนี้" 1 ครั้งต่อขนาดจอ -> เข้าประวัติของบัญชีนั้น และนับใน Dashboard แอดมิน

ผลลัพธ์: e2e/screenshots/<วันเวลา>/<desktop|mobile>/NN_ชื่อขั้น.png + report.md
กล้องในเบราว์เซอร์ทดสอบเป็นกล้องจำลองที่ใช้รูปจาก --photo (หน้าลองแว่น / Golden Ratio จึงเห็นหน้าคนจริง)
"""
import argparse
import getpass
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    sys.exit("ติดตั้งก่อน:  python -m pip install playwright pillow  แล้ว  python -m playwright install chromium")

for _stream in (sys.stdout, sys.stderr):      # คอนโซล Windows บางตัวพิมพ์ภาษาไทย / ✓ ไม่ได้
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_URL = "https://ttub19.github.io/appUface/"
VIEWPORTS = {
    "desktop": dict(viewport={"width": 1366, "height": 860}, device_scale_factor=1, is_mobile=False, has_touch=False),
    "mobile": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True),
}
SLOW = 180_000   # Render ฟรีหลับได้ — ปลุกครั้งแรก + โหลดโมเดลอาจนานเกือบ 2 นาที


# ---------------------------------------------------------------------------
def photo_to_y4m(photo: Path, out: Path, w: int = 1280, h: int = 720) -> Path:
    """ทำรูปให้เป็น 'วิดีโอ' 1 เฟรม (Y4M) ให้ Chromium ใช้แทนกล้อง — วนเฟรมเดิมไปเรื่อย ๆ"""
    im = Image.open(photo).convert("RGB")
    im.thumbnail((w, h))
    canvas = Image.new("RGB", (w, h), (128, 128, 128))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    y, cb, cr = canvas.convert("YCbCr").split()
    cb, cr = cb.resize((w // 2, h // 2)), cr.resize((w // 2, h // 2))
    with open(out, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
        f.write(b"FRAME\n" + y.tobytes() + cb.tobytes() + cr.tobytes())
    return out


class Walk:
    def __init__(self, page, out: Path, label: str):
        self.page, self.out, self.label = page, out, label
        self.n = 0
        self.rows = []          # (step, ok, seconds, note)
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: m.type == "error" and self.errors.append(f"console: {m.text[:200]}"))

    # ---- helpers ----
    def shot(self, name: str, full: bool = True):
        self.n += 1
        path = self.out / f"{self.n:02d}_{name}.png"
        self.page.screenshot(path=str(path), full_page=full)
        return path

    def js(self, code: str, arg=None):
        return self.page.evaluate(code, arg)

    def go(self, screen: str, wait: float = 1.2):
        self.js("s => window.navigateTo(s)", screen)
        self.page.wait_for_timeout(int(wait * 1000))
        self.js("() => window.scrollTo(0, 0)")

    def visible(self, sel: str) -> bool:
        loc = self.page.locator(sel)
        return loc.count() > 0 and loc.first.is_visible()

    def wait_screen(self, *ids, timeout=SLOW):
        self.page.wait_for_function(
            "ids => ids.some(id => { const e = document.getElementById(id); return e && !e.classList.contains('hidden'); })",
            arg=list(ids), timeout=timeout)

    def step(self, name: str, fn):
        t = time.time()
        before = len(self.errors)
        try:
            note = fn() or ""
            ok = True
        except Exception as exc:     # ขั้นไหนพังก็ถ่ายรูปไว้แล้วไปขั้นต่อไป
            ok, note = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            try:
                self.shot(f"ERROR_{name}")
            except Exception:
                pass
        new_err = self.errors[before:]
        if new_err:
            note = (note + " | " if note else "") + f"JS error {len(new_err)}: " + new_err[0][:150]
        dt = time.time() - t
        self.rows.append((name, ok, dt, note))
        print(f"  [{self.label}] {'✓' if ok else '✗'} {name} ({dt:.1f}s) {note}")
        return ok


# ---------------------------------------------------------------------------
def walkthrough(w: Walk, cfg):
    p = w.page

    def login_screen():
        p.goto(cfg.url, wait_until="domcontentloaded", timeout=60_000)
        w.wait_screen("screen-login", timeout=60_000)
        p.wait_for_timeout(1500)
        w.shot("login")

    def login():
        p.fill("#auth-email", cfg.email)
        p.fill("#auth-password", cfg.password)
        p.click("#auth-submit-btn")
        try:
            w.wait_screen("screen-main", "screen-admin")
        except PWTimeout:
            err = p.locator("#auth-error").inner_text() if w.visible("#auth-error") else "หมดเวลารอ"
            raise RuntimeError(f"ล็อกอินไม่สำเร็จ: {err}")
        p.wait_for_timeout(1500)
        role = w.js("() => { const n = document.getElementById('nav-admin'); return n && !n.classList.contains('hidden') ? 'admin' : 'member/barber'; }")
        return f"role={role}"

    def home():
        w.go("screen-main")
        w.shot("home")

    def scan():
        w.go("screen-main")
        p.set_input_files("#file-input", str(cfg.photo))
        p.wait_for_timeout(1500)
        w.shot("scan_photo_selected", full=False)
        p.locator('#screen-main [onclick="runAI()"]').first.click()
        w.wait_screen("screen-result")
        p.wait_for_timeout(2500)
        res = w.js("() => [document.getElementById('res-shape')?.textContent, document.getElementById('res-confidence')?.textContent]")
        w.shot("scan_result")
        return f"ผล: {(res[0] or '').strip()} ({(res[1] or '').strip()})"

    def save_scan():
        if not w.visible("#screen-result"):
            raise RuntimeError("ไม่อยู่หน้าผลวิเคราะห์ (ขั้นวิเคราะห์ไม่สำเร็จ)")
        w.js("() => document.getElementById('btn-save').scrollIntoView({block: 'center'})")
        p.click("#btn-save")
        w.wait_screen("screen-history", timeout=60_000)   # บันทึกสำเร็จแล้วแอปพาไปหน้าประวัติเอง
        p.wait_for_timeout(2000)
        w.shot("scan_saved_history")

    def guide():
        w.go("screen-guide")
        w.shot("guide")
        w.js("() => window.openGuideShape('oval')")
        p.wait_for_timeout(1000)
        w.shot("guide_oval_detail", full=False)
        w.js("() => window.closeGuideShape && window.closeGuideShape()")

    def golden():
        w.go("screen-golden")
        p.click("#golden-open-btn")
        p.wait_for_timeout(8000)          # เปิดกล้องจำลอง + โหลด MediaPipe + วัดค่า
        w.shot("golden_ratio")
        w.js("() => window.stopGoldenSession()")

    def glasses():
        w.go("screen-glasses")
        items = p.locator("#glasses-selector button")
        if items.count() > 2:
            items.nth(2).click()
        p.click("#glasses-open-btn")
        p.wait_for_timeout(8000)
        w.shot("glasses_live", full=False)
        if w.visible("#glasses-controls"):
            p.locator('#glasses-controls [onclick="captureGlassesPhoto()"]').click()
            p.wait_for_timeout(1500)
            w.shot("glasses_photo", full=False)
        w.js("() => window.stopGlassesSession()")

    def hair():
        w.go("screen-hairstyle")
        w.shot("hair_start")
        p.set_input_files("#hair-file", str(cfg.photo))
        p.wait_for_function("() => !document.getElementById('hair-stage').classList.contains('hidden')", timeout=90_000)
        p.wait_for_timeout(2500)
        items = p.locator("#hair-selector button")
        if items.count() > 1:
            items.nth(1).click()
            p.wait_for_timeout(2500)
        stage = "() => document.getElementById('hair-stage').scrollIntoView({block: 'center'})"
        w.js(stage)
        w.shot("hair_tryon", full=False)
        colors = p.locator("#hair-color-filter button")
        if colors.count() > 2:
            colors.nth(2).click()
            p.wait_for_timeout(1500)
            w.js(stage)
            w.shot("hair_color", full=False)
        # ห้ามกด #hair-ai-btn ("สร้างภาพสมจริง") — และถูกบล็อกที่ network อีกชั้น
        w.js("() => window.stopHairSession && window.stopHairSession()")

    def booking():
        w.go("screen-booking", wait=3)
        w.shot("booking_discover")
        open_btn = p.locator('#screen-booking button:has-text("ดูร้านและจอง")')
        if open_btn.count():
            open_btn.first.click()
            p.wait_for_timeout(3000)
            w.shot("booking_shop_detail")          # ดูเฉยๆ ไม่กดส่งคำขอจอง
        w.js("() => window.showBookingTab('mine')")
        p.wait_for_timeout(2500)
        w.shot("booking_mine")
        if w.visible("#booking-manage-tab"):
            w.js("() => window.showBookingTab('manage')")
            p.wait_for_timeout(3000)
            w.shot("booking_manage")

    def simple(screen, name, wait=2.0):
        def run():
            w.go(screen, wait=wait)
            w.shot(name)
        return run

    def admin():
        if w.js("() => { const n = document.getElementById('nav-admin'); return !n || n.classList.contains('hidden'); }"):
            return "ข้าม (บัญชีนี้ไม่ใช่ admin)"
        w.go("screen-admin", wait=3)
        w.shot("admin_dashboard")

    def mobile_menu():
        w.go("screen-main")
        w.js("() => window.openMobileMenu()")
        p.wait_for_timeout(800)
        w.shot("mobile_menu", full=False)
        w.js("() => window.closeMobileMenu()")

    w.step("login_screen", login_screen)
    if not w.step("login", login):
        return
    w.step("home", home)
    if w.step("scan_and_result", scan):
        w.step("save_scan", save_scan)
    w.step("guide", guide)
    w.step("golden_ratio", golden)
    w.step("glasses", glasses)
    if not cfg.skip_hair:
        w.step("hair_2d_only", hair)
    w.step("booking", booking)
    w.step("favorites", simple("screen-favorites", "favorites"))
    w.step("history", simple("screen-history", "history"))
    w.step("profile", simple("screen-profile", "profile"))
    w.step("contact_admin", simple("screen-feedback", "contact_admin"))
    w.step("admin", admin)
    if w.label == "mobile":
        w.step("mobile_menu", mobile_menu)


# ---------------------------------------------------------------------------
def new_context(browser, label):
    return browser.new_context(**VIEWPORTS[label], locale="th-TH", timezone_id="Asia/Bangkok",
                               permissions=["camera", "geolocation"],
                               geolocation={"latitude": 15.2448, "longitude": 104.8473})


def run(cfg, context_hook=None):
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    root = Path(cfg.out) / stamp
    root.mkdir(parents=True, exist_ok=True)
    cam = photo_to_y4m(cfg.photo, root / "_camera.y4m")
    labels = ["desktop", "mobile"] if cfg.viewport == "both" else [cfg.viewport]
    blocked = []
    report = [f"# uShape E2E walkthrough — {datetime.now():%Y-%m-%d %H:%M}", "",
              f"- URL: {cfg.url}", f"- บัญชี: {cfg.email}", f"- รูปทดสอบ: {cfg.photo.name}", ""]
    all_ok = True
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not cfg.headed, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={cam}"])
        for label in labels:
            out = root / label
            out.mkdir(exist_ok=True)
            ctx = new_context(browser, label)

            def guard(route, request):
                if request.method == "POST" and request.url.rstrip("/").endswith("/hair/realistic"):
                    blocked.append(request.url)
                    return route.abort()
                return route.fallback()
            ctx.route("**/hair/realistic*", guard)
            if context_hook:
                context_hook(ctx, label)
            page = ctx.new_page()
            print(f"\n== {label} ==")
            w = Walk(page, out, label)
            walkthrough(w, cfg)
            ctx.close()

            passed = sum(ok for _, ok, _, _ in w.rows)
            all_ok &= passed == len(w.rows)
            report += [f"## {label}: ผ่าน {passed}/{len(w.rows)} ขั้น", "",
                       "| ขั้น | ผล | เวลา (วินาที) | หมายเหตุ |", "|---|---|---|---|"]
            report += [f"| {n} | {'ผ่าน' if ok else '**ไม่ผ่าน**'} | {dt:.1f} | {note.replace('|', '/')} |"
                       for n, ok, dt, note in w.rows]
            report.append("")
        browser.close()
    report.append(f"- เรียก AI ทรงผมสมจริง: {'ไม่มี (0 ครั้ง)' if not blocked else f'ถูกบล็อก {len(blocked)} ครั้ง'}")
    (root / "report.md").write_text("\n".join(report), encoding="utf-8")
    (root / "_camera.y4m").unlink(missing_ok=True)
    print(f"\nภาพหน้าจอ + report.md อยู่ที่: {root.resolve()}")
    return all_ok, root


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="uShape E2E walkthrough with screenshots")
    ap.add_argument("--url", default=os.environ.get("E2E_URL", DEFAULT_URL))
    ap.add_argument("--email", default=os.environ.get("E2E_EMAIL"))
    ap.add_argument("--photo", default=os.environ.get("E2E_PHOTO"), help="รูปหน้าตรง (jpg/png)")
    ap.add_argument("--viewport", choices=["desktop", "mobile", "both"], default="both")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "screenshots"))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--skip-hair", action="store_true")
    cfg = ap.parse_args(argv)
    if not cfg.email:
        cfg.email = input("อีเมลที่ใช้ล็อกอิน: ").strip()
    if not cfg.photo:
        cfg.photo = input("path รูปหน้าตรง: ").strip().strip('"')
    cfg.photo = Path(cfg.photo)
    if cfg.photo.is_dir():          # ให้ชี้เป็นโฟลเดอร์ได้ — ใช้รูปแรกในโฟลเดอร์
        imgs = sorted(f for f in cfg.photo.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        if not imgs:
            sys.exit(f"ในโฟลเดอร์นี้ไม่มีไฟล์รูป: {cfg.photo}")
        cfg.photo = imgs[0]
        print(f"ใช้รูป: {cfg.photo}")
    if not cfg.photo.is_file():
        sys.exit(f"ไม่พบไฟล์รูป: {cfg.photo}")
    cfg.password = os.environ.get("E2E_PASSWORD") or getpass.getpass("รหัสผ่าน (ไม่แสดงบนจอ): ")
    return cfg


if __name__ == "__main__":
    ok, _ = run(parse_args())
    sys.exit(0 if ok else 1)
