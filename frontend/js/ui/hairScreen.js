import { showToast } from "./toast.js";
import { appState } from "../state.js";
import { faceData } from "../data.js";
import { navigateTo } from "./navigation.js";
import { HairPhotoSession } from "../hairTryOn/hairTryOnEngine.js";
import {
    HAIR_CATEGORIES, HAIR_STYLES, filterHairStyles, genderCategory, getHairStyle, loadHairImage, preloadHairImages, recommendHairStyles,
} from "../hairTryOn/hairAssets.js";
import { DEFAULT_HAIR_COLOR, HAIR_COLORS, getHairColor } from "../hairTryOn/hairLook.js";

// v2: auto-fit recalibrated from a live screenshot, so a size saved against the old fit is discarded
const SIZE_STORAGE_KEY = "hairSizePercent.v2";
const COLOR_STORAGE_KEY = "hairColor.v1";
const SIZE_MIN = 75, SIZE_MAX = 125;
const DESKTOP_MIN_WIDTH = 1024;

let currentHairStyle = HAIR_STYLES[0].id;
let shapeFilter = "all";
let categoryFilter = "all";
let autoFilteredFor = null;
let eraseEnabled = true;
let hairSession = null;      // HairPhotoSession: one photo, re-rendered on every change
let cameraStream = null;     // only while taking the photo
let hairOffsetPercent = 0;   // manual up/down, % of face width (+ = lower)
let hairSizePercent = loadHairSize();
let hairColorId = loadHairColor();

const shapeLabel = (id) => faceData.find((f) => f.id === id)?.name || id;
const shortShape = (id) => shapeLabel(id).replace(/\s*\(.*\)/, "");
const analysedShape = () => appState.resultObj?.id || null;
/** Gender the user asked recommendations for (set by the analysis; else read from the home screen). */
const genderPref = () => appState.genderPref || document.getElementById("gender-pref")?.value || null;
const isRecommended = (style, shape = analysedShape()) => Boolean(shape) && style.faceShapes.includes(shape);

function loadHairColor() {
    try {
        const id = localStorage.getItem(COLOR_STORAGE_KEY);
        return HAIR_COLORS.some((c) => c.id === id) ? id : DEFAULT_HAIR_COLOR;
    } catch { return DEFAULT_HAIR_COLOR; }
}

function clampSize(value) {
    const n = Math.round(Number(value));
    return Number.isFinite(n) ? Math.min(SIZE_MAX, Math.max(SIZE_MIN, n)) : 100;
}

function loadHairSize() {
    try { return clampSize(localStorage.getItem(SIZE_STORAGE_KEY) ?? 100); }
    catch { return 100; }
}

function syncHairSizeUI() {
    const input = document.getElementById("hair-size");
    const output = document.getElementById("hair-size-value");
    if (input) input.value = String(hairSizePercent);
    if (output) output.textContent = hairSizePercent === 100 ? "พอดีหัว" : `${hairSizePercent}%`;
}

/** Manual fine-tune on top of the automatic head-width fit (75–125%). */
export function setHairSize(value) {
    hairSizePercent = clampSize(value);
    try { localStorage.setItem(SIZE_STORAGE_KEY, String(hairSizePercent)); } catch { /* storage unavailable */ }
    hairSession?.setSizeScale(hairSizePercent / 100);
    syncHairSizeUI();
}

/** Manual up/down for the still photo (−15 … +15 % of the face width). */
export function setHairOffset(value) {
    const n = Math.round(Number(value));
    hairOffsetPercent = Number.isFinite(n) ? Math.min(15, Math.max(-15, n)) : 0;
    hairSession?.setOffsetY(hairOffsetPercent / 100);
    const input = document.getElementById("hair-offset");
    const output = document.getElementById("hair-offset-value");
    if (input) input.value = String(hairOffsetPercent);
    if (output) output.textContent = hairOffsetPercent === 0 ? "อัตโนมัติ" : `${hairOffsetPercent > 0 ? "ลง" : "ขึ้น"} ${Math.abs(hairOffsetPercent)}%`;
}

/** Called by navigation when the screen opens: pre-filter to the user's analysed face shape and gender. */
export function initHairScreen() {
    const analysed = analysedShape();
    if (analysed && autoFilteredFor !== analysed) {
        autoFilteredFor = analysed;
        shapeFilter = analysed;
        categoryFilter = genderCategory(genderPref());
        const best = recommendHairStyles({ shape: analysed, gender: genderPref(), limit: 1 })[0];
        if (best && !isRecommended(getHairStyle(currentHairStyle), analysed)) currentHairStyle = best.id;
    }
    syncHairSizeUI();
    setHairOffset(hairOffsetPercent);
    setStageState(hairSession?.hasPhoto ? "photo" : "empty");
    renderShapeFilter();
    renderCategoryFilter();
    renderColorFilter();
    renderHairSelector();
    renderStyleInfo();
}

/**
 * "ลองทรงนี้" from the result screen: open the hair screen with this style
 * selected and put it straight onto the photo that was just analysed.
 */
export function tryHairStyle(id) {
    if (appState.currentRole === "guest") {
        showToast("เข้าสู่ระบบหรือสมัครสมาชิกฟรี เพื่อลองทรงผมผ่านกล้อง", "info");
        return;
    }
    navigateTo("screen-hairstyle");
    const style = getHairStyle(id);
    currentHairStyle = style.id;
    if (!filterHairStyles({ shape: shapeFilter, category: categoryFilter }).some((s) => s.id === style.id)) {
        categoryFilter = "all";
        if (!style.faceShapes.includes(shapeFilter)) shapeFilter = "all";
        renderShapeFilter();
        renderCategoryFilter();
    }
    renderHairSelector();
    renderStyleInfo();
    document.querySelector(`.hair-btn[data-style="${style.id}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
    if (hairSession?.hasPhoto) selectHairStyle(style.id);
    else if (appState.selectedImg) useScanPhotoForHair();
}

/** Result screen: the best hairstyles for the analysed face shape, each with a try-on button. */
export function renderResultHairRecs(shapeId, gender) {
    const root = document.getElementById("res-hair-recs");
    if (!root) return;
    const styles = recommendHairStyles({ shape: shapeId, gender, limit: 4 });
    root.replaceChildren(...styles.map((style) => {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "hair-rec";
        card.addEventListener("click", () => tryHairStyle(style.id));
        const img = document.createElement("img");
        img.src = style.src; img.alt = ""; img.loading = "lazy";
        const name = document.createElement("strong");
        name.textContent = style.name;
        const why = document.createElement("small");
        why.textContent = style.desc;
        const cta = document.createElement("span");
        cta.className = "hair-rec-cta";
        cta.innerHTML = appState.currentRole === "guest"
            ? '<i class="fa-solid fa-lock" aria-hidden="true"></i> เข้าสู่ระบบเพื่อลอง'
            : 'ลองทรงนี้ <i class="fa-solid fa-arrow-right" aria-hidden="true"></i>';
        card.append(img, name, why, cta);
        return card;
    }));
}

export function setHairShapeFilter(shape) {
    shapeFilter = shape;
    renderShapeFilter();
    renderCategoryFilter(); // counts depend on the face-shape filter
    renderHairSelector();
}

export function setHairCategory(category) {
    categoryFilter = HAIR_CATEGORIES.some((c) => c.id === category) ? category : "all";
    renderCategoryFilter();
    renderHairSelector();
}

function chip(label, active, onClick, count) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `chip chip-filter${active ? " is-active" : ""}`;
    btn.setAttribute("aria-pressed", String(active));
    btn.append(label);
    if (count !== undefined) {
        const n = document.createElement("span");
        n.className = "chip-count";
        n.textContent = String(count);
        btn.append(n);
    }
    btn.addEventListener("click", onClick);
    return btn;
}

function renderShapeFilter() {
    const root = document.getElementById("hair-shape-filter");
    if (!root) return;
    root.replaceChildren(...["all", ...faceData.map((f) => f.id)].map((id) =>
        chip(id === "all" ? "ทุกรูปหน้า" : shortShape(id), id === shapeFilter, () => setHairShapeFilter(id))));

    const note = document.getElementById("hair-recommend-note");
    if (!note) return;
    const analysed = analysedShape();
    note.classList.remove("hidden");
    if (analysed) {
        note.textContent = `ผลวิเคราะห์ล่าสุดของคุณ: ${shapeLabel(analysed)} — ทรงที่มีป้าย "แนะนำ" อยู่ด้านบนสุด`;
    } else {
        const text = document.createElement("span");
        text.textContent = "ยังไม่รู้รูปหน้าของคุณ วิเคราะห์ก่อนเพื่อดูว่าทรงไหนเหมาะ";
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "btn btn-secondary btn-sm";
        btn.textContent = "วิเคราะห์รูปหน้า";
        btn.addEventListener("click", () => navigateTo("screen-main"));
        note.replaceChildren(text, btn);
    }
}

/** Hair colour chips (recolours the hairpiece; "เหมือนผมฉัน" samples your own hair from the camera). */
function renderColorFilter() {
    const root = document.getElementById("hair-color-filter");
    if (!root) return;
    root.replaceChildren(...HAIR_COLORS.map((c) => {
        const btn = chip("", c.id === hairColorId, () => setHairColor(c.id));
        const dot = document.createElement("span");
        dot.className = `color-dot${c.id === "mine" ? " is-auto" : ""}`;
        const rgb = c.rgb || c.swatch;
        if (rgb) dot.style.background = `rgb(${rgb.join(",")})`;
        dot.setAttribute("aria-hidden", "true");
        btn.prepend(dot);
        btn.append(c.label);
        return btn;
    }));
}

export function setHairColor(id) {
    hairColorId = getHairColor(id).id;
    try { localStorage.setItem(COLOR_STORAGE_KEY, hairColorId); } catch { /* storage unavailable */ }
    hairSession?.setHairColor(hairColorId);
    renderColorFilter();
}

/** What's selected and whether it suits the analysed face shape. */
function renderStyleInfo() {
    const root = document.getElementById("hair-style-info");
    if (!root) return;
    const style = getHairStyle(currentHairStyle);
    const analysed = analysedShape();
    const name = document.createElement("strong");
    name.textContent = style.name;
    const desc = document.createElement("p");
    desc.textContent = style.desc;
    const fit = document.createElement("p");
    fit.className = "hair-fit";
    if (!analysed) {
        fit.classList.add("is-unknown");
        fit.textContent = `เหมาะกับหน้า${style.faceShapes.map(shortShape).map((n) => n.replace(/^หน้า/, "")).join(" / ")}`;
    } else if (isRecommended(style, analysed)) {
        fit.classList.add("is-good");
        fit.innerHTML = '<i class="fa-solid fa-circle-check" aria-hidden="true"></i>';
        fit.append(` แนะนำสำหรับ${shortShape(analysed)}ของคุณ`);
    } else {
        fit.classList.add("is-meh");
        fit.innerHTML = '<i class="fa-solid fa-circle-info" aria-hidden="true"></i>';
        fit.append(` ไม่ใช่ทรงที่แนะนำสำหรับ${shortShape(analysed)} — ลองเทียบกับทรงที่มีป้าย "แนะนำ"`);
    }
    root.replaceChildren(name, desc, fit);
}

function renderCategoryFilter() {
    const root = document.getElementById("hair-category-filter");
    if (!root) return;
    root.replaceChildren(...HAIR_CATEGORIES.map(({ id, label }) =>
        chip(label, id === categoryFilter, () => setHairCategory(id), filterHairStyles({ shape: shapeFilter, category: id }).length)));
}

function renderHairSelector() {
    const root = document.getElementById("hair-selector");
    if (!root) return;
    const analysed = analysedShape();
    // recommended for the analysed face first (strongest match first), then the rest in catalogue order
    const styles = filterHairStyles({ shape: shapeFilter, category: categoryFilter })
        .map((s, order) => ({ s, order, rank: isRecommended(s, analysed) ? s.faceShapes.indexOf(analysed) : 99 }))
        .sort((a, b) => a.rank - b.rank || a.order - b.order)
        .map(({ s }) => s);
    root.replaceChildren(...styles.map((style) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.style = style.id;
        btn.className = "hair-btn option-card is-vertical";
        btn.addEventListener("click", () => selectHairStyle(style.id));

        const thumb = document.createElement("img");
        thumb.src = style.src;
        thumb.alt = "";
        thumb.loading = "lazy";
        const title = document.createElement("strong");
        title.textContent = style.name;
        const desc = document.createElement("small");
        desc.textContent = style.desc;
        btn.append(thumb, title, desc);

        if (analysed && style.faceShapes.includes(analysed)) {
            const badge = document.createElement("span");
            badge.className = "option-badge";
            badge.textContent = "แนะนำ";
            btn.append(badge);
        }
        return btn;
    }));
    if (!styles.length) {
        const empty = document.createElement("p");
        empty.className = "tryon-empty";
        empty.textContent = "ยังไม่มีทรงผมที่ตรงกับทั้งรูปหน้าและหมวดนี้ ลองเลือก \"ทั้งหมด\"";
        root.append(empty);
    }
    root.scrollTop = 0;
    root.scrollLeft = 0;
    highlightSelected();
}

function highlightSelected() {
    document.querySelectorAll(".hair-btn").forEach((btn) => {
        const active = btn.dataset.style === currentHairStyle;
        btn.classList.toggle("is-selected", active);
        btn.setAttribute("aria-pressed", String(active));
    });
}

export async function selectHairStyle(id) {
    const style = getHairStyle(id);
    currentHairStyle = style.id;
    highlightSelected();
    renderStyleInfo();
    if (!hairSession) return;
    try {
        const image = await loadHairImage(style);
        if (hairSession && currentHairStyle === style.id) hairSession.setStyle(style, image);
    } catch (err) {
        showToast(err.message, "error");
    }
}

export function setHairErase(enabled) {
    eraseEnabled = Boolean(enabled);
    hairSession?.setEraseEnabled(eraseEnabled);
}

// ---------------------------------------------------------------------------
// Still-photo flow: choose a photo (analysed photo / upload / camera snapshot)
// -> AI runs once -> every style, colour and size change re-renders instantly.
// ---------------------------------------------------------------------------
function setStageState(state) {
    // state: "empty" | "camera" | "photo"
    const show = (id, on) => document.getElementById(id)?.classList.toggle("hidden", !on);
    show("hair-placeholder", state === "empty");
    show("hair-source-actions", state === "empty");
    show("hair-video", state === "camera");
    show("hair-camera-controls", state === "camera");
    show("hair-stage", state === "photo");
    show("hair-controls", state === "photo");
    const scan = document.getElementById("hair-use-scan");
    if (scan) scan.classList.toggle("hidden", !appState.selectedImg);
}

function setLoading(text) {
    const loading = document.getElementById("hair-loading");
    if (!loading) return;
    loading.classList.toggle("hidden", !text);
    if (text) document.getElementById("hair-loading-text").textContent = text;
}

function imageFromSrc(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("อ่านไฟล์รูปไม่ได้ ลองใช้รูป JPG หรือ PNG"));
        img.src = src;
    });
}

/** Runs the AI on a photo and shows the selected hairstyle on it. */
async function usePhoto(source, { mirror = false } = {}) {
    setLoading("กำลังวิเคราะห์ใบหน้าและเส้นผมในรูป...");
    try {
        hairSession ??= new HairPhotoSession({ canvas: document.getElementById("hair-stage") });
        const style = getHairStyle(currentHairStyle);
        const [, image] = await Promise.all([hairSession.load(), loadHairImage(style)]);
        hairSession.setEraseEnabled(eraseEnabled);
        hairSession.setSizeScale(hairSizePercent / 100);
        hairSession.setOffsetY(hairOffsetPercent / 100);
        hairSession.setHairColor(hairColorId);
        hairSession.setStyle(style, image);
        hairSession.setPhoto(source, { mirror });
        setStageState("photo");
        syncHairSizeUI();
        setHairOffset(hairOffsetPercent);
        preloadHairImages(filterHairStyles({ shape: shapeFilter, category: categoryFilter }));
        if (window.innerWidth < DESKTOP_MIN_WIDTH) {
            document.querySelector("#screen-hairstyle .ar-stage")?.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    } catch (err) {
        console.error(err);
        showToast(err.message || "ประมวลผลรูปไม่สำเร็จ กรุณาลองรูปอื่น", "error");
        if (!hairSession?.hasPhoto) setStageState("empty");
    } finally {
        setLoading(null);
    }
}

/** The photo analysed on the home screen (no need to upload again). */
export async function useScanPhotoForHair() {
    if (!appState.selectedImg) {
        showToast("ยังไม่มีรูปที่วิเคราะห์ไว้ กรุณาอัปโหลดหรือถ่ายรูป", "info");
        return;
    }
    stopHairCamera();
    try { await usePhoto(await imageFromSrc(appState.selectedImg)); }
    catch (err) { showToast(err.message, "error"); }
}

export function openHairPhotoPicker() {
    document.getElementById("hair-file")?.click();
}

export async function handleHairPhotoUpload(input) {
    const file = input.files?.[0];
    input.value = ""; // picking the same file again should still work
    if (!file) return;
    if (!file.type.startsWith("image/")) {
        showToast("กรุณาเลือกไฟล์รูปภาพ", "error");
        return;
    }
    stopHairCamera();
    const url = URL.createObjectURL(file);
    try { await usePhoto(await imageFromSrc(url)); }
    catch (err) { showToast(err.message, "error"); }
    finally { URL.revokeObjectURL(url); }
}

/** Camera only to take the photo; the try-on itself runs on the still frame. */
export async function startHairCamera() {
    if (cameraStream) return;
    if (!navigator.mediaDevices?.getUserMedia) {
        showToast("เบราว์เซอร์นี้เปิดกล้องไม่ได้ ใช้การอัปโหลดรูปแทน", "error");
        return;
    }
    setLoading("กำลังขออนุญาตใช้กล้อง...");
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
        });
        const video = document.getElementById("hair-video");
        video.srcObject = cameraStream;
        await video.play();
        setStageState("camera");
        loadHairPhotoModelsInBackground(); // warm the models while the user poses
    } catch (err) {
        stopHairCamera();
        const denied = ["NotAllowedError", "PermissionDeniedError"].includes(err?.name);
        showToast(denied ? "ไม่ได้รับสิทธิ์ใช้กล้อง ใช้การอัปโหลดรูปแทนได้" : (err.message || "เปิดกล้องไม่สำเร็จ"), "error");
        setStageState(hairSession?.hasPhoto ? "photo" : "empty");
    } finally {
        setLoading(null);
    }
}

function loadHairPhotoModelsInBackground() {
    hairSession ??= new HairPhotoSession({ canvas: document.getElementById("hair-stage") });
    hairSession.load().catch(() => {});
}

export async function snapHairPhoto() {
    const video = document.getElementById("hair-video");
    if (!cameraStream || !video || video.readyState < 2) {
        showToast("กล้องยังไม่พร้อม กรุณารอสักครู่", "error");
        return;
    }
    const frame = document.createElement("canvas");
    frame.width = video.videoWidth; frame.height = video.videoHeight;
    frame.getContext("2d").drawImage(video, 0, 0);
    stopHairCamera();
    await usePhoto(frame, { mirror: true });   // like the preview the user just saw
}

export function stopHairCamera() {
    if (cameraStream) cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
    const video = document.getElementById("hair-video");
    if (video) video.srcObject = null;
    if (document.getElementById("hair-camera-controls") && !document.getElementById("hair-camera-controls").classList.contains("hidden")) {
        setStageState(hairSession?.hasPhoto ? "photo" : "empty");
    }
}

/** Save the photo with the new hairstyle as a JPEG file. */
export function captureHairPhoto() {
    const dataUrl = hairSession?.capture();
    if (!dataUrl) {
        showToast("ยังไม่มีรูปให้บันทึก", "error");
        return;
    }
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `ushape-${currentHairStyle}.jpg`;
    document.body.append(a);
    a.click();
    a.remove();
    showToast("บันทึกรูปทรงผมแล้ว", "success");
}

/** Back to choosing a photo. */
export function resetHairPhoto() {
    stopHairCamera();
    hairSession?.clear();
    setStageState("empty");
}

/** Called when leaving the screen: the camera must not stay on; the photo is kept for coming back. */
export function stopHairSession() {
    stopHairCamera();
}
