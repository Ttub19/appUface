import { showToast } from "./toast.js";
import { FaceARSession } from "../cameraUtils.js";
import { ArRenderer, ArTransformSmoother, GLASSES_SHAPES, GLASSES_STYLE_CONFIG, GLASSES_TEMPLES, estimateYaw, templeArm } from "../arRenderer.js";
import {
    GLASSES_CATEGORIES, DEFAULT_GLASSES_STYLE, glassesInCategory, isInCategory, getGlassesItem, resolveGlassesStyle,
} from "../glassesCatalog.js";

// v2: auto-fit was recalibrated, so a size saved against the old fit is discarded
const SIZE_STORAGE_KEY = "glassesSizePercent.v2";
const SIZE_MIN = 75, SIZE_MAX = 125;
const DESKTOP_MIN_WIDTH = 1024;

let currentGlassesStyle = DEFAULT_GLASSES_STYLE;
let currentCategory = "all";
let glassesSession = null;
let lastTrackedAt = 0;
let lastPose = null; // { cx, cy, angle, scale } of the frame currently on screen, in video pixels
let capturing = false;
let glassesSizePercent = loadGlassesSize();
const glassesSmoother = new ArTransformSmoother();
const imageCache = new Map();

function clampSize(value) {
    const n = Math.round(Number(value));
    return Number.isFinite(n) ? Math.min(SIZE_MAX, Math.max(SIZE_MIN, n)) : 100;
}

function loadGlassesSize() {
    try { return clampSize(localStorage.getItem(SIZE_STORAGE_KEY) ?? 100); }
    catch { return 100; }
}

function syncGlassesSizeUI() {
    const input = document.getElementById("glasses-size");
    const output = document.getElementById("glasses-size-value");
    if (input) input.value = String(glassesSizePercent);
    if (output) output.textContent = glassesSizePercent === 100 ? "พอดีหน้า" : `${glassesSizePercent}%`;
}

/** Manual fine-tune on top of the automatic face-width fit (75–125%). */
export function setGlassesSize(value) {
    glassesSizePercent = clampSize(value);
    try { localStorage.setItem(SIZE_STORAGE_KEY, String(glassesSizePercent)); } catch { /* storage unavailable */ }
    syncGlassesSizeUI();
}

function loadImage(src) {
    if (!imageCache.has(src)) {
        imageCache.set(src, new Promise((resolve, reject) => {
            const img = new Image();
            img.decoding = "async";
            img.onload = () => resolve(img);
            img.onerror = () => { imageCache.delete(src); reject(new Error(`โหลดรูปแว่นไม่สำเร็จ: ${src}`)); };
            img.src = src;
        }));
    }
    return imageCache.get(src);
}

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------- Picker: category chips + frame cards ---------- */

function renderCategoryChips() {
    const root = document.getElementById("glasses-category-filter");
    if (!root) return;
    root.innerHTML = GLASSES_CATEGORIES.map(({ id, label }) => {
        const active = id === currentCategory;
        return `<button type="button" class="chip chip-filter${active ? " is-active" : ""}" data-category="${id}" aria-pressed="${active}">`
            + `${escapeHtml(label)}<span class="chip-count">${glassesInCategory(id).length}</span></button>`;
    }).join("");
}

function renderGlassesList() {
    const root = document.getElementById("glasses-selector");
    if (!root) return;
    root.innerHTML = glassesInCategory(currentCategory).map(({ id, name, desc }) => {
        const active = id === currentGlassesStyle;
        return `<button type="button" class="glass-btn option-card is-vertical${active ? " is-selected" : ""}" data-style="${id}" aria-pressed="${active}">`
            + `<img src="assets/glasses/${id}.png?v=3" alt="" loading="lazy" decoding="async">`
            + `<strong>${escapeHtml(name)}</strong><small>${escapeHtml(desc)}</small></button>`;
    }).join("");
}

/** Shows only the frames in one category ("all" shows everything). */
export function setGlassesCategory(categoryId) {
    currentCategory = GLASSES_CATEGORIES.some((c) => c.id === categoryId) ? categoryId : "all";
    renderCategoryChips();
    renderGlassesList();
    const list = document.getElementById("glasses-selector");
    if (list) { list.scrollTop = 0; list.scrollLeft = 0; }
}

function initGlassesPicker() {
    const chips = document.getElementById("glasses-category-filter");
    const list = document.getElementById("glasses-selector");
    if (!chips || !list) return;
    chips.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-category]");
        if (btn) setGlassesCategory(btn.dataset.category);
    });
    list.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-style]");
        if (btn) selectGlassesStyle(btn.dataset.style);
    });
    renderCategoryChips();
    renderGlassesList();
}

/** Accepts current or legacy style ids (the face-shape guide still uses the old ones). */
export function selectGlassesStyle(style) {
    currentGlassesStyle = resolveGlassesStyle(style);
    // If the frame isn't visible under the current filter (e.g. opened from the
    // face-shape guide), switch to its own category so the selection shows.
    if (!isInCategory(currentGlassesStyle, currentCategory)) {
        currentCategory = getGlassesItem(currentGlassesStyle)?.categories[0] || "all";
        renderCategoryChips();
        renderGlassesList();
    }
    document.querySelectorAll(".glass-btn").forEach((btn) => {
        const active = btn.dataset.style === currentGlassesStyle;
        btn.classList.toggle("is-selected", active);
        btn.setAttribute("aria-pressed", String(active));
    });
    document.getElementById("glasses-shape-inner").innerHTML = GLASSES_SHAPES[currentGlassesStyle] || GLASSES_SHAPES[DEFAULT_GLASSES_STYLE];
    loadImage(`assets/glasses/${currentGlassesStyle}.png?v=3`).catch(() => {}); // warm the cache for capture
}

/* ---------- Camera session ---------- */

export async function startGlassesSession() {
    if (glassesSession) return;
    const openBtn = document.getElementById("glasses-open-btn");
    const loading = document.getElementById("glasses-loading");
    const placeholder = document.getElementById("glasses-placeholder");
    const video = document.getElementById("glasses-video");
    const overlay = document.getElementById("glasses-overlay");
    const preview = document.getElementById("glasses-preview-container");
    const controls = document.getElementById("glasses-controls");

    selectGlassesStyle(currentGlassesStyle); // draw a shape immediately
    syncGlassesSizeUI();
    glassesSmoother.reset();
    lastTrackedAt = 0;
    lastPose = null;

    loading.classList.remove("hidden");
    openBtn.disabled = true;
    openBtn.classList.add("opacity-60", "cursor-not-allowed");

    // On phones/tablets the picker sits under the camera; bring the camera into view.
    if (window.innerWidth < DESKTOP_MIN_WIDTH) {
        document.querySelector("#screen-glasses .ar-stage")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    glassesSession = new FaceARSession({ video, onFrame: onGlassesFrame });

    try {
        await glassesSession.start();
        video.classList.remove("hidden");
        overlay.classList.remove("hidden");
        placeholder.classList.add("hidden");
        preview.classList.add("hidden");
        controls.classList.remove("hidden");
        openBtn.classList.add("hidden");
        showToast("เปิดกล้องสำเร็จ เลือกสไตล์แว่นและลองสวมใส่", "success");
    } catch (err) {
        console.error(err);
        glassesSession?.stop();
        glassesSession = null;
        video.classList.add("hidden");
        overlay.classList.add("hidden");
        placeholder.classList.remove("hidden");
        controls.classList.add("hidden");
        showToast(err.message || "ไม่สามารถเข้าใช้งานกล้องได้ กรุณาให้สิทธิ์การเข้าถึงกล้องและลองใหม่", "error");
    } finally {
        loading.classList.add("hidden");
        openBtn.disabled = false;
        openBtn.classList.remove("opacity-60", "cursor-not-allowed");
    }
}

function onGlassesFrame(landmarks, w, h) {
    const svg = document.getElementById("glasses-overlay");
    const group = document.getElementById("glasses-shape-group");
    const hint = document.getElementById("glasses-face-hint");
    if (!svg || !group) return;

    const vb = svg.viewBox.baseVal;
    if (vb.width !== w || vb.height !== h) svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

    if (!landmarks) {
        group.style.opacity = "0";
        drawTemple(null);
        hint.classList.remove("hidden");
        glassesSmoother.reset();
        lastTrackedAt = 0;
        lastPose = null;
        return;
    }
    try {
        const fit = GLASSES_STYLE_CONFIG[currentGlassesStyle] || GLASSES_STYLE_CONFIG[DEFAULT_GLASSES_STYLE];
        const raw = ArRenderer.getEyeTransform(landmarks, w, h, { verticalOffsetRatio: fit.verticalOffsetRatio });
        raw.yaw = estimateYaw(landmarks, w, h);
        const t = glassesSmoother.update(raw);   // yaw is smoothed too, so the arm doesn't flicker
        const scale = ArRenderer.fitGlassesScale(t, fit, glassesSizePercent / 100);
        ArRenderer.applyTransform(group, t.cx, t.cy, t.angle, scale);
        const arm = templeArm(t, scale, GLASSES_TEMPLES[currentGlassesStyle]);
        drawTemple(arm);
        lastPose = { cx: t.cx, cy: t.cy, angle: t.angle, scale, arm };
        lastTrackedAt = performance.now();
        hint.classList.add("hidden");
        group.style.opacity = "1";
    } catch (error) {
        console.error("Glasses tracking failed:", error);
        group.style.opacity = "0";
        drawTemple(null);
        hint.classList.remove("hidden");
    }
}

/** The visible temple arm (ขาแว่น) as an SVG polygon behind the frame; null hides it. */
function drawTemple(arm) {
    const el = document.getElementById("glasses-temple");
    if (!el) return;
    if (!arm) { el.setAttribute("opacity", "0"); return; }
    el.setAttribute("points", arm.points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" "));
    el.setAttribute("fill", arm.color);
    el.setAttribute("opacity", arm.opacity.toFixed(3));
}

function drawTempleOnto(ctx, arm) {
    if (!arm) return;
    ctx.save();
    ctx.globalAlpha = arm.opacity;
    ctx.fillStyle = arm.color;
    ctx.strokeStyle = "rgba(0,0,0,.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    arm.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
}

/**
 * Draws the frame onto the photo canvas with the same transform as on screen.
 * (Rasterizing the SVG overlay instead would drop the frame: an SVG rendered
 * as an image is not allowed to load external files like assets/glasses/*.png.)
 */
async function drawGlassesOnto(ctx) {
    const el = document.querySelector("#glasses-shape-inner image");
    if (!el || !lastPose) return false;
    const src = el.getAttribute("href") || el.getAttribute("xlink:href");
    const img = await loadImage(src);
    const num = (name) => Number(el.getAttribute(name)) || 0;
    drawTempleOnto(ctx, lastPose.arm);   // arm first: the frame's end piece covers the joint
    ctx.save();
    ctx.translate(lastPose.cx, lastPose.cy);
    ctx.rotate(lastPose.angle);
    ctx.scale(lastPose.scale, lastPose.scale);
    ctx.drawImage(img, num("x"), num("y"), num("width"), num("height"));
    ctx.restore();
    return true;
}

export async function captureGlassesPhoto() {
    if (capturing) return;
    const video = document.getElementById("glasses-video");
    if (!video || video.readyState < 2) {
        showToast("กล้องยังไม่พร้อมให้ถ่ายรูป กรุณารอสักครู่", "error");
        return;
    }
    if (!lastPose || !lastTrackedAt || performance.now() - lastTrackedAt > 500) {
        showToast("ยังติดตามใบหน้าไม่ได้ กรุณาหันหน้าตรงเข้าหากล้อง", "error");
        return;
    }
    capturing = true;
    try {
        const canvas = document.getElementById("glasses-canvas");
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height); // freeze this frame first
        const drawn = await drawGlassesOnto(ctx);
        if (!drawn) throw new Error("no frame to draw");
        finishGlassesCapture(canvas);
    } catch (error) {
        console.error("Glasses capture failed:", error);
        showToast("ถ่ายรูปพร้อมแว่นไม่สำเร็จ ลองกดถ่ายอีกครั้ง", "error");
    } finally {
        capturing = false;
    }
}

function finishGlassesCapture(canvas) {
    document.getElementById("glasses-preview").src = canvas.toDataURL("image/jpeg", 0.92);
    if (glassesSession) { glassesSession.stop(); glassesSession = null; }
    glassesSmoother.reset();
    lastTrackedAt = 0;
    lastPose = null;
    document.getElementById("glasses-video").classList.add("hidden");
    document.getElementById("glasses-overlay").classList.add("hidden");
    document.getElementById("glasses-controls").classList.add("hidden");
    document.getElementById("glasses-placeholder").classList.add("hidden");
    document.getElementById("glasses-preview-container").classList.remove("hidden");
    document.getElementById("glasses-open-btn").classList.remove("hidden");
    showToast("ถ่ายรูปพร้อมแว่น AR สำเร็จแล้ว!", "success");
}

export function stopGlassesSession() {
    if (glassesSession) { glassesSession.stop(); glassesSession = null; }
    glassesSmoother.reset();
    lastTrackedAt = 0;
    lastPose = null;
    document.getElementById("glasses-video").classList.add("hidden");
    document.getElementById("glasses-overlay").classList.add("hidden");
    document.getElementById("glasses-shape-group").style.opacity = "0";
    drawTemple(null);
    document.getElementById("glasses-controls").classList.add("hidden");
    document.getElementById("glasses-preview-container").classList.add("hidden");
    document.getElementById("glasses-face-hint").classList.add("hidden");
    document.getElementById("glasses-placeholder").classList.remove("hidden");
    document.getElementById("glasses-open-btn").classList.remove("hidden");
}

// Module scripts run after the document is parsed, so the picker can render now.
if (typeof document !== "undefined") initGlassesPicker();
