import { showToast } from "./toast.js";
import { FaceARSession } from "../cameraUtils.js";
import { ArRenderer, ArTransformSmoother, GLASSES_REF_EYE_SPAN, GLASSES_SHAPES, GLASSES_STYLE_CONFIG } from "../arRenderer.js";

let currentGlassesStyle = "classic";
let glassesSession = null;
let lastTrackedAt = 0;
const glassesSmoother = new ArTransformSmoother();

export function selectGlassesStyle(style) {
    currentGlassesStyle = style;
    document.querySelectorAll(".glass-btn").forEach((btn) => {
        const active = btn.dataset.style === style;
        btn.classList.toggle("border-[#4274D9]", active);
        btn.classList.toggle("bg-[#D0E7E6]", active);
        btn.classList.toggle("border-transparent", !active);
        btn.classList.toggle("bg-slate-50", !active);
    });
    document.getElementById("glasses-shape-inner").innerHTML = GLASSES_SHAPES[style] || GLASSES_SHAPES.classic;
}

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
    glassesSmoother.reset();
    lastTrackedAt = 0;

    loading.classList.remove("hidden");
    openBtn.disabled = true;
    openBtn.classList.add("opacity-60", "cursor-not-allowed");

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
        hint.classList.remove("hidden");
        glassesSmoother.reset();
        lastTrackedAt = 0;
        return;
    }
    try {
        const fit = GLASSES_STYLE_CONFIG[currentGlassesStyle] || GLASSES_STYLE_CONFIG.classic;
        const raw = ArRenderer.getEyeTransform(landmarks, w, h, { verticalOffsetRatio: fit.verticalOffsetRatio });
        const t = glassesSmoother.update(raw);
        ArRenderer.applyTransform(group, t.cx, t.cy, t.angle, t.span / GLASSES_REF_EYE_SPAN * fit.scale);
        lastTrackedAt = performance.now();
        hint.classList.add("hidden");
        group.style.opacity = "1";
    } catch (error) {
        console.error("Glasses tracking failed:", error);
        group.style.opacity = "0";
        hint.classList.remove("hidden");
    }
}

export function captureGlassesPhoto() {
    const video = document.getElementById("glasses-video");
    const svg = document.getElementById("glasses-overlay");
    if (!video || video.readyState < 2) {
        showToast("กล้องยังไม่พร้อมให้ถ่ายรูป กรุณารอสักครู่", "error");
        return;
    }
    if (!lastTrackedAt || performance.now() - lastTrackedAt > 500) {
        showToast("ยังติดตามใบหน้าไม่ได้ กรุณาหันหน้าตรงเข้าหากล้อง", "error");
        return;
    }
    const canvas = document.getElementById("glasses-canvas");
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, width, height);

    try {
        // Rasterize the tracked SVG overlay on top of the freeze-frame so the
        // saved photo matches exactly what was on screen when captured.
        const svgData = new XMLSerializer().serializeToString(svg);
        const svgImg = new Image();
        svgImg.onload = () => { ctx.drawImage(svgImg, 0, 0, width, height); finishGlassesCapture(canvas); };
        svgImg.onerror = () => finishGlassesCapture(canvas);
        svgImg.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
    } catch (e) {
        console.error("Failed to rasterize glasses overlay onto capture:", e);
        finishGlassesCapture(canvas);
    }
}

function finishGlassesCapture(canvas) {
    document.getElementById("glasses-preview").src = canvas.toDataURL("image/jpeg", 0.92);
    if (glassesSession) { glassesSession.stop(); glassesSession = null; }
    glassesSmoother.reset();
    lastTrackedAt = 0;
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
    document.getElementById("glasses-video").classList.add("hidden");
    document.getElementById("glasses-overlay").classList.add("hidden");
    document.getElementById("glasses-shape-group").style.opacity = "0";
    document.getElementById("glasses-controls").classList.add("hidden");
    document.getElementById("glasses-preview-container").classList.add("hidden");
    document.getElementById("glasses-face-hint").classList.add("hidden");
    document.getElementById("glasses-placeholder").classList.remove("hidden");
    document.getElementById("glasses-open-btn").classList.remove("hidden");
}
