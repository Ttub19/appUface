import { appState } from "../state.js";
import { showToast } from "./toast.js";
import { FaceARSession } from "../cameraUtils.js";
import { GoldenRatioCalc, GoldenRatioStabilizer } from "../goldenRatioCalc.js";
import { ArRenderer } from "../arRenderer.js";
import { clamp } from "../landmarks.js";
import { apiClient } from "../apiClient.js";

let goldenSession = null;
let goldenFrameCount = 0;
let lastMetrics = null;
const goldenStabilizer = new GoldenRatioStabilizer();

function setGoldenSaveEnabled(enabled) {
    const button = document.getElementById("golden-save-btn");
    button.disabled = !enabled;
    button.classList.toggle("opacity-50", !enabled);
    button.classList.toggle("cursor-not-allowed", !enabled);
}

export async function startGoldenSession() {
    const openBtn = document.getElementById("golden-open-btn");
    const stopBtn = document.getElementById("golden-stop-btn");
    const loading = document.getElementById("golden-loading");
    const placeholder = document.getElementById("golden-placeholder");
    const video = document.getElementById("golden-video");
    const canvas = document.getElementById("golden-canvas");

    goldenFrameCount = 0;
    lastMetrics = null;
    goldenStabilizer.reset();
    loading.classList.remove("hidden");
    openBtn.disabled = true;
    openBtn.classList.add("opacity-60", "cursor-not-allowed");

    goldenSession = new FaceARSession({ video, onFrame: onGoldenFrame });

    try {
        await goldenSession.start();
        placeholder.classList.add("hidden");
        video.classList.remove("hidden");
        canvas.classList.remove("hidden");
        openBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");
        document.getElementById("golden-save-btn").classList.remove("hidden");
        setGoldenSaveEnabled(false);
        showToast("เปิดกล้องสำเร็จ กำลังวิเคราะห์สัดส่วนใบหน้าแบบเรียลไทม์", "success");
    } catch (err) {
        console.error(err);
        goldenSession = null;
        showToast(err.message || "ไม่สามารถเข้าใช้งานกล้องได้ กรุณาให้สิทธิ์การเข้าถึงกล้องแล้วลองใหม่", "error");
    } finally {
        loading.classList.add("hidden");
        openBtn.disabled = false;
        openBtn.classList.remove("opacity-60", "cursor-not-allowed");
    }
}

export function stopGoldenSession() {
    if (goldenSession) { goldenSession.stop(); goldenSession = null; }
    lastMetrics = null;
    goldenFrameCount = 0;
    goldenStabilizer.reset();
    setGoldenSaveEnabled(false);
    const els = ["golden-video", "golden-canvas"].map((id) => document.getElementById(id));
    els.forEach((el) => el && el.classList.add("hidden"));
    document.getElementById("golden-placeholder").classList.remove("hidden");
    document.getElementById("golden-open-btn").classList.remove("hidden");
    document.getElementById("golden-save-btn").classList.add("hidden");
    document.getElementById("golden-stop-btn").classList.add("hidden");
    document.getElementById("golden-face-hint").classList.add("hidden");
}

function onGoldenFrame(landmarks, w, h) {
    const canvas = document.getElementById("golden-canvas");
    const hint = document.getElementById("golden-face-hint");
    if (!canvas) return;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext("2d");

    if (!landmarks) {
        ctx.clearRect(0, 0, w, h);
        hint.classList.remove("hidden");
        lastMetrics = null;
        goldenStabilizer.reset();
        setGoldenSaveEnabled(false);
        return;
    }
    hint.classList.add("hidden");
    ArRenderer.drawGoldenMesh(ctx, landmarks, w, h);

    // Throttle the score UI update — smooth enough to feel live without
    // thrashing layout on every single video frame (~30fps).
    goldenFrameCount++;
    if (goldenFrameCount % 6 !== 0) return;
    try {
        const frameMetrics = GoldenRatioCalc.compute(landmarks, w, h);
        if (!frameMetrics.quality.isUsable) {
            lastMetrics = null;
            goldenStabilizer.reset();
            setGoldenSaveEnabled(false);
            hint.classList.remove("hidden");
            return;
        }
        const metrics = goldenStabilizer.push(frameMetrics);
        updateGoldenScoreUI(metrics);
        lastMetrics = metrics.stable ? metrics : null;
        setGoldenSaveEnabled(metrics.stable);
    } catch (e) {
        console.error("Golden ratio calculation failed:", e);
        lastMetrics = null;
        goldenStabilizer.reset();
        setGoldenSaveEnabled(false);
    }
}

function updateGoldenScoreUI(m) {
    document.getElementById("golden-overall-score").innerText = Math.round(m.overall);
    document.getElementById("golden-overall-label").innerText = m.overall >= 85
        ? "สัดส่วนใกล้เคียงอัตราส่วนทองคำมาก"
        : m.overall >= 65 ? "สัดส่วนอยู่ในเกณฑ์ดี" : "มีความแตกต่างจากอัตราส่วนทองคำอยู่บ้าง";

    const thirdIds = ["golden-third-1", "golden-third-2", "golden-third-3"];
    const barIds = ["golden-bar-1", "golden-bar-2", "golden-bar-3"];
    m.thirds.forEach((pct, i) => {
        document.getElementById(thirdIds[i]).innerText = pct.toFixed(1) + "%";
        document.getElementById(barIds[i]).style.width = clamp(pct, 0, 100) + "%";
    });

    document.getElementById("golden-ratio-value").innerHTML = `${m.ratio.toFixed(2)} : 1 <span class="text-slate-400">(1.618)</span>`;
    document.getElementById("golden-bar-ratio").style.width = clamp(m.ratioScore, 0, 100) + "%";
    document.getElementById("golden-symmetry-value").innerText = Math.round(m.symmetryScore) + "%";
    document.getElementById("golden-bar-symmetry").style.width = clamp(m.symmetryScore, 0, 100) + "%";
}

/** Persists the most recently computed Golden Ratio metrics snapshot to the backend. */
export async function saveGoldenRatioResult() {
    if (appState.currentRole === "guest") {
        showToast("ต้องเข้าสู่ระบบเป็นสมาชิกก่อนจึงจะบันทึกผลวิเคราะห์ได้", "error");
        return;
    }
    if (!lastMetrics?.stable) {
        showToast("ผลยังไม่นิ่ง กรุณาหันหน้าตรงและอยู่นิ่งสักครู่", "error");
        return;
    }
    try {
        await apiClient.saveGoldenRatio({
            forehead_pct: lastMetrics.thirds[0],
            midface_pct: lastMetrics.thirds[1],
            lower_pct: lastMetrics.thirds[2],
            length_width_ratio: lastMetrics.ratio,
            ratio_score: lastMetrics.ratioScore,
            symmetry_score: lastMetrics.symmetryScore,
            overall_score: lastMetrics.overall,
        });
        showToast("บันทึกผลวิเคราะห์ Golden Ratio สำเร็จ", "success");
    } catch (err) {
        console.error(err);
        showToast(err.message || "บันทึกผลวิเคราะห์ไม่สำเร็จ กรุณาลองใหม่อีกครั้ง", "error");
    }
}
