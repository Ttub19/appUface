import { showToast } from "./toast.js";
import { appState } from "../state.js";
import { faceData } from "../data.js";
import { HairTryOnSession } from "../hairTryOn/hairTryOnEngine.js";
import { HAIR_STYLES, getHairStyle, loadHairImage, stylesForFaceShape } from "../hairTryOn/hairAssets.js";

let currentHairStyle = HAIR_STYLES[0].id;
let shapeFilter = "all";
let autoFilteredFor = null;
let eraseEnabled = true;
let hairSession = null;

const shapeLabel = (id) => faceData.find((f) => f.id === id)?.name || id;

/** Called by navigation when the screen opens: pre-filter to the user's analysed face shape. */
export function initHairScreen() {
    const analysed = appState.resultObj?.id;
    if (analysed && autoFilteredFor !== analysed) {
        autoFilteredFor = analysed;
        shapeFilter = analysed;
        const recommended = stylesForFaceShape(analysed);
        if (recommended.length && !recommended.some((s) => s.id === currentHairStyle)) currentHairStyle = recommended[0].id;
    }
    renderShapeFilter();
    renderHairSelector();
}

export function setHairShapeFilter(shape) {
    shapeFilter = shape;
    renderShapeFilter();
    renderHairSelector();
}

function renderShapeFilter() {
    const root = document.getElementById("hair-shape-filter");
    if (!root) return;
    root.replaceChildren(...["all", ...faceData.map((f) => f.id)].map((id) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = id === "all" ? "ทั้งหมด" : shapeLabel(id).replace(/\s*\(.*\)/, "");
        btn.className = `text-xs px-3 py-1.5 rounded-full border transition ${id === shapeFilter
            ? "bg-[#293681] text-white border-[#293681]"
            : "bg-white text-slate-600 border-slate-200 hover:border-[#4274D9]"}`;
        btn.addEventListener("click", () => setHairShapeFilter(id));
        return btn;
    }));

    const note = document.getElementById("hair-recommend-note");
    const analysed = appState.resultObj?.id;
    note.classList.toggle("hidden", !analysed);
    if (analysed) note.textContent = `ผลวิเคราะห์ล่าสุดของคุณ: ${shapeLabel(analysed)} — ทรงที่มีป้าย "แนะนำ" เหมาะกับคุณ`;
}

function renderHairSelector() {
    const root = document.getElementById("hair-selector");
    if (!root) return;
    const analysed = appState.resultObj?.id;
    const styles = stylesForFaceShape(shapeFilter);
    root.replaceChildren(...styles.map((style) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.style = style.id;
        btn.className = "hair-btn relative text-left p-3 rounded-2xl border-2 transition flex flex-col gap-2";
        btn.addEventListener("click", () => selectHairStyle(style.id));

        const thumb = document.createElement("img");
        thumb.src = style.src;
        thumb.alt = "";
        thumb.className = "w-full h-24 object-contain bg-[#eef4ff] rounded-xl";
        const title = document.createElement("span");
        title.className = "font-semibold text-slate-800 text-sm";
        title.textContent = style.name;
        const desc = document.createElement("span");
        desc.className = "text-[11px] text-slate-500 leading-snug";
        desc.textContent = style.desc;
        btn.append(thumb, title, desc);

        if (analysed && style.faceShapes.includes(analysed)) {
            const badge = document.createElement("span");
            badge.className = "absolute top-2 right-2 text-[10px] font-bold bg-[#293681] text-white px-2 py-0.5 rounded-full";
            badge.textContent = "แนะนำ";
            btn.append(badge);
        }
        return btn;
    }));
    if (!styles.length) root.textContent = "ยังไม่มีทรงผมในหมวดนี้";
    highlightSelected();
}

function highlightSelected() {
    document.querySelectorAll(".hair-btn").forEach((btn) => {
        const active = btn.dataset.style === currentHairStyle;
        btn.classList.toggle("border-[#4274D9]", active);
        btn.classList.toggle("bg-[#D0E7E6]", active);
        btn.classList.toggle("border-transparent", !active);
        btn.classList.toggle("bg-slate-50", !active);
    });
}

export async function selectHairStyle(id) {
    const style = getHairStyle(id);
    currentHairStyle = style.id;
    highlightSelected();
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

export async function startHairSession() {
    if (hairSession) return;
    const openBtn = document.getElementById("hair-open-btn");
    const loading = document.getElementById("hair-loading");
    const loadingText = document.getElementById("hair-loading-text");
    const hint = document.getElementById("hair-face-hint");

    loadingText.textContent = "กำลังโหลดโมเดล AI และขออนุญาตกล้อง...";
    loading.classList.remove("hidden");
    openBtn.disabled = true;
    openBtn.classList.add("opacity-60", "cursor-not-allowed");

    const session = new HairTryOnSession({
        video: document.getElementById("hair-video"),
        canvas: document.getElementById("hair-stage"),
        onFaceChange: (visible) => hint.classList.toggle("hidden", visible),
    });
    hairSession = session;
    session.setEraseEnabled(eraseEnabled);

    try {
        const style = getHairStyle(currentHairStyle);
        const [image] = await Promise.all([loadHairImage(style), session.start()]);
        if (hairSession !== session) return; // stopped while loading
        session.setStyle(style, image);
        document.getElementById("hair-stage").classList.remove("hidden");
        document.getElementById("hair-placeholder").classList.add("hidden");
        document.getElementById("hair-preview-container").classList.add("hidden");
        document.getElementById("hair-controls").classList.remove("hidden");
        showToast("เปิดกล้องสำเร็จ เลือกทรงผมและลองเปลี่ยนดูได้เลย", "success");
    } catch (err) {
        if (hairSession === session) stopHairSession();
        if (err?.name !== "AbortError") {
            console.error(err);
            showToast(err.message || "ไม่สามารถเปิดกล้องหรือโหลดโมเดล AI ได้ กรุณาลองใหม่", "error");
        }
    } finally {
        loading.classList.add("hidden");
        openBtn.disabled = false;
        openBtn.classList.remove("opacity-60", "cursor-not-allowed");
    }
}

export function captureHairPhoto() {
    const dataUrl = hairSession?.capture();
    if (!dataUrl) {
        showToast("กล้องยังไม่พร้อมให้ถ่ายรูป กรุณารอสักครู่", "error");
        return;
    }
    stopHairSession();
    document.getElementById("hair-preview").src = dataUrl;
    document.getElementById("hair-placeholder").classList.add("hidden");
    document.getElementById("hair-preview-container").classList.remove("hidden");
    showToast("ถ่ายรูปทรงผมสำเร็จแล้ว!", "success");
}

export function stopHairSession() {
    if (hairSession) { hairSession.stop(); hairSession = null; }
    document.getElementById("hair-stage")?.classList.add("hidden");
    document.getElementById("hair-controls")?.classList.add("hidden");
    document.getElementById("hair-preview-container")?.classList.add("hidden");
    document.getElementById("hair-face-hint")?.classList.add("hidden");
    document.getElementById("hair-placeholder")?.classList.remove("hidden");
}
