import { appState } from "../state.js";
import { showToast } from "./toast.js";
import { navigateTo } from "./navigation.js";
import { applyRolePermissions } from "./auth.js";
import { apiClient } from "../apiClient.js";
import { updateFavBadge } from "./favoritesScreen.js";
import { faceData } from "../data.js";
import { renderResultHairRecs } from "./hairScreen.js";

/** Brings the photo frame into view when it's off-screen (e.g. on phones, below the buttons). */
function revealPreviewFrame() {
    const frame = document.querySelector(".home-frame");
    if (!frame) return;
    const r = frame.getBoundingClientRect();
    if (r.top < 0 || r.bottom > window.innerHeight) frame.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ---------- Image size ----------
// Phone photos are 3-12 MB. Sending them as-is is slow on mobile data, can time out,
// and makes the server (512 MB on Render's free plan) run out of memory. The model
// only looks at a 320 px face crop, so 1280 px on the long side loses nothing.
const MAX_UPLOAD_SIDE = 1280;

/** Draws an image source onto a canvas no larger than MAX_UPLOAD_SIDE and returns a JPEG data URL. */
function toUploadDataUrl(source, width, height, { mirror = false } = {}) {
    const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    if (mirror) { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }   // same as the mirrored preview
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", .9);
}

/** File -> downscaled JPEG data URL (the browser applies the photo's EXIF rotation when decoding). */
function fileToUploadDataUrl(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            URL.revokeObjectURL(url);
            try { resolve(toUploadDataUrl(img, img.naturalWidth, img.naturalHeight)); } catch (err) { reject(err); }
        };
        img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("unreadable image")); };
        img.src = url;
    });
}

// ---------- Upload ----------
export async function handleFileUpload(e) {
    const input = e.target;
    const file = input.files[0];
    input.value = ""; // let the same file be picked again
    if (!file) return;
    stopCamera();
    try {
        appState.selectedImg = await fileToUploadDataUrl(file);
    } catch {
        showToast("เปิดรูปนี้ไม่ได้ กรุณาใช้ไฟล์ JPG หรือ PNG", "error");
        return;
    }
    document.getElementById("preview-placeholder").classList.add("hidden");
    document.getElementById("preview-container").classList.remove("hidden");
    document.getElementById("main-preview").src = appState.selectedImg;
    revealPreviewFrame();
}

// ---------- Live camera ----------
export async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showToast("เบราว์เซอร์นี้ไม่รองรับการเปิดกล้อง", "error");
        return;
    }
    try {
        // ask for a sharper picture than the default 640x480 (phones pick the closest they support)
        appState.cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 1280 } },
        });
        document.getElementById("preview-placeholder").classList.add("hidden");
        document.getElementById("preview-container").classList.add("hidden");
        const video = document.getElementById("camera-video");
        video.srcObject = appState.cameraStream;
        video.classList.remove("hidden");
        document.getElementById("camera-guide").classList.remove("hidden");
        document.getElementById("camera-controls").classList.remove("hidden");
        revealPreviewFrame();
    } catch (err) {
        showToast("ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาตการใช้กล้องของเบราว์เซอร์", "error");
    }
}

export function capturePhoto() {
    const video = document.getElementById("camera-video");
    appState.selectedImg = toUploadDataUrl(video, video.videoWidth || 480, video.videoHeight || 480, { mirror: true });

    stopCamera();
    document.getElementById("preview-container").classList.remove("hidden");
    document.getElementById("main-preview").src = appState.selectedImg;
    showToast("ถ่ายภาพเรียบร้อย พร้อมวิเคราะห์โครงหน้า", "success");
}

export function stopCamera() {
    if (appState.cameraStream) {
        appState.cameraStream.getTracks().forEach((t) => t.stop());
        appState.cameraStream = null;
    }
    document.getElementById("camera-video").classList.add("hidden");
    document.getElementById("camera-guide")?.classList.add("hidden");
    document.getElementById("camera-controls").classList.add("hidden");
    if (!appState.selectedImg) document.getElementById("preview-placeholder").classList.remove("hidden");
}

// ---------- Model version picker ----------
// The server can hold more than one trained model (v3 ResNet50 = "Core", v6 ConvNeXt trained on a
// person-disjoint split = "Vision"). GET /scans/models says which ones exist and for which genders;
// the choice is remembered per browser. A model that has no file for the chosen gender is shown
// disabled and the default model is used instead.
const MODEL_PREF_KEY = "ushape.scanModel";
const GENDER_TH = { male: "ผู้ชาย", female: "ผู้หญิง" };
const FALLBACK_MODELS = [
    { id: "v3", label: "Core", badge: "แนะนำ", tech: "ResNet50",
      detail: "โมเดลหลักของ uShape ใช้งานมาตั้งแต่แรก รองรับทั้งผู้ชายและผู้หญิง", genders: ["female", "male"], default: true },
    { id: "v6", label: "Vision", badge: "ทดลอง", tech: "ConvNeXt",
      detail: "โมเดลใหม่ที่ฝึกให้อ่านโครงหน้าของคนที่ไม่เคยเห็นมาก่อน ผลอาจต่างจาก Core ลองเทียบกันได้", genders: ["female", "male"], default: false },
];
// Shown until GET /scans/models answers (Render's free server can take ~1 min to wake up), so the
// picker never disappears; the server's list replaces it, and the result line says which model answered.
let scanModels = FALLBACK_MODELS;

function savedModel() {
    try { return localStorage.getItem(MODEL_PREF_KEY); } catch { return null; }
}

function chosenGender() {
    return document.getElementById("gender-pref")?.value || "";
}

function defaultModel() {
    return scanModels.find((m) => m.default) || scanModels[0];
}

function supports(model, gender) {
    return !gender || !Array.isArray(model.genders) || model.genders.includes(gender);
}

/** The model the next scan will really use: the user's choice if it covers the chosen gender, else the default. */
export function currentScanModel() {
    const wanted = scanModels.find((m) => m.id === (appState.scanModel || savedModel()));
    const gender = chosenGender();
    return (wanted && supports(wanted, gender) ? wanted : defaultModel()).id;
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
}

function renderModelPicker() {
    const wrap = document.getElementById("model-pref-wrap");
    const group = document.getElementById("model-pref");
    if (!wrap || !group) return;
    wrap.classList.toggle("hidden", scanModels.length < 2);       // one model: nothing to choose
    const selected = currentScanModel();
    const gender = chosenGender();
    group.innerHTML = "";
    for (const m of scanModels) {
        const usable = supports(m, gender);
        const card = el("button", "model-card");
        card.type = "button";
        card.dataset.model = m.id;
        card.setAttribute("role", "radio");
        card.setAttribute("aria-checked", String(m.id === selected));
        card.tabIndex = m.id === selected ? 0 : -1;
        if (!usable) card.setAttribute("aria-disabled", "true");

        const head = el("span", "model-card-head");
        head.append(el("span", "model-card-dot"), el("strong", "model-card-name", m.label));
        if (m.badge) head.append(el("span", `model-badge${m.default ? " is-default" : ""}`, m.badge));
        card.append(head);
        card.append(el("span", "model-card-desc", m.detail || ""));
        if (!usable) {
            const fallback = defaultModel().label;
            card.append(el("span", "model-card-note", `ยังไม่รองรับ${GENDER_TH[gender] || "เพศนี้"} จะใช้โมเดล ${fallback} แทน`));
        } else if (m.tech) {
            card.append(el("span", "model-card-tech", m.tech));
        }
        card.addEventListener("click", () => { if (usable) setScanModel(m.id); });
        group.appendChild(card);
    }
}

export function setScanModel(id) {
    if (!scanModels.some((m) => m.id === id)) return;
    appState.scanModel = id;
    try { localStorage.setItem(MODEL_PREF_KEY, id); } catch { /* private mode: keep in memory only */ }
    renderModelPicker();
}

export async function loadScanModels(attempt = 1) {
    renderModelPicker();                                   // fallback list right away
    try {
        const list = await apiClient.listModels();
        if (Array.isArray(list) && list.length) scanModels = list;
        renderModelPicker();
    } catch (err) {
        console.warn(`GET /scans/models failed (attempt ${attempt}), keeping the built-in list:`, err);
        if (attempt < 4) setTimeout(() => loadScanModels(attempt + 1), 15000);   // server may be waking up
    }
}

// arrow keys move between the usable options (radio-group behaviour)
document.addEventListener("keydown", (e) => {
    if (!e.target.closest?.("#model-pref") || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    const gender = chosenGender();
    const ids = scanModels.filter((m) => supports(m, gender)).map((m) => m.id);
    if (!ids.length) return;
    const i = Math.max(0, ids.indexOf(currentScanModel()));
    const next = ids[(i + (e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1) + ids.length) % ids.length];
    setScanModel(next);
    document.querySelector(`#model-pref [data-model="${next}"]`)?.focus();
});
document.addEventListener("change", (e) => { if (e.target.id === "gender-pref") renderModelPicker(); });

function modelLabel(version) {
    return scanModels.find((m) => m.id === version)?.label || version || "";
}

// ---------- AI analysis (real backend call, full loading/success/error state handling) ----------
export async function runAI() {
    if (!appState.selectedImg) {
        showToast("กรุณาอัปโหลดรูปภาพ หรือถ่ายภาพจากกล้องก่อน", "error");
        return;
    }
    const spinner = document.getElementById("loading-spinner");
    spinner.classList.remove("hidden");
    try {
        const gender = document.getElementById("gender-pref").value;
        const model = scanModels.length > 1 ? currentScanModel() : null;
        const prediction = await apiClient.predictFaceShape(appState.selectedImg, gender, model);
        // The backend owns the prediction (shape id + confidence); the rich style
        // recommendations (hair/glasses/jewelry tips) are presentation content that
        // lives in the front-end's local faceData table, keyed by the same shape id.
        const styleInfo = faceData.find((s) => s.id === prediction.face_shape) || faceData[0];
        appState.resultObj = styleInfo;
        appState.lastConfidence = prediction.confidence;
        appState.genderPref = gender;

        document.getElementById("res-photo").src = appState.selectedImg;
        document.getElementById("res-shape").innerText = prediction.face_shape_name;
        document.getElementById("res-confidence").innerText = prediction.confidence.toFixed(1) + "%";
        document.querySelector(".result-photo")?.style.setProperty("--pct", `${Math.max(0, Math.min(100, prediction.confidence))}%`);
        renderRunnerUp(prediction.ranking);
        appState.lastModelVersion = prediction.model_version || "";
        const resModel = document.getElementById("res-model");
        if (resModel) {
            resModel.classList.toggle("hidden", !prediction.model_version);
            document.getElementById("res-model-name").innerText = modelLabel(prediction.model_version);
        }
        document.getElementById("tip-hair").innerText = styleInfo.hair[gender];
        renderResultHairRecs(styleInfo.id, gender);
        document.getElementById("tip-glass").innerText = styleInfo.glass;
        document.getElementById("tip-jewelry").innerText = styleInfo.jewelry;
        document.getElementById("tip-beard").innerText = gender === "male" ? styleInfo.beard : styleInfo.makeup;
        document.getElementById("beard-label").innerHTML = gender === "male"
            ? '<span class="tile-icon"><i class="fa-solid fa-user-check"></i></span>หนวดเครา'
            : '<span class="tile-icon"><i class="fa-solid fa-spa"></i></span>การแต่งหน้า';

        updateFavoriteButtonState();
        applyRolePermissions();
        navigateTo("screen-result");
    } catch (err) {
        console.error("predictFaceShape failed:", err);
        showToast(err.message || "เกิดข้อผิดพลาดในการวิเคราะห์ กรุณาลองใหม่อีกครั้ง", "error");
    } finally {
        spinner.classList.add("hidden");
    }
}

/**
 * Face shapes overlap (an oval face can lean heart-shaped), so the result also names the
 * runner-up the model gave real weight to. ranking = every shape, most likely first.
 */
export const RUNNER_UP_MIN_CONFIDENCE = 10;

export function pickRunnerUp(ranking) {
    const second = Array.isArray(ranking) ? ranking[1] : null;
    return second && Number(second.confidence) >= RUNNER_UP_MIN_CONFIDENCE ? second : null;
}

function renderRunnerUp(ranking) {
    const el = document.getElementById("res-second");
    if (!el) return;
    const second = pickRunnerUp(ranking);
    appState.lastRunnerUp = second;
    el.classList.toggle("hidden", !second);
    if (!second) return;
    document.getElementById("res-second-name").innerText = second.face_shape_name;
    document.getElementById("res-second-confidence").innerText = `${Number(second.confidence).toFixed(1)}%`;
}

export function shareResult() {
    if (!appState.resultObj) return;
    showToast(`คัดลอกลิงก์ผลลัพธ์ "${appState.resultObj.name}" ไปยังคลิปบอร์ดแล้ว`, "info");
}

function findMatchingFavorite() {
    return appState.favorites.find(
        (f) => f.face_shape === appState.resultObj.id && f.image_data_url === appState.selectedImg
    );
}

function updateFavoriteButtonState() {
    const favBtn = document.getElementById("btn-favorite");
    const isFavorited = Boolean(findMatchingFavorite());
    favBtn.innerHTML = isFavorited ? '<i class="fa-solid fa-heart"></i>' : '<i class="fa-regular fa-heart"></i>';
    favBtn.classList.toggle("text-rose-500", isFavorited);
    favBtn.classList.toggle("text-slate-400", !isFavorited);
    favBtn.setAttribute("aria-pressed", String(isFavorited));
}

// ---------- Favorites toggle (result screen heart button) ----------
export async function toggleFavorite() {
    if (appState.currentRole === "guest") {
        showToast("ต้องเข้าสู่ระบบเป็นสมาชิกก่อนจึงจะบันทึกรายการโปรดได้", "error");
        return;
    }
    if (!appState.resultObj || !appState.selectedImg) return;

    const existing = findMatchingFavorite();
    try {
        if (existing) {
            await apiClient.deleteFavorite(existing.id);
            appState.favorites = appState.favorites.filter((f) => f.id !== existing.id);
            showToast("นำออกจากรายการโปรดแล้ว", "info");
        } else {
            const created = await apiClient.createFavorite({
                face_shape: appState.resultObj.id,
                face_shape_name: appState.resultObj.name,
                confidence: document.getElementById("res-confidence").innerText,
                image_data_url: appState.selectedImg,
            });
            appState.favorites.unshift(created);
            showToast("บันทึกเป็นรายการโปรดแล้ว", "success");
        }
        updateFavoriteButtonState();
        updateFavBadge();
    } catch (err) {
        console.error(err);
        showToast(err.message || "ทำรายการไม่สำเร็จ กรุณาลองใหม่อีกครั้ง", "error");
    }
}
