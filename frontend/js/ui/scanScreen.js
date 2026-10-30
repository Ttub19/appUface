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
        const prediction = await apiClient.predictFaceShape(appState.selectedImg, gender);
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
