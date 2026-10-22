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

// ---------- Upload ----------
export function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    stopCamera();
    const reader = new FileReader();
    reader.onload = function (ev) {
        appState.selectedImg = ev.target.result;
        document.getElementById("preview-placeholder").classList.add("hidden");
        document.getElementById("preview-container").classList.remove("hidden");
        document.getElementById("main-preview").src = appState.selectedImg;
        revealPreviewFrame();
    };
    reader.readAsDataURL(file);
}

// ---------- Live camera ----------
export async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showToast("เบราว์เซอร์นี้ไม่รองรับการเปิดกล้อง", "error");
        return;
    }
    try {
        appState.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        document.getElementById("preview-placeholder").classList.add("hidden");
        document.getElementById("preview-container").classList.add("hidden");
        const video = document.getElementById("camera-video");
        video.srcObject = appState.cameraStream;
        video.classList.remove("hidden");
        document.getElementById("camera-controls").classList.remove("hidden");
        revealPreviewFrame();
    } catch (err) {
        showToast("ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาตการใช้กล้องของเบราว์เซอร์", "error");
    }
}

export function capturePhoto() {
    const video = document.getElementById("camera-video");
    const canvas = document.getElementById("camera-canvas");
    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    appState.selectedImg = canvas.toDataURL("image/png");

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
        const prediction = await apiClient.predictFaceShape(appState.selectedImg);
        // The backend owns the prediction (shape id + confidence); the rich style
        // recommendations (hair/glasses/jewelry tips) are presentation content that
        // lives in the front-end's local faceData table, keyed by the same shape id.
        const styleInfo = faceData.find((s) => s.id === prediction.face_shape) || faceData[0];
        appState.resultObj = styleInfo;
        appState.lastConfidence = prediction.confidence;

        const gender = document.getElementById("gender-pref").value;
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
