import { appState } from "../state.js";
import { showToast } from "./toast.js";
import { apiClient } from "../apiClient.js";

let currentAccent = "#4274D9";

/** Applies an accent color to the CSS variables + highlights the matching swatch. Does not save to the backend. */
export function applyAccent(color) {
    currentAccent = color;
    const lightColors = ["#95CCDD", "#D0E7E6"];
    const isLight = lightColors.includes(color.toUpperCase()) || lightColors.includes(color);
    document.documentElement.style.setProperty("--accent", color);
    document.documentElement.style.setProperty("--accent-dark", "#293681");
    document.documentElement.style.setProperty("--accent-text", isLight ? "#293681" : "#ffffff");

    document.querySelectorAll(".accent-swatch").forEach((el) => {
        const swatchColor = (el.getAttribute("onclick") || "").match(/'(#[0-9A-Fa-f]{6})'/)?.[1];
        el.classList.toggle("active", swatchColor?.toUpperCase() === color.toUpperCase());
    });
}

/** Click handler for the accent swatch buttons — applies immediately, persisted next time "บันทึกการตั้งค่า" is pressed. */
export function setAccent(color, btn) {
    applyAccent(color);
    showToast("เปลี่ยนสีหลักของแอปแล้ว (กดบันทึกการตั้งค่าเพื่อเก็บไว้ถาวร)", "info");
}

/** Loads the current user's profile into the form. Called on entering the profile screen. */
export async function renderProfileScreen() {
    if (appState.currentRole === "guest") return;
    try {
        const user = appState.currentUser || (await apiClient.getProfile());
        document.getElementById("profile-name").value = user.display_name;
        document.getElementById("profile-email").value = user.email;

        const toggle = document.getElementById("profile-notify-toggle");
        const dot = document.getElementById("profile-notify-dot");
        toggle.classList.toggle("bg-[#293681]", user.notify_on_complete);
        toggle.classList.toggle("bg-slate-300", !user.notify_on_complete);
        dot.classList.toggle("translate-x-5", user.notify_on_complete);

        applyAccent(user.theme_accent);
    } catch (err) {
        console.error(err);
        showToast(err.message || "ไม่สามารถโหลดโปรไฟล์ได้", "error");
    }
}

export async function saveProfile() {
    const name = document.getElementById("profile-name").value.trim();
    if (!name) {
        showToast("กรุณากรอกชื่อที่แสดง", "error");
        return;
    }
    const notifyOn = document.getElementById("profile-notify-dot").classList.contains("translate-x-5");

    try {
        const updated = await apiClient.updateProfile({
            display_name: name,
            notify_on_complete: notifyOn,
            theme_accent: currentAccent,
        });
        appState.currentUser = updated;
        showToast("บันทึกโปรไฟล์เรียบร้อยแล้ว", "success");
    } catch (err) {
        console.error(err);
        showToast(err.message || "บันทึกโปรไฟล์ไม่สำเร็จ", "error");
    }
}
