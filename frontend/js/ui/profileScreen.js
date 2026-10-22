import { appState } from "../state.js";
import { showToast } from "./toast.js";
import { apiClient } from "../apiClient.js";

const DEFAULT_ACCENT = "#3E5C9A";
// accent -> its pressed/hover shade, all from the brand palette
const ACCENT_SWATCHES = { "#3E5C9A": "#264478", "#264478": "#141E3C", "#141E3C": "#0B1228" };
let currentAccent = DEFAULT_ACCENT;

/** Applies an accent color to the CSS variables + highlights the matching swatch. Does not save to the backend.
 *  Accents saved with an older palette fall back to the brand default. */
export function applyAccent(color) {
    const normalized = String(color || "").toUpperCase();
    currentAccent = normalized in ACCENT_SWATCHES ? normalized : DEFAULT_ACCENT;
    document.documentElement.style.setProperty("--accent", currentAccent);
    document.documentElement.style.setProperty("--accent-dark", ACCENT_SWATCHES[currentAccent]);
    document.documentElement.style.setProperty("--accent-text", "#ffffff");

    document.querySelectorAll(".accent-swatch").forEach((el) => {
        const swatchColor = (el.getAttribute("onclick") || "").match(/'(#[0-9A-Fa-f]{6})'/)?.[1];
        el.classList.toggle("active", swatchColor?.toUpperCase() === currentAccent);
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
        toggle.classList.toggle("is-on", user.notify_on_complete);
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
