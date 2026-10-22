/**
 * "ติดต่อแอดมิน / ส่งความคิดเห็น" screen (everyone, guests included) and the
 * admin inbox on the dashboard.
 */
import { appState } from "../state.js";
import { apiClient } from "../apiClient.js";
import { escapeHtml as e } from "../bookingUtils.js";
import { showToast } from "./toast.js";

export const FEEDBACK_CATEGORIES = Object.freeze({
    suggestion: "ข้อเสนอแนะ",
    bug: "แจ้งปัญหาการใช้งาน",
    contact: "ติดต่อแอดมิน",
    other: "อื่น ๆ",
});
const STATUS = Object.freeze({
    new: ["รอแอดมินอ่าน", "is-new"],
    read: ["แอดมินอ่านแล้ว", "is-read"],
    resolved: ["ดำเนินการแล้ว", "is-resolved"],
});
const SCREEN_NAMES = Object.freeze({
    "screen-main": "หน้าแรก", "screen-result": "ผลวิเคราะห์", "screen-guide": "คู่มือรูปหน้า", "screen-golden": "Golden Ratio",
    "screen-hairstyle": "ลองทรงผม", "screen-glasses": "ลองแว่นตา", "screen-booking": "จองคิว", "screen-favorites": "รายการโปรด",
    "screen-history": "ประวัติการสแกน", "screen-profile": "โปรไฟล์", "screen-admin": "Dashboard",
});

let fromScreen = null;
let adminFilter = "new";
const thaiDate = (v) => new Date(v).toLocaleString("th-TH", { dateStyle: "medium", timeStyle: "short" });
const badge = (s) => `<span class="feedback-status ${STATUS[s]?.[1] || ""}">${STATUS[s]?.[0] || e(s)}</span>`;

/** Called by navigateTo() before switching, so the form can say which screen the problem was on. */
export function rememberFeedbackOrigin(screenId) {
    fromScreen = screenId && screenId !== "screen-feedback" ? screenId : fromScreen;
}

export function initFeedbackScreen() {
    const guest = !appState.currentUser;
    document.getElementById("feedback-guest-fields").classList.toggle("hidden", !guest);
    document.getElementById("feedback-mine-card").classList.toggle("hidden", guest);
    const origin = document.getElementById("feedback-origin");
    origin.textContent = fromScreen && SCREEN_NAMES[fromScreen] ? `ส่งจากหน้า: ${SCREEN_NAMES[fromScreen]}` : "";
    updateFeedbackCount();
    if (!guest) renderMyFeedback();
}

export function updateFeedbackCount() {
    const msg = document.getElementById("feedback-message");
    document.getElementById("feedback-count").textContent = `${msg.value.trim().length}/2000`;
}

export async function submitFeedback(event) {
    event?.preventDefault();
    const btn = document.getElementById("feedback-submit");
    const subject = document.getElementById("feedback-subject").value.trim();
    const message = document.getElementById("feedback-message").value.trim();
    const payload = {
        category: document.getElementById("feedback-category").value,
        subject, message,
        page: fromScreen ? SCREEN_NAMES[fromScreen] || fromScreen : null,
    };
    if (subject.length < 3) return showToast("กรุณากรอกหัวข้ออย่างน้อย 3 ตัวอักษร", "error");
    if (message.length < 10) return showToast("กรุณาเล่ารายละเอียดอย่างน้อย 10 ตัวอักษร", "error");
    if (!appState.currentUser) {
        payload.contact_name = document.getElementById("feedback-name").value.trim();
        payload.contact_email = document.getElementById("feedback-email").value.trim();
        if (!payload.contact_name || !payload.contact_email) return showToast("กรุณากรอกชื่อและอีเมล เพื่อให้แอดมินติดต่อกลับได้", "error");
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(payload.contact_email)) return showToast("รูปแบบอีเมลไม่ถูกต้อง", "error");
    }
    btn.disabled = true;
    try {
        await apiClient.sendFeedback(payload);
        document.getElementById("feedback-form").reset();
        updateFeedbackCount();
        showToast("ส่งข้อความถึงแอดมินแล้ว ขอบคุณสำหรับความคิดเห็น", "success");
        if (appState.currentUser) await renderMyFeedback();
    } catch (err) {
        showToast(err.message || "ส่งข้อความไม่สำเร็จ", "error");
    } finally {
        btn.disabled = false;
    }
}

async function renderMyFeedback() {
    const box = document.getElementById("feedback-mine");
    try {
        const items = await apiClient.myFeedback();
        box.innerHTML = items.length ? items.map((x) => `
            <article class="feedback-item">
                <header><strong>${e(x.subject)}</strong>${badge(x.status)}</header>
                <p class="feedback-meta">${FEEDBACK_CATEGORIES[x.category] || e(x.category)} · ${thaiDate(x.created_at)}</p>
                <p class="feedback-body">${e(x.message)}</p>
                ${x.admin_note ? `<div class="feedback-reply"><span>คำตอบจากแอดมิน</span><p>${e(x.admin_note)}</p></div>` : ""}
            </article>`).join("") : `<p class="feedback-empty">ยังไม่มีข้อความที่ส่ง</p>`;
    } catch (err) {
        box.innerHTML = `<p class="feedback-empty">โหลดข้อความไม่ได้</p>`;
    }
}

// ---------------- admin inbox ----------------
export async function loadAdminFeedback(filter = adminFilter) {
    adminFilter = filter;
    document.querySelectorAll("[data-feedback-filter]").forEach((b) => {
        const on = b.dataset.feedbackFilter === filter;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-pressed", String(on));
    });
    const box = document.getElementById("admin-feedback-list");
    if (!box) return;
    try {
        const items = await apiClient.adminFeedback(filter === "all" ? null : filter);
        if (filter === "new") document.getElementById("admin-feedback-count").textContent = items.length;
        box.innerHTML = items.length ? items.map((x) => `
            <article class="feedback-item">
                <header><strong>${e(x.subject)}</strong>${badge(x.status)}</header>
                <p class="feedback-meta">${FEEDBACK_CATEGORIES[x.category] || e(x.category)} · ${e(x.contact_name)} · ${e(x.contact_email)}${x.user_id ? "" : " (ผู้ใช้ทั่วไป)"}${x.page ? ` · หน้า ${e(x.page)}` : ""} · ${thaiDate(x.created_at)}</p>
                <p class="feedback-body">${e(x.message)}</p>
                ${x.admin_note ? `<div class="feedback-reply"><span>คำตอบของแอดมิน</span><p>${e(x.admin_note)}</p></div>` : ""}
                <div class="feedback-actions">
                    ${x.status === "new" ? `<button class="btn btn-secondary btn-sm" onclick="setFeedbackStatus('${x.id}','read')">อ่านแล้ว</button>` : ""}
                    ${x.status !== "resolved" ? `<button class="btn btn-primary btn-sm" onclick="setFeedbackStatus('${x.id}','resolved',true)">ตอบกลับ / ปิดเรื่อง</button>` : `<button class="btn btn-secondary btn-sm" onclick="setFeedbackStatus('${x.id}','read')">เปิดเรื่องอีกครั้ง</button>`}
                </div>
            </article>`).join("") : `<p class="feedback-empty">ไม่มีข้อความในหมวดนี้</p>`;
    } catch (err) {
        box.innerHTML = `<p class="feedback-empty">${e(err.message || "โหลดข้อความไม่ได้")}</p>`;
    }
}

export async function setFeedbackStatus(id, status, askReply = false) {
    let note;
    if (askReply) {
        note = prompt("ข้อความตอบกลับถึงผู้ส่ง (เว้นว่างได้)", "");
        if (note === null) return;
    }
    try {
        await apiClient.updateFeedback(id, { status, ...(note !== undefined ? { admin_note: note } : {}) });
        showToast(status === "resolved" ? "ปิดเรื่องแล้ว" : "อัปเดตสถานะแล้ว", "success");
        await loadAdminFeedback(adminFilter);
        if (adminFilter !== "new") {
            const fresh = await apiClient.adminFeedback("new");
            document.getElementById("admin-feedback-count").textContent = fresh.length;
        }
    } catch (err) {
        showToast(err.message || "อัปเดตไม่สำเร็จ", "error");
    }
}
