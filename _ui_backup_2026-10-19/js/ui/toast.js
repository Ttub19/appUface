/**
 * Shows a transient toast notification in the bottom-right toast tray.
 * @param {string} message
 * @param {"success"|"error"|"info"} [type="success"]
 */
export function showToast(message, type = "success") {
    const styles = {
        success: "bg-emerald-50 border-emerald-200 text-emerald-700",
        error: "bg-rose-50 border-rose-200 text-rose-700",
        info: "bg-[#D0E7E6] border-[#95CCDD] text-[#293681]"
    };
    const icons = { success: "fa-circle-check", error: "fa-triangle-exclamation", info: "fa-circle-info" };
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast-enter flex items-center gap-3 px-5 py-4 rounded-2xl border backdrop-blur-md shadow-xl text-sm font-medium max-w-sm ${styles[type] || styles.info}`;
    toast.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.transition = "opacity .3s, transform .3s";
        toast.style.opacity = "0";
        toast.style.transform = "translateY(8px)";
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}
