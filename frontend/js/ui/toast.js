/**
 * Shows a transient toast notification in the bottom-right toast tray.
 * @param {string} message
 * @param {"success"|"error"|"info"} [type="success"]
 */
export function showToast(message, type = "success") {
    const icons = { success: "fa-circle-check", error: "fa-circle-exclamation", info: "fa-circle-info" };
    const kind = icons[type] ? type : "info";
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast toast-enter toast-${kind}`;
    toast.setAttribute("role", kind === "error" ? "alert" : "status");
    toast.innerHTML = `<i class="fa-solid ${icons[kind]}"></i><span></span>`;
    toast.querySelector("span").textContent = message; // server text is shown as text, never as HTML
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.transition = "opacity .3s, transform .3s";
        toast.style.opacity = "0";
        toast.style.transform = "translateY(8px)";
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}
