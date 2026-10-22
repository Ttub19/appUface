import { appState } from "../state.js";
import { showToast } from "./toast.js";
import { apiClient } from "../apiClient.js";

/** Re-fetches the current user's favorites from the backend into appState.favorites. No-op for guests. */
export async function refreshFavoritesCache() {
    if (appState.currentRole === "guest") {
        appState.favorites = [];
        updateFavBadge();
        return;
    }
    try {
        appState.favorites = await apiClient.listFavorites();
    } catch (err) {
        console.error("Failed to load favorites:", err);
        appState.favorites = [];
    }
    updateFavBadge();
}

export function updateFavBadge() {
    const badge = document.getElementById("fav-badge");
    badge.innerText = appState.favorites.length;
    badge.classList.toggle("hidden", appState.favorites.length === 0);
}

export async function renderFavorites() {
    const grid = document.getElementById("favorites-grid");
    const empty = document.getElementById("favorites-empty");
    await refreshFavoritesCache();

    grid.innerHTML = "";
    empty.classList.toggle("hidden", appState.favorites.length > 0);

    appState.favorites.forEach((f, idx) => {
        const card = document.createElement("div");
        card.className = "card media-card";
        card.innerHTML = `
            <img src="${f.image_data_url || ""}" alt="">
            <div class="p-4 flex flex-col gap-3">
                <div class="flex justify-between items-start gap-3">
                    <div class="min-w-0">
                        <h4 class="font-semibold text-[15px] truncate">${f.face_shape_name}</h4>
                        <p class="text-xs text-[color:var(--ink-4)] mt-0.5">${new Date(f.created_at).toLocaleString("th-TH")}</p>
                    </div>
                    <span class="chip chip-accent shrink-0">${f.confidence}</span>
                </div>
                <button onclick="removeFavorite(${idx})" class="btn btn-danger-ghost btn-sm self-start -ml-3"><i class="fa-regular fa-trash-can"></i>นำออก</button>
            </div>`;
        grid.appendChild(card);
    });
}

export async function removeFavorite(idx) {
    const favorite = appState.favorites[idx];
    if (!favorite) return;
    try {
        await apiClient.deleteFavorite(favorite.id);
        appState.favorites.splice(idx, 1);
        updateFavBadge();
        await renderFavorites();
        showToast("นำออกจากรายการโปรดแล้ว", "info");
    } catch (err) {
        console.error(err);
        showToast(err.message || "นำออกจากรายการโปรดไม่สำเร็จ", "error");
    }
}
