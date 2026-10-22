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
        card.className = "bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden";
        card.innerHTML = `
            <img src="${f.image_data_url || ""}" class="w-full h-40 object-cover bg-slate-100">
            <div class="p-5 space-y-2">
                <div class="flex justify-between items-start">
                    <h4 class="font-bold text-slate-800 text-sm font-display">${f.face_shape_name}</h4>
                    <span class="text-[10px] bg-[#D0E7E6] text-[#293681] border border-[#95CCDD] px-2 py-1 rounded-full">${f.confidence}</span>
                </div>
                <p class="text-[10px] text-slate-400">${new Date(f.created_at).toLocaleString("th-TH")}</p>
                <button onclick="removeFavorite(${idx})" class="w-full text-xs bg-slate-50 hover:bg-rose-50 hover:text-rose-600 text-slate-500 py-2 rounded-xl transition"><i class="fa-solid fa-trash mr-2"></i>นำออก</button>
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
