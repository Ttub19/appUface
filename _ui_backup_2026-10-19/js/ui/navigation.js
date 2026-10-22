import { stopCamera } from "./scanScreen.js";
import { stopGoldenSession } from "./goldenScreen.js";
import { stopGlassesSession } from "./glassesScreen.js";
import { initHairScreen, stopHairSession } from "./hairScreen.js";
import { renderGuide } from "./guideScreen.js";
import { renderFavorites } from "./favoritesScreen.js";
import { renderHistoryTable } from "./historyScreen.js";
import { initCharts } from "./adminScreen.js";
import { renderProfileScreen } from "./profileScreen.js";
import { closeShopChat, initBookingScreen } from "./bookingScreen.js";

const SCREEN_IDS = [
    "screen-login", "app-content", "screen-main", "screen-result", "screen-golden",
    "screen-guide", "screen-favorites", "screen-history", "screen-admin",
    "screen-profile", "screen-glasses", "screen-hairstyle", "screen-booking"
];

const NAV_MAP = [
    ["nav-main", "screen-main"], ["nav-golden", "screen-golden"], ["nav-glasses", "screen-glasses"],
    ["nav-hairstyle", "screen-hairstyle"], ["nav-guide", "screen-guide"], ["nav-favorites", "screen-favorites"],
    ["nav-history", "screen-history"], ["nav-admin", "screen-admin"], ["nav-profile", "screen-profile"], ["nav-booking", "screen-booking"]
];

export function openMobileMenu() {
    document.getElementById("sidebar").classList.add("sidebar-open");
    document.getElementById("sidebar-overlay").classList.add("overlay-visible");
}

export function closeMobileMenu() {
    document.getElementById("sidebar").classList.remove("sidebar-open");
    document.getElementById("sidebar-overlay").classList.remove("overlay-visible");
}

/** Stops every AR/webcam session across all screens — called on every navigation. */
export function stopAllCameras() {
    stopCamera();
    stopGoldenSession();
    stopGlassesSession();
    stopHairSession();
}

/**
 * Shows the screen with the given id and hides all others, updates the
 * active nav-button styling, and runs each screen's own "on enter" hook.
 * @param {string} id - target screen element id, e.g. "screen-main"
 */
export function navigateTo(id) {
    SCREEN_IDS.forEach((s) => {
        const el = document.getElementById(s);
        if (el) el.classList.add("hidden");
    });
    const target = document.getElementById(id);
    if (target) target.classList.remove("hidden");
    if (id !== "screen-login") document.getElementById("app-content").classList.remove("hidden");

    NAV_MAP.forEach(([navId, screenId]) => {
        const btn = document.getElementById(navId);
        if (!btn) return;
        const active = id === screenId;
        btn.classList.toggle("bg-[#4274D9]", active);
        btn.classList.toggle("text-white", active);
        btn.classList.toggle("border-[#95CCDD]", active);
        btn.classList.toggle("border-transparent", !active);
        btn.classList.toggle("text-[#95CCDD]", !active);
    });

    stopAllCameras();
    closeShopChat();
    if (id === "screen-admin") initCharts();
    if (id === "screen-guide") renderGuide();
    if (id === "screen-hairstyle") initHairScreen();
    if (id === "screen-favorites") renderFavorites();
    if (id === "screen-history") renderHistoryTable();
    if (id === "screen-profile") renderProfileScreen();
    if (id === "screen-booking") initBookingScreen();

    closeMobileMenu();
}
