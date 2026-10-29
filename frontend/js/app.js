/**
 * app.js — application entry point.
 *
 * The HTML markup uses plain inline `onclick="fn()"` / `onchange="fn()"`
 * attributes (kept from the original mockup for simplicity), which look up
 * `fn` on `window`. Since this file and everything it imports are ES modules
 * (not classic global-scope scripts), each handler function has to be
 * explicitly attached to `window` here.
 */
import { handleAuthSubmit, toggleAuthMode, toggleBarberRegistrationFields, continueAsGuest, handleLogout, restoreSessionOnLoad } from "./ui/auth.js";
import { navigateTo, openMobileMenu, closeMobileMenu, goHome } from "./ui/navigation.js";
import {
    handleFileUpload, startCamera, capturePhoto, stopCamera,
    runAI, shareResult, toggleFavorite
} from "./ui/scanScreen.js";
import { removeFavorite } from "./ui/favoritesScreen.js";
import { openGuideShape, closeGuideShape, tryHairForShape, tryGlassesForShape } from "./ui/guideScreen.js";
import { startGoldenSession, stopGoldenSession, saveGoldenRatioResult } from "./ui/goldenScreen.js";
import { selectGlassesStyle, startGlassesSession, captureGlassesPhoto, stopGlassesSession, setGlassesSize } from "./ui/glassesScreen.js";
import { selectHairStyle, captureHairPhoto, stopHairSession, setHairErase, setHairShapeFilter, setHairSize, setHairColor, tryHairStyle, setHairOffset, useScanPhotoForHair, openHairPhotoPicker, handleHairPhotoUpload, startHairCamera, snapHairPhoto, resetHairPhoto } from "./ui/hairScreen.js";
import { saveToDB, removeHistoryRow, filterHistory, exportCSV } from "./ui/historyScreen.js";
import { saveProfile, setAccent } from "./ui/profileScreen.js";
import { approveBarberApplication, rejectBarberApplication } from "./ui/adminScreen.js";
import { submitFeedback, updateFeedbackCount, loadAdminFeedback, setFeedbackStatus } from "./ui/feedbackScreen.js";
import { showBookingTab,useBookingLocation,searchBarberShops,openShopBooking,loadBookingSlots,selectBookingSlot,submitBarberBooking,cancelMyBooking,fillNewShopLocation,submitNewShop,selectManagedShop,addShopBarber,addShopService,saveShopHours,manageBookingAction,openCustomerShopChat,openShopChat,sendShopChat,closeShopChat } from "./ui/bookingScreen.js";

Object.assign(window, {
    // auth / navigation
    handleAuthSubmit, toggleAuthMode, toggleBarberRegistrationFields, continueAsGuest, handleLogout,
    navigateTo, openMobileMenu, closeMobileMenu, goHome,
    // scan & analyze
    handleFileUpload, startCamera, capturePhoto, stopCamera, runAI, shareResult, toggleFavorite,
    // favorites
    removeFavorite,
    // face-shape guide
    openGuideShape, closeGuideShape, tryHairForShape, tryGlassesForShape,
    // Golden Ratio Analysis
    startGoldenSession, stopGoldenSession, saveGoldenRatioResult,
    // Virtual Glasses
    selectGlassesStyle, startGlassesSession, captureGlassesPhoto, stopGlassesSession, setGlassesSize,
    // Virtual Hairstyle
    selectHairStyle, captureHairPhoto, stopHairSession, setHairErase, setHairShapeFilter, setHairSize, setHairColor, tryHairStyle,
    setHairOffset, useScanPhotoForHair, openHairPhotoPicker, handleHairPhotoUpload, startHairCamera, snapHairPhoto, resetHairPhoto,
    // history
    saveToDB, removeHistoryRow, filterHistory, exportCSV,
    // profile
    saveProfile, setAccent, approveBarberApplication, rejectBarberApplication,
    // feedback / contact admin
    submitFeedback, updateFeedbackCount, loadAdminFeedback, setFeedbackStatus,
    // booking
    showBookingTab,useBookingLocation,searchBarberShops,openShopBooking,loadBookingSlots,selectBookingSlot,submitBarberBooking,cancelMyBooking,fillNewShopLocation,submitNewShop,selectManagedShop,addShopBarber,addShopService,saveShopHours,manageBookingAction,openCustomerShopChat,openShopChat,sendShopChat,closeShopChat
});

// ---- On page load: if a JWT is already stored (from a previous visit),
// restore that session and skip straight past the login screen. Otherwise
// applyRolePermissions() just sets up the guest-level UI on the login screen. ----
restoreSessionOnLoad();
