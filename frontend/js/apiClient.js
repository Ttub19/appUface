/**
 * apiClient.js — thin fetch wrapper for the FastAPI backend.
 *
 * Base URL resolution:
 *   - Defaults to "/api", which works out of the box under the bundled
 *     docker-compose setup (nginx reverse-proxies /api/* to the backend,
 *     so the front-end and API share one origin — no CORS needed).
 *   - Can be overridden by setting `window.__API_BASE_URL__` in a small
 *     inline <script> before app.js loads — useful when running the
 *     front-end standalone (e.g. `npx serve`) against a backend on a
 *     different port, e.g. `http://localhost:8000`.
 */
const API_BASE_URL = window.__API_BASE_URL__ || "/api";
const TOKEN_STORAGE_KEY = "faceShapeAi.accessToken";
function wsUrl(path){const u=new URL(`${API_BASE_URL}${path}`,window.location.origin);u.protocol=u.protocol==="https:"?"wss:":"ws:";const t=getToken();if(t)u.searchParams.set("token",t);return u.toString()}

function getToken() {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
}

function setToken(token) {
    if (token) localStorage.setItem(TOKEN_STORAGE_KEY, token);
    else localStorage.removeItem(TOKEN_STORAGE_KEY);
}

/**
 * @param {string} method
 * @param {string} path - e.g. "/scans"
 * @param {Object} [options]
 * @param {any} [options.body] - JSON-serializable request body
 * @param {boolean} [options.auth=true] - attach the Authorization header if a token is stored
 * @returns {Promise<any>} parsed JSON body, or null for 204 No Content
 * @throws {Error} with a user-facing Thai message on any non-2xx response
 */
async function request(method, path, { body, auth = true } = {}) {
    const headers = { "Content-Type": "application/json" };
    if (auth) {
        const token = getToken();
        if (token) headers.Authorization = `Bearer ${token}`;
    }

    let response;
    try {
        response = await fetch(`${API_BASE_URL}${path}`, {
            method,
            headers,
            body: body !== undefined ? JSON.stringify(body) : undefined,
        });
    } catch (networkErr) {
        console.error("Network error calling", path, networkErr);
        throw new Error("เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ เซิร์ฟเวอร์อาจกำลังเริ่มทำงาน กรุณาลองใหม่อีกครั้งใน 1 นาที");
    }

    if (!response.ok) {
        throw new Error(await extractErrorMessage(response));
    }
    if (response.status === 204) return null;
    return response.json();
}

async function extractErrorMessage(response) {
    try {
        const data = await response.json();
        if (typeof data?.detail === "string") return data.detail;
        if (Array.isArray(data?.detail)) {
            // FastAPI validation error format: [{ loc, msg, type }, ...]
            return data.detail.map((d) => d.msg).join(", ");
        }
    } catch {
        // response body wasn't JSON — fall through to the generic message
    }
    return `เกิดข้อผิดพลาด (HTTP ${response.status})`;
}

export const apiClient = {
    getToken,
    setToken,
    isAuthenticated() {
        return Boolean(getToken());
    },
    logout() {
        setToken(null);
    },

    // ---- Feedback / contact admin ----
    sendFeedback(payload) {
        return request("POST", "/feedback", { body: payload });
    },
    myFeedback() {
        return request("GET", "/feedback/mine");
    },
    adminFeedback(status = null) {
        return request("GET", status ? `/admin/feedback?status=${encodeURIComponent(status)}` : "/admin/feedback");
    },
    updateFeedback(id, payload) {
        return request("PATCH", `/admin/feedback/${encodeURIComponent(id)}`, { body: payload });
    },

    // ---- Auth ----
    register(payload) {
        return request("POST", "/auth/register", { body: payload, auth: false });
    },

    async login(email, password) {
        const form = new URLSearchParams();
        form.set("username", email);
        form.set("password", password);

        let response;
        try {
            response = await fetch(`${API_BASE_URL}/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: form,
            });
        } catch (networkErr) {
            console.error("Network error during login", networkErr);
            throw new Error("เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ กรุณาตรวจสอบการเชื่อมต่อเครือข่าย");
        }
        if (!response.ok) {
            throw new Error(await extractErrorMessage(response));
        }
        const data = await response.json();
        setToken(data.access_token);
        return data;
    },

    me() {
        return request("GET", "/auth/me");
    },

    // ---- Scans ----
    predictFaceShape(imageDataUrl) {
        return request("POST", "/scans/predict", { body: { image_data_url: imageDataUrl }, auth: false });
    },
    createScan(payload) {
        return request("POST", "/scans", { body: payload });
    },
    listScans({ q, shape } = {}) {
        const params = new URLSearchParams();
        if (q) params.set("q", q);
        if (shape) params.set("shape", shape);
        const qs = params.toString();
        return request("GET", `/scans${qs ? `?${qs}` : ""}`);
    },
    deleteScan(id) {
        return request("DELETE", `/scans/${id}`);
    },
    async exportScansCsv() {
        const token = getToken();
        const response = await fetch(`${API_BASE_URL}/scans/export.csv`, {
            headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!response.ok) throw new Error(await extractErrorMessage(response));
        return response.blob();
    },

    // ---- Favorites ----
    createFavorite(payload) {
        return request("POST", "/favorites", { body: payload });
    },
    listFavorites() {
        return request("GET", "/favorites");
    },
    deleteFavorite(id) {
        return request("DELETE", `/favorites/${id}`);
    },

    // ---- Golden Ratio ----
    saveGoldenRatio(payload) {
        return request("POST", "/golden-ratio", { body: payload });
    },
    listGoldenRatio() {
        return request("GET", "/golden-ratio");
    },

    // ---- Profile ----
    getProfile() {
        return request("GET", "/profile");
    },
    updateProfile(payload) {
        return request("PUT", "/profile", { body: payload });
    },

    // ---- Barber shop booking ----
    listShops(p={}) { const q=new URLSearchParams(); for(const [k,v] of Object.entries({q:p.q,specialty:p.specialty,latitude:p.latitude,longitude:p.longitude,radius_km:p.radiusKm})) if(v!==undefined&&v!==null&&v!=="")q.set(k,v); return request("GET",`/shops?${q}`,{auth:false}); },
    getShop(id){return request("GET",`/shops/${id}`);}, myShops(){return request("GET","/shops/mine");}, createShop(p){return request("POST","/shops",{body:p});},
    addShopStaff(id,p){return request("POST",`/shops/${id}/staff`,{body:p});}, createShopService(id,p){return request("POST",`/shops/${id}/services`,{body:p});}, setShopBusinessHours(id,hours){return request("PUT",`/shops/${id}/business-hours`,{body:{hours}});},
    getShopAvailability(id,p){const q=new URLSearchParams({date:p.date,service_id:p.serviceId,barber_member_id:p.barberMemberId});return request("GET",`/shops/${id}/availability?${q}`,{auth:false});},
    createBooking(p){return request("POST","/bookings",{body:p});}, listMyBookings(){return request("GET","/bookings/mine");}, listShopBookings(id){return request("GET",`/bookings/shop/${id}`);},
    confirmBooking(id){return request("POST",`/bookings/${id}/confirm`);}, rejectBooking(id,reason){return request("POST",`/bookings/${id}/reject`,{body:{reason}});}, cancelBooking(id,reason){return request("POST",`/bookings/${id}/cancel`,{body:{reason}});}, completeBooking(id){return request("POST",`/bookings/${id}/complete`);},
    listShopConversations(id){return request("GET",`/shop-chat/shops/${id}/conversations`);}, getShopMessages(id,c){return request("GET",`/shop-chat/shops/${id}/${c}`);}, postShopMessage(id,c,body){return request("POST",`/shop-chat/shops/${id}/${c}`,{body:{body}});}, openShopChatSocket(id,c){return new WebSocket(wsUrl(`/shop-chat/ws/${id}/${c}`));},

    // ---- Admin ----
    getAdminStats() { return request("GET", "/admin/stats"); },
    getBarberApplications(status="pending") { return request("GET",`/admin/barber-applications?status=${encodeURIComponent(status)}`); },
    approveBarberApplication(id,note=null) { return request("POST",`/admin/barber-applications/${id}/approve`,{body:{review_note:note}}); },
    rejectBarberApplication(id,note) { return request("POST",`/admin/barber-applications/${id}/reject`,{body:{review_note:note}}); },
};
