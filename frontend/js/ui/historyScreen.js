import { appState } from "../state.js";
import { showToast } from "./toast.js";
import { navigateTo } from "./navigation.js";
import { apiClient } from "../apiClient.js";

export async function saveToDB() {
    if (appState.currentRole === "guest") {
        showToast("ต้องเข้าสู่ระบบเป็นสมาชิกก่อนจึงจะบันทึกประวัติได้", "error");
        return;
    }
    try {
        await apiClient.createScan({
            face_shape: appState.resultObj.id,
            face_shape_name: appState.resultObj.name,
            confidence: appState.lastConfidence,
            gender_pref: document.getElementById("gender-pref").value,
            image_data_url: appState.selectedImg,
        });
        showToast("บันทึกข้อมูลเข้าสู่ตาราง 'face_analytics' ใน PostgreSQL สำเร็จ", "success");
        navigateTo("screen-history");
    } catch (err) {
        console.error(err);
        showToast(err.message || "บันทึกประวัติไม่สำเร็จ กรุณาลองใหม่อีกครั้ง", "error");
    }
}

/** Loads the current user's scan history from the backend and rebuilds the table. Called on entering the screen. */
export async function renderHistoryTable() {
    if (appState.currentRole === "guest") {
        appState.historyCount = 0;
        updateHistoryEmptyState();
        return;
    }
    try {
        const scans = await apiClient.listScans();
        appState.historyCount = scans.length;
        const body = document.getElementById("history-body");
        body.innerHTML = "";
        scans.forEach((scan) => {
            const row = document.createElement("tr");
            row.dataset.shape = scan.face_shape;
            row.dataset.id = scan.id;
            row.innerHTML = `
                <td><img src="${scan.image_data_url || ""}" alt="" class="thumb"></td>
                <td class="font-semibold text-[color:var(--ink)]">${scan.face_shape_name}</td>
                <td><span class="chip chip-accent tabular-nums">${scan.confidence.toFixed(1)}%</span></td>
                <td class="text-[13px] text-[color:var(--ink-3)] tabular-nums">${new Date(scan.created_at).toLocaleString("th-TH")}</td>
                <td><button class="icon-btn icon-btn-plain hover:!text-[color:var(--danger)]" onclick="removeHistoryRow(this)" aria-label="ลบประวัตินี้" title="ลบ"><i class="fa-regular fa-trash-can"></i></button></td>
            `;
            body.appendChild(row);
        });
        updateHistoryEmptyState();
    } catch (err) {
        console.error(err);
        showToast(err.message || "ไม่สามารถโหลดประวัติได้", "error");
    }
}

/** Removes a row from the history table (bound to each row's trash button) and deletes it on the backend. */
export async function removeHistoryRow(buttonEl) {
    const row = buttonEl.closest("tr");
    const id = row?.dataset.id;
    if (!id) return;
    try {
        await apiClient.deleteScan(id);
        row.remove();
        appState.historyCount = Math.max(0, appState.historyCount - 1);
        updateHistoryEmptyState();
    } catch (err) {
        console.error(err);
        showToast(err.message || "ลบประวัติไม่สำเร็จ กรุณาลองใหม่อีกครั้ง", "error");
    }
}

export function updateHistoryEmptyState() {
    document.getElementById("history-empty").classList.toggle("hidden", appState.historyCount > 0);
    filterHistory();
}

export function filterHistory() {
    const query = document.getElementById("history-search").value.trim().toLowerCase();
    const shapeFilter = document.getElementById("history-filter").value;
    const rows = document.querySelectorAll("#history-body tr");
    let visibleCount = 0;

    rows.forEach((row) => {
        const nameCell = row.children[1] ? row.children[1].innerText.toLowerCase() : "";
        const matchesQuery = !query || nameCell.includes(query);
        const matchesShape = !shapeFilter || row.dataset.shape === shapeFilter;
        const show = matchesQuery && matchesShape;
        row.classList.toggle("hidden", !show);
        if (show) visibleCount++;
    });

    document.getElementById("history-no-match").classList.toggle("hidden", !(appState.historyCount > 0 && visibleCount === 0));
}

export async function exportCSV() {
    try {
        const blob = await apiClient.exportScansCsv();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "scan_history.csv";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        showToast("Export ไฟล์ CSV สำเร็จ", "success");
    } catch (err) {
        console.error(err);
        showToast(err.message || "ไม่สามารถ export CSV ได้", "error");
    }
}
