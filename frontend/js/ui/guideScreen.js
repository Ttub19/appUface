import { faceData, faceGuide } from "../data.js";
import { appState } from "../state.js";
import { navigateTo, syncNavMirrors } from "./navigation.js";
import { setHairShapeFilter } from "./hairScreen.js";
import { selectGlassesStyle } from "./glassesScreen.js";

const SHORT_NAME = { oval: "หน้ารูปไข่", round: "หน้ากลม", square: "หน้าเหลี่ยม", heart: "หน้ารูปหัวใจ", oblong: "หน้ายาว" };
const MEASURE_LABEL = { A: "หน้าผาก", B: "โหนกแก้ม", C: "กราม", D: "ความยาวหน้า" };

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/** Illustrated reference image per shape (assets/faceshapes/<id>.png), replacing
 *  the earlier code-drawn SVG diagrams — reuses the "face-diagram" CSS class so
 *  it gets the same sizing rules (width:100%, max-width in the detail view). */
const faceDiagramImg = (id, alt) => `<img class="face-diagram" src="assets/faceshapes/${id}.png" alt="${esc(alt)}" loading="lazy">`;
const isMine = (id) => appState.resultObj?.id === id;
const mineBadge = (id) => (isMine(id) ? `<span class="chip chip-accent">รูปหน้าของคุณ</span>` : "");

/** Entry from navigation: always lands on the index of the five shapes. */
export function renderGuide() {
    renderGuideIndex();
    showIndex();
}

function renderGuideIndex() {
    const grid = document.getElementById("guide-grid");
    grid.innerHTML = faceData.map((shape) => {
        const g = faceGuide[shape.id];
        return `
            <button type="button" class="guide-card" onclick="openGuideShape('${shape.id}')">
                <span class="guide-card-figure">${faceDiagramImg(shape.id, SHORT_NAME[shape.id])}</span>
                <span class="guide-card-body">
                    <span class="flex items-center gap-2 flex-wrap"><strong>${SHORT_NAME[shape.id]}</strong><em>${g.en}</em>${mineBadge(shape.id)}</span>
                    <span class="guide-card-text">${esc(g.traits[0])}</span>
                    <span class="guide-card-more">อ่านต่อ <i class="fa-solid fa-arrow-right text-xs"></i></span>
                </span>
            </button>`;
    }).join("");
}

function showIndex() {
    document.getElementById("guide-index").classList.remove("hidden");
    document.getElementById("guide-detail").classList.add("hidden");
}

/** Opens the single-page article for one face shape. */
export function openGuideShape(id) {
    const idx = faceData.findIndex((f) => f.id === id);
    if (idx < 0) return;
    if (document.getElementById("screen-guide").classList.contains("hidden")) navigateTo("screen-guide");
    renderGuideIndex();

    const shape = faceData[idx];
    const g = faceGuide[id];
    const prev = faceData[(idx - 1 + faceData.length) % faceData.length];
    const next = faceData[(idx + 1) % faceData.length];
    const glassName = shape.glassStyle.replace("-", " ").replace(/\b\w/g, (c) => c.toUpperCase());

    const hairColumn = (label, h) => `
        <div class="guide-col">
            <h4>${label}</h4>
            <ul class="guide-list is-good">${h.good.map(([t, why]) => `<li><i class="fa-solid fa-check"></i><div><strong>${esc(t)}</strong><span>${esc(why)}</span></div></li>`).join("")}</ul>
            <ul class="guide-list is-avoid">${h.avoid.map(([t, why]) => `<li><i class="fa-solid fa-xmark"></i><div><strong>${esc(t)}</strong><span>${esc(why)}</span></div></li>`).join("")}</ul>
        </div>`;
    const simpleList = (items, good) => `<ul class="guide-list ${good ? "is-good" : "is-avoid"}">${items.map((t) => `<li><i class="fa-solid ${good ? "fa-check" : "fa-xmark"}"></i><div><span class="!text-[15px] !text-[color:var(--ink-2)]">${esc(t)}</span></div></li>`).join("")}</ul>`;

    document.getElementById("guide-detail").innerHTML = `
        <nav class="guide-topbar" aria-label="นำทางคู่มือ">
            <button type="button" class="guide-back" onclick="closeGuideShape()"><i class="fa-solid fa-arrow-left"></i>คู่มือรูปหน้า</button>
            <span class="guide-pager">
                <button type="button" class="icon-btn" onclick="openGuideShape('${prev.id}')" aria-label="ก่อนหน้า: ${SHORT_NAME[prev.id]}"><i class="fa-solid fa-chevron-left"></i></button>
                <span>${idx + 1} / ${faceData.length}</span>
                <button type="button" class="icon-btn" onclick="openGuideShape('${next.id}')" aria-label="ถัดไป: ${SHORT_NAME[next.id]}"><i class="fa-solid fa-chevron-right"></i></button>
            </span>
        </nav>

        <header class="guide-hero">
            <div>
                <p class="home-kicker">รูปหน้าแบบที่ ${idx + 1} จาก ${faceData.length}</p>
                <h1 class="guide-title">${SHORT_NAME[id]} <span>${g.en}</span></h1>
                ${isMine(id) ? `<p class="notice notice-info mt-4"><i class="fa-solid fa-circle-info mt-1"></i><span>ผลวิเคราะห์ล่าสุดของคุณคือ${SHORT_NAME[id]}</span></p>` : ""}
                <p class="guide-summary">${esc(g.summary)}</p>
                <h3 class="guide-h3">จุดสังเกต</h3>
                <ol class="guide-traits">${g.traits.map((t) => `<li>${esc(t)}</li>`).join("")}</ol>
            </div>
            <figure class="guide-figure">
                ${faceDiagramImg(id, `ภาพเค้าโครง${SHORT_NAME[id]} พร้อมเส้นวัด A ถึง D`)}
                <dl class="guide-measures">
                    ${Object.entries(g.measures).map(([k, v]) => `<div><dt><b>${k}</b>${MEASURE_LABEL[k]}</dt><dd>${esc(v)}</dd></div>`).join("")}
                </dl>
            </figure>
        </header>

        <section class="guide-section">
            <h2>เช็กด้วยตัวเองหน้ากระจก</h2>
            <ol class="home-steps !mt-0">
                <li><b>01</b><div><strong>รวบผมให้เห็นกรอบหน้า</strong><span>ยืนตรงหน้ากระจกในที่แสงพอ มองตรง ไม่เงยหรือก้ม</span></div></li>
                <li><b>02</b><div><strong>วัด 4 จุดตามภาพ</strong><span>ใช้สายวัดหรือนิ้ว วัดความกว้างหน้าผาก (A) โหนกแก้ม (B) กราม (C) และความยาวจากไรผมถึงปลายคาง (D)</span></div></li>
                <li><b>03</b><div><strong>เทียบกับลักษณะของรูปหน้านี้</strong><span>${esc(g.verdict)}</span></div></li>
            </ol>
        </section>

        <section class="guide-section">
            <div class="guide-section-head">
                <h2>ทรงผม</h2>
                <button type="button" class="btn btn-primary btn-sm" data-mirror="nav-hairstyle" onclick="tryHairForShape('${id}')"><i class="fa-solid fa-scissors"></i>ลองทรงผมสำหรับ${SHORT_NAME[id]}</button>
            </div>
            <p class="guide-principle"><b>หลักการ</b>${esc(g.principle)}</p>
            <div class="guide-cols">
                ${hairColumn("ผู้ชาย", g.hair.male)}
                ${hairColumn("ผู้หญิง", g.hair.female)}
            </div>
        </section>

        <section class="guide-section">
            <div class="guide-section-head">
                <h2>แว่นตาและเครื่องประดับ</h2>
                <button type="button" class="btn btn-secondary btn-sm" data-mirror="nav-glasses" onclick="tryGlassesForShape('${id}')"><i class="fa-solid fa-glasses"></i>ลองแว่นทรง ${glassName}</button>
            </div>
            <div class="guide-cols">
                <div class="guide-col"><h4>แว่นตา</h4>${simpleList(g.glasses.good, true)}${simpleList(g.glasses.avoid, false)}</div>
                <div class="guide-col"><h4>ต่างหูและเครื่องประดับ</h4>${simpleList(g.jewelry.good, true)}${simpleList(g.jewelry.avoid, false)}</div>
            </div>
        </section>

        <section class="guide-section">
            <h2>หนวดเคราและการแต่งหน้า</h2>
            <div class="guide-cols">
                <div class="guide-col"><h4>หนวดเครา</h4><p class="guide-note">${esc(g.beard)}</p></div>
                <div class="guide-col"><h4>แต่งหน้า</h4><p class="guide-note">${esc(g.makeup)}</p></div>
            </div>
        </section>

        <nav class="guide-footer-nav" aria-label="รูปหน้าอื่น">
            <button type="button" onclick="openGuideShape('${prev.id}')"><span>ก่อนหน้า</span><strong><i class="fa-solid fa-arrow-left text-sm"></i>${SHORT_NAME[prev.id]}</strong></button>
            <button type="button" onclick="openGuideShape('${next.id}')" class="text-right"><span>ถัดไป</span><strong>${SHORT_NAME[next.id]}<i class="fa-solid fa-arrow-right text-sm"></i></strong></button>
        </nav>`;

    document.getElementById("guide-index").classList.add("hidden");
    document.getElementById("guide-detail").classList.remove("hidden");
    syncNavMirrors();
    window.scrollTo({ top: 0 });
}

export function closeGuideShape() {
    showIndex();
    window.scrollTo({ top: 0 });
}

/** Opens the hairstyle try-on pre-filtered to this face shape. */
export function tryHairForShape(id) {
    navigateTo("screen-hairstyle");
    setHairShapeFilter(id);
}

/** Opens the glasses try-on with the style recommended for this face shape. */
export function tryGlassesForShape(id) {
    const shape = faceData.find((f) => f.id === id);
    navigateTo("screen-glasses");
    if (shape) selectGlassesStyle(shape.glassStyle);
}
