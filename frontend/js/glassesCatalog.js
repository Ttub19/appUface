/**
 * Glasses catalog for the try-on picker: display names, categories and the
 * aliases for style ids used before the photo-based frames replaced the
 * hand-drawn SVG ones (still referenced by the face-shape guide data).
 * Pure data + helpers, no DOM, so it can be unit tested in Node.
 */

export const GLASSES_CATEGORIES = Object.freeze([
    { id: "all", label: "ทั้งหมด" },
    { id: "square", label: "ทรงเหลี่ยม" },
    { id: "round", label: "ทรงกลม / รี" },
    { id: "cateye", label: "Cat Eye" },
    { id: "aviator", label: "Aviator / สปอร์ต" },
    { id: "browline", label: "Browline" },
    { id: "geometric", label: "เรขาคณิต" },
    { id: "rimless", label: "ไร้กรอบ" },
]);

/** Listed in display order; the first category of each item is its primary one. */
export const GLASSES_CATALOG = Object.freeze([
    { id: "classic-tortoise", name: "Classic", desc: "ทรงเหลี่ยมคลาสสิก กระดองเต่า", categories: ["square"] },
    { id: "wayfarer-black", name: "Wayfarer", desc: "กรอบหนาโมเดิร์น สีดำ", categories: ["square"] },
    { id: "square-black", name: "Square ดำ", desc: "ทรงเหลี่ยมคม มั่นใจ", categories: ["square"] },
    { id: "square-green", name: "Square เขียว", desc: "ทรงเหลี่ยม สีเขียว", categories: ["square"] },
    { id: "square-bold-tortoise", name: "Square Bold", desc: "ทรงเหลี่ยมหนา กระดองเต่า", categories: ["square"] },
    { id: "square-oversized-tortoise", name: "Square Oversized", desc: "ทรงเหลี่ยมใหญ่", categories: ["square"] },
    { id: "clear-square", name: "Square ใส", desc: "ทรงเหลี่ยม กรอบใส", categories: ["square"] },
    { id: "colorblock-square", name: "Square สองโทน", desc: "ทรงเหลี่ยม สองสี", categories: ["square"] },
    { id: "rect-thin-metal", name: "Rectangle เมทัล", desc: "สี่เหลี่ยมผืนผ้า ลวดบาง", categories: ["square"] },
    { id: "round-tortoise", name: "Round กระดองเต่า", desc: "ทรงกลม วินเทจ", categories: ["round"] },
    { id: "round-gold-wire", name: "Round ทอง", desc: "ทรงกลม ลวดบาง", categories: ["round"] },
    { id: "round-gunmetal", name: "Round เมทัล", desc: "ทรงกลม สีเมทัลลิค", categories: ["round"] },
    { id: "oval-gold-wire", name: "Oval ทอง", desc: "ทรงรี ลวดบาง", categories: ["round"] },
    { id: "cateye-black", name: "Cat Eye", desc: "ปลายเชิด สีดำ", categories: ["cateye"] },
    { id: "cateye-red", name: "Cat Eye แดง", desc: "ปลายเชิด สีแดง", categories: ["cateye"] },
    { id: "cateye-amber", name: "Cat Eye ส้ม", desc: "ปลายเชิด สีอำพัน", categories: ["cateye"] },
    { id: "aviator-gold", name: "Aviator", desc: "ทรงหยดน้ำ สีทอง", categories: ["aviator"] },
    { id: "sport-wrap", name: "Sport Wrap", desc: "ทรงสปอร์ตโค้งรอบ", categories: ["aviator"] },
    { id: "browline-gold", name: "Browline ทอง", desc: "บนหนา ล่างบาง", categories: ["browline"] },
    { id: "browline-gold2", name: "Browline ทองเข้ม", desc: "บนหนา ล่างบาง", categories: ["browline"] },
    { id: "browline-silver", name: "Browline เงิน", desc: "บนหนา ล่างบาง", categories: ["browline"] },
    { id: "hexagon-gold", name: "Hexagon ทอง", desc: "หกเหลี่ยม ลวดบาง", categories: ["geometric"] },
    { id: "hexagon-gunmetal", name: "Hexagon เมทัล", desc: "หกเหลี่ยม สีเมทัลลิค", categories: ["geometric"] },
    { id: "hexagon-tortoise", name: "Hexagon กระดองเต่า", desc: "หกเหลี่ยม กรอบหนา", categories: ["geometric"] },
    { id: "rimless-silver-bar", name: "Rimless เงิน", desc: "ไร้กรอบ บาร์บนบาง", categories: ["rimless", "round"] },
    { id: "rimless-gunmetal-square", name: "Rimless เหลี่ยม", desc: "ไร้กรอบ ทรงเหลี่ยม", categories: ["rimless", "square"] },
    { id: "rimless-gunmetal-aviator", name: "Rimless Aviator", desc: "ไร้กรอบ ทรงหยดน้ำ", categories: ["rimless", "aviator"] },
    { id: "rimless-rosegold-cateye", name: "Rimless Cat Eye", desc: "ไร้กรอบ สีโรสโกลด์", categories: ["rimless", "cateye"] },
]);

export const DEFAULT_GLASSES_STYLE = "classic-tortoise";

/** Style ids from the old hand-drawn set -> closest photo frame. */
export const LEGACY_GLASSES_ALIASES = Object.freeze({
    classic: "classic-tortoise",
    wayfarer: "wayfarer-black",
    aviator: "aviator-gold",
    "cat-eye": "cateye-black",
    oversized: "square-oversized-tortoise",
    round: "round-tortoise",
    square: "square-black",
    browline: "browline-gold",
    hexagon: "hexagon-tortoise",
});

const BY_ID = new Map(GLASSES_CATALOG.map((item) => [item.id, item]));

export function getGlassesItem(id) {
    return BY_ID.get(id) || null;
}

/** Maps any known id (current or legacy) to a catalog id; unknown -> default. */
export function resolveGlassesStyle(id) {
    if (BY_ID.has(id)) return id;
    const alias = LEGACY_GLASSES_ALIASES[id];
    return alias && BY_ID.has(alias) ? alias : DEFAULT_GLASSES_STYLE;
}

export function glassesInCategory(categoryId) {
    if (!categoryId || categoryId === "all") return [...GLASSES_CATALOG];
    return GLASSES_CATALOG.filter((item) => item.categories.includes(categoryId));
}

export function isInCategory(id, categoryId) {
    if (!categoryId || categoryId === "all") return BY_ID.has(id);
    return Boolean(BY_ID.get(id)?.categories.includes(categoryId));
}
