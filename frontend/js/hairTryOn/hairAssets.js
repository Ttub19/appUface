/**
 * Hairstyle catalogue for the real-time try-on.
 *
 * Every style is a photo-real hairpiece cut out of a generated product shot
 * (transparent WebP in assets/hair/real/). Each image carries its own
 * calibration, measured from the face opening in the picture:
 *   - `viewBox`: the image's pixel size (it is drawn 1:1 in its own pixels)
 *   - `anchor`: the hairline point that sits on MediaPipe landmark 10. For
 *     styles with bangs the real hairline is hidden, so it is placed a fixed
 *     distance above the bottom of the fringe.
 *   - `faceWidthPx`: width of the face opening at temple level = the width
 *     the visible face should fill.
 *
 * `faceShapes` uses the same ids as data.js / the backend classifier.
 * `categories` drive the picker's chips (see HAIR_CATEGORIES).
 * `depth` = how far the hair mass sits behind the forehead, as a fraction of
 * head width (used to counter-shift the asset when the head turns).
 */

/** Template the original vector styles were drawn on (kept for any SVG style added later). */
export const HAIR_ASSET_VIEWBOX = Object.freeze({ width: 400, height: 560 });

/**
 * Live-camera calibration (from a real try-on screenshot):
 * - Size: a face opening of N px should be drawn where the temple landmarks
 *   (127/356) are 0.92·N apart. (0.8 from the glasses fit was 15% too big for
 *   hair — the user had to drag the size slider to its 75% minimum.)
 * - Height: MediaPipe landmark 10 is on the mid-forehead, not at the real
 *   hairline: it sits 0.235 face-widths below it. Without this every style
 *   sat too low and bangs covered the eyes.
 */
export const HAIR_LANDMARK_WIDTH_RATIO = .92;
export const LANDMARK10_BELOW_HAIRLINE = .235;

export const HAIR_CATEGORIES = Object.freeze([
    { id: "all", label: "ทั้งหมด" },
    { id: "men", label: "ทรงผู้ชาย" },
    { id: "women", label: "ทรงผู้หญิง" },
    { id: "short", label: "ผมสั้น" },
    { id: "medium", label: "ผมประบ่า" },
    { id: "long", label: "ผมยาว" },
    { id: "bangs", label: "มีหน้าม้า" },
]);

const style = ({ calib, ...s }) => Object.freeze({
    depth: .4,
    ...s,
    src: `assets/hair/real/${s.id}.webp`,
    viewBox: Object.freeze({ width: calib[0], height: calib[1] }),
    // calib[3] is the hairline; the point that goes on landmark 10 is a bit lower
    anchor: Object.freeze({ x: calib[2], y: calib[3] + LANDMARK10_BELOW_HAIRLINE * calib[4] }),
    hairlineY: calib[3],
    faceWidthPx: calib[4],
    headWidth: calib[4] * HAIR_LANDMARK_WIDTH_RATIO,
});

// calib: [width, height, anchorX, anchorY, faceWidthPx] — from the cut-out pipeline
export const HAIR_STYLES = Object.freeze([
    // ---- men ----
    style({ id: "pompadour", name: "Pompadour", gender: "male", categories: ["men", "short"], calib: [491, 529, 244.5, 250, 364],
        faceShapes: ["oval", "square"], desc: "เปิดหน้าผาก เพิ่มวอลลุ่มด้านบน ช่วยบาลานซ์กรามเหลี่ยม" }),
    style({ id: "quiff", name: "Quiff", gender: "male", categories: ["men", "short"], calib: [495, 496, 251, 236, 360],
        faceShapes: ["round", "oval"], desc: "ยกผมด้านหน้าให้สูง ทำให้หน้ากลมดูยาวขึ้น" }),
    style({ id: "side-part", name: "Side Part", gender: "male", categories: ["men", "short"], depth: .35, calib: [489, 477, 242.5, 194, 364],
        faceShapes: ["heart", "oblong"], desc: "แสกข้างเรียบ ไม่เพิ่มความสูง เหมาะกับหน้ายาว/หน้าผากกว้าง" }),
    style({ id: "two-block", name: "Two Block", gender: "male", categories: ["men", "short", "bangs"], calib: [495, 405, 244.5, 221, 360],
        faceShapes: ["oval", "oblong", "heart"], desc: "ทรงเกาหลี ด้านบนยาวปรกหน้าผาก ด้านข้างสั้น" }),
    style({ id: "middle-part", name: "Middle Part", gender: "male", categories: ["men", "short"], calib: [508, 550, 254.8, 135.8, 329],
        faceShapes: ["oval", "square", "heart"], desc: "แสกกลางเปิดหน้าผากเป็นทรงม่าน ช่วยให้หน้าดูเรียว" }),
    style({ id: "comma-hair", name: "Comma Hair", gender: "male", categories: ["men", "short", "bangs"], calib: [482, 463, 241, 109.2, 339],
        faceShapes: ["round", "oval", "oblong"], desc: "หน้าม้าปัดข้างม้วนปลายเป็นรูปลูกน้ำ ตัดความกลมของหน้า" }),
    style({ id: "french-crop", name: "French Crop", gender: "male", categories: ["men", "short", "bangs"], calib: [451, 399, 225.5, 193.7, 358],
        faceShapes: ["oval", "square", "oblong"], desc: "หน้าม้าสั้นตัดตรง ด้านข้างไล่สั้น ดูแลง่าย" }),
    style({ id: "messy-fringe", name: "Messy Fringe", gender: "male", categories: ["men", "short", "bangs"], calib: [526, 548, 261, 214, 330],
        faceShapes: ["oval", "oblong", "heart"], desc: "ผมยุ่งเป็นธรรมชาติ หน้าม้ายาวปรกหน้าผาก ลุคสบาย ๆ" }),
    style({ id: "slick-back", name: "Slick Back", gender: "male", categories: ["men", "short"], calib: [494, 487, 246, 196, 368],
        faceShapes: ["oval", "round", "square"], desc: "เสยหลังเรียบเงา เปิดหน้าผาก ทำให้หน้าดูยาวขึ้น" }),
    style({ id: "buzz-cut", name: "Buzz Cut", gender: "male", categories: ["men", "short"], calib: [468, 425, 234.5, 171, 376.7],
        faceShapes: ["oval", "square"], desc: "สั้นเกรียนทั้งหัว โชว์โครงหน้าชัด" }),
    // ---- women ----
    style({ id: "long-layers", name: "Long Layers", gender: "female", categories: ["women", "long"], depth: .3, calib: [508, 1203, 261.5, 107, 207],
        faceShapes: ["oval", "heart"], desc: "ผมยาวไล่เลเยอร์กรอบหน้า เพิ่มน้ำหนักช่วงล่างให้หน้ารูปหัวใจ" }),
    style({ id: "long-wavy", name: "Long Wavy", gender: "female", categories: ["women", "long"], depth: .3, calib: [558, 1209, 291, 137, 234.2],
        faceShapes: ["square", "oblong", "oval"], desc: "ผมยาวลอนคลื่นแสกข้าง ช่วยลดความเหลี่ยมของกราม" }),
    style({ id: "long-curly", name: "Long Curly", gender: "female", categories: ["women", "long"], depth: .3, calib: [556, 1204, 286.5, 110, 207],
        faceShapes: ["oblong", "square", "heart"], desc: "ผมยาวลอนดัดแน่น เพิ่มวอลลุ่มด้านข้าง เหมาะกับหน้ายาว" }),
    style({ id: "long-curtain", name: "Long + Curtain Bangs", gender: "female", categories: ["women", "long", "bangs"], depth: .3, calib: [475, 1304, 237, 143.6, 243],
        faceShapes: ["oblong", "heart", "square"], desc: "ผมยาวตรงกับหน้าม้าแสกกลาง กรอบหน้าให้ดูเรียว" }),
    style({ id: "hime-cut", name: "Hime Cut", gender: "female", categories: ["women", "long", "bangs"], depth: .3, calib: [473, 1309, 237, 159, 225],
        faceShapes: ["round", "oblong", "oval"], desc: "หน้าม้าตรงกับผมข้างแก้มตัดเสมอ ช่วยให้แก้มดูเรียว" }),
    style({ id: "bangs-bob", name: "Bob + Bangs", gender: "female", categories: ["women", "medium", "bangs"], depth: .3, calib: [464, 576, 234, 148.2, 266],
        faceShapes: ["oblong", "square", "round"], desc: "บ๊อบระดับคางกับหน้าม้าตรง ตัดความยาวใบหน้า" }),
    style({ id: "lob-curtain", name: "Lob + Curtain Bangs", gender: "female", categories: ["women", "medium", "bangs"], depth: .3, calib: [536, 712, 271, 159.5, 256],
        faceShapes: ["round", "square", "heart"], desc: "ผมยาวประบ่ากับหน้าม้าแสกกลาง กรอบหน้าดูนุ่ม" }),
    style({ id: "shaggy-bob", name: "Shaggy Bob", gender: "female", categories: ["women", "medium", "bangs"], depth: .3, calib: [554, 555, 275, 123.5, 258],
        faceShapes: ["oval", "square", "round"], desc: "บ๊อบซอยเป็นชั้นกับหน้าม้าบางเบา ลุคสบาย ๆ" }),
    style({ id: "wolf-cut", name: "Wolf Cut", gender: "female", categories: ["women", "medium", "bangs"], depth: .3, calib: [503, 800, 246.5, 200.3, 258],
        faceShapes: ["oval", "square", "heart"], desc: "ไล่เลเยอร์ฟูด้านบน ปลายซอยแตก ลุคเท่สบาย ๆ" }),
    style({ id: "pixie-cut", name: "Pixie Cut", gender: "female", categories: ["women", "short", "bangs"], calib: [406, 397, 208, 175.4, 265],
        faceShapes: ["oval", "heart"], desc: "ผมสั้นพิกซี่ หน้าม้าปัดข้าง โชว์โครงหน้าและลำคอ" }),
]);

export function getHairStyle(id) {
    return HAIR_STYLES.find((s) => s.id === id) || HAIR_STYLES[0];
}

export function stylesForFaceShape(shape) {
    return shape && shape !== "all" ? HAIR_STYLES.filter((s) => s.faceShapes.includes(shape)) : [...HAIR_STYLES];
}

/**
 * Best styles for a face shape, best first: the order of a style's `faceShapes`
 * says how strongly it suits each shape (first = primary recommendation).
 * `gender` ("male"/"female") keeps to that gender's styles when any match.
 */
export function recommendHairStyles({ shape, gender = null, limit = 4 } = {}) {
    if (!shape || shape === "all") return [];
    const matches = HAIR_STYLES.filter((s) => s.faceShapes.includes(shape));
    const byGender = gender ? matches.filter((s) => s.gender === gender) : matches;
    const pool = byGender.length ? byGender : matches;
    return pool
        .map((s, order) => ({ s, rank: s.faceShapes.indexOf(shape), order }))
        .sort((a, b) => a.rank - b.rank || a.order - b.order)
        .slice(0, limit)
        .map(({ s }) => s);
}

/** Category chip that matches a gender preference ("male" -> "men"). */
export const genderCategory = (gender) => (gender === "male" ? "men" : gender === "female" ? "women" : "all");

/** Styles matching both the face-shape filter and the category chip ("all" = no filter). */
export function filterHairStyles({ shape = "all", category = "all" } = {}) {
    return stylesForFaceShape(shape).filter((s) => category === "all" || s.categories.includes(category));
}

const imageCache = new Map();

/** Loads (and caches) a style's image. Resolves to an HTMLImageElement ready for drawImage. */
export function loadHairImage(style) {
    if (!imageCache.has(style.id)) {
        const promise = new Promise((resolve, reject) => {
            const img = new Image();
            img.decoding = "async";
            img.onload = () => (img.decode ? img.decode().catch(() => {}) : Promise.resolve()).then(() => resolve(img));
            img.onerror = () => { imageCache.delete(style.id); reject(new Error(`โหลดภาพทรงผม ${style.name} ไม่สำเร็จ`)); };
            img.src = style.src;
        });
        imageCache.set(style.id, promise);
    }
    return imageCache.get(style.id);
}

/** Warms the cache for the given styles in the background so switching is instant. */
export function preloadHairImages(styles) {
    styles.forEach((s) => loadHairImage(s).catch(() => {}));
}
