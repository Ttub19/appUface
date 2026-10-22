/**
 * Hairstyle catalogue for the real-time try-on.
 *
 * Every asset is an original vector drawing (no photos of real people) on a
 * shared 400×560 canvas laid out around the same reference face:
 *   - hairline / landmark 10 anchor at (200, 168)
 *   - temple-to-temple width 200 (x = 100 … 300), chin at y ≈ 423
 * so one set of placement maths fits every style. To add a style, draw it on
 * that template (SVG or a transparent PNG exported at 800×1120 / 2x) and add
 * an entry here — `src` can point at either format.
 *
 * `faceShapes` uses the same ids as data.js / the backend classifier.
 * `depth` = how far the hair mass sits behind the forehead, as a fraction of
 * head width (used to counter-shift the asset when the head turns).
 */
export const HAIR_ASSET_VIEWBOX = Object.freeze({ width: 400, height: 560 });

const BASE = Object.freeze({ anchor: { x: 200, y: 168 }, headWidth: 200, depth: .4 });

export const HAIR_STYLES = Object.freeze([
    {
        ...BASE, id: "pompadour", name: "Pompadour", gender: "male",
        faceShapes: ["oval", "square"],
        desc: "เปิดหน้าผาก เพิ่มวอลลุ่มด้านบน ช่วยบาลานซ์กรามเหลี่ยม",
        src: "assets/hair/pompadour.svg",
    },
    {
        ...BASE, id: "quiff", name: "Quiff", gender: "male",
        faceShapes: ["round", "oval"],
        desc: "ยกผมด้านหน้าให้สูง ทำให้หน้ากลมดูยาวขึ้น",
        src: "assets/hair/quiff.svg",
    },
    {
        ...BASE, id: "side-part", name: "Side Part", gender: "male",
        faceShapes: ["heart", "oblong"],
        desc: "แสกข้างเรียบ ไม่เพิ่มความสูง เหมาะกับหน้ายาว/หน้าผากกว้าง",
        src: "assets/hair/side-part.svg", depth: .35,
    },
    {
        ...BASE, id: "long-layers", name: "Long Layers", gender: "female",
        faceShapes: ["oval", "heart"],
        desc: "ผมยาวไล่เลเยอร์กรอบหน้า เพิ่มน้ำหนักช่วงล่างให้หน้ารูปหัวใจ",
        src: "assets/hair/long-layers.svg", depth: .3,
    },
    {
        ...BASE, id: "bangs-bob", name: "Bob + Bangs", gender: "female",
        faceShapes: ["oblong", "square", "round"],
        desc: "บ๊อบระดับคางกับหน้าม้าตรง ตัดความยาวใบหน้า",
        src: "assets/hair/bangs-bob.svg", depth: .3,
    },
]);

export function getHairStyle(id) {
    return HAIR_STYLES.find((s) => s.id === id) || HAIR_STYLES[0];
}

export function stylesForFaceShape(shape) {
    return shape && shape !== "all" ? HAIR_STYLES.filter((s) => s.faceShapes.includes(shape)) : [...HAIR_STYLES];
}

const imageCache = new Map();

/** Loads (and caches) a style's image. Resolves to an HTMLImageElement ready for drawImage. */
export function loadHairImage(style) {
    if (!imageCache.has(style.id)) {
        const promise = new Promise((resolve, reject) => {
            const img = new Image();
            img.decoding = "async";
            img.onload = () => resolve(img);
            img.onerror = () => { imageCache.delete(style.id); reject(new Error(`โหลดภาพทรงผม ${style.name} ไม่สำเร็จ`)); };
            img.src = style.src;
        });
        imageCache.set(style.id, promise);
    }
    return imageCache.get(style.id);
}
