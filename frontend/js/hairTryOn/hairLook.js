/**
 * Makes a cut-out hairpiece look like it belongs in the camera frame instead
 * of a sticker on top of it:
 *   - colour: recolour the studio-brown asset to a preset or to the user's own
 *     hair colour (sampled from the segmentation mask), keeping its shading;
 *   - edges: feather the cut-out alpha so strands fade instead of ending hard;
 *   - contact shadow: a soft shadow baked under the hair where it meets the
 *     forehead — the cue that it sits ON the head, not in front of it;
 *   - lighting: scale brightness with the scene, measured from the face skin.
 *
 * Pixel maths (recolorPixels, lightingFactor, quantizeColor) is pure so it runs
 * under `node --test`; prepareHairCanvas needs a DOM canvas.
 */
import { clamp } from "../landmarks.js";

export const HAIR_COLORS = Object.freeze([
    { id: "mine", label: "เหมือนผมฉัน", rgb: null },
    { id: "original", label: "สีของทรง", rgb: null, swatch: [54, 44, 40] },
    { id: "black", label: "ดำ", rgb: [26, 24, 24] },
    { id: "dark-brown", label: "น้ำตาลเข้ม", rgb: [62, 46, 37] },
    { id: "brown", label: "น้ำตาล", rgb: [98, 72, 54] },
    { id: "ash", label: "เทาหม่น", rgb: [116, 110, 104] },
    { id: "blonde", label: "บลอนด์", rgb: [182, 146, 100] },
    { id: "wine", label: "แดงไวน์", rgb: [104, 40, 46] },
]);

export const DEFAULT_HAIR_COLOR = "mine";

export function getHairColor(id) {
    return HAIR_COLORS.find((c) => c.id === id) || HAIR_COLORS[0];
}

const luminance = (r, g, b) => .299 * r + .587 * g + .114 * b;

/** Soft highlight roll-off so light targets (blonde) keep strand detail instead of clipping. */
function softClip(v) {
    return v <= 200 ? v : 200 + 55 * (1 - Math.exp(-(v - 200) / 55));
}

/**
 * Recolours RGBA pixels in place to `target` while keeping the asset's own
 * light/dark strand pattern: each pixel's luminance relative to the asset's
 * mean luminance scales the target colour.
 * @param {Uint8ClampedArray} data RGBA
 * @param {number[]} target [r,g,b]
 */
export function recolorPixels(data, target) {
    let sum = 0, n = 0;
    for (let i = 0; i < data.length; i += 4) {
        if (data[i + 3] < 128) continue;
        sum += luminance(data[i], data[i + 1], data[i + 2]); n++;
    }
    if (!n) return data;
    const mean = Math.max(1, sum / n);
    // Light colours keep less of the dark asset's strand contrast, or blonde looks like metal foil.
    const contrast = .85 - .3 * clamp((luminance(...target) - 60) / 100, 0, 1);
    for (let i = 0; i < data.length; i += 4) {
        if (!data[i + 3]) continue;
        const ratio = 1 + (luminance(data[i], data[i + 1], data[i + 2]) / mean - 1) * contrast;
        for (let c = 0; c < 3; c++) data[i + c] = softClip(target[c] * Math.max(0, ratio));
    }
    return data;
}

/**
 * Brightness multiplier for the hair so it follows the scene exposure.
 * The assets are lit like a studio shot of skin at ~150 luminance; a dim
 * room makes the face (and so the real hair) darker, and the new hair should
 * follow. Square-rooted and clamped: lighting cues, not a full re-exposure.
 */
export function lightingFactor(skinRgb) {
    if (!skinRgb) return 1;
    const l = luminance(skinRgb[0], skinRgb[1], skinRgb[2]);
    return clamp(Math.sqrt(l / 150), .7, 1.15);
}

/** Rounds a sampled colour to a coarse grid so tiny frame-to-frame changes don't rebuild the hair image. */
export function quantizeColor(rgb, step = 12) {
    return rgb ? rgb.map((v) => clamp(Math.round(v / step) * step, 0, 255)) : null;
}

/** Hair colour actually used for a preset id, given the user's sampled hair colour (if any). */
export function resolveHairRgb(colorId, sampledHair) {
    if (colorId === "mine") return quantizeColor(sampledHair);
    return getHairColor(colorId).rgb;
}

function makeCanvas(w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    return c;
}

const filterSupport = (() => {
    let ok = null;
    return () => {
        if (ok === null) {
            try {
                const ctx = makeCanvas(1, 1).getContext("2d");
                ctx.filter = "blur(1px)";
                ok = ctx.filter === "blur(1px)";
            } catch { ok = false; }
        }
        return ok;
    };
})();

/** Feather radius in asset pixels (assets are ~2× the on-screen size, so ~0.6px on screen). */
const FEATHER_PX = 1.4;

/**
 * Builds the image actually drawn for a style: optional recolour, feathered
 * edges and a baked contact shadow. Same pixel grid as the source (anchor
 * coordinates are unchanged); only extra rows are added at the bottom for
 * the shadow of the hair tips.
 * @param {CanvasImageSource & {width:number,height:number}} img
 * @param {{rgb?: number[]|null, faceWidthPx: number}} opts
 * @returns {HTMLCanvasElement}
 */
export function prepareHairCanvas(img, { rgb = null, faceWidthPx }) {
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    const hair = makeCanvas(w, h);
    const hctx = hair.getContext("2d", { willReadFrequently: Boolean(rgb) });
    hctx.drawImage(img, 0, 0, w, h);
    if (rgb) {
        const data = hctx.getImageData(0, 0, w, h);
        recolorPixels(data.data, rgb);
        hctx.putImageData(data, 0, 0);
    }
    const filters = filterSupport();
    if (filters) {
        // alpha × blurred(alpha): solid areas stay solid, the outer strands fade out
        const soft = makeCanvas(w, h);
        const sctx = soft.getContext("2d");
        sctx.filter = `blur(${FEATHER_PX}px)`;
        sctx.drawImage(hair, 0, 0);
        hctx.globalCompositeOperation = "destination-in";
        hctx.drawImage(soft, 0, 0);
        hctx.globalCompositeOperation = "source-over";
    }

    const shadowOffset = Math.round(.025 * faceWidthPx);
    const shadowBlur = .035 * faceWidthPx;
    const pad = filters ? Math.ceil(shadowOffset + 2 * shadowBlur) : 0;
    const out = makeCanvas(w, h + pad);
    const ctx = out.getContext("2d");
    if (filters) {
        ctx.save();
        ctx.filter = `blur(${shadowBlur.toFixed(1)}px) brightness(0)`;
        ctx.globalAlpha = .42;
        ctx.drawImage(hair, 0, shadowOffset);
        ctx.restore();
    }
    ctx.drawImage(hair, 0, 0);
    return out;
}

export { filterSupport as canvasFilterSupported };
