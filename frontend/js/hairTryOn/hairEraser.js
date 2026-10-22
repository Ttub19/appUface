import { clamp } from "../landmarks.js";

const COLOR_EMA = .2;
/** Grow the hair mask by this many pixels (in segmentation-frame px) to catch flyaways the segmenter misses. */
const DILATE_PX = 2;
/** Pixels with (dilated) confidence at or below this are trusted as "not hair" when sampling fill colours. */
const FREE_CONF = .2;
/** Skin samples further than this (RGB distance) from the cheek/forehead tone are brows, eyes or glasses: not used for the forehead fill. */
const SKIN_TOLERANCE = 70;

function smoothstep(edge0, edge1, x) {
    const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
}

function ellipseTester(e) {
    const c = Math.cos(e.angle), s = Math.sin(e.angle), irx = 1 / e.rx, iry = 1 / e.ry;
    return (x, y) => {
        const dx = x - e.cx, dy = y - e.cy;
        const u = (dx * c + dy * s) * irx, v = (-dx * s + dy * c) * iry;
        return u * u + v * v <= 1;
    };
}

function blend(prev, next) {
    if (!next) return prev;
    if (!prev) return next;
    return prev.map((v, i) => v + (next[i] - v) * COLOR_EMA);
}

/** Separable max filter (square dilation) of a w×h float mask. */
export function dilateMask(conf, width, height, radius = DILATE_PX) {
    if (radius <= 0) return conf;
    const tmp = new Float32Array(width * height), out = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
        const row = y * width;
        for (let x = 0; x < width; x++) {
            let m = 0;
            for (let k = Math.max(0, x - radius); k <= Math.min(width - 1, x + radius); k++) if (conf[row + k] > m) m = conf[row + k];
            tmp[row + x] = m;
        }
    }
    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
            let m = 0;
            for (let k = Math.max(0, y - radius); k <= Math.min(height - 1, y + radius); k++) if (tmp[k * width + x] > m) m = tmp[k * width + x];
            out[y * width + x] = m;
        }
    }
    return out;
}

/**
 * Push-pull hole filling: every pixel gets a colour smoothly interpolated
 * from the nearest valid pixels in 2D (Gortler et al., "The Lumigraph", 1996).
 * Unlike a per-row fill this has no streaks, and large holes get a soft
 * gradient of the surrounding colours instead of one flat average.
 * @param {Uint8ClampedArray} frame RGBA source
 * @param {Uint8Array} valid 1 = use this pixel as a sample
 * @param {Uint8Array|null} [need] only these full-resolution pixels have to be filled (saves time)
 * @returns {Float32Array|null} RGB per pixel, or null when nothing is valid
 */
export function pushPullFill(frame, valid, width, height, need = null) {
    // level 0: valid pixels carry their colour with weight 1
    let w = width, h = height;
    let col = new Float32Array(w * h * 3), wt = new Float32Array(w * h);
    let any = false;
    for (let i = 0; i < w * h; i++) {
        if (!valid[i]) continue;
        any = true;
        wt[i] = 1;
        col[i * 3] = frame[i * 4]; col[i * 3 + 1] = frame[i * 4 + 1]; col[i * 3 + 2] = frame[i * 4 + 2];
    }
    if (!any) return null;
    const levels = [{ w, h, col, wt }];
    // pull: average valid children into coarser levels until 1×1
    while (w > 1 || h > 1) {
        const nw = Math.ceil(w / 2), nh = Math.ceil(h / 2);
        const ncol = new Float32Array(nw * nh * 3), nwt = new Float32Array(nw * nh);
        for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
            let sw = 0, r = 0, g = 0, b = 0;
            for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
                const cx = 2 * x + dx, cy = 2 * y + dy;
                if (cx >= w || cy >= h) continue;
                const ci = cy * w + cx, cw = wt[ci];
                if (!cw) continue;
                sw += cw; r += col[ci * 3] * cw; g += col[ci * 3 + 1] * cw; b += col[ci * 3 + 2] * cw;
            }
            const ni = y * nw + x;
            if (sw > 0) {
                ncol[ni * 3] = r / sw; ncol[ni * 3 + 1] = g / sw; ncol[ni * 3 + 2] = b / sw;
                nwt[ni] = Math.min(1, sw);
            }
        }
        w = nw; h = nh; col = ncol; wt = nwt;
        levels.push({ w, h, col, wt });
    }
    // push: fill each level's gaps from the (already filled) coarser level, bilinearly
    for (let l = levels.length - 2; l >= 0; l--) {
        const fine = levels[l], coarse = levels[l + 1];
        for (let y = 0; y < fine.h; y++) {
            const cyf = clamp((y + .5) / 2 - .5, 0, coarse.h - 1);
            const y0 = Math.floor(cyf), y1 = Math.min(coarse.h - 1, y0 + 1), fy = cyf - y0;
            for (let x = 0; x < fine.w; x++) {
                const i = y * fine.w + x, a = fine.wt[i];
                if (a >= 1 || (l === 0 && need && !need[i])) continue;
                const cxf = clamp((x + .5) / 2 - .5, 0, coarse.w - 1);
                const x0 = Math.floor(cxf), x1 = Math.min(coarse.w - 1, x0 + 1), fx = cxf - x0;
                for (let c = 0; c < 3; c++) {
                    const v00 = coarse.col[(y0 * coarse.w + x0) * 3 + c], v10 = coarse.col[(y0 * coarse.w + x1) * 3 + c];
                    const v01 = coarse.col[(y1 * coarse.w + x0) * 3 + c], v11 = coarse.col[(y1 * coarse.w + x1) * 3 + c];
                    const up = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
                    fine.col[i * 3 + c] = a * fine.col[i * 3 + c] + (1 - a) * up;
                }
                fine.wt[i] = 1;
            }
        }
    }
    return levels[0].col;
}

/**
 * Builds the RGBA "cover-up" layer that hides the user's real hair before the
 * new hairstyle is drawn. The segmenter's hair mask is dilated a little (to
 * catch flyaways) and each hair pixel is painted with:
 *   - inside the face ellipse (real bangs on the forehead): skin, filled from
 *     the surrounding skin pixels that match the cheek/forehead tone;
 *   - outside it: background, filled from the surrounding non-hair pixels.
 * Both fills are push-pull interpolations, so they follow the local shading
 * of the wall / forehead instead of being one flat colour.
 *
 * Pure function — no DOM — so it runs under `node --test`.
 *
 * @param {Object} p
 * @param {Float32Array} p.conf - hair confidence 0..1, length w*h
 * @param {Uint8ClampedArray} p.frame - RGBA pixels of the same w×h frame
 * @param {number} p.width @param {number} p.height
 * @param {{cx,cy,rx,ry,angle}} p.ellipse - face ellipse in this w×h space
 * @param {{x:number,y:number}} [p.skinPoint] - forehead point (legacy single sample)
 * @param {{x:number,y:number}[]} [p.skinPoints] - skin sample points (forehead + cheeks)
 * @param {{skin?:number[], bg?:number[], hair?:number[]}} [p.previous] - colours from last frame
 * @returns {{rgba:Uint8ClampedArray, skin:number[]|null, bg:number[]|null, hair:number[]|null, coverage:number}}
 */
export function buildEraseLayer({ conf, frame, width, height, ellipse, skinPoint, skinPoints, previous = {} }) {
    const n = width * height;
    if (conf.length < n || frame.length < n * 4) throw new RangeError("mask/frame size mismatch");
    const inFace = ellipseTester(ellipse);
    const face = new Uint8Array(n);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) face[y * width + x] = inFace(x, y) ? 1 : 0;

    // Skin tone: small patches around the sample points, skipping hair-covered pixels.
    const r = Math.max(2, Math.round(width * .012));
    const skinSum = [0, 0, 0]; let skinN = 0;
    for (const pt of (skinPoints || [skinPoint]).filter(Boolean)) {
        for (let y = Math.round(pt.y) - r; y <= Math.round(pt.y) + r; y++) {
            for (let x = Math.round(pt.x) - r; x <= Math.round(pt.x) + r; x++) {
                if (x < 0 || y < 0 || x >= width || y >= height) continue;
                const i = y * width + x;
                if (conf[i] > .3) continue;
                skinSum[0] += frame[i * 4]; skinSum[1] += frame[i * 4 + 1]; skinSum[2] += frame[i * 4 + 2]; skinN++;
            }
        }
    }
    const skin = skinN ? skinSum.map((v) => v / skinN) : null;

    // The user's own hair colour: confident hair pixels away from the face.
    const hairSum = [0, 0, 0]; let hairN = 0, hairCount = 0;
    for (let i = 0; i < n; i++) {
        if (conf[i] > .5) hairCount++;
        if (conf[i] > .85 && !face[i]) {
            hairSum[0] += frame[i * 4]; hairSum[1] += frame[i * 4 + 1]; hairSum[2] += frame[i * 4 + 2]; hairN++;
        }
    }
    const hair = hairN >= 30 ? hairSum.map((v) => v / hairN) : null;

    const skinColor = blend(previous.skin, skin);
    const hairColor = blend(previous.hair, hair);
    const rgba = new Uint8ClampedArray(n * 4);
    let bgColor = previous.bg || null;
    if (hairCount) {
        const mask = dilateMask(conf, width, height);
        const bgValid = new Uint8Array(n), skinValid = new Uint8Array(n);
        const tol2 = SKIN_TOLERANCE * SKIN_TOLERANCE;
        for (let i = 0; i < n; i++) {
            if (mask[i] > FREE_CONF) continue;
            if (!face[i]) { bgValid[i] = 1; continue; }
            if (!skinColor) { skinValid[i] = 1; continue; }
            const dr = frame[i * 4] - skinColor[0], dg = frame[i * 4 + 1] - skinColor[1], db = frame[i * 4 + 2] - skinColor[2];
            if (dr * dr + dg * dg + db * db <= tol2) skinValid[i] = 1;
        }
        const needBg = new Uint8Array(n), needSkin = new Uint8Array(n);
        for (let i = 0; i < n; i++) if (mask[i] > .2) (face[i] ? needSkin : needBg)[i] = 1;
        const needAll = new Uint8Array(n);
        for (let i = 0; i < n; i++) needAll[i] = needBg[i] | needSkin[i];
        // each fill only computes the pixels it will paint; if one side has no samples
        // at all, the other fill covers every hair pixel instead
        let skinFill = pushPullFill(frame, skinValid, width, height, needSkin);
        const bgFill = pushPullFill(frame, bgValid, width, height, skinFill ? needBg : needAll);
        if (!bgFill && skinFill) skinFill = pushPullFill(frame, skinValid, width, height, needAll);

        if (bgFill) { // average background colour, kept for callers and as a fallback
            let s0 = 0, s1 = 0, s2 = 0, k = 0;
            for (let i = 0; i < n; i++) if (bgValid[i]) { s0 += frame[i * 4]; s1 += frame[i * 4 + 1]; s2 += frame[i * 4 + 2]; k++; }
            bgColor = blend(previous.bg, [s0 / k, s1 / k, s2 / k]);
        }
        const flatSkin = skinColor || bgColor;
        for (let i = 0; i < n; i++) {
            const alpha = smoothstep(.2, .5, mask[i]);
            if (alpha <= 0) continue;
            const o = i * 4;
            const src = face[i] ? (skinFill || bgFill) : (bgFill || skinFill);
            if (src) {
                rgba[o] = src[i * 3]; rgba[o + 1] = src[i * 3 + 1]; rgba[o + 2] = src[i * 3 + 2];
            } else if (flatSkin) {
                rgba[o] = flatSkin[0]; rgba[o + 1] = flatSkin[1]; rgba[o + 2] = flatSkin[2];
            } else continue;
            rgba[o + 3] = alpha * 255;
        }
    }
    return { rgba, skin: skinColor || null, bg: bgColor || null, hair: hairColor || null, coverage: hairCount / n };
}

/** Holds the latest erase layer on an offscreen canvas and draws it scaled + feathered. */
export class HairEraser {
    constructor() {
        this.canvas = document.createElement("canvas");
        this.ctx = this.canvas.getContext("2d");
        this.colors = {};
        this.ready = false;
    }
    reset() { this.ready = false; this.colors = {}; }

    update(params) {
        const { rgba, skin, bg, hair } = buildEraseLayer({ ...params, previous: this.colors });
        this.colors = { skin, bg, hair };
        if (this.canvas.width !== params.width || this.canvas.height !== params.height) {
            this.canvas.width = params.width;
            this.canvas.height = params.height;
        }
        this.ctx.putImageData(new ImageData(rgba, params.width, params.height), 0, 0);
        this.ready = true;
    }

    /** Draws onto a w×h target; the blur both feathers edges and slightly dilates the cover. */
    draw(ctx, w, h) {
        if (!this.ready) return;
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        if ("filter" in ctx) ctx.filter = `blur(${Math.max(2, Math.round(w / this.canvas.width * 2))}px)`;
        ctx.drawImage(this.canvas, 0, 0, w, h);
        ctx.restore();
    }
}
