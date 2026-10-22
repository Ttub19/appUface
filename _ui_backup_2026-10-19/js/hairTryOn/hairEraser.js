import { clamp } from "../landmarks.js";

const COLOR_EMA = .2;

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

/**
 * Builds the RGBA "cover-up" layer that hides the user's real hair before the
 * new hairstyle is drawn. Each hair pixel (from the segmenter's confidence
 * mask) is painted with either sampled forehead skin (inside the face
 * ellipse, e.g. real bangs) or sampled background (outside it), with soft
 * alpha edges. Colours are smoothed over time so they don't flicker.
 *
 * Pure function — no DOM — so it runs under `node --test`.
 *
 * @param {Object} p
 * @param {Float32Array} p.conf - hair confidence 0..1, length w*h
 * @param {Uint8ClampedArray} p.frame - RGBA pixels of the same w×h frame
 * @param {number} p.width @param {number} p.height
 * @param {{cx,cy,rx,ry,angle}} p.ellipse - face ellipse in this w×h space
 * @param {{x:number,y:number}} p.skinPoint - forehead point in this w×h space
 * @param {{skin?:number[], bg?:number[]}} [p.previous] - colours from last frame
 * @returns {{rgba:Uint8ClampedArray, skin:number[]|null, bg:number[]|null, coverage:number}}
 */
export function buildEraseLayer({ conf, frame, width, height, ellipse, skinPoint, previous = {} }) {
    const n = width * height;
    if (conf.length < n || frame.length < n * 4) throw new RangeError("mask/frame size mismatch");
    const inFace = ellipseTester(ellipse);

    let minX = width, minY = height, maxX = -1, maxY = -1, hairCount = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (conf[y * width + x] > .5) {
                hairCount++;
                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (y < minY) minY = y; if (y > maxY) maxY = y;
            }
        }
    }

    // Skin: small patch around the forehead, skipping pixels covered by hair.
    const r = Math.max(2, Math.round(width * .012));
    const skinSum = [0, 0, 0]; let skinN = 0;
    for (let y = Math.round(skinPoint.y) - r; y <= Math.round(skinPoint.y) + r; y++) {
        for (let x = Math.round(skinPoint.x) - r; x <= Math.round(skinPoint.x) + r; x++) {
            if (x < 0 || y < 0 || x >= width || y >= height) continue;
            const i = y * width + x;
            if (conf[i] > .3) continue;
            skinSum[0] += frame[i * 4]; skinSum[1] += frame[i * 4 + 1]; skinSum[2] += frame[i * 4 + 2]; skinN++;
        }
    }
    const skin = skinN ? skinSum.map((v) => v / skinN) : null;

    // Background: non-hair, non-face pixels around (and above) the hair region.
    const bgSum = [0, 0, 0]; let bgN = 0;
    const sampleBg = (x0, y0, x1, y1, step) => {
        for (let y = Math.max(0, y0); y <= Math.min(height - 1, y1); y += step) {
            for (let x = Math.max(0, x0); x <= Math.min(width - 1, x1); x += step) {
                const i = y * width + x;
                if (conf[i] > .05 || inFace(x, y)) continue;
                bgSum[0] += frame[i * 4]; bgSum[1] += frame[i * 4 + 1]; bgSum[2] += frame[i * 4 + 2]; bgN++;
            }
        }
    };
    if (hairCount) {
        const pad = Math.round((maxX - minX + 1) * .25);
        sampleBg(minX - pad, minY - pad, maxX + pad, Math.min(maxY, Math.round(ellipse.cy)), 2);
    }
    if (bgN < 20) sampleBg(0, 0, width - 1, Math.round(ellipse.cy), 3);
    const bg = bgN ? bgSum.map((v) => v / bgN) : null;

    const skinColor = blend(previous.skin, skin);
    const bgColor = blend(previous.bg, bg) || skinColor;
    const rgba = new Uint8ClampedArray(n * 4);
    if (hairCount && (skinColor || bgColor)) {
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = y * width + x;
                const alpha = smoothstep(.25, .65, conf[i]);
                if (alpha <= 0) continue;
                const color = (inFace(x, y) && skinColor) || bgColor;
                const o = i * 4;
                rgba[o] = color[0]; rgba[o + 1] = color[1]; rgba[o + 2] = color[2]; rgba[o + 3] = alpha * 255;
            }
        }
    }
    return { rgba, skin: skinColor || null, bg: bgColor || null, coverage: hairCount / n };
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
        const { rgba, skin, bg } = buildEraseLayer({ ...params, previous: this.colors });
        this.colors = { skin, bg };
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
