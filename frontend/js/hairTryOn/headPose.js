import { LM, clamp, isLandmarkPoint } from "../landmarks.js";

const EPS = 1e-6;
/** Real hairline height above landmark 10, in face widths (same calibration as the hair placement). */
const FOREHEAD_ABOVE_LM10 = .235;
const lerp = (a, b, t) => a + (b - a) * t;
const angleDelta = (a, b) => Math.atan2(Math.sin(b - a), Math.cos(b - a));

/**
 * Extracts yaw/pitch/roll (radians) from MediaPipe Face Landmarker's
 * `facialTransformationMatrixes[i].data` (4x4, flattened).
 *
 * The layout is detected rather than assumed: in a column-major matrix the
 * translation lives at [12..14] and the bottom row [3,7,11] is zero; in a
 * row-major one it's the other way round. Rotation columns are normalised so
 * any uniform scale baked into the pose matrix doesn't skew the angles.
 * Decomposition order is Y (yaw) · X (pitch) · Z (roll).
 *
 * @param {ArrayLike<number>} data
 * @returns {{yaw:number,pitch:number,roll:number}|null}
 */
export function eulerFromTransformMatrix(data) {
    if (!data || data.length < 16) return null;
    for (let i = 0; i < 16; i++) if (!Number.isFinite(data[i])) return null;
    const colMajor = Math.abs(data[3]) + Math.abs(data[7]) + Math.abs(data[11])
        <= Math.abs(data[12]) + Math.abs(data[13]) + Math.abs(data[14]);
    const raw = (r, c) => (colMajor ? data[c * 4 + r] : data[r * 4 + c]);
    const norms = [0, 1, 2].map((c) => Math.hypot(raw(0, c), raw(1, c), raw(2, c)));
    if (norms.some((n) => n <= EPS)) return null;
    const m = (r, c) => raw(r, c) / norms[c];
    return {
        pitch: Math.asin(clamp(-m(1, 2), -1, 1)),
        yaw: Math.atan2(m(0, 2), m(2, 2)),
        roll: Math.atan2(m(1, 0), m(1, 1)),
    };
}

function px(lm, index, w, h) {
    const p = lm?.[index];
    if (!isLandmarkPoint(p)) throw new RangeError(`invalid landmark ${index}`);
    return { x: p.x * w, y: p.y * h };
}

/**
 * Where to draw a 2D hairstyle asset so it follows the head.
 *
 * - Position: the asset's hairline anchor sits on landmark 10 (top of face oval).
 * - Roll: from the temple line in image space (always matches what's on screen).
 * - Yaw/pitch magnitude: from the 3D transformation matrix when available
 *   (accurate angle); yaw *direction* comes from the nose offset in the image,
 *   so it stays correct whether or not the preview is mirrored.
 * - Scale: temple width is divided by cos(yaw) to recover the true head width,
 *   so turning the head doesn't shrink the hair.
 * - Turning the head moves the forehead more than the skull silhouette, so the
 *   hair is shifted back toward the head's rotation axis by `asset.depth`.
 *
 * @param {Array<{x:number,y:number}>} lm - normalised landmarks
 * @param {number} w @param {number} h - frame size in px
 * @param {{yaw:number,pitch:number}|null} pose - from eulerFromTransformMatrix
 * @param {{headWidth:number, depth?:number}} asset
 * @param {number|null} [headWidthPx] - true head width from HeadScaleFusion (else temples / cos(yaw))
 */
export function computeHairPlacement(lm, w, h, pose, asset, headWidthPx = null) {
    if (![w, h].every(Number.isFinite) || w <= 0 || h <= 0) throw new RangeError("invalid frame size");
    if (!(asset?.headWidth > 0)) throw new RangeError("asset.headWidth required");
    const top = px(lm, LM.FOREHEAD, w, h);
    const a = px(lm, LM.TEMPLE_L, w, h), b = px(lm, LM.TEMPLE_R, w, h);
    const nose = px(lm, LM.NOSE_TIP, w, h);
    const [left, right] = a.x <= b.x ? [a, b] : [b, a];
    const dx = right.x - left.x, dy = right.y - left.y;
    const projectedWidth = Math.hypot(dx, dy);
    if (projectedWidth <= EPS) throw new RangeError("temple landmarks are too close");

    const roll = Math.atan2(dy, dx);
    const ux = dx / projectedWidth, uy = dy / projectedWidth;
    const mid = { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 };
    const noseOffset = ((nose.x - mid.x) * ux + (nose.y - mid.y) * uy) / projectedWidth;
    const direction = Math.abs(noseOffset) < .01 ? 0 : Math.sign(noseOffset);

    const yawAbs = pose && Number.isFinite(pose.yaw)
        ? Math.min(Math.abs(pose.yaw), 1.1)
        : Math.asin(clamp(Math.abs(noseOffset) * 2.2, 0, .9));
    const pitchAbs = pose && Number.isFinite(pose.pitch) ? Math.min(Math.abs(pose.pitch), .9) : 0;

    const headWidth = headWidthPx > 0 ? headWidthPx : projectedWidth / Math.max(Math.cos(yawAbs), .55);
    const scale = headWidth / asset.headWidth;
    const shift = -direction * Math.sin(yawAbs) * headWidth * (asset.depth ?? .4);

    return {
        x: top.x + ux * shift,
        y: top.y + uy * shift,
        rotation: roll,
        scaleX: scale * (1 - (1 - Math.cos(yawAbs)) * .25),
        scaleY: scale * (1 - (1 - Math.cos(pitchAbs)) * .5),
        yaw: direction * yawAbs,
        pitch: pitchAbs,
    };
}

/**
 * Rotated ellipse approximating the face (forehead → chin, cheek → cheek),
 * in the pixel space of a frame of size w×h. Used to decide whether erased
 * hair should be filled with skin (inside) or background (outside).
 */
export function faceEllipse(lm, w, h) {
    const lm10 = px(lm, LM.FOREHEAD, w, h), chin = px(lm, LM.CHIN, w, h);
    const cl = px(lm, LM.CHEEK_L, w, h), cr = px(lm, LM.CHEEK_R, w, h);
    // Landmark 10 is mid-forehead, not the hairline: extend the top up to the
    // real hairline so erased bangs are filled with skin, not background.
    const cheekWidth = Math.hypot(cr.x - cl.x, cr.y - cl.y);
    const len = Math.hypot(chin.x - lm10.x, chin.y - lm10.y) || 1;
    const lift = FOREHEAD_ABOVE_LM10 * cheekWidth;
    const top = { x: lm10.x - (chin.x - lm10.x) / len * lift, y: lm10.y - (chin.y - lm10.y) / len * lift };
    const angle = Math.atan2(chin.y - top.y, chin.x - top.x) - Math.PI / 2;
    return {
        cx: (top.x + chin.x) / 2,
        cy: (top.y + chin.y) / 2,
        rx: Math.hypot(cr.x - cl.x, cr.y - cl.y) / 2 * .98,
        ry: Math.hypot(chin.x - top.x, chin.y - top.y) / 2 * 1.04,
        angle,
    };
}

export function insideEllipse(e, x, y) {
    const c = Math.cos(e.angle), s = Math.sin(e.angle);
    const dx = x - e.cx, dy = y - e.cy;
    const u = (dx * c + dy * s) / e.rx, v = (-dx * s + dy * c) / e.ry;
    return u * u + v * v <= 1;
}

const smoothingAlpha = (cutoffHz, dtSec) => 1 / (1 + 1 / (2 * Math.PI * cutoffHz * dtSec));

/**
 * One Euro filter channel (Casiez et al., CHI 2012): the cutoff frequency
 * rises with speed, so a still head gets heavy smoothing (no jitter) while a
 * fast turn gets almost none (no lag). `beta` is in units of 1/(value/s).
 */
class OneEuroChannel {
    constructor({ minCutoff, beta, dCutoff = 1 }) {
        Object.assign(this, { minCutoff, beta, dCutoff });
        this.reset();
    }
    reset() { this.value = null; this.speed = 0; }
    /** @param {number} delta - raw change since the last filtered value (lets angles wrap) */
    filter(delta, dtSec) {
        const rawSpeed = delta / dtSec;
        this.speed = lerp(this.speed, rawSpeed, smoothingAlpha(this.dCutoff, dtSec));
        const cutoff = this.minCutoff + this.beta * Math.abs(this.speed);
        this.value += delta * smoothingAlpha(cutoff, dtSec);
        return this.value;
    }
}

/**
 * Smooths a hair placement frame to frame (rotation wraps). Uses a One Euro
 * filter per channel so the hair stays steady when you hold still and keeps
 * up when you move; after a gap (face lost / tab hidden) it snaps instead of
 * sliding in from the old spot.
 */
export class PlacementSmoother {
    constructor({ minCutoff = 1.2, snapAfterMs = 300 } = {}) {
        this.snapAfterMs = snapAfterMs;
        this.channels = {
            // beta raised from .008: at .008 the hair visibly trailed the head on quick moves
            x: new OneEuroChannel({ minCutoff, beta: .015 }),
            y: new OneEuroChannel({ minCutoff, beta: .015 }),
            scaleX: new OneEuroChannel({ minCutoff, beta: 4 }),
            scaleY: new OneEuroChannel({ minCutoff, beta: 4 }),
            rotation: new OneEuroChannel({ minCutoff, beta: 3 }),
        };
        this.reset();
    }
    reset() {
        this.state = null; this.lastTimestamp = null;
        Object.values(this.channels).forEach((c) => c.reset());
    }
    update(target, timestamp) {
        const keys = ["x", "y", "rotation", "scaleX", "scaleY"];
        if (!target || !keys.every((k) => Number.isFinite(target[k]))) return null;
        const elapsed = this.lastTimestamp === null ? Infinity : Math.max(0, timestamp - this.lastTimestamp);
        this.lastTimestamp = timestamp;
        if (!this.state || elapsed >= this.snapAfterMs) {
            this.state = { ...target };
            for (const k of keys) { this.channels[k].value = target[k]; this.channels[k].speed = 0; }
            return { ...this.state };
        }
        if (elapsed <= 0) return { ...this.state };
        const dt = elapsed / 1000;
        for (const k of ["x", "y", "scaleX", "scaleY"]) this.state[k] = this.channels[k].filter(target[k] - this.state[k], dt);
        this.state.rotation = this.channels.rotation.filter(angleDelta(this.state.rotation, target.rotation), dt);
        this.state.yaw = target.yaw;
        this.state.pitch = target.pitch;
        return { ...this.state };
    }
}

/**
 * Where the user's real hairline is, measured on the hair-segmentation mask:
 * walk up from landmark 10 along the face's vertical axis and find the first
 * band that is mostly hair. Foreheads differ a lot between people, and a fixed
 * offset made the new hair float on small foreheads and cover the eyes on big
 * ones. Returns the distance above landmark 10 in temple widths, or null when
 * it can't be measured (no hair found, or hair already covering landmark 10 =
 * bangs, where the real hairline is hidden).
 * @param {{conf:Float32Array, width:number, height:number, landmarks:Array<{x:number,y:number}>}} p
 */
export function measureHairlineLift({ conf, width, height, landmarks }) {
    let top, chin, tl, tr;
    try {
        top = px(landmarks, LM.FOREHEAD, width, height); chin = px(landmarks, LM.CHIN, width, height);
        tl = px(landmarks, LM.TEMPLE_L, width, height); tr = px(landmarks, LM.TEMPLE_R, width, height);
    } catch { return null; }
    const temple = Math.hypot(tr.x - tl.x, tr.y - tl.y);
    const len = Math.hypot(top.x - chin.x, top.y - chin.y);
    if (temple < 8 || len < 8) return null;
    const ux = (top.x - chin.x) / len, uy = (top.y - chin.y) / len; // towards the crown
    const vx = (tr.x - tl.x) / temple, vy = (tr.y - tl.y) / temple; // across the forehead
    const half = .1 * temple, step = Math.max(1, half / 4);
    const at = (x, y) => {
        const xi = Math.round(x), yi = Math.round(y);
        return xi < 0 || yi < 0 || xi >= width || yi >= height ? 0 : conf[yi * width + xi];
    };
    for (let t = 0; t <= .6 * temple; t += 1) {
        let sum = 0, n = 0;
        for (let k = -half; k <= half; k += step) { sum += at(top.x + ux * t + vx * k, top.y + uy * t + vy * k); n++; }
        if (sum / n > .5) {
            const lift = t / temple;
            return lift < .12 ? null : lift; // hair at landmark 10 = bangs: real hairline unknown
        }
    }
    return null;
}

/**
 * Head width that stays steady when the head turns. Temple width / cos(yaw)
 * drifts at larger yaw (the far temple landmark is guessed by the model), so
 * it's blended with a second estimate from the face's vertical size
 * (landmark 10 -> nose base, unaffected by talking), whose ratio to the head
 * width is learned while the user faces the camera.
 */
export class HeadScaleFusion {
    constructor() { this.reset(); }
    reset() { this.ratio = null; }

    /** @returns {number} head width in px for computeHairPlacement */
    update(lm, w, h, pose) {
        const a = px(lm, LM.TEMPLE_L, w, h), b = px(lm, LM.TEMPLE_R, w, h);
        const top = px(lm, LM.FOREHEAD, w, h), base = px(lm, LM.NOSE_BASE, w, h);
        const yaw = Math.min(Math.abs(pose?.yaw ?? 0), 1.1), pitch = Math.min(Math.abs(pose?.pitch ?? 0), .9);
        const fromWidth = Math.hypot(b.x - a.x, b.y - a.y) / Math.max(Math.cos(yaw), .55);
        const vertical = Math.hypot(base.x - top.x, base.y - top.y) / Math.max(Math.cos(pitch), .6);
        if (!(vertical > 0)) return fromWidth;
        if (yaw < .15 && pitch < .3) {
            const r = fromWidth / vertical;
            this.ratio = this.ratio === null ? r : this.ratio + (r - this.ratio) * .05;
        }
        if (this.ratio === null) return fromWidth;
        const weight = clamp((yaw - .15) / .35, 0, 1) * .7;
        return (1 - weight) * fromWidth + weight * this.ratio * vertical;
    }
}
