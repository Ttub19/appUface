import { LM, clamp, isLandmarkPoint } from "../landmarks.js";

const EPS = 1e-6;
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
 */
export function computeHairPlacement(lm, w, h, pose, asset) {
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

    const headWidth = projectedWidth / Math.max(Math.cos(yawAbs), .55);
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
    const top = px(lm, LM.FOREHEAD, w, h), chin = px(lm, LM.CHIN, w, h);
    const cl = px(lm, LM.CHEEK_L, w, h), cr = px(lm, LM.CHEEK_R, w, h);
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

/** Frame-rate-independent low-pass filter for a hair placement (rotation wraps). */
export class PlacementSmoother {
    constructor({ timeConstantMs = 70, snapAfterMs = 300 } = {}) {
        this.timeConstantMs = timeConstantMs;
        this.snapAfterMs = snapAfterMs;
        this.reset();
    }
    reset() { this.state = null; this.lastTimestamp = null; }
    update(target, timestamp) {
        const keys = ["x", "y", "rotation", "scaleX", "scaleY"];
        if (!target || !keys.every((k) => Number.isFinite(target[k]))) return null;
        const elapsed = this.lastTimestamp === null ? Infinity : Math.max(0, timestamp - this.lastTimestamp);
        this.lastTimestamp = timestamp;
        if (!this.state || elapsed >= this.snapAfterMs) { this.state = { ...target }; return { ...this.state }; }
        const t = clamp(1 - Math.exp(-elapsed / Math.max(this.timeConstantMs, 1)), 0, 1);
        for (const k of ["x", "y", "scaleX", "scaleY"]) this.state[k] = lerp(this.state[k], target[k], t);
        this.state.rotation += angleDelta(this.state.rotation, target.rotation) * t;
        this.state.yaw = target.yaw;
        this.state.pitch = target.pitch;
        return { ...this.state };
    }
}
