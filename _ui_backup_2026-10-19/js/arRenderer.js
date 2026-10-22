import { LM, isLandmarkPoint } from "./landmarks.js";

const EPS = 1e-6;
const average = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const lerp = (a, b, t) => a + (b - a) * t;
const angleDelta = (a, b) => Math.atan2(Math.sin(b - a), Math.cos(b - a));

/**
 * ArRenderer — turns raw MediaPipe landmarks into a 2D transform (position,
 * rotation, scale) so glasses SVG graphics track the face, plus the
 * live measurement-mesh drawing used on the Golden Ratio screen.
 */
export const ArRenderer = {
    _dist(a, b, w, h) {
        const dx = (a.x - b.x) * w, dy = (a.y - b.y) * h;
        return Math.sqrt(dx * dx + dy * dy);
    },

    _pixel(lm, index, w, h) {
        const p = lm?.[index];
        if (!isLandmarkPoint(p)) throw new RangeError(`invalid landmark ${index}`);
        return { x: p.x * w, y: p.y * h };
    },

    /** Uses geometric eye centers, with a light optional iris correction. */
    getEyeTransform(lm, w, h, { verticalOffsetRatio = .035 } = {}) {
        if (![w, h].every(Number.isFinite) || w <= 0 || h <= 0) throw new RangeError("invalid frame size");
        const eyeA = average(this._pixel(lm, LM.EYE_L_OUTER, w, h), this._pixel(lm, LM.EYE_L_INNER, w, h));
        const eyeB = average(this._pixel(lm, LM.EYE_R_OUTER, w, h), this._pixel(lm, LM.EYE_R_INNER, w, h));
        let [left, right] = eyeA.x <= eyeB.x ? [eyeA, eyeB] : [eyeB, eyeA];
        if (isLandmarkPoint(lm?.[LM.IRIS_A]) && isLandmarkPoint(lm?.[LM.IRIS_B])) {
            const irisA = this._pixel(lm, LM.IRIS_A, w, h), irisB = this._pixel(lm, LM.IRIS_B, w, h);
            const [irisLeft, irisRight] = irisA.x <= irisB.x ? [irisA, irisB] : [irisB, irisA];
            left = { x: lerp(left.x, irisLeft.x, .25), y: lerp(left.y, irisLeft.y, .25) };
            right = { x: lerp(right.x, irisRight.x, .25), y: lerp(right.y, irisRight.y, .25) };
        }
        const dx = right.x - left.x, dy = right.y - left.y, span = Math.hypot(dx, dy);
        if (span <= EPS) throw new RangeError("eye landmarks are too close");
        const angle = Math.atan2(dy, dx), middle = average(left, right), offset = span * verticalOffsetRatio;
        return { cx: middle.x - Math.sin(angle) * offset, cy: middle.y + Math.cos(angle) * offset, angle, span };
    },

    applyTransform(group, cx, cy, angleRad, scaleX, scaleY = scaleX) {
        if (!group || ![cx, cy, angleRad, scaleX, scaleY].every(Number.isFinite) || scaleX <= 0 || scaleY <= 0) return false;
        group.setAttribute("transform", `translate(${cx.toFixed(3)} ${cy.toFixed(3)}) rotate(${(angleRad * 180 / Math.PI).toFixed(3)}) scale(${scaleX.toFixed(5)} ${scaleY.toFixed(5)})`);
        return true;
    },

    /** Draws the live golden-ratio measurement mesh (landmark dots + guide lines) on a canvas. */
    drawGoldenMesh(ctx, lm, w, h) {
        ctx.clearRect(0, 0, w, h);
        ctx.save();

        ctx.fillStyle = "rgba(66,116,217,0.35)";
        for (let i = 0; i < lm.length; i += 3) {
            ctx.beginPath();
            ctx.arc(lm[i].x * w, lm[i].y * h, 1.4, 0, Math.PI * 2);
            ctx.fill();
        }

        const vertPts = [LM.FOREHEAD, LM.GLABELLA, LM.NOSE_BASE, LM.CHIN].map((i) => ({ x: lm[i].x * w, y: lm[i].y * h }));
        ctx.strokeStyle = "#95CCDD";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(vertPts[0].x, vertPts[0].y);
        for (let i = 1; i < vertPts.length; i++) ctx.lineTo(vertPts[i].x, vertPts[i].y);
        ctx.stroke();
        ctx.setLineDash([]);

        const cheekL = { x: lm[LM.CHEEK_L].x * w, y: lm[LM.CHEEK_L].y * h };
        const cheekR = { x: lm[LM.CHEEK_R].x * w, y: lm[LM.CHEEK_R].y * h };
        ctx.strokeStyle = "#4274D9";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cheekL.x, cheekL.y);
        ctx.lineTo(cheekR.x, cheekR.y);
        ctx.stroke();

        ctx.fillStyle = "#293681";
        [...vertPts, cheekL, cheekR].forEach((p) => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
        ctx.restore();
    }
};

/** Frame-rate-independent low-pass filter for stable AR placement. */
export class ArTransformSmoother {
    constructor({ timeConstantMs = 85, snapAfterMs = 350 } = {}) {
        this.timeConstantMs = timeConstantMs;
        this.snapAfterMs = snapAfterMs;
        this.reset();
    }
    reset() { this.state = null; this.lastTimestamp = null; }
    update(target, timestamp = globalThis.performance?.now?.() ?? Date.now()) {
        if (!target || ![target.cx, target.cy, target.angle, target.span].every(Number.isFinite)) return null;
        const next = { ...target };
        if (!this.state || this.lastTimestamp === null) {
            this.state = next; this.lastTimestamp = timestamp; return { ...this.state };
        }
        const elapsed = Math.max(0, timestamp - this.lastTimestamp);
        this.lastTimestamp = timestamp;
        if (elapsed >= this.snapAfterMs) { this.state = next; return { ...this.state }; }
        const t = Math.max(0, Math.min(1, 1 - Math.exp(-elapsed / Math.max(this.timeConstantMs, 1))));
        this.state.cx = lerp(this.state.cx, next.cx, t);
        this.state.cy = lerp(this.state.cy, next.cy, t);
        this.state.span = lerp(this.state.span, next.span, t);
        this.state.angle += angleDelta(this.state.angle, next.angle) * t;
        return { ...this.state };
    }
}

/**
 * Reference eye-to-eye span (px) each glasses shape below was hand-drawn at;
 * the live scale factor is the tracked eye span divided by this constant.
 */
export const GLASSES_REF_EYE_SPAN = 180;
export const GLASSES_STYLE_CONFIG = Object.freeze({
    classic: { scale: 1.08, verticalOffsetRatio: .035 },
    aviator: { scale: 1.08, verticalOffsetRatio: .055 },
    "cat-eye": { scale: 1.06, verticalOffsetRatio: .015 },
    oversized: { scale: 1.02, verticalOffsetRatio: .04 },
    wayfarer: { scale: 1.04, verticalOffsetRatio: .025 },
});

/** SVG path/shape markup for each glasses style, centered on the origin (0,0). */
export const GLASSES_SHAPES = {
    classic: `<ellipse cx="-90" cy="0" rx="45" ry="35" fill="rgba(15,23,42,0.45)" stroke="#4274D9" stroke-width="4"/><ellipse cx="90" cy="0" rx="45" ry="35" fill="rgba(15,23,42,0.45)" stroke="#4274D9" stroke-width="4"/><path d="M-45 0 L45 0" stroke="#4274D9" stroke-width="4" fill="none"/><path d="M-135 0 Q-150 -10 -165 0" stroke="#4274D9" stroke-width="3" fill="none"/><path d="M135 0 Q150 -10 165 0" stroke="#4274D9" stroke-width="3" fill="none"/>`,
    aviator: `<path d="M-135 20 L-100 -25 Q-65 -45 -45 -25 L-10 20 Z" fill="rgba(15,23,42,0.45)" stroke="#1E90FF" stroke-width="4"/><path d="M10 20 L45 -25 Q80 -45 100 -25 L135 20 Z" fill="rgba(15,23,42,0.45)" stroke="#1E90FF" stroke-width="4"/><path d="M-10 0 L10 0" stroke="#1E90FF" stroke-width="4" fill="none"/><path d="M-135 0 Q-150 -10 -165 0" stroke="#1E90FF" stroke-width="3" fill="none"/><path d="M135 0 Q150 -10 165 0" stroke="#1E90FF" stroke-width="3" fill="none"/>`,
    "cat-eye": `<path d="M-140 15 Q-100 -30 -50 -12 L-30 22 Z" fill="rgba(15,23,42,0.45)" stroke="#FF1493" stroke-width="4"/><path d="M140 15 Q100 -30 50 -12 L30 22 Z" fill="rgba(15,23,42,0.45)" stroke="#FF1493" stroke-width="4"/><path d="M-30 5 L30 5" stroke="#FF1493" stroke-width="4" fill="none"/><path d="M-150 0 Q-170 -15 -180 -25" stroke="#FF1493" stroke-width="3" fill="none"/><path d="M150 0 Q170 -15 180 -25" stroke="#FF1493" stroke-width="3" fill="none"/>`,
    oversized: `<ellipse cx="-100" cy="0" rx="70" ry="60" fill="rgba(15,23,42,0.45)" stroke="#32CD32" stroke-width="4"/><ellipse cx="100" cy="0" rx="70" ry="60" fill="rgba(15,23,42,0.45)" stroke="#32CD32" stroke-width="4"/><path d="M-30 0 L30 0" stroke="#32CD32" stroke-width="4" fill="none"/><path d="M-170 0 Q-190 0 -200 0" stroke="#32CD32" stroke-width="3" fill="none"/><path d="M170 0 Q190 0 200 0" stroke="#32CD32" stroke-width="3" fill="none"/>`,
    wayfarer: `<path d="M-155 -25 L-85 -35 Q-55 -30 -15 -25 L-15 25 L-155 20 Z" fill="rgba(15,23,42,0.45)" stroke="#111827" stroke-width="4"/><path d="M15 -25 L85 -35 Q115 -30 155 -25 L155 25 L15 20 Z" fill="rgba(15,23,42,0.45)" stroke="#111827" stroke-width="4"/><path d="M-15 0 L15 0" stroke="#111827" stroke-width="4" fill="none"/><path d="M-180 0 Q-200 0 -210 0" stroke="#111827" stroke-width="3" fill="none"/><path d="M180 0 Q200 0 210 0" stroke="#111827" stroke-width="3" fill="none"/>`
};
