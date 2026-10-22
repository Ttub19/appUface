import { LM, isLandmarkPoint } from "./landmarks.js";

const EPS = 1e-6;
const average = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const lerp = (a, b, t) => a + (b - a) * t;
const angleDelta = (a, b) => Math.atan2(Math.sin(b - a), Math.cos(b - a));

// Distance between landmarks 127/356 vs. eye-center span, as seen on a live
// webcam (~1.8-1.9). Outside MIN..MAX the temple landmarks are treated as
// unreliable (e.g. strong head turn) and the typical ratio is used instead.
const FACE_TO_EYE_SPAN_TYPICAL = 1.85;
const FACE_TO_EYE_SPAN_MIN = 1.4;
const FACE_TO_EYE_SPAN_MAX = 2.6;

/**
 * How wide the whole glasses image (front + hinge stubs) is drawn relative
 * to the 127/356 landmark distance. Those mesh points sit ~20% inside the
 * visible edge of the face on a real camera, so 1.25 makes the frame span the
 * visible face width at eye level, like a correctly sized real pair
 * (calibrated from a live try-on screenshot: landmark width ≈172px vs.
 * visible face ≈220px).
 */
export const GLASSES_FACE_FIT_RATIO = 1.25;

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
        return {
            cx: middle.x - Math.sin(angle) * offset, cy: middle.y + Math.cos(angle) * offset,
            angle, span, faceWidth: this.getFaceWidth(lm, w, h, span),
        };
    },

    /**
     * Face width at eye level (temple to temple, landmarks 127/356) — the
     * width a glasses front should roughly match. Falls back to an
     * eye-span-based estimate when the temple points are missing or implausible
     * (e.g. one side hidden on a strong head turn).
     */
    getFaceWidth(lm, w, h, span) {
        const fallback = span * FACE_TO_EYE_SPAN_TYPICAL;
        if (!isLandmarkPoint(lm?.[LM.TEMPLE_L]) || !isLandmarkPoint(lm?.[LM.TEMPLE_R])) return fallback;
        const width = this._dist(lm[LM.TEMPLE_L], lm[LM.TEMPLE_R], w, h);
        const ratio = width / span;
        return ratio >= FACE_TO_EYE_SPAN_MIN && ratio <= FACE_TO_EYE_SPAN_MAX ? width : fallback;
    },

    /**
     * Scale for a glasses style so its frame spans the wearer's face width
     * (times the user's manual size tweak). Styles without a measured
     * frameWidth fall back to the old eye-span scaling.
     */
    fitGlassesScale(transform, fit, userScale = 1) {
        const user = Number.isFinite(userScale) && userScale > 0 ? userScale : 1;
        const styleScale = fit?.scale ?? 1;
        if (fit?.frameWidth > 0 && transform?.faceWidth > 0) {
            return transform.faceWidth * GLASSES_FACE_FIT_RATIO / fit.frameWidth * styleScale * user;
        }
        return transform.span / GLASSES_REF_EYE_SPAN * styleScale * user;
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

        ctx.fillStyle = "rgba(255,255,255,0.45)";
        for (let i = 0; i < lm.length; i += 3) {
            ctx.beginPath();
            ctx.arc(lm[i].x * w, lm[i].y * h, 1.4, 0, Math.PI * 2);
            ctx.fill();
        }

        const vertPts = [LM.FOREHEAD, LM.GLABELLA, LM.NOSE_BASE, LM.CHIN].map((i) => ({ x: lm[i].x * w, y: lm[i].y * h }));
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(vertPts[0].x, vertPts[0].y);
        for (let i = 1; i < vertPts.length; i++) ctx.lineTo(vertPts[i].x, vertPts[i].y);
        ctx.stroke();
        ctx.setLineDash([]);

        const cheekL = { x: lm[LM.CHEEK_L].x * w, y: lm[LM.CHEEK_L].y * h };
        const cheekR = { x: lm[LM.CHEEK_R].x * w, y: lm[LM.CHEEK_R].y * h };
        ctx.strokeStyle = "#93A8CF";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cheekL.x, cheekL.y);
        ctx.lineTo(cheekR.x, cheekR.y);
        ctx.stroke();

        ctx.fillStyle = "#3E5C9A";
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
        if (Number.isFinite(next.faceWidth)) {
            this.state.faceWidth = Number.isFinite(this.state.faceWidth) ? lerp(this.state.faceWidth, next.faceWidth, t) : next.faceWidth;
        }
        this.state.angle += angleDelta(this.state.angle, next.angle) * t;
        return { ...this.state };
    }
}

/**
 * Reference eye-to-eye span (px) every glasses shape below is calibrated
 * against; the live scale factor is the tracked eye span divided by this
 * constant. For the raster styles, each image's own lens-center span was
 * measured directly from the source photo and baked into its x/y/width/
 * height below, so at scale=1 the photo's two lens centers land exactly on
 * the tracked eye centers.
 */
export const GLASSES_REF_EYE_SPAN = 180;

/**
 * Style tuning for the 28 photo-based styles. frameWidth is the image's
 * drawn width in the shape's local units (same as its <image width>), used by
 * ArRenderer.fitGlassesScale to size the frame to the wearer's face width.
 * verticalOffsetRatio nudges the eye-line anchor slightly down toward the
 * nose bridge; scale is a small per-style fine-tune multiplier.
 */
export const GLASSES_STYLE_CONFIG = Object.freeze({
    "aviator-gold": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 352.25 },
    "browline-gold": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 349.12 },
    "browline-gold2": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 345.96 },
    "browline-silver": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 343.5 },
    "cateye-amber": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 336.73 },
    "cateye-black": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 337.58 },
    "cateye-red": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 341.83 },
    "classic-tortoise": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 350.29 },
    "clear-square": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 335.76 },
    "colorblock-square": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 352.47 },
    "hexagon-gold": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 345.58 },
    "hexagon-gunmetal": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 350.76 },
    "hexagon-tortoise": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 356.22 },
    "oval-gold-wire": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 347.57 },
    "rect-thin-metal": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 348.12 },
    "rimless-silver-bar": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 365.3 },
    "rimless-gunmetal-square": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 352.8 },
    "rimless-gunmetal-aviator": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 392.9 },
    "rimless-rosegold-cateye": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 348.2 },
    "round-gold-wire": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 349.78 },
    "round-gunmetal": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 353.63 },
    "round-tortoise": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 360.69 },
    "sport-wrap": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 412.79 },
    "square-black": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 354.08 },
    "square-bold-tortoise": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 347.47 },
    "square-green": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 354.37 },
    "square-oversized-tortoise": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 333.61 },
    "wayfarer-black": { scale: 1.0, verticalOffsetRatio: .02, frameWidth: 352.18 },
});

/**
 * SVG markup for each glasses style: a single <image> referencing a cropped,
 * transparent-background photo in assets/glasses/. Each photo's two lens
 * centers were located (background-distance masking + nose-bridge gap
 * detection), and x/y/width/height below place and scale the image so that,
 * at scale=1, those two lens centers sit at exactly the tracked eye-center
 * positions (i.e. GLASSES_REF_EYE_SPAN apart) around local origin (0,0) —
 * the same convention the previous hand-drawn shapes used.
 */
export const GLASSES_SHAPES = {
    "aviator-gold": `<image href="assets/glasses/aviator-gold.png" x="-173.31" y="-80.12" width="352.25" height="153.88" preserveAspectRatio="none"/>`,
    "browline-gold": `<image href="assets/glasses/browline-gold.png" x="-171.51" y="-52.02" width="349.12" height="134.69" preserveAspectRatio="none"/>`,
    "browline-gold2": `<image href="assets/glasses/browline-gold2.png" x="-170.97" y="-52.51" width="345.96" height="133.67" preserveAspectRatio="none"/>`,
    "browline-silver": `<image href="assets/glasses/browline-silver.png" x="-170.9" y="-51.0" width="343.5" height="127.59" preserveAspectRatio="none"/>`,
    "cateye-amber": `<image href="assets/glasses/cateye-amber.png" x="-169.42" y="-65.59" width="336.73" height="145.66" preserveAspectRatio="none"/>`,
    "cateye-black": `<image href="assets/glasses/cateye-black.png" x="-168.51" y="-65.4" width="337.58" height="142.14" preserveAspectRatio="none"/>`,
    "cateye-red": `<image href="assets/glasses/cateye-red.png" x="-171.01" y="-58.66" width="341.83" height="133.23" preserveAspectRatio="none"/>`,
    "classic-tortoise": `<image href="assets/glasses/classic-tortoise.png" x="-173.0" y="-62.98" width="350.29" height="131.36" preserveAspectRatio="none"/>`,
    "clear-square": `<image href="assets/glasses/clear-square.png" x="-167.96" y="-62.09" width="335.76" height="134.65" preserveAspectRatio="none"/>`,
    "colorblock-square": `<image href="assets/glasses/colorblock-square.png" x="-175.21" y="-67.01" width="352.47" height="143.19" preserveAspectRatio="none"/>`,
    "hexagon-gold": `<image href="assets/glasses/hexagon-gold.png" x="-173.12" y="-79.58" width="345.58" height="145.51" preserveAspectRatio="none"/>`,
    "hexagon-gunmetal": `<image href="assets/glasses/hexagon-gunmetal.png" x="-176.35" y="-70.1" width="350.76" height="134.63" preserveAspectRatio="none"/>`,
    "hexagon-tortoise": `<image href="assets/glasses/hexagon-tortoise.png" x="-175.61" y="-85.95" width="356.22" height="166.23" preserveAspectRatio="none"/>`,
    "oval-gold-wire": `<image href="assets/glasses/oval-gold-wire.png" x="-173.75" y="-61.14" width="347.57" height="114.09" preserveAspectRatio="none"/>`,
    "rect-thin-metal": `<image href="assets/glasses/rect-thin-metal.png" x="-175.45" y="-54.48" width="348.12" height="107.88" preserveAspectRatio="none"/>`,
    "rimless-silver-bar": `<image href="assets/glasses/rimless-silver-bar.png" x="-178.1" y="-55.8" width="365.3" height="115.4" preserveAspectRatio="none"/>`,
    "rimless-gunmetal-square": `<image href="assets/glasses/rimless-gunmetal-square.png" x="-173.7" y="-68.9" width="352.8" height="134.6" preserveAspectRatio="none"/>`,
    "rimless-gunmetal-aviator": `<image href="assets/glasses/rimless-gunmetal-aviator.png" x="-191.4" y="-59.5" width="392.9" height="138.6" preserveAspectRatio="none"/>`,
    "rimless-rosegold-cateye": `<image href="assets/glasses/rimless-rosegold-cateye.png" x="-170.6" y="-52.3" width="348.2" height="112.1" preserveAspectRatio="none"/>`,
    "round-gold-wire": `<image href="assets/glasses/round-gold-wire.png" x="-174.1" y="-73.89" width="349.78" height="138.44" preserveAspectRatio="none"/>`,
    "round-gunmetal": `<image href="assets/glasses/round-gunmetal.png" x="-177.12" y="-86.19" width="353.63" height="158.66" preserveAspectRatio="none"/>`,
    "round-tortoise": `<image href="assets/glasses/round-tortoise.png" x="-176.18" y="-62.86" width="360.69" height="134.69" preserveAspectRatio="none"/>`,
    "sport-wrap": `<image href="assets/glasses/sport-wrap.png" x="-204.45" y="-78.01" width="412.79" height="159.89" preserveAspectRatio="none"/>`,
    "square-black": `<image href="assets/glasses/square-black.png" x="-174.29" y="-65.26" width="354.08" height="141.08" preserveAspectRatio="none"/>`,
    "square-bold-tortoise": `<image href="assets/glasses/square-bold-tortoise.png" x="-171.98" y="-65.71" width="347.47" height="138.11" preserveAspectRatio="none"/>`,
    "square-green": `<image href="assets/glasses/square-green.png" x="-176.74" y="-66.78" width="354.37" height="144.1" preserveAspectRatio="none"/>`,
    "square-oversized-tortoise": `<image href="assets/glasses/square-oversized-tortoise.png" x="-165.82" y="-82.36" width="333.61" height="164.2" preserveAspectRatio="none"/>`,
    "wayfarer-black": `<image href="assets/glasses/wayfarer-black.png" x="-175.1" y="-62.09" width="352.18" height="134.37" preserveAspectRatio="none"/>`,
};
