import { LM, clamp, isLandmarkPoint } from "./landmarks.js";

const PHI = (1 + Math.sqrt(5)) / 2;
const EPS = 1e-6;
const REQUIRED = [
    LM.FOREHEAD, LM.GLABELLA, LM.NOSE_BRIDGE, LM.NOSE_TIP, LM.NOSE_BASE, LM.CHIN,
    LM.CHEEK_L, LM.CHEEK_R, LM.EYE_L_OUTER, LM.EYE_L_INNER,
    LM.EYE_R_INNER, LM.EYE_R_OUTER, LM.BROW_L_OUTER, LM.BROW_L_INNER,
    LM.BROW_R_INNER, LM.BROW_R_OUTER, LM.MOUTH_L, LM.MOUTH_R, LM.JAW_L, LM.JAW_R,
];
const PAIRS = [
    [LM.EYE_L_OUTER, LM.EYE_R_OUTER, 1], [LM.EYE_L_INNER, LM.EYE_R_INNER, 1],
    [LM.BROW_L_OUTER, LM.BROW_R_OUTER, .8], [LM.BROW_L_INNER, LM.BROW_R_INNER, .8],
    [LM.CHEEK_L, LM.CHEEK_R, 1.2], [LM.MOUTH_L, LM.MOUTH_R, .9], [LM.JAW_L, LM.JAW_R, .8],
];

function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function average(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
function thirdsScore(values) {
    const ideal = 100 / 3;
    return clamp(100 - values.reduce((sum, value) => sum + Math.abs(value - ideal), 0) / 3 * 6, 0, 100);
}
function ratioScore(value) { return clamp(100 - Math.abs(value - PHI) * 60, 0, 100); }

/**
 * Heuristic geometry comparison, not an attractiveness score or diagnosis.
 * Measurements are aligned to the eye axis to reduce head-roll error. Landmark
 * 10 is the top of the tracked face oval, not a reliably detected hairline.
 */
export const GoldenRatioCalc = {
    PHI,
    compute(lm, w, h) {
        if (!Array.isArray(lm)) throw new TypeError("landmarks must be an array");
        if (![w, h].every(Number.isFinite) || w <= 0 || h <= 0) throw new RangeError("invalid frame size");
        for (const index of REQUIRED) {
            if (!isLandmarkPoint(lm[index])) throw new RangeError(`missing or invalid face landmark at index ${index}`);
        }
        const point = (index) => ({ x: lm[index].x * w, y: lm[index].y * h });
        const eyeA = average(point(LM.EYE_L_OUTER), point(LM.EYE_L_INNER));
        const eyeB = average(point(LM.EYE_R_OUTER), point(LM.EYE_R_INNER));
        const [leftEye, rightEye] = eyeA.x <= eyeB.x ? [eyeA, eyeB] : [eyeB, eyeA];
        const dx = rightEye.x - leftEye.x;
        const dy = rightEye.y - leftEye.y;
        const eyeSpan = Math.hypot(dx, dy);
        if (eyeSpan <= EPS) throw new RangeError("eye landmarks are too close together");

        const ux = dx / eyeSpan, uy = dy / eyeSpan;
        const origin = average(leftEye, rightEye);
        const projectPoint = (p) => {
            const px = p.x - origin.x, py = p.y - origin.y;
            return { x: px * ux + py * uy, y: px * -uy + py * ux };
        };
        const project = (index) => projectPoint(point(index));
        const vertical = [LM.FOREHEAD, LM.GLABELLA, LM.NOSE_BASE, LM.CHIN].map(project);
        const rawSegments = [
            vertical[1].y - vertical[0].y,
            vertical[2].y - vertical[1].y,
            vertical[3].y - vertical[2].y,
        ];
        const ordered = rawSegments.every((value) => value > eyeSpan * .04);
        const segments = rawSegments.map(Math.abs);
        const faceHeight = segments.reduce((sum, value) => sum + value, 0);
        const cheekA = project(LM.CHEEK_L), cheekB = project(LM.CHEEK_R);
        const faceWidth = Math.abs(cheekB.x - cheekA.x);
        if (faceHeight <= EPS || faceWidth <= EPS) throw new RangeError("face dimensions are too small");

        const thirds = segments.map((value) => value / faceHeight * 100);
        const ratio = faceHeight / faceWidth;
        const scoreThirds = thirdsScore(thirds);
        const scoreRatio = ratioScore(ratio);
        const midlineX = median([
            project(LM.FOREHEAD).x, project(LM.GLABELLA).x, project(LM.NOSE_BRIDGE).x,
            project(LM.NOSE_BASE).x, project(LM.CHIN).x, (cheekA.x + cheekB.x) / 2, 0,
        ]);
        let deviation = 0, weights = 0;
        for (const [aIndex, bIndex, weight] of PAIRS) {
            const a = project(aIndex), b = project(bIndex);
            const horizontal = Math.abs(Math.abs(a.x - midlineX) - Math.abs(b.x - midlineX)) / faceWidth;
            const verticalDiff = Math.abs(a.y - b.y) / faceHeight;
            deviation += (horizontal * .75 + verticalDiff * .25) * weight;
            weights += weight;
        }
        const symmetryScore = clamp(100 - deviation / weights * 260, 0, 100);

        const rollDeg = Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);
        const nose = project(LM.NOSE_TIP);
        const projectedLeft = projectPoint(leftEye), projectedRight = projectPoint(rightEye);
        const noseLeft = Math.abs(nose.x - projectedLeft.x);
        const noseRight = Math.abs(projectedRight.x - nose.x);
        const yawAsymmetry = Math.abs(noseLeft - noseRight) / Math.max(noseLeft + noseRight, EPS);
        const noseBetweenEyes = nose.x > projectedLeft.x && nose.x < projectedRight.x;
        const faceSizeRatio = faceWidth / Math.min(w, h);
        const inside = [LM.FOREHEAD, LM.CHIN, LM.CHEEK_L, LM.CHEEK_R]
            .map((index) => lm[index]).every((p) => p.x >= .015 && p.x <= .985 && p.y >= .015 && p.y <= .985);
        const reasons = [];
        if (!ordered) reasons.push("vertical-order");
        if (rollDeg > 20) reasons.push("roll");
        if (!noseBetweenEyes || yawAsymmetry > .32) reasons.push("yaw");
        if (faceSizeRatio < .18) reasons.push("face-too-small");
        if (!inside) reasons.push("face-clipped");
        const qualityScore = clamp(((1 - clamp(rollDeg / 28, 0, 1)) * .25 +
            (1 - clamp(yawAsymmetry / .4, 0, 1)) * .35 +
            clamp((faceSizeRatio - .12) / .18, 0, 1) * .2 + (inside ? .1 : 0) + (ordered ? .1 : 0)) * 100, 0, 100);
        const overall = clamp(scoreThirds * .4 + scoreRatio * .35 + symmetryScore * .25, 0, 100);
        return {
            thirds, ratio, thirdsScore: scoreThirds, ratioScore: scoreRatio, symmetryScore, overall,
            quality: { isUsable: reasons.length === 0, score: qualityScore, reasons, rollDeg, yawAsymmetry, faceSizeRatio },
        };
    },
};

/** Median sliding window used to prevent saving a noisy single frame. */
export class GoldenRatioStabilizer {
    constructor({ windowSize = 15, minSamples = 7, minStabilityScore = 65 } = {}) {
        if (!Number.isInteger(windowSize) || windowSize < 3) throw new RangeError("windowSize must be at least 3");
        if (!Number.isInteger(minSamples) || minSamples < 2 || minSamples > windowSize) throw new RangeError("invalid minSamples");
        this.windowSize = windowSize;
        this.minSamples = minSamples;
        this.minStabilityScore = minStabilityScore;
        this.reset();
    }
    reset() { this.samples = []; }
    push(metrics) {
        if (!metrics?.quality?.isUsable) return null;
        this.samples.push(metrics);
        if (this.samples.length > this.windowSize) this.samples.shift();
        const rawThirds = [0, 1, 2].map((i) => median(this.samples.map((s) => s.thirds[i])));
        const total = rawThirds.reduce((sum, value) => sum + value, 0);
        const thirds = rawThirds.map((value) => value * 100 / Math.max(total, EPS));
        const ratio = median(this.samples.map((s) => s.ratio));
        const symmetryScore = median(this.samples.map((s) => s.symmetryScore));
        const scoreThirds = thirdsScore(thirds), scoreRatio = ratioScore(ratio);
        const overall = clamp(scoreThirds * .4 + scoreRatio * .35 + symmetryScore * .25, 0, 100);
        const center = median(this.samples.map((s) => s.overall));
        const mad = median(this.samples.map((s) => Math.abs(s.overall - center)));
        const stabilityScore = clamp(100 - mad * 15, 0, 100);
        return {
            thirds, ratio, thirdsScore: scoreThirds, ratioScore: scoreRatio, symmetryScore, overall,
            quality: { isUsable: true, reasons: [], score: median(this.samples.map((s) => s.quality.score)) },
            sampleCount: this.samples.length,
            stabilityScore,
            stable: this.samples.length >= this.minSamples && stabilityScore >= this.minStabilityScore,
        };
    }
}
