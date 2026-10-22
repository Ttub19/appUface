/** MediaPipe Face Mesh indices used by measurement and AR. */
export const LM = Object.freeze({
    FOREHEAD: 10,       // top of tracked face oval; not the true hairline
    FOREHEAD_CENTER: 151, // mid-forehead skin, used to sample skin tone
    GLABELLA: 168,
    NOSE_BRIDGE: 6,
    NOSE_TIP: 1,
    NOSE_BASE: 2,
    CHIN: 152,
    CHEEK_L: 234,
    CHEEK_R: 454,
    TEMPLE_L: 127,
    TEMPLE_R: 356,
    JAW_L: 172,
    JAW_R: 397,
    EYE_L_OUTER: 33,
    EYE_L_INNER: 133,
    EYE_R_INNER: 362,
    EYE_R_OUTER: 263,
    BROW_L_OUTER: 70,
    BROW_L_INNER: 107,
    BROW_R_INNER: 336,
    BROW_R_OUTER: 300,
    MOUTH_L: 61,
    MOUTH_R: 291,
    IRIS_A: 468, // optional when refineLandmarks=true
    IRIS_B: 473,
});

export function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

export function isLandmarkPoint(point) {
    return Boolean(point) && Number.isFinite(point.x) && Number.isFinite(point.y);
}
