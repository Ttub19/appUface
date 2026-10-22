import test from "node:test";
import assert from "node:assert/strict";
import { LM } from "../js/landmarks.js";
import { computeHairPlacement, eulerFromTransformMatrix, faceEllipse, insideEllipse, PlacementSmoother, measureHairlineLift, HeadScaleFusion } from "../js/hairTryOn/headPose.js";
import { buildEraseLayer, dilateMask, pushPullFill } from "../js/hairTryOn/hairEraser.js";
import { HAIR_STYLES, HAIR_CATEGORIES, HAIR_LANDMARK_WIDTH_RATIO, LANDMARK10_BELOW_HAIRLINE, stylesForFaceShape, filterHairStyles, recommendHairStyles, genderCategory } from "../js/hairTryOn/hairAssets.js";
import { HAIR_COLORS, recolorPixels, lightingFactor, quantizeColor, resolveHairRgb } from "../js/hairTryOn/hairLook.js";
import { existsSync, readFileSync } from "node:fs";

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

/** 4x4 pose = T · S · Ry(yaw) · Rx(pitch) · Rz(roll), returned in both layouts. */
function poseMatrix(yaw, pitch, roll, scale = 1.3) {
    const [cy, sy, cx, sx, cz, sz] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch), Math.cos(roll), Math.sin(roll)];
    const mul = (A, B) => A.map((row, i) => B[0].map((_, j) => row.reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
    const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
    const Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]];
    const Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
    const R = mul(mul(Ry, Rx), Rz).map((row) => row.map((v) => v * scale));
    const t = [1.5, -2, -48];
    const M = [[...R[0], t[0]], [...R[1], t[1]], [...R[2], t[2]], [0, 0, 0, 1]];
    return { rowMajor: M.flat(), colMajor: [0, 1, 2, 3].flatMap((c) => M.map((row) => row[c])) };
}

function face({ noseX = .5, angle = 0 } = {}) {
    const lm = Array.from({ length: 478 }, () => ({ x: .5, y: .4, z: 0 }));
    const set = (i, x, y) => { lm[i] = { x, y, z: 0 }; };
    set(LM.FOREHEAD, .5, .2); set(LM.FOREHEAD_CENTER, .5, .26);
    set(LM.TEMPLE_L, .4, .35); set(LM.TEMPLE_R, .6, .35);
    set(LM.CHEEK_L, .39, .42); set(LM.CHEEK_R, .61, .42);
    set(LM.NOSE_TIP, noseX, .45); set(LM.CHIN, .5, .6);
    if (!angle) return lm;
    const c = Math.cos(angle), s = Math.sin(angle);
    return lm.map((p) => ({ ...p, x: .5 + (p.x - .5) * c - (p.y - .4) * s, y: .4 + (p.x - .5) * s + (p.y - .4) * c }));
}

const ASSET = { headWidth: 200, depth: .4 };

test("euler angles from pose matrix (column- and row-major, scaled)", () => {
    const { rowMajor, colMajor } = poseMatrix(.4, -.2, .15);
    for (const data of [rowMajor, colMajor]) {
        const e = eulerFromTransformMatrix(data);
        close(e.yaw, .4); close(e.pitch, -.2); close(e.roll, .15);
    }
});

test("euler rejects malformed matrices", () => {
    assert.equal(eulerFromTransformMatrix(null), null);
    assert.equal(eulerFromTransformMatrix(new Array(16).fill(0)), null);
    assert.equal(eulerFromTransformMatrix([NaN, ...new Array(15).fill(1)]), null);
});

test("frontal face: hair anchored on landmark 10 at 1:1 scale", () => {
    const p = computeHairPlacement(face(), 1000, 1000, { yaw: 0, pitch: 0 }, ASSET);
    close(p.x, 500); close(p.y, 200); close(p.rotation, 0); close(p.scaleX, 1); close(p.scaleY, 1);
});

test("head roll rotates the hair with the temple line", () => {
    const p = computeHairPlacement(face({ angle: .3 }), 1000, 1000, null, ASSET);
    close(p.rotation, .3, 1e-9);
});

test("yaw keeps true head size and counter-shifts hair toward the skull", () => {
    const right = computeHairPlacement(face({ noseX: .55 }), 1000, 1000, { yaw: .5, pitch: 0 }, ASSET);
    const left = computeHairPlacement(face({ noseX: .45 }), 1000, 1000, { yaw: -.5, pitch: 0 }, ASSET);
    assert.ok(right.x < 500 && left.x > 500, "shift opposes the nose direction");
    close(right.x - 500, -(left.x - 500));
    assert.ok(right.scaleY > 1.1, "projected width is corrected by cos(yaw)");
    assert.ok(right.scaleX < right.scaleY, "mild horizontal foreshortening");
    assert.equal(Math.sign(right.yaw), 1);
});

test("pitch only squashes vertically", () => {
    const p = computeHairPlacement(face(), 1000, 1000, { yaw: 0, pitch: .5 }, ASSET);
    close(p.scaleX, 1); assert.ok(p.scaleY < 1);
});

test("face ellipse contains the nose and excludes the frame corner", () => {
    const e = faceEllipse(face(), 1000, 1000);
    assert.ok(insideEllipse(e, 500, 450));
    assert.ok(!insideEllipse(e, 10, 10));
});

test("placement smoother wraps rotation and snaps after a gap", () => {
    const s = new PlacementSmoother({ timeConstantMs: 50, snapAfterMs: 300 });
    const base = { x: 0, y: 0, scaleX: 1, scaleY: 1 };
    s.update({ ...base, rotation: Math.PI - .02 }, 0);
    assert.ok(Math.abs(s.update({ ...base, rotation: -Math.PI + .02 }, 16).rotation) > 3);
    close(s.update({ ...base, x: 100, rotation: 0 }, 1000).x, 100);
});

test("erase layer fills hair with background outside the face and skin inside", () => {
    const W = 20, H = 20, SKIN = [200, 150, 120], BG = [20, 200, 20];
    const ellipse = { cx: 10, cy: 12, rx: 5, ry: 6, angle: 0 };
    const conf = new Float32Array(W * H);
    const frame = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const hair = y < 4 || (y === 7 && x === 10); // a hair cap + one strand over the forehead
        conf[i] = hair ? 1 : 0;
        const c = hair ? [0, 0, 0] : insideEllipse(ellipse, x, y) ? SKIN : BG;
        frame.set([...c, 255], i * 4);
    }
    const out = buildEraseLayer({ conf, frame, width: W, height: H, ellipse, skinPoint: { x: 10, y: 11 } });
    const px = (x, y) => Array.from(out.rgba.slice((y * W + x) * 4, (y * W + x) * 4 + 4));
    assert.deepEqual(px(2, 1), [...BG, 255]);
    assert.deepEqual(px(10, 7), [...SKIN, 255]);
    assert.equal(px(2, 15)[3], 0, "non-hair stays transparent");
    close(out.coverage, (W * 4 + 1) / (W * H));
});

test("every face shape has at least one hairstyle", () => {
    for (const shape of ["oval", "round", "square", "heart", "oblong"]) assert.ok(stylesForFaceShape(shape).length > 0, shape);
    assert.equal(stylesForFaceShape("all").length, HAIR_STYLES.length);
});

test("every hairstyle image is on disk and its calibration fits inside it", () => {
    const ids = new Set();
    for (const s of HAIR_STYLES) {
        assert.ok(!ids.has(s.id), `duplicate ${s.id}`); ids.add(s.id);
        assert.ok(existsSync(new URL(`../${s.src}`, import.meta.url)), s.src);
        assert.ok(s.anchor.x > 0 && s.anchor.x < s.viewBox.width, `${s.id} anchor x`);
        assert.ok(s.anchor.y > 0 && s.anchor.y < s.viewBox.height, `${s.id} anchor y`);
        assert.ok(s.faceWidthPx > 100 && s.faceWidthPx < s.viewBox.width, `${s.id} face width`);
    }
});

test("categories: every style is in at least one and every chip has styles", () => {
    const known = new Set(HAIR_CATEGORIES.map((c) => c.id));
    for (const s of HAIR_STYLES) {
        assert.ok(s.categories.length && s.categories.every((c) => known.has(c) && c !== "all"), s.id);
        assert.ok(s.categories.includes(s.gender === "male" ? "men" : "women"), `${s.id} gender chip`);
    }
    for (const c of HAIR_CATEGORIES) assert.ok(filterHairStyles({ category: c.id }).length > 0, c.id);
    assert.equal(filterHairStyles().length, HAIR_STYLES.length);
});

test("face-shape and category filters combine", () => {
    const both = filterHairStyles({ shape: "round", category: "women" });
    assert.ok(both.length > 0);
    for (const s of both) assert.ok(s.faceShapes.includes("round") && s.categories.includes("women"), s.id);
});

test("live-camera calibration: size ratio and hairline above landmark 10", () => {
    for (const s of HAIR_STYLES) {
        close(s.headWidth, s.faceWidthPx * HAIR_LANDMARK_WIDTH_RATIO);
        // landmark 10 sits below the hairline, so the anchor is lower than the hairline
        close(s.anchor.y - s.hairlineY, s.faceWidthPx * LANDMARK10_BELOW_HAIRLINE);
    }
});

test("smoother: steady head is smoothed, fast move is followed closely", () => {
    const base = { y: 0, rotation: 0, scaleX: 1, scaleY: 1 };
    // jitter of ±2px around 100 at 30fps should be damped
    const s = new PlacementSmoother();
    let t = 0, last;
    s.update({ ...base, x: 100 }, t);
    for (let i = 0; i < 30; i++) { t += 33; last = s.update({ ...base, x: 100 + (i % 2 ? 2 : -2) }, t); }
    assert.ok(Math.abs(last.x - 100) < 1.2, `jitter damped: ${last.x}`);
    // a fast 300px move should be mostly caught up within ~100ms
    const f = new PlacementSmoother(); t = 0;
    f.update({ ...base, x: 0 }, t);
    for (let i = 1; i <= 3; i++) { t += 33; last = f.update({ ...base, x: 100 * i }, t); }
    assert.ok(last.x > 200, `follows fast motion: ${last.x}`);
});

test("erase fill uses the background beside the head on each side, not one average", () => {
    const W = 30, H = 10, LEFT = [230, 240, 180], RIGHT = [250, 250, 250], SKIN = [200, 150, 120];
    const ellipse = { cx: 15, cy: 30, rx: 4, ry: 4, angle: 0 }; // face far below: every pixel here is "outside"
    const conf = new Float32Array(W * H), frame = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, hair = x >= 8 && x < 22;
        conf[i] = hair ? 1 : 0;
        frame.set([...(hair ? [0, 0, 0] : x < 15 ? LEFT : RIGHT), 255], i * 4);
    }
    const out = buildEraseLayer({ conf, frame, width: W, height: H, ellipse, skinPoint: { x: 15, y: 30 } });
    const px = (x, y) => Array.from(out.rgba.slice((y * W + x) * 4, (y * W + x) * 4 + 3));
    const dist = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
    assert.ok(dist(px(9, 5), LEFT) < dist(px(9, 5), RIGHT), `left edge filled from the left: ${px(9, 5)}`);
    assert.ok(dist(px(20, 5), RIGHT) < dist(px(20, 5), LEFT), `right edge filled from the right: ${px(20, 5)}`);
    assert.ok(px(15, 5)[0] > LEFT[0] && px(15, 5)[0] < RIGHT[0], "the middle blends smoothly between the two sides");
});


// ---------------------------------------------------------------- recommendations
test("recommendations: best match first, gender respected, all suit the face shape", () => {
    for (const shape of ["oval", "round", "square", "heart", "oblong"]) {
        for (const gender of ["male", "female"]) {
            const recs = recommendHairStyles({ shape, gender });
            assert.ok(recs.length >= 3 && recs.length <= 4, `${shape}/${gender}: ${recs.length}`);
            for (const s of recs) assert.ok(s.faceShapes.includes(shape) && s.gender === gender, `${shape}/${gender} ${s.id}`);
            const ranks = recs.map((s) => s.faceShapes.indexOf(shape));
            assert.deepEqual(ranks, [...ranks].sort((a, b) => a - b), `${shape}/${gender} sorted by match strength`);
        }
    }
    assert.deepEqual(recommendHairStyles({ shape: null }), []);
    assert.equal(genderCategory("male"), "men");
    assert.equal(genderCategory("female"), "women");
    assert.equal(genderCategory(null), "all");
});

// ---------------------------------------------------------------- hair look
test("recolour keeps the strand shading and moves the mean to the target colour", () => {
    const px = [[40, 30, 25, 255], [80, 60, 50, 255], [60, 45, 38, 255], [0, 0, 0, 0]];
    const data = new Uint8ClampedArray(px.flat());
    recolorPixels(data, [98, 72, 54]);
    const lum = (i) => .299 * data[i] + .587 * data[i + 1] + .114 * data[i + 2];
    assert.ok(lum(0) < lum(8) && lum(8) < lum(4), "dark strand stays darker than light strand");
    assert.ok(Math.abs(data[8] - 98) < 12 && Math.abs(data[9] - 72) < 12, `mid strand near target: ${data.slice(8, 11)}`);
    assert.deepEqual(Array.from(data.slice(12)), [0, 0, 0, 0], "transparent pixels untouched");
    const blonde = new Uint8ClampedArray([200, 200, 200, 255, 20, 20, 20, 255]);
    recolorPixels(blonde, [182, 146, 100]);
    assert.ok(blonde[0] < 255, "highlights roll off instead of clipping to white");
});

test("lighting factor follows the scene and stays within limits", () => {
    close(lightingFactor([150, 150, 150]), 1, 1e-9);
    assert.ok(lightingFactor([60, 50, 45]) < 1 && lightingFactor([60, 50, 45]) >= .7);
    assert.ok(lightingFactor([255, 255, 255]) <= 1.15);
    assert.equal(lightingFactor(null), 1);
});

test("hair colour presets resolve, 'mine' uses the sampled colour quantised", () => {
    assert.ok(HAIR_COLORS.some((c) => c.id === "mine") && HAIR_COLORS.some((c) => c.id === "original"));
    assert.equal(resolveHairRgb("original", [10, 10, 10]), null);
    assert.deepEqual(resolveHairRgb("black", null), HAIR_COLORS.find((c) => c.id === "black").rgb);
    assert.equal(resolveHairRgb("mine", null), null, "no sample yet -> original colours");
    assert.deepEqual(resolveHairRgb("mine", [31, 25, 19.9]), [36, 24, 24]);
    assert.deepEqual(quantizeColor([31, 25, 19.9]), resolveHairRgb("mine", [33, 27, 21]), "tiny changes don't rebuild");
});

// ---------------------------------------------------------------- eraser internals
test("mask dilation grows hair by the radius in both directions", () => {
    const W = 9, H = 9, m = new Float32Array(W * H);
    m[4 * W + 4] = 1;
    const d = dilateMask(m, W, H, 2);
    assert.equal(d[2 * W + 2], 1); assert.equal(d[6 * W + 6], 1);
    assert.equal(d[1 * W + 4], 0); assert.equal(d[4 * W + 7], 0);
});

test("push-pull fill: exact on samples, smooth in holes, null when nothing valid", () => {
    const W = 16, H = 4, frame = new Uint8ClampedArray(W * H * 4), valid = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, v = x < 4 ? 0 : x >= 12 ? 200 : 77;
        frame.set([v, v, v, 255], i * 4);
        valid[i] = x < 4 || x >= 12 ? 1 : 0;
    }
    const out = pushPullFill(frame, valid, W, H);
    close(out[(1 * W + 1) * 3], 0); close(out[(1 * W + 14) * 3], 200);
    const row = Array.from({ length: 8 }, (_, k) => out[(1 * W + 4 + k) * 3]);
    for (let k = 1; k < row.length; k++) assert.ok(row[k] >= row[k - 1] - 1e-6, `monotonic ramp across the hole: ${row.map((v) => v.toFixed(0))}`);
    assert.ok(row[0] < 100 && row[7] > 100, "hole ends take the colour of their side");
    assert.equal(pushPullFill(frame, new Uint8Array(W * H), W, H), null);
});

test("erase layer samples the user's hair colour away from the face", () => {
    const W = 24, H = 24, HAIR = [30, 22, 18];
    const ellipse = { cx: 12, cy: 15, rx: 5, ry: 6, angle: 0 };
    const conf = new Float32Array(W * H), frame = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, hair = y < 6;
        conf[i] = hair ? 1 : 0;
        frame.set([...(hair ? HAIR : insideEllipse(ellipse, x, y) ? [200, 150, 120] : [90, 160, 90]), 255], i * 4);
    }
    const out = buildEraseLayer({ conf, frame, width: W, height: H, ellipse, skinPoint: { x: 12, y: 14 } });
    assert.deepEqual(out.hair.map(Math.round), HAIR);
});

// ---------------------------------------------------------------- placement
test("hairline: measured above landmark 10 in temple widths; bangs and bald give null", () => {
    const W = 200, H = 200, lm = face(); // lm10 at (100,40), temples 40px apart, chin below
    const mask = (hairTopY) => {
        const conf = new Float32Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (y <= hairTopY) conf[y * W + x] = 1;
        return conf;
    };
    const lift = measureHairlineLift({ conf: mask(28), width: W, height: H, landmarks: lm });
    assert.ok(Math.abs(lift - 12 / 40) < .03, `hair 12px above lm10 / 40px temples: ${lift}`);
    assert.equal(measureHairlineLift({ conf: mask(41), width: W, height: H, landmarks: lm }), null, "bangs over landmark 10");
    assert.equal(measureHairlineLift({ conf: new Float32Array(W * H), width: W, height: H, landmarks: lm }), null, "no hair");
});

test("head scale fusion: frontal = temple width, turned head stays near the learned size", () => {
    const f = new HeadScaleFusion();
    const lm = face();
    lm[LM.NOSE_BASE] = { x: .5, y: .48, z: 0 };
    for (let i = 0; i < 60; i++) close(f.update(lm, 1000, 1000, { yaw: 0, pitch: 0 }), 200, 1e-6);
    // turned 0.6 rad: temples foreshortened by cos, but the model misplaces the far temple (width 20% short)
    const turned = lm.map((p) => ({ ...p }));
    turned[LM.TEMPLE_L] = { x: .5 - .1 * Math.cos(.6) * .8, y: .35, z: 0 };
    turned[LM.TEMPLE_R] = { x: .5 + .1 * Math.cos(.6) * .8, y: .35, z: 0 };
    const naive = 200 * .8;
    const fused = f.update(turned, 1000, 1000, { yaw: .6, pitch: 0 });
    assert.ok(Math.abs(fused - 200) < Math.abs(naive - 200), `fused ${fused.toFixed(1)} closer to 200 than ${naive}`);
    const fresh = new HeadScaleFusion();
    close(fresh.update(turned, 1000, 1000, { yaw: .6, pitch: 0 }), naive, 1e-6);
});

test("placement uses an explicit head width when given", () => {
    const p = computeHairPlacement(face(), 1000, 1000, { yaw: 0, pitch: 0 }, ASSET, 300);
    close(p.scaleX, 1.5); close(p.scaleY, 1.5);
});
