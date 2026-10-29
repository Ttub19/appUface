import test from "node:test";
import assert from "node:assert/strict";
import { ArTransformSmoother, GLASSES_SHAPES, GLASSES_TEMPLES, TEMPLE_LENGTH_RATIO, estimateYaw, templeArm } from "../js/arRenderer.js";
import { LM } from "../js/landmarks.js";

function face({ nose = .5 } = {}) {
    const lm = Array.from({ length: 478 }, () => ({ x: .5, y: .5, z: 0 }));
    lm[LM.TEMPLE_L] = { x: .35, y: .35, z: 0 };
    lm[LM.TEMPLE_R] = { x: .65, y: .35, z: 0 };
    lm[LM.NOSE_TIP] = { x: nose, y: .45, z: 0 };
    lm[LM.CHEEK_L] = { x: .30, y: .40, z: 0 };
    lm[LM.CHEEK_R] = { x: .70, y: .40, z: 0 };
    return lm;
}
const T = (yaw, angle = 0) => ({ cx: 500, cy: 350, angle, span: 180, yaw, faceWidth: 400 });
const temple = { l: [-160, -20], r: [160, -20], width: 9, color: "#111111" };

test("every glasses style has temple data inside its image", () => {
    assert.deepEqual(Object.keys(GLASSES_TEMPLES).sort(), Object.keys(GLASSES_SHAPES).sort());
    for (const [id, t] of Object.entries(GLASSES_TEMPLES)) {
        assert.ok(t.l[0] < 0 && t.r[0] > 0, id);
        assert.ok(t.width >= 4 && t.width <= 12 && /^#[0-9a-f]{6}$/.test(t.color), id);
    }
});

test("yaw estimate: frontal ~0, sign follows the nose", () => {
    assert.ok(Math.abs(estimateYaw(face(), 1000, 1000)) < 1e-9);
    assert.ok(estimateYaw(face({ nose: .56 }), 1000, 1000) > .2);
    assert.ok(estimateYaw(face({ nose: .44 }), 1000, 1000) < -.2);
});

test("no arm when facing the camera", () => {
    assert.equal(templeArm(T(0), 1, temple), null);
    assert.equal(templeArm(T(.08), 1, temple), null);
});

test("turned head: only the arm on the visible side, pointing outwards and slightly down", () => {
    const arm = templeArm(T(.4), 1, temple);
    assert.equal(arm.side, "left");
    assert.ok(arm.opacity > .99);
    const end = [(arm.points[1][0] + arm.points[2][0]) / 2, (arm.points[1][1] + arm.points[2][1]) / 2];
    const len = TEMPLE_LENGTH_RATIO * 400 * Math.sin(.4);
    assert.ok(Math.abs(end[0] - (340 - len)) < 1, String(end));
    assert.ok(end[1] > 330 && end[1] < 330 + len * .3, String(end));
    const right = templeArm(T(-.4), 1, temple);
    assert.equal(right.side, "right");
    assert.ok(right.points[1][0] > 660 + len - 2);
});

test("arm follows head roll", () => {
    const arm = templeArm(T(.4, .3), 1, temple);
    const dy = (arm.points[1][1] + arm.points[2][1]) / 2 - (arm.points[0][1] + arm.points[3][1]) / 2;
    assert.ok(dy < 0, "rolled head: the outward arm on the left rises"); // left side goes up when angle > 0
});

test("longer arm when turned further", () => {
    const len = (a) => Math.hypot(a.points[1][0] - a.points[0][0], a.points[1][1] - a.points[0][1]);
    assert.ok(len(templeArm(T(.6), 1, temple)) > len(templeArm(T(.3), 1, temple)));
});

test("adaptive smoother: steadier when still, quicker when moving", () => {
    const still = new ArTransformSmoother({ timeConstantMs: 85 });
    still.update({ cx: 0, cy: 0, span: 100, angle: 0 }, 0);
    const a = still.update({ cx: 1, cy: 0, span: 100, angle: 0 }, 33).cx;           // 1 px jitter
    const fast = new ArTransformSmoother({ timeConstantMs: 85 });
    fast.update({ cx: 0, cy: 0, span: 100, angle: 0 }, 0);
    const b = fast.update({ cx: 100, cy: 0, span: 100, angle: 0 }, 33).cx / 100;    // a whole eye-span
    assert.ok(a < b, `${a} vs ${b}`);
    const yaw = still.update({ cx: 1, cy: 0, span: 100, angle: 0, yaw: .3 }, 66);
    assert.equal(yaw.yaw, .3);   // new fields start at their value
});
