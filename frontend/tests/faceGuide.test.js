import test from "node:test";
import assert from "node:assert/strict";
import { FACE_PROPORTIONS, DIAGRAM_VIEWBOX, outlinePoints, smoothClosedPath, faceDiagramSVG } from "../js/faceDiagram.js";
import { faceData, faceGuide } from "../js/data.js";

const length = (p) => p.chin.y - p.top;

test("every classifier face shape has a diagram and a guide page", () => {
    for (const { id } of faceData) {
        assert.ok(FACE_PROPORTIONS[id], `missing proportions for ${id}`);
        const g = faceGuide[id];
        assert.ok(g, `missing guide for ${id}`);
        for (const key of ["A", "B", "C", "D"]) assert.ok(g.measures[key], `${id} missing measure ${key}`);
        assert.ok(g.traits.length >= 3 && g.hair.male.good.length && g.hair.female.good.length && g.verdict);
    }
});

test("diagram proportions match what the guide text claims", () => {
    const { oval, round, square, heart, oblong } = FACE_PROPORTIONS;
    // oval: cheekbones widest, forehead a bit wider than jaw, length ~1.5x width
    assert.ok(oval.cheek.w > oval.forehead.w && oval.forehead.w > oval.jaw.w);
    assert.ok(Math.abs(length(oval) / oval.cheek.w - 1.5) < 0.15);
    // round: length close to width
    assert.ok(length(round) / round.cheek.w < 1.25);
    // square: forehead, cheek and jaw within ~5% of each other
    const sq = [square.forehead.w, square.cheek.w, square.jaw.w];
    assert.ok(Math.max(...sq) / Math.min(...sq) < 1.05);
    // heart: forehead widest, jaw clearly narrower
    assert.ok(heart.forehead.w > heart.cheek.w && heart.jaw.w < heart.forehead.w * 0.65);
    // oblong: clearly longer than wide (>= 1.7x)
    assert.ok(length(oblong) / oblong.cheek.w >= 1.7);
});

test("outline is symmetric and stays inside the canvas", () => {
    for (const [id, p] of Object.entries(FACE_PROPORTIONS)) {
        const pts = outlinePoints(p);
        const cx = DIAGRAM_VIEWBOX.width / 2;
        const xs = pts.map((q) => q.x - cx).sort((a, b) => a - b);
        xs.forEach((x, i) => assert.ok(Math.abs(x + xs[xs.length - 1 - i]) < 1e-9, `${id} not symmetric`));
        for (const q of pts) assert.ok(q.x > 0 && q.x < DIAGRAM_VIEWBOX.width && q.y > 0 && q.y < DIAGRAM_VIEWBOX.height, `${id} point outside`);
        // Curve (C) segments by default; a shape whose chin is a flat edge (square)
        // legitimately closes with straight-line (L) segments instead — that's the
        // fix, not a regression, so both command letters are valid here.
        assert.match(smoothClosedPath(pts), /^M[\d. ]+([CL][\d. ]+)+Z$/);
    }
});

test("square's jaw is a real flat edge, not a curve pretending to be one", () => {
    const pts = outlinePoints(FACE_PROPORTIONS.square);
    const path = smoothClosedPath(pts);
    assert.ok(path.includes("L"), "square should close its chin with straight lines");
});

test("round and oblong taper to a blunt, non-pointed chin while heart stays sharp", () => {
    // A low chin.lift relative to chin.w keeps the spline's tangent at the
    // center point close to horizontal, which is what makes the curve round
    // through instead of coming to a point — see the file header for why.
    const flatness = (p) => (p.chin.lift || 0) / p.chin.w;
    assert.ok(flatness(FACE_PROPORTIONS.round) < flatness(FACE_PROPORTIONS.oval));
    assert.ok(flatness(FACE_PROPORTIONS.oval) < flatness(FACE_PROPORTIONS.heart));
    assert.ok(FACE_PROPORTIONS.round.chin.w / FACE_PROPORTIONS.round.jaw.w > 0.85, "round chin should stay nearly as wide as its jaw");
    assert.ok(FACE_PROPORTIONS.heart.chin.w / FACE_PROPORTIONS.heart.jaw.w < 0.35, "heart chin should taper to a narrow point");
});

test("labels toggle the A-D measurement guides", () => {
    const withLabels = faceDiagramSVG("heart");
    const plain = faceDiagramSVG("heart", { labels: false });
    for (const l of ["A", "B", "C", "D"]) assert.ok(withLabels.includes(`>${l}</text>`));
    assert.ok(!plain.includes("fd-measure"));
    assert.equal(faceDiagramSVG("triangle"), "");
});