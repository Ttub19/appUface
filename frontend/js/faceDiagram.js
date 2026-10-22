/**
 * Line-drawn reference diagrams for the five face shapes (face-shape guide).
 *
 * Each shape is described by the four measurements the guide teaches people to
 * check in a mirror — forehead (A), cheekbones (B), jawline (C) and face
 * length (D) — on a shared 240×320 canvas.
 *
 * The outline is a closed curve through those points, but — unlike a single
 * uniform spline through every point — the chin/jaw segment is built
 * differently per shape (`chinType`), because that is the part people
 * actually use to tell the five shapes apart. Running every shape through
 * identical smoothing math made them all render as the same rounded teardrop
 * with slightly different proportions, which is the bug this file now fixes:
 *
 *  - "point" (oval / round / heart / oblong): the two sides of the jaw taper
 *    through a chin point to a single vertex at the center. How sharp that
 *    taper LOOKS is controlled by `chin.w` (how wide the jaw still is right
 *    before it meets center) and `chin.lift` (how far above the final chin.y
 *    that point sits): a small width + big lift makes the spline's tangent at
 *    the center point steep → a sharp V (heart). A width close to the jaw's
 *    own width + a tiny lift makes that tangent nearly horizontal → the curve
 *    rounds through smoothly with no point at all (round, and a narrower
 *    version of the same idea for oblong). A middling width/lift gives the
 *    soft, slightly-tapered point of a true oval.
 *  - "flat" (square): the jaw is closed with straight line segments instead
 *    of a curve at all, so it actually has flat edges and real corners,
 *    which a spline can never produce no matter how its control points are
 *    tuned.
 */

export const DIAGRAM_VIEWBOX = Object.freeze({ width: 240, height: 320 });
const CX = 120;

/**
 * y = vertical position, w = full width at that height.
 * chinType: "point" | "flat" — see file header.
 * chin.lift only applies to "point"; chin.w means "taper width" for "point"
 * and "flat-bottom width" for "flat".
 */
export const FACE_PROPORTIONS = Object.freeze({
    oval:   { top: 40, forehead: { y: 98, w: 128 }, cheek: { y: 152, w: 156 }, jaw: { y: 214, w: 116 }, chin: { y: 272, w: 60, lift: 8 }, chinType: "point" },
    round:  { top: 50, forehead: { y: 100, w: 150 }, cheek: { y: 158, w: 182 }, jaw: { y: 220, w: 160 }, chin: { y: 266, w: 146, lift: 4 }, chinType: "point" },
    square: { top: 40, forehead: { y: 92, w: 170 }, cheek: { y: 150, w: 176 }, jaw: { y: 228, w: 170 }, chin: { y: 262, w: 150 }, chinType: "flat", jawUpper: 198 },
    heart:  { top: 36, forehead: { y: 92, w: 180 }, cheek: { y: 150, w: 168 }, jaw: { y: 222, w: 104 }, chin: { y: 290, w: 24, lift: 16 }, chinType: "point" },
    oblong: { top: 18, forehead: { y: 84, w: 140 }, cheek: { y: 150, w: 146 }, jaw: { y: 248, w: 134 }, chin: { y: 294, w: 120, lift: 5 }, chinType: "point", jawUpper: 215 },
});

const r1 = (n) => Math.round(n * 10) / 10;
const mirror = (pt) => ({ x: 2 * CX - pt.x, y: pt.y });

/** The right-hand half of the outline, from the brow down to the jaw (not the chin — that's shape-specific, see outlinePoints). */
function rightChain(p) {
    return [
        { x: CX + p.forehead.w * 0.32, y: p.top + (p.forehead.y - p.top) * 0.18 },
        { x: CX + p.forehead.w / 2, y: p.forehead.y },
        { x: CX + p.cheek.w / 2, y: p.cheek.y },
        ...(p.jawUpper ? [{ x: CX + p.jaw.w / 2, y: p.jawUpper }] : []),
        { x: CX + p.jaw.w / 2, y: p.jaw.y },
    ];
}

/**
 * Clockwise outline points, starting at the top of the forehead.
 * Points may carry an `in` hint ("line") read by smoothClosedPath to say
 * how the segment ARRIVING at that point is drawn; no hint = smoothed curve
 * (the default, used for every shape except square's flat jaw).
 */
export function outlinePoints(p) {
    const right = rightChain(p); // ends at jaw-right
    const jawRight = right[right.length - 1];
    const jawLeft = mirror(jawRight);

    let chin;
    if (p.chinType === "flat") {
        const halfW = p.chin.w / 2;
        chin = [
            { x: CX + halfW, y: p.chin.y, in: "line" },
            { x: CX - halfW, y: p.chin.y, in: "line" },
        ];
    } else {
        const chinRight = { x: CX + p.chin.w / 2, y: p.chin.y - (p.chin.lift || 0) };
        chin = [chinRight, { x: CX, y: p.chin.y }, mirror(chinRight)];
    }
    const jawLeftPoint = p.chinType === "flat" ? { ...jawLeft, in: "line" } : jawLeft;

    const leftChain = right.slice(0, -1).reverse().map(mirror); // cheek..brow mirrored, jaw excluded (added explicitly above)
    return [{ x: CX, y: p.top }, ...right, ...chin, jawLeftPoint, ...leftChain];
}

/**
 * Closed outline path through the points. Each segment is a Catmull-Rom
 * cubic by default; a point marked `in: "line"` is instead joined to the
 * previous point with a straight line, which is how square gets real flat
 * edges and corners that no spline tuning can produce.
 */
export function smoothClosedPath(pts) {
    const n = pts.length;
    let d = `M${r1(pts[0].x)} ${r1(pts[0].y)}`;
    for (let i = 0; i < n; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % n];
        if (p2.in === "line") {
            d += `L${r1(p2.x)} ${r1(p2.y)}`;
            continue;
        }
        const p0 = pts[(i - 1 + n) % n];
        const p3 = pts[(i + 2) % n];
        const c1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 };
        const c2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 };
        d += `C${r1(c1.x)} ${r1(c1.y)} ${r1(c2.x)} ${r1(c2.y)} ${r1(p2.x)} ${r1(p2.y)}`;
    }
    return d + "Z";
}

function measureLine(label, y, w) {
    const x1 = CX - w / 2, x2 = CX + w / 2;
    return `<g class="fd-measure">
        <line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}"/>
        <line x1="${x1}" y1="${y - 5}" x2="${x1}" y2="${y + 5}"/><line x1="${x2}" y1="${y - 5}" x2="${x2}" y2="${y + 5}"/>
        <circle cx="${x1 - 13}" cy="${y}" r="9"/><text x="${x1 - 13}" y="${y + 3.5}">${label}</text>
    </g>`;
}

/**
 * @param {string} shapeId - oval | round | square | heart | oblong
 * @param {{labels?: boolean, title?: string}} [opts] - labels: draw the A–D measurement guides
 * @returns {string} SVG markup
 */
export function faceDiagramSVG(shapeId, opts = {}) {
    const p = FACE_PROPORTIONS[shapeId];
    if (!p) return "";
    const { labels = true, title = "" } = opts;
    const outline = smoothClosedPath(outlinePoints(p));

    // soft facial features so the outline reads as a face, not a blob
    const eyeY = p.cheek.y - 22, browY = eyeY - 15;
    const mouthY = r1(p.cheek.y + (p.jaw.y - p.cheek.y) * 0.62);
    const neckX = Math.max(p.jaw.w, 90) * 0.3;
    const features = `<g class="fd-feature">
        <path d="M${CX - 46} ${browY}q14 -7 28 0M${CX + 18} ${browY}q14 -7 28 0"/>
        <path d="M${CX - 45} ${eyeY}q13 -9 26 0q-13 7 -26 0ZM${CX + 19} ${eyeY}q13 -9 26 0q-13 7 -26 0Z"/>
        <path d="M${CX - 2} ${eyeY + 6}l-7 ${r1((mouthY - eyeY) * 0.62)}q7 4 15 0"/>
        <path d="M${CX - 17} ${mouthY}q17 9 34 0"/>
        <path d="M${CX - neckX} ${p.chin.y - 12}L${CX - neckX - 4} 320M${CX + neckX} ${p.chin.y - 12}L${CX + neckX + 4} 320"/>
    </g>`;

    const widest = Math.max(p.forehead.w, p.cheek.w, p.jaw.w);
    const lx = r1(CX + widest / 2 + 20);
    const guides = labels ? `
        ${measureLine("A", p.forehead.y, p.forehead.w)}
        ${measureLine("B", p.cheek.y, p.cheek.w)}
        ${measureLine("C", p.jaw.y, p.jaw.w)}
        <g class="fd-measure">
            <line x1="${lx}" y1="${p.top}" x2="${lx}" y2="${p.chin.y}"/>
            <line x1="${lx - 5}" y1="${p.top}" x2="${lx + 5}" y2="${p.top}"/><line x1="${lx - 5}" y1="${p.chin.y}" x2="${lx + 5}" y2="${p.chin.y}"/>
            <circle cx="${lx}" cy="${r1((p.top + p.chin.y) / 2)}" r="9"/><text x="${lx}" y="${r1((p.top + p.chin.y) / 2 + 3.5)}">D</text>
        </g>` : "";

    const a11y = title ? `role="img" aria-label="${title}"` : `aria-hidden="true"`;
    return `<svg class="face-diagram" viewBox="0 0 ${DIAGRAM_VIEWBOX.width} ${DIAGRAM_VIEWBOX.height}" ${a11y}>
        <line class="fd-axis" x1="${CX}" y1="4" x2="${CX}" y2="316"/>
        <path class="fd-outline" d="${outline}"/>
        ${features}
        ${guides}
    </svg>`;
}