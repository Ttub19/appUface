import test from "node:test";
import assert from "node:assert/strict";
import { GLASSES_SHAPES, GLASSES_STYLE_CONFIG } from "../js/arRenderer.js";
import {
    GLASSES_CATALOG, GLASSES_CATEGORIES, LEGACY_GLASSES_ALIASES, DEFAULT_GLASSES_STYLE,
    resolveGlassesStyle, glassesInCategory, isInCategory,
} from "../js/glassesCatalog.js";

test("catalog and renderer list the same frames", () => {
    const catalogIds = GLASSES_CATALOG.map((g) => g.id).sort();
    assert.deepEqual(catalogIds, Object.keys(GLASSES_SHAPES).sort());
    assert.deepEqual(catalogIds, Object.keys(GLASSES_STYLE_CONFIG).sort());
    assert.equal(new Set(catalogIds).size, catalogIds.length);
});

test("every frame belongs to known categories and every category has frames", () => {
    const known = new Set(GLASSES_CATEGORIES.map((c) => c.id));
    for (const g of GLASSES_CATALOG) {
        assert.ok(g.categories.length > 0, g.id);
        for (const c of g.categories) assert.ok(known.has(c) && c !== "all", `${g.id}: ${c}`);
    }
    for (const c of GLASSES_CATEGORIES) assert.ok(glassesInCategory(c.id).length > 0, c.id);
    assert.equal(glassesInCategory("all").length, GLASSES_CATALOG.length);
});

test("legacy and unknown style ids resolve to real frames", () => {
    for (const [legacy, target] of Object.entries(LEGACY_GLASSES_ALIASES)) {
        assert.equal(resolveGlassesStyle(legacy), target);
        assert.ok(GLASSES_SHAPES[target], target);
    }
    assert.equal(resolveGlassesStyle("cateye-red"), "cateye-red");
    assert.equal(resolveGlassesStyle("nope"), DEFAULT_GLASSES_STYLE);
});

test("multi-category frames show up in each of their categories", () => {
    assert.ok(isInCategory("rimless-rosegold-cateye", "rimless"));
    assert.ok(isInCategory("rimless-rosegold-cateye", "cateye"));
    assert.ok(!isInCategory("rimless-rosegold-cateye", "square"));
    assert.ok(isInCategory("rimless-rosegold-cateye", "all"));
});
