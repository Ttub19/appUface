import { LM } from "../landmarks.js";
import { HAIR_ASSET_VIEWBOX, HAIR_LANDMARK_WIDTH_RATIO, LANDMARK10_BELOW_HAIRLINE } from "./hairAssets.js";
import { HairEraser } from "./hairEraser.js";
import { DEFAULT_HAIR_COLOR, canvasFilterSupported, lightingFactor, prepareHairCanvas, resolveHairRgb } from "./hairLook.js";
import {
    HeadScaleFusion, PlacementSmoother, computeHairPlacement, eulerFromTransformMatrix, faceEllipse, measureHairlineLift,
} from "./headPose.js";

// MediaPipe Tasks Vision: one pinned build for the whole app (see mediapipeTasks.js)
import { FACE_LANDMARKER_MODEL, createTask, loadVision } from "../mediapipeTasks.js";
const HAIR_SEGMENTER_MODEL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite";

/** Width the frame is downscaled to before hair segmentation (mask cost scales with this). */
const SEGMENT_WIDTH = 384;
/** Run segmentation every N rendered frames; the last mask is reused in between. */
const SEGMENT_EVERY = 2;
/** Crossfade length when switching hairstyles. */
const STYLE_FADE_MS = 220;
/** Accepted range for the measured hairline (face widths above landmark 10); outside it the default is kept. */
const HAIRLINE_LIFT_RANGE = [.17, .33];
/** "เหมือนผมฉัน": rebuild the recoloured hair at most this often while the sampled colour settles. */
const RECOLOR_MIN_INTERVAL_MS = 1200;
const PREPARED_CACHE_SIZE = 12;

let modelsPromise = null;

/**
 * Loads MediaPipe Tasks Vision + both models once per page; later sessions
 * reuse them so re-opening the camera is instant. A failed load is not cached.
 */
export function loadHairModels() {
    modelsPromise ??= (async () => {
        const { vision, fileset } = await loadVision();
        const [faceLandmarker, hairSegmenter] = await Promise.all([
            createTask(vision.FaceLandmarker, fileset, {
                baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL },
                runningMode: "VIDEO",
                numFaces: 1,
                outputFacialTransformationMatrixes: true,
                outputFaceBlendshapes: false,
            }),
            createTask(vision.ImageSegmenter, fileset, {
                baseOptions: { modelAssetPath: HAIR_SEGMENTER_MODEL },
                runningMode: "VIDEO",
                outputCategoryMask: false,
                outputConfidenceMasks: true,
            }),
        ]);
        return { faceLandmarker, hairSegmenter };
    })().catch((error) => {
        modelsPromise = null;
        throw error;
    });
    return modelsPromise;
}

/**
 * One live try-on session: webcam → Face Landmarker (pose) + Hair Segmenter
 * (real-hair mask) → composited onto a single canvas:
 *   1. mirrored camera frame
 *   2. erase layer covering the real hair (skin / background colour)
 *   3. the selected hairstyle asset, placed from landmarks + 3D head pose
 * Because everything lives on one canvas, capture is just canvas.toDataURL().
 */
export class HairTryOnSession {
    constructor({ video, canvas, mirror = true, onFaceChange = () => {} }) {
        if (!video || !canvas) throw new TypeError("HairTryOnSession requires video and canvas");
        this.video = video;
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.mirror = mirror;
        this.onFaceChange = onFaceChange;
        this.segCanvas = document.createElement("canvas");
        this.segCtx = this.segCanvas.getContext("2d", { willReadFrequently: true });
        this.eraser = new HairEraser();
        this.smoother = new PlacementSmoother();
        this.style = null;
        this.styleImage = null;   // prepared canvas actually drawn (recoloured, feathered, shadowed)
        this.rawImage = null;     // the style's source image
        this.colorId = DEFAULT_HAIR_COLOR;
        this.colorKey = null;
        this.lastRecolorAt = 0;
        this.prepared = new Map();
        this.scaleFusion = new HeadScaleFusion();
        this.hairlineLift = LANDMARK10_BELOW_HAIRLINE;
        this.lastPose = null;
        this.fading = null; // previous style while it crossfades out
        this.eraseEnabled = true;
        this.sizeScale = 1;
        this.running = false;
        this.generation = 0;
        this.frameCount = 0;
        this.lastTimestamp = 0;
        this.lastVideoTime = -1;
        this.faceVisible = null;
        this.rafId = null;
    }

    async start() {
        if (this.running) throw new Error("กล้องกำลังทำงานอยู่แล้ว");
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("เบราว์เซอร์นี้ไม่รองรับการเปิดกล้อง");
        const generation = ++this.generation;

        this.models = await loadHairModels();
        if (generation !== this.generation) throw new DOMException("cancelled", "AbortError");

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false,
            });
        } catch (error) {
            if (["NotAllowedError", "PermissionDeniedError"].includes(error?.name)) throw new Error("ไม่ได้รับสิทธิ์ใช้กล้อง กรุณาอนุญาตแล้วลองใหม่");
            if (["NotFoundError", "DevicesNotFoundError"].includes(error?.name)) throw new Error("ไม่พบอุปกรณ์กล้อง");
            throw error;
        }
        if (generation !== this.generation) {
            stream.getTracks().forEach((t) => t.stop());
            throw new DOMException("cancelled", "AbortError");
        }
        this.video.srcObject = stream;
        await this.video.play();

        this.running = true;
        this.frameCount = 0;
        this.lastVideoTime = -1;
        this.eraser.reset();
        this.smoother.reset();
        this.scaleFusion.reset();
        this.hairlineLift = LANDMARK10_BELOW_HAIRLINE;
        this.rafId = requestAnimationFrame(() => this._loop(generation));
    }

    /**
     * Switches hairstyle with a short crossfade: the previous style keeps its
     * own smoother and fades out while the new one fades in, so changing
     * styles doesn't pop.
     * @param {object} style - entry from HAIR_STYLES @param {HTMLImageElement} image
     */
    setStyle(style, image) {
        if (this.style && this.styleImage && this.style.id !== style.id && this.faceVisible) {
            this.fading = { style: this.style, image: this.styleImage, smoother: this.smoother, start: performance.now() };
            this.smoother = new PlacementSmoother();
        } else {
            this.fading = null;
            this.smoother.reset();
        }
        this.style = style;
        this.rawImage = image;
        this.styleImage = this._preparedFor(style, image);
    }

    /** Hair colour preset id from HAIR_COLORS ("mine" = sampled from the user's own hair). */
    setHairColor(colorId) {
        this.colorId = colorId || DEFAULT_HAIR_COLOR;
        if (this.style && this.rawImage) this.styleImage = this._preparedFor(this.style, this.rawImage);
    }

    /** Recoloured + feathered + shadowed canvas for a style, cached per colour. */
    _preparedFor(style, image) {
        const rgb = resolveHairRgb(this.colorId, this.eraser.colors.hair);
        const key = `${style.id}|${rgb ? rgb.join(",") : "original"}`;
        this.colorKey = rgb ? rgb.join(",") : "original";
        this.lastRecolorAt = performance.now();
        let canvas = this.prepared.get(key);
        if (!canvas) {
            try {
                canvas = prepareHairCanvas(image, { rgb, faceWidthPx: style.faceWidthPx || 300 });
            } catch (error) {
                console.warn("hair image preparation failed, drawing the original", error);
                return image;
            }
            this.prepared.set(key, canvas);
            if (this.prepared.size > PREPARED_CACHE_SIZE) this.prepared.delete(this.prepared.keys().next().value);
        }
        return canvas;
    }

    /** Manual size tweak on top of the automatic head fit (1 = auto). Scales around the hairline anchor. */
    setSizeScale(scale) {
        this.sizeScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
    }

    /** Only toggles drawing the cover-up; segmentation keeps running for hair colour and hairline. */
    setEraseEnabled(enabled) {
        this.eraseEnabled = Boolean(enabled);
    }

    capture(type = "image/jpeg", quality = .92) {
        if (!this.canvas.width) return null;
        return this.canvas.toDataURL(type, quality);
    }

    stop() {
        this.running = false;
        this.generation += 1;
        if (this.rafId !== null) cancelAnimationFrame(this.rafId);
        this.rafId = null;
        const stream = this.video.srcObject;
        if (stream?.getTracks) stream.getTracks().forEach((t) => t.stop());
        this.video.srcObject = null;
        this.eraser.reset();
        this.smoother.reset();
        this.scaleFusion.reset();
        this.fading = null;
        this.faceVisible = null;
        this.prepared.clear();
    }

    _nextTimestamp() {
        // MediaPipe VIDEO mode requires strictly increasing timestamps.
        this.lastTimestamp = Math.max(performance.now(), this.lastTimestamp + 1);
        return this.lastTimestamp;
    }

    _loop(generation) {
        if (!this.running || generation !== this.generation) return;
        try {
            this._renderFrame();
        } catch (error) {
            console.error("Hair try-on frame failed:", error);
        }
        this.rafId = requestAnimationFrame(() => this._loop(generation));
    }

    _renderFrame() {
        const video = this.video;
        if (video.readyState < 2 || !video.videoWidth) return;
        if (video.currentTime === this.lastVideoTime) return; // no new camera frame
        this.lastVideoTime = video.currentTime;

        const w = video.videoWidth, h = video.videoHeight;
        if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
        const ts = this._nextTimestamp();
        const face = this.models.faceLandmarker.detectForVideo(video, ts);
        const landmarks = face.faceLandmarks?.[0] || null;
        const matrix = face.facialTransformationMatrixes?.[0]?.data || null;

        const visible = Boolean(landmarks);
        if (visible !== this.faceVisible) { this.faceVisible = visible; this.onFaceChange(visible); }

        if (visible && this.frameCount % SEGMENT_EVERY === 0) this._segmentHair(landmarks, ts);
        this.frameCount++;

        const ctx = this.ctx;
        ctx.save();
        if (this.mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
        ctx.drawImage(video, 0, 0, w, h);
        if (visible) {
            if (this.eraseEnabled) this.eraser.draw(ctx, w, h);
            this._drawHair(ctx, landmarks, matrix, w, h, ts);
        } else {
            this.smoother.reset();
            this.fading = null;
        }
        ctx.restore();
    }

    _segmentHair(landmarks, ts) {
        const sw = SEGMENT_WIDTH;
        const sh = Math.round(sw * this.video.videoHeight / this.video.videoWidth);
        if (this.segCanvas.width !== sw || this.segCanvas.height !== sh) { this.segCanvas.width = sw; this.segCanvas.height = sh; }
        this.segCtx.drawImage(this.video, 0, 0, sw, sh);
        const frame = this.segCtx.getImageData(0, 0, sw, sh).data;
        const ellipse = faceEllipse(landmarks, sw, sh);
        // forehead + both cheeks: cheeks are never under bangs, so skin is always found
        const skinPoints = [LM.FOREHEAD_CENTER, LM.CHEEK_SKIN_L, LM.CHEEK_SKIN_R]
            .map((i) => landmarks[i]).filter(Boolean).map((p) => ({ x: p.x * sw, y: p.y * sh }));

        // Masks are only valid inside the callback, so copy what we need there.
        this.models.hairSegmenter.segmentForVideo(this.segCanvas, ts, (result) => {
            const masks = result.confidenceMasks;
            if (!masks?.length) return;
            const mask = masks[masks.length - 1]; // hair_segmenter: [background, hair]
            if (mask.width !== sw || mask.height !== sh) return;
            const conf = mask.getAsFloat32Array();
            this.eraser.update({ conf, frame, width: sw, height: sh, ellipse, skinPoint: skinPoints[0], skinPoints });
            this._updateHairline(conf, sw, sh, landmarks);
            this._maybeRecolor();
        });
    }

    /** Follows the user's real hairline (measured while facing the camera), smoothed. */
    _updateHairline(conf, width, height, landmarks) {
        const pose = this.lastPose;
        if (pose && (Math.abs(pose.yaw) > .35 || Math.abs(pose.pitch) > .3)) return;
        const lift = measureHairlineLift({ conf, width, height, landmarks });
        if (lift === null) return;
        const target = Math.min(HAIRLINE_LIFT_RANGE[1], Math.max(HAIRLINE_LIFT_RANGE[0], lift * HAIR_LANDMARK_WIDTH_RATIO));
        this.hairlineLift += (target - this.hairlineLift) * .15;
    }

    /** "เหมือนผมฉัน": once the sampled hair colour has moved to a new step, rebuild the hair image. */
    _maybeRecolor() {
        if (this.colorId !== "mine" || !this.style || !this.rawImage) return;
        const rgb = resolveHairRgb("mine", this.eraser.colors.hair);
        const key = rgb ? rgb.join(",") : "original";
        if (key === this.colorKey || performance.now() - this.lastRecolorAt < RECOLOR_MIN_INTERVAL_MS) return;
        this.styleImage = this._preparedFor(this.style, this.rawImage);
    }

    _drawHair(ctx, landmarks, matrix, w, h, ts) {
        if (!this.style || !this.styleImage) return;
        const pose = matrix ? eulerFromTransformMatrix(matrix) : null;
        this.lastPose = pose;
        const headWidthPx = this.scaleFusion.update(landmarks, w, h, pose);
        const now = performance.now();
        let fadeIn = 1;
        if (this.fading) {
            const t = Math.min(1, (now - this.fading.start) / STYLE_FADE_MS);
            fadeIn = t * t * (3 - 2 * t);
            if (t >= 1) this.fading = null;
            else this._drawStyle(ctx, this.fading.style, this.fading.image, this.fading.smoother, landmarks, w, h, pose, ts, 1 - fadeIn, headWidthPx);
        }
        this._drawStyle(ctx, this.style, this.styleImage, this.smoother, landmarks, w, h, pose, ts, fadeIn, headWidthPx);
    }

    _drawStyle(ctx, style, image, smoother, landmarks, w, h, pose, ts, alpha, headWidthPx) {
        const target = computeHairPlacement(landmarks, w, h, pose, style, headWidthPx);
        const p = smoother.update(target, ts);
        if (!p || alpha <= 0) return;
        // Prepared canvases are 1:1 with the asset pixels (plus shadow rows at the bottom);
        // a raw vector image falls back to its viewBox.
        const vb = style.viewBox || HAIR_ASSET_VIEWBOX;
        const [dw, dh] = image instanceof HTMLCanvasElement ? [image.width, image.height] : [vb.width, vb.height];
        // Landmark 10 goes this far below the asset's hairline: measured from the user's own forehead when possible.
        const anchorY = Number.isFinite(style.hairlineY) ? style.hairlineY + this.hairlineLift * style.faceWidthPx : style.anchor.y;
        ctx.save();
        ctx.globalAlpha = alpha;
        if (this.colorId !== "mine" && canvasFilterSupported()) {
            // follow the room's exposure; "mine" is sampled from the frame so it already matches
            ctx.filter = `brightness(${lightingFactor(this.eraser.colors.skin).toFixed(3)})`;
        }
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rotation);
        ctx.scale(p.scaleX * this.sizeScale, p.scaleY * this.sizeScale);
        ctx.drawImage(image, -style.anchor.x, -anchorY, dw, dh);
        ctx.restore();
    }
}

// ---------------------------------------------------------------------------
// Still-photo try-on (the default since v3): the same pipeline run ONCE on one
// photo instead of every camera frame. With a single frame there is nothing to
// jitter, segmentation can run at a higher resolution, and the result is the
// same every time for the same photo + settings.
// ---------------------------------------------------------------------------
const PHOTO_MAX_SIDE = 1280;        // photos are downscaled to this before processing
const PHOTO_SEGMENT_WIDTH = 512;    // hair mask resolution (live mode uses 384)
export const NO_FACE_IN_PHOTO = "ไม่พบใบหน้าในรูป กรุณาใช้รูปหน้าตรงที่เห็นใบหน้าและหน้าผากชัดเจน";

let photoModelsPromise = null;

/** Same two models as the live mode, in IMAGE running mode (one-shot detect / segment). */
export function loadHairPhotoModels() {
    photoModelsPromise ??= (async () => {
        const { vision, fileset } = await loadVision();
        const [faceLandmarker, hairSegmenter] = await Promise.all([
            createTask(vision.FaceLandmarker, fileset, {
                baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL },
                runningMode: "IMAGE",
                numFaces: 1,
                outputFacialTransformationMatrixes: true,
                outputFaceBlendshapes: false,
            }),
            createTask(vision.ImageSegmenter, fileset, {
                baseOptions: { modelAssetPath: HAIR_SEGMENTER_MODEL },
                runningMode: "IMAGE",
                outputCategoryMask: false,
                outputConfidenceMasks: true,
            }),
        ]);
        return { faceLandmarker, hairSegmenter };
    })().catch((error) => {
        photoModelsPromise = null;
        throw error;
    });
    return photoModelsPromise;
}

/**
 * Try-on on a still photo:
 *   setPhoto() — landmarks + head pose + hair mask + erase layer + hairline, computed once
 *   render()   — photo → erase layer → hairstyle (re-run on every style / colour / size change)
 */
export class HairPhotoSession {
    constructor({ canvas }) {
        if (!canvas) throw new TypeError("HairPhotoSession requires a canvas");
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.base = document.createElement("canvas");
        this.segCanvas = document.createElement("canvas");
        this.segCtx = this.segCanvas.getContext("2d", { willReadFrequently: true });
        this.eraser = new HairEraser();
        this.prepared = new Map();
        this.models = null;
        this.face = null;          // { landmarks, pose, headWidthPx }
        this.style = null;
        this.rawImage = null;
        this.styleImage = null;
        this.colorId = DEFAULT_HAIR_COLOR;
        this.hairlineLift = LANDMARK10_BELOW_HAIRLINE;
        this.eraseEnabled = true;
        this.sizeScale = 1;
        this.offsetY = 0;          // manual up/down, in face widths (+ = lower)
        this.realistic = null;     // AI-generated image shown instead of the 2D try-on
        this.photoId = 0;
    }

    async load() {
        this.models ??= await loadHairPhotoModels();
        return this;
    }

    get hasPhoto() { return Boolean(this.face); }

    /**
     * Analyse a photo (img / canvas / video frame). mirror = flip like a selfie preview.
     * Throws NO_FACE_IN_PHOTO when no face is found (the previous photo is kept).
     */
    setPhoto(source, { mirror = false } = {}) {
        if (!this.models) throw new Error("ยังโหลดโมเดลไม่เสร็จ");
        const sw = source.naturalWidth || source.videoWidth || source.width;
        const sh = source.naturalHeight || source.videoHeight || source.height;
        if (!(sw > 0 && sh > 0)) throw new Error("อ่านรูปไม่ได้");
        const k = Math.min(1, PHOTO_MAX_SIDE / Math.max(sw, sh));
        const w = Math.round(sw * k), h = Math.round(sh * k);
        const base = document.createElement("canvas");
        base.width = w; base.height = h;
        const bctx = base.getContext("2d");
        if (mirror) { bctx.translate(w, 0); bctx.scale(-1, 1); }
        bctx.drawImage(source, 0, 0, w, h);

        const result = this.models.faceLandmarker.detect(base);
        const landmarks = result.faceLandmarks?.[0] || null;
        if (!landmarks) throw new Error(NO_FACE_IN_PHOTO);
        const matrix = result.facialTransformationMatrixes?.[0]?.data || null;
        const pose = matrix ? eulerFromTransformMatrix(matrix) : null;

        // hair mask on a downscaled copy (same as live mode, just sharper)
        const segW = Math.min(PHOTO_SEGMENT_WIDTH, w), segH = Math.round(segW * h / w);
        this.segCanvas.width = segW; this.segCanvas.height = segH;
        this.segCtx.drawImage(base, 0, 0, segW, segH);
        const frame = this.segCtx.getImageData(0, 0, segW, segH).data;
        let conf = null;
        this.models.hairSegmenter.segment(this.segCanvas, (res) => {
            const masks = res.confidenceMasks;
            const mask = masks?.[masks.length - 1]; // hair_segmenter: [background, hair]
            if (mask && mask.width === segW && mask.height === segH) conf = new Float32Array(mask.getAsFloat32Array());
        });

        this.base = base;
        this.realistic = null;
        this.photoId = (this.photoId || 0) + 1;     // cache key for generated images
        this.eraser.reset();
        this.hairlineLift = LANDMARK10_BELOW_HAIRLINE;
        if (conf) {
            const skinPoints = [LM.FOREHEAD_CENTER, LM.CHEEK_SKIN_L, LM.CHEEK_SKIN_R]
                .map((i) => landmarks[i]).filter(Boolean).map((p) => ({ x: p.x * segW, y: p.y * segH }));
            this.eraser.update({ conf, frame, width: segW, height: segH, ellipse: faceEllipse(landmarks, segW, segH), skinPoint: skinPoints[0], skinPoints });
            const facingCamera = !pose || (Math.abs(pose.yaw) <= .35 && Math.abs(pose.pitch) <= .3);
            const lift = facingCamera ? measureHairlineLift({ conf, width: segW, height: segH, landmarks }) : null;
            if (lift !== null) {
                this.hairlineLift = Math.min(HAIRLINE_LIFT_RANGE[1], Math.max(HAIRLINE_LIFT_RANGE[0], lift * HAIR_LANDMARK_WIDTH_RATIO));
            }
        }
        this.face = { landmarks, pose, headWidthPx: new HeadScaleFusion().update(landmarks, w, h, pose) };
        this.prepared.clear();                       // "เหมือนผมฉัน" depends on this photo's hair colour
        if (this.style && this.rawImage) this.styleImage = this._preparedFor(this.style, this.rawImage);
        this.render();
        return { width: w, height: h, hairFound: Boolean(conf && this.eraser.colors.hair) };
    }

    setStyle(style, image) {
        this.realistic = null;
        this.style = style;
        this.rawImage = image;
        this.styleImage = this._preparedFor(style, image);
        this.render();
    }

    setHairColor(colorId) {
        this.realistic = null;
        this.colorId = colorId || DEFAULT_HAIR_COLOR;
        if (this.style && this.rawImage) this.styleImage = this._preparedFor(this.style, this.rawImage);
        this.render();
    }

    setSizeScale(scale) { this.realistic = null; this.sizeScale = Number.isFinite(scale) && scale > 0 ? scale : 1; this.render(); }
    setOffsetY(faceWidths) { this.realistic = null; this.offsetY = Number.isFinite(faceWidths) ? faceWidths : 0; this.render(); }
    setEraseEnabled(enabled) { this.realistic = null; this.eraseEnabled = Boolean(enabled); this.render(); }

    _preparedFor(style, image) {
        const rgb = resolveHairRgb(this.colorId, this.eraser.colors.hair);
        const key = `${style.id}|${rgb ? rgb.join(",") : "original"}`;
        let canvas = this.prepared.get(key);
        if (!canvas) {
            try {
                canvas = prepareHairCanvas(image, { rgb, faceWidthPx: style.faceWidthPx || 300 });
            } catch (error) {
                console.warn("hair image preparation failed, drawing the original", error);
                return image;
            }
            this.prepared.set(key, canvas);
            if (this.prepared.size > PREPARED_CACHE_SIZE) this.prepared.delete(this.prepared.keys().next().value);
        }
        return canvas;
    }

    /** The 2D try-on: photo -> erase layer -> hairstyle. */
    _render2D(ctx) {
        const { width: w, height: h } = this.base;
        ctx.save();
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(this.base, 0, 0);
        if (this.eraseEnabled) this.eraser.draw(ctx, w, h);
        this._drawHair(ctx, true);
        ctx.restore();
    }

    /** Draws the chosen hairstyle in photo space (lighting = match the room, off for masks). */
    _drawHair(ctx, lighting) {
        if (!this.style || !this.styleImage) return;
        const { width: w, height: h } = this.base;
        const { landmarks, pose, headWidthPx } = this.face;
        const p = computeHairPlacement(landmarks, w, h, pose, this.style, headWidthPx);
        const style = this.style, image = this.styleImage;
        const vb = style.viewBox || HAIR_ASSET_VIEWBOX;
        const [dw, dh] = image instanceof HTMLCanvasElement ? [image.width, image.height] : [vb.width, vb.height];
        const fw = style.faceWidthPx || 300;
        const anchorY = (Number.isFinite(style.hairlineY) ? style.hairlineY + this.hairlineLift * fw : style.anchor.y) - this.offsetY * fw;
        ctx.save();
        if (lighting && this.colorId !== "mine" && canvasFilterSupported()) {
            ctx.filter = `brightness(${lightingFactor(this.eraser.colors.skin).toFixed(3)})`;
        }
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rotation);
        ctx.scale(p.scaleX * this.sizeScale, p.scaleY * this.sizeScale);
        ctx.drawImage(image, -style.anchor.x, -anchorY, dw, dh);
        ctx.restore();
    }

    render() {
        if (!this.face) return;
        const src = this.realistic;
        const [w, h] = src ? [src.width, src.height] : [this.base.width, this.base.height];
        if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
        if (src) {
            this.ctx.clearRect(0, 0, w, h);
            this.ctx.drawImage(src, 0, 0);
        } else {
            this._render2D(this.ctx);
        }
    }

    // ---- "สร้างภาพสมจริง": inputs for the generative model and putting its answer back ----

    /**
     * Square (photo px) around the face and the whole hairstyle (long hair included),
     * kept inside the photo where possible so the model sees real pixels, not padding.
     */
    headCrop(mask = this.hairMask()) {
        const { width: w, height: h } = this.base;
        const lm = this.face.landmarks;
        const xs = [LM.FOREHEAD, LM.CHIN, LM.TEMPLE_L, LM.TEMPLE_R].map((i) => lm[i].x * w);
        const ys = [LM.FOREHEAD, LM.CHIN, LM.TEMPLE_L, LM.TEMPLE_R].map((i) => lm[i].y * h);
        let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
        // hair bounding box from a small copy of the mask
        const k = Math.min(1, 256 / Math.max(w, h));
        const sw = Math.max(1, Math.round(w * k)), sh = Math.max(1, Math.round(h * k));
        const small = document.createElement("canvas");
        small.width = sw; small.height = sh;
        const sctx = small.getContext("2d", { willReadFrequently: true });
        sctx.drawImage(mask, 0, 0, sw, sh);
        const a = sctx.getImageData(0, 0, sw, sh).data;
        for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
            if (a[(y * sw + x) * 4 + 3] > 40) {
                x0 = Math.min(x0, x / k); x1 = Math.max(x1, (x + 1) / k);
                y0 = Math.min(y0, y / k); y1 = Math.max(y1, (y + 1) / k);
            }
        }
        let size = Math.round(Math.max(x1 - x0, y1 - y0) * 1.12);
        size = Math.min(size, Math.max(w, h));
        const place = (lo, hi, limit) => {
            let start = (lo + hi) / 2 - size / 2;
            if (size <= limit) start = Math.min(Math.max(start, 0), limit - size);
            return Math.round(start);
        };
        return { x: place(x0, x1, w), y: place(y0, y1, h), size };
    }

    /**
     * Where the model may repaint: the new hairstyle plus the old hair that was
     * covered up, grown a little so the new hair blends into the photo.
     * Returns an alpha canvas (white, alpha = repaint) in photo space.
     */
    hairMask() {
        const { width: w, height: h } = this.base;
        const m = document.createElement("canvas");
        m.width = w; m.height = h;
        const ctx = m.getContext("2d");
        if (this.eraseEnabled && this.eraser.ready) ctx.drawImage(this.eraser.canvas, 0, 0, w, h);
        this._drawHair(ctx, false);
        ctx.globalCompositeOperation = "source-in";
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, w, h);
        return m;
    }

    /**
     * patch mode (inpainting): 512x512 head crop of the 2D try-on + its mask.
     * full mode: the untouched photo + the chosen hairstyle on white as a reference.
     */
    realisticInputs(mode, size = 512) {
        if (!this.face) throw new Error("ยังไม่มีรูป");
        if (mode === "full") {
            const k = Math.min(1, 1024 / Math.max(this.base.width, this.base.height));
            const photo = document.createElement("canvas");
            photo.width = Math.round(this.base.width * k); photo.height = Math.round(this.base.height * k);
            photo.getContext("2d").drawImage(this.base, 0, 0, photo.width, photo.height);
            const img = this.styleImage;
            const ref = document.createElement("canvas");
            const rk = Math.min(1, 768 / Math.max(img.width, img.height));
            ref.width = Math.round(img.width * rk); ref.height = Math.round(img.height * rk);
            const rctx = ref.getContext("2d");
            rctx.fillStyle = "#fff"; rctx.fillRect(0, 0, ref.width, ref.height);
            rctx.drawImage(img, 0, 0, ref.width, ref.height);
            return { mode, photo: photo.toDataURL("image/jpeg", .92), reference: ref.toDataURL("image/png") };
        }
        const hairMask = this.hairMask();
        const crop = this.headCrop(hairMask);
        const full = document.createElement("canvas");
        full.width = this.base.width; full.height = this.base.height;
        this._render2D(full.getContext("2d"));
        const guide = document.createElement("canvas");
        guide.width = guide.height = size;
        const gctx = guide.getContext("2d");
        gctx.fillStyle = "#808080"; gctx.fillRect(0, 0, size, size);
        gctx.drawImage(full, crop.x, crop.y, crop.size, crop.size, 0, 0, size, size);

        // mask: crop -> 512, soften + grow a little, then white-on-black for the model
        const alpha = document.createElement("canvas");
        alpha.width = alpha.height = size;
        const actx = alpha.getContext("2d");
        actx.filter = `blur(${Math.round(size / 90)}px)`;
        actx.drawImage(hairMask, crop.x, crop.y, crop.size, crop.size, 0, 0, size, size);
        actx.filter = "none";
        const px = actx.getImageData(0, 0, size, size);
        for (let i = 3; i < px.data.length; i += 4) px.data[i] = Math.min(255, px.data[i] * 2.2);   // grow
        actx.putImageData(px, 0, 0);
        const mask = document.createElement("canvas");
        mask.width = mask.height = size;
        const mctx = mask.getContext("2d");
        mctx.fillStyle = "#000"; mctx.fillRect(0, 0, size, size);
        mctx.drawImage(alpha, 0, 0);
        return { mode, crop, alpha, full, guide: guide.toDataURL("image/jpeg", .93), mask: mask.toDataURL("image/png") };
    }

    /** patch mode: the model's 512 crop goes back into the full-resolution photo only where the mask allows. */
    composePatch(resultImage, inputs) {
        const { width: w, height: h } = this.base;
        const out = document.createElement("canvas");
        out.width = w; out.height = h;
        const octx = out.getContext("2d");
        // the 2D try-on as it was when the request was sent (settings may have changed since)
        if (inputs.full) octx.drawImage(inputs.full, 0, 0); else this._render2D(octx);
        const { crop, alpha } = inputs;
        const patch = document.createElement("canvas");
        patch.width = patch.height = crop.size;
        const pctx = patch.getContext("2d");
        pctx.drawImage(resultImage, 0, 0, crop.size, crop.size);
        pctx.globalCompositeOperation = "destination-in";
        pctx.filter = `blur(${Math.max(1, Math.round(crop.size / 200))}px)`;   // feathered seam
        pctx.drawImage(alpha, 0, 0, crop.size, crop.size);
        octx.drawImage(patch, crop.x, crop.y);
        return out;
    }

    /** Show a finished realistic image instead of the 2D try-on (null = back to 2D). */
    showRealistic(image) {
        this.realistic = image || null;
        this.render();
    }

    capture(type = "image/jpeg", quality = .92) {
        return this.face ? this.canvas.toDataURL(type, quality) : null;
    }

    clear() {
        this.face = null;
        this.realistic = null;
        this.eraser.reset();
        this.prepared.clear();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}
