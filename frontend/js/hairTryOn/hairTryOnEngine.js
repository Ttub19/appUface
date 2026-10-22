import { LM } from "../landmarks.js";
import { HAIR_ASSET_VIEWBOX, HAIR_LANDMARK_WIDTH_RATIO, LANDMARK10_BELOW_HAIRLINE } from "./hairAssets.js";
import { HairEraser } from "./hairEraser.js";
import { DEFAULT_HAIR_COLOR, canvasFilterSupported, lightingFactor, prepareHairCanvas, resolveHairRgb } from "./hairLook.js";
import {
    HeadScaleFusion, PlacementSmoother, computeHairPlacement, eulerFromTransformMatrix, faceEllipse, measureHairlineLift,
} from "./headPose.js";

// Pinned so the JS bundle and its WASM files always match.
const TASKS_VISION_VERSION = "0.10.14";
const TASKS_VISION_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;
const FACE_LANDMARKER_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
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

async function createTask(TaskClass, fileset, options) {
    try {
        return await TaskClass.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: "GPU" } });
    } catch (gpuError) {
        console.warn(`${TaskClass.name}: GPU delegate unavailable, falling back to CPU`, gpuError);
        return TaskClass.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: "CPU" } });
    }
}

/**
 * Loads MediaPipe Tasks Vision + both models once per page; later sessions
 * reuse them so re-opening the camera is instant. A failed load is not cached.
 */
export function loadHairModels() {
    modelsPromise ??= (async () => {
        const vision = await import(`${TASKS_VISION_BASE}/vision_bundle.mjs`);
        const fileset = await vision.FilesetResolver.forVisionTasks(`${TASKS_VISION_BASE}/wasm`);
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
