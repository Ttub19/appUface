import { LM } from "../landmarks.js";
import { HAIR_ASSET_VIEWBOX } from "./hairAssets.js";
import { HairEraser } from "./hairEraser.js";
import { PlacementSmoother, computeHairPlacement, eulerFromTransformMatrix, faceEllipse } from "./headPose.js";

// Pinned so the JS bundle and its WASM files always match.
const TASKS_VISION_VERSION = "0.10.14";
const TASKS_VISION_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;
const FACE_LANDMARKER_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const HAIR_SEGMENTER_MODEL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite";

/** Width the frame is downscaled to before hair segmentation (mask cost scales with this). */
const SEGMENT_WIDTH = 384;
/** Run segmentation every N rendered frames; the last mask is reused in between. */
const SEGMENT_EVERY = 2;

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
        this.styleImage = null;
        this.eraseEnabled = true;
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
        this.rafId = requestAnimationFrame(() => this._loop(generation));
    }

    /** @param {object} style - entry from HAIR_STYLES @param {HTMLImageElement} image */
    setStyle(style, image) {
        this.style = style;
        this.styleImage = image;
        this.smoother.reset();
    }

    setEraseEnabled(enabled) {
        this.eraseEnabled = Boolean(enabled);
        if (!this.eraseEnabled) this.eraser.reset();
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
        this.faceVisible = null;
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

        if (visible && this.eraseEnabled && this.frameCount % SEGMENT_EVERY === 0) this._segmentHair(landmarks, ts);
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
        const skin = landmarks[LM.FOREHEAD_CENTER];

        // Masks are only valid inside the callback, so copy what we need there.
        this.models.hairSegmenter.segmentForVideo(this.segCanvas, ts, (result) => {
            const masks = result.confidenceMasks;
            if (!masks?.length) return;
            const mask = masks[masks.length - 1]; // hair_segmenter: [background, hair]
            if (mask.width !== sw || mask.height !== sh) return;
            this.eraser.update({
                conf: mask.getAsFloat32Array(),
                frame,
                width: sw,
                height: sh,
                ellipse,
                skinPoint: { x: skin.x * sw, y: skin.y * sh },
            });
        });
    }

    _drawHair(ctx, landmarks, matrix, w, h, ts) {
        if (!this.style || !this.styleImage) return;
        const pose = matrix ? eulerFromTransformMatrix(matrix) : null;
        const target = computeHairPlacement(landmarks, w, h, pose, this.style);
        const p = this.smoother.update(target, ts);
        if (!p) return;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rotation);
        ctx.scale(p.scaleX, p.scaleY);
        ctx.drawImage(this.styleImage, -this.style.anchor.x, -this.style.anchor.y, HAIR_ASSET_VIEWBOX.width, HAIR_ASSET_VIEWBOX.height);
        ctx.restore();
    }
}
