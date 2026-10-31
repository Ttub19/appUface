import { FACE_LANDMARKER_MODEL, createTask, loadVision } from "./mediapipeTasks.js";

let trackerPromise = null;

/**
 * Face Landmarker (VIDEO mode) shared by the glasses and golden-ratio screens.
 * 478 landmarks = the old Face Mesh with refineLandmarks (468 + 10 iris points),
 * same order and normalised x/y/z, so the screens' landmark code is unchanged.
 * Kept loaded after the first use: opening the camera again is instant.
 */
function loadFaceTracker() {
    trackerPromise ??= (async () => {
        const { vision, fileset } = await loadVision();
        return createTask(vision.FaceLandmarker, fileset, {
            baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL },
            runningMode: "VIDEO",
            numFaces: 1,
            minFaceDetectionConfidence: .6,
            minFacePresenceConfidence: .6,
            minTrackingConfidence: .6,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
        });
    })().catch((error) => {
        trackerPromise = null;
        throw error;
    });
    return trackerPromise;
}

/** Owns one webcam + face-landmark pipeline. onFrame(landmarks | null, videoWidth, videoHeight) runs per video frame. */
export class FaceARSession {
    constructor({ video, onFrame }) {
        if (!video || typeof onFrame !== "function") throw new TypeError("FaceARSession requires video and onFrame");
        this.video = video;
        this.onFrame = onFrame;
        this.tracker = null;
        this.stream = null;
        this.running = false;
        this.generation = 0;
        this.rafId = 0;
        this.lastVideoTime = -1;
        this.lastTimestamp = 0;
    }

    async start() {
        if (this.running) throw new Error("กล้องกำลังทำงานอยู่แล้ว");
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("เบราว์เซอร์นี้ไม่รองรับการเปิดกล้อง");

        const generation = ++this.generation;
        this.running = true;
        try {
            // camera permission + model download in parallel
            const [stream, tracker] = await Promise.all([
                navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false,
                }),
                loadFaceTracker().catch((error) => {
                    console.error(error);
                    throw new Error("โหลดตัวตรวจจับใบหน้าไม่สำเร็จ กรุณาตรวจสอบอินเทอร์เน็ตแล้วลองใหม่");
                }),
            ]);
            if (!this.running || generation !== this.generation) {   // stopped while loading
                stream.getTracks().forEach((track) => track.stop());
                return;
            }
            this.stream = stream;
            this.tracker = tracker;
            this.video.srcObject = stream;
            this.video.muted = true;
            this.video.playsInline = true;
            await this.video.play();
        } catch (error) {
            this.stop();
            if (["NotAllowedError", "PermissionDeniedError"].includes(error?.name)) {
                throw new Error("ไม่ได้รับสิทธิ์ใช้กล้อง กรุณาอนุญาตแล้วลองใหม่");
            }
            if (["NotFoundError", "DevicesNotFoundError"].includes(error?.name)) throw new Error("ไม่พบอุปกรณ์กล้อง");
            throw error;
        }
        this.lastVideoTime = -1;
        this.rafId = requestAnimationFrame(() => this.tick(generation));
    }

    tick(generation) {
        if (!this.running || generation !== this.generation) return;
        const v = this.video;
        if (v.readyState >= 2 && v.currentTime !== this.lastVideoTime) {
            this.lastVideoTime = v.currentTime;
            // detectForVideo needs strictly increasing timestamps
            const now = Math.max(performance.now(), this.lastTimestamp + 1);
            this.lastTimestamp = now;
            let landmarks = null;
            try {
                landmarks = this.tracker.detectForVideo(v, now)?.faceLandmarks?.[0] || null;
            } catch (error) {
                console.error("FaceLandmarker.detectForVideo failed:", error);
            }
            try { this.onFrame(landmarks, v.videoWidth || 1280, v.videoHeight || 720); }
            catch (error) { console.error("FaceARSession onFrame failed:", error); }
        }
        this.rafId = requestAnimationFrame(() => this.tick(generation));
    }

    stop() {
        this.running = false;
        this.generation += 1;
        if (this.rafId) cancelAnimationFrame(this.rafId);
        this.rafId = 0;
        const stream = this.stream || this.video?.srcObject;
        if (stream?.getTracks) stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
        this.tracker = null;            // the shared landmarker stays loaded for the next session
        if (this.video) this.video.srcObject = null;
    }
}
