const FACE_MESH_ASSET_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh";

/** Owns one webcam + MediaPipe Face Mesh pipeline. */
export class FaceARSession {
    constructor({ video, onFrame }) {
        if (!video || typeof onFrame !== "function") throw new TypeError("FaceARSession requires video and onFrame");
        this.video = video;
        this.onFrame = onFrame;
        this.faceMesh = null;
        this.camera = null;
        this.running = false;
        this.sendInFlight = false;
        this.generation = 0;
    }

    async start() {
        if (this.running) throw new Error("กล้องกำลังทำงานอยู่แล้ว");
        if (typeof FaceMesh === "undefined" || typeof Camera === "undefined") {
            throw new Error("ไลบรารี MediaPipe โหลดไม่สำเร็จ กรุณาตรวจสอบอินเทอร์เน็ตแล้วลองใหม่");
        }
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("เบราว์เซอร์นี้ไม่รองรับการเปิดกล้อง");

        const generation = ++this.generation;
        this.faceMesh = new FaceMesh({ locateFile: (file) => `${FACE_MESH_ASSET_BASE}/${file}` });
        this.faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: .6,
            minTrackingConfidence: .6,
        });
        this.faceMesh.onResults((results) => {
            if (!this.running || generation !== this.generation) return;
            const w = this.video.videoWidth || results?.image?.width || 1280;
            const h = this.video.videoHeight || results?.image?.height || 720;
            try { this.onFrame(results?.multiFaceLandmarks?.[0] || null, w, h); }
            catch (error) { console.error("FaceARSession onFrame failed:", error); }
        });
        this.camera = new Camera(this.video, {
            onFrame: async () => {
                if (!this.running || generation !== this.generation || this.sendInFlight) return;
                this.sendInFlight = true;
                try { await this.faceMesh.send({ image: this.video }); }
                catch (error) { if (this.running) console.error("FaceMesh.send failed:", error); }
                finally { this.sendInFlight = false; }
            },
            width: 1280,
            height: 720,
        });
        this.running = true;
        try { await this.camera.start(); }
        catch (error) {
            this.stop();
            if (["NotAllowedError", "PermissionDeniedError"].includes(error?.name)) {
                throw new Error("ไม่ได้รับสิทธิ์ใช้กล้อง กรุณาอนุญาตแล้วลองใหม่");
            }
            if (["NotFoundError", "DevicesNotFoundError"].includes(error?.name)) throw new Error("ไม่พบอุปกรณ์กล้อง");
            throw error;
        }
    }

    stop() {
        this.running = false;
        this.sendInFlight = false;
        this.generation += 1;
        try { this.camera?.stop(); } catch (error) { console.error(error); }
        try { this.faceMesh?.close(); } catch (error) { console.error(error); }
        this.camera = null;
        this.faceMesh = null;
        const stream = this.video?.srcObject;
        if (stream?.getTracks) stream.getTracks().forEach((track) => track.stop());
        if (this.video) this.video.srcObject = null;
    }
}
