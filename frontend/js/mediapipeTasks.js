/**
 * One place for MediaPipe Tasks Vision (pinned version + loader).
 *
 * The whole app uses this build only. The older @mediapipe/face_mesh "solutions"
 * script must NOT be loaded on the same page: both are Emscripten WebAssembly
 * builds that read/write the same global `Module`, and once Face Mesh had run,
 * loading Tasks Vision crashed with "abort(Module.arguments has been replaced ...)".
 */

// Pinned so the JS bundle and its WASM files always match.
export const TASKS_VISION_VERSION = "0.10.14";
export const TASKS_VISION_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;
export const FACE_LANDMARKER_MODEL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let visionPromise = null;

/** { vision, fileset } loaded once per page (a failed load is retried next time). */
export function loadVision() {
    visionPromise ??= (async () => {
        const vision = await import(`${TASKS_VISION_BASE}/vision_bundle.mjs`);
        const fileset = await vision.FilesetResolver.forVisionTasks(`${TASKS_VISION_BASE}/wasm`);
        return { vision, fileset };
    })().catch((error) => {
        visionPromise = null;
        throw error;
    });
    return visionPromise;
}

/** createFromOptions on the GPU delegate, falling back to CPU where WebGL isn't usable. */
export async function createTask(TaskClass, fileset, options) {
    try {
        return await TaskClass.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: "GPU" } });
    } catch (gpuError) {
        console.warn(`${TaskClass.name}: GPU delegate unavailable, falling back to CPU`, gpuError);
        return TaskClass.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: "CPU" } });
    }
}
