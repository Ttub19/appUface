/**
 * Shared, mutable application state.
 * Exported as a single object (rather than several `let` bindings) so any
 * module can `import { appState } from "./state.js"` and mutate a property
 * directly — reassigning an imported `let` binding from another module isn't
 * allowed in ES modules, but mutating a shared object's properties is.
 */
export const appState = {
    selectedImg: null,
    resultObj: null,
    lastConfidence: null,
    genderPref: null, // "male" / "female": who the recommendations are for (set by the analysis)
    cameraStream: null,
    currentRole: "guest",
    currentUser: null,
    historyCount: 0,
    favorites: []
};
