# Golden Ratio & Glasses AR review

## Fixed
- Corrected the malformed MediaPipe asset URL.
- Added landmark/frame validation, roll-aligned measurements, pose/size quality gates, and multi-pair symmetry.
- Added median multi-frame stabilization; only stable Golden Ratio results can be saved.
- Moved glasses from outer-eye anchors to eye centers, added iris-assisted placement and frame-rate-independent smoothing.
- Fixed duplicate camera sessions and the capture bug that hid the preview immediately after taking a photo.
- Matched SVG/canvas cover behavior to the video and added stricter API validation.
- Added Node unit tests and syntax checks.

## Run checks
```bash
cd frontend
npm run check
npm test

cd ../backend
python -m compileall app
```

The score is a heuristic geometry comparison, not a medical assessment or attractiveness score. MediaPipe landmark 10 is the top of the tracked face oval, not the true hairline.
