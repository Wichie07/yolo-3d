# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO-3D is a real-time pseudo-3D object detection system combining YOLO26 (2D detection) with Depth Anything v2 (monocular depth estimation) to produce 3D bounding boxes and Bird's Eye View (BEV) visualization from video input.

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run.py
```

All configuration lives at the top of `run.py`'s `main()` function — input source, model sizes, thresholds, and feature toggles (tracking, BEV, pseudo-3D boxes). There are no tests.

## Architecture

The pipeline processes video frames sequentially:

```
Video frame
  → ObjectDetector (YOLO26)      → 2D boxes, class IDs, track IDs
  → DepthEstimator (Depth Anything v2) → per-frame depth map (0–1 normalized)
  → BBox3DEstimator               → 3D box params (location, dimensions, orientation)
  → Visualization                 → 3D wireframe overlay + BirdEyeView panel
  → Output file / display
```

**Key modules:**

- `run.py` — orchestrates the full pipeline, handles video I/O, FPS display, and graceful GPU error recovery
- `detection_model.py` — `ObjectDetector` wraps Ultralytics YOLO26; returns `[bbox, score, class_id, object_id]` per frame
- `depth_model.py` — `DepthEstimator` wraps Hugging Face Depth Anything v2; provides normalized depth maps and point/region depth queries
- `bbox3d_utils.py` — `BBox3DEstimator` converts 2D boxes + depth into pseudo-3D parameters using camera intrinsics; `BirdEyeView` renders the top-down view; both include per-object Kalman filtering for temporal smoothing
- `load_camera_params.py` — loads camera intrinsic/extrinsic matrices from JSON or falls back to KITTI-like defaults; defines class-specific default dimensions for 30+ object classes

## Device Handling

- Prefers CUDA → MPS (Apple Silicon) → CPU
- Depth model is forced to CPU on MPS due to limitations
- GPU errors during inference trigger a CPU fallback automatically

## Depth-to-Distance Mapping

Normalized depth (0–1) maps to 1–10 m display range. Depth sampling strategy is class-specific: center-point for persons/animals, median of bounding box region for rigid objects.

## 3D Box Visualization

Boxes are pseudo-3D wireframes: the front face corresponds to the 2D detection box, the back face is offset by estimated depth. Overlays show class name, confidence, depth value, and track ID.
