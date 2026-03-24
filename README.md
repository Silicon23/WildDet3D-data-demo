# Tinyval 3D Bounding Box Visualization Server

A static HTML visualization for the Tinyval dataset (5,000 images) designed to be hosted on GitHub Pages. Displays 3D bounding boxes from multiple models overlaid on images and in interactive 3D point cloud views.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Data Sources](#data-sources)
- [Data Preparation](#data-preparation)
- [Coordinate Systems](#coordinate-systems)
- [Known Issues](#known-issues)
- [Troubleshooting History](#troubleshooting-history)

---

## Overview

This visualization server displays 3D bounding box predictions from multiple models on the Tinyval dataset. It provides:
- A gallery view organized by scene categories (indoor/outdoor hierarchy)
- Detailed image pages with filtered and exploratory bounding box views
- Interactive 3D point cloud visualization with Three.js

## Features

### Main Page (Gallery View)
- Collapsible tree structure organized by scene category
- Image cards showing source dataset (COCO, LVIS, V3Det, Obj365)
- Click any image to open its detail page

### Image Detail Page

#### Section 1: Filtered Scored Boxes
- **Score threshold slider** (0-11 range)
- **Filtering logic**: For each object, select the single best box with `vlm_total_score >= threshold`
- **Tie-breaker priority**: sam3d > algorithm > 3d_mood > detany3d
- **Visualizations**:
  1. 2D bboxes (ground truth) overlaid on RGB
  2. 3D bboxes projected to 2D overlaid on RGB
  3. Interactive 3D point cloud + 3D bboxes

#### Section 2: Explore All Boxes
- **Object buttons**: Select one object at a time (objects with ≥1 3D box)
- **Model toggles**: la3d, sam3d, algorithm, 3d_mood, detany3d (independent selection)
- **Score comparison table**: Shows VLM scores (6 metrics + total) for each model
- **Visualizations**: Same 3 views as Section 1, but for selected object/models

### Color Scheme
| Model | Color | Hex |
|-------|-------|-----|
| la3d | Purple | #a855f7 |
| sam3d | Green | #22c55e |
| algorithm | Orange | #f97316 |
| 3d_mood | Red | #ef4444 |
| detany3d | Blue | #3b82f6 |

---

## Directory Structure

```
visualization/
├── index.html              # Main gallery page
├── image.html              # Detail page template
├── css/
│   └── styles.css          # All styling
├── js/
│   ├── app.js              # Main application logic
│   ├── three-viewer.js     # Three.js point cloud/bbox viewer
│   ├── overlay-renderer.js # 2D canvas overlay rendering
│   └── scene-tree.js       # Collapsible scene hierarchy
├── scripts/
│   └── prepare_data.py     # Data preparation script
├── data/                   # Generated data (gitignored, ~15GB)
│   ├── index.json          # Master index with all image metadata
│   ├── images/             # SR images (2560x1576 or similar)
│   ├── pointclouds/        # PLY files (ASCII format, ~250k points each)
│   ├── camera/             # Camera parameters JSON
│   ├── boxes_scored/       # Step 3.5 merged scored boxes
│   └── boxes_unscored/     # Step 2.9 merged unscored boxes
└── README.md               # This file
```

---

## Data Sources

### Pipeline Outputs (v4 experiment)
All source data comes from the 3D bounding box detection pipeline at:
```
/weka/oe-training-default/weikaih/3d_boundingbox_detection/single_frame_data/experiment/
```

| Step | Directory | Description |
|------|-----------|-------------|
| Step 0 | `v4_sr/` | Super-resolution images (4x upscaled) |
| Step 1 | `v4_depth/` | Depth maps (.npy) + camera parameters |
| Step 2.9 | `v4_unify/` | Unscored boxes (sam3d, 3d_mood, detany3d) |
| Step 2.9 | `v4_unify_algorithm/` | Unscored algorithm boxes |
| Step 2.9 | `v4_unify_la3d/` | Unscored LA3D boxes |
| Step 3 | `v4_score/` | Scored boxes (sam3d, 3d_mood, detany3d) |
| Step 3 | `v4_score_algorithm/` | Scored algorithm boxes |
| Step 3 | `v4_score_la3d/` | Scored LA3D boxes |
| Step 3.5 | `v4_score_merged/` | Merged scored boxes (4 models: sam3d, 3d_mood, detany3d, algorithm) |
| Step 3.5 | `v4_score_merged_la3d/` | **Final merged scored boxes (all 5 models including LA3D)** |

### Scene Classifications
Scene category labels come from:
```
/weka/oe-training-default/weikaih/3d_boundingbox_detection/scene_background_diversity/
```

Each dataset has separate classification files:
- COCO: `coco/output/{train,val}/coco_{train,val}_classifications.jsonl`
- V3Det: `v3det/output/{train,val}/v3det_{train,val}_classifications.jsonl`
- Obj365: `output_scene_tags/classifications_full.jsonl` (train), `val_output/obj365_val_classifications.jsonl` (val)

**Important**: LVIS images use COCO classification files since LVIS images are from COCO.

### Tinyval Dataset
```
/weka/oe-training-default/weikaih/3d_boundingbox_detection/single_frame_data/unified_datasets/unified_tinyval_5k.json
```

Contains 5,000 images from COCO, LVIS, V3Det, and Obj365 with unified annotation format.

---

## Data Preparation

### Running the Script

```bash
cd /weka/oe-training-default/weikaih/3d_boundingbox_detection/single_frame_data/visualization
/weka/oe-training-default/jieyuz2/improve_segments/miniconda3/bin/python scripts/prepare_data.py --workers 16
```

Options:
- `--workers N`: Number of parallel workers (default: 8)
- `--limit N`: Process only first N images (for testing)
- `--output-dir PATH`: Custom output directory

### What It Does

1. **Loads Tinyval annotations** from unified JSON
2. **Loads scene classifications** keyed by `(dataset, image_id)` to avoid ID collisions
3. **For each image**:
   - Copies SR image from `v4_sr`
   - Copies camera parameters from `v4_depth`
   - Generates point cloud PLY from depth map (downsample factor: 2, ~250k points)
4. **Copies bounding box results**:
   - Scored boxes from `v4_score_merged_la3d` (all 5 models pre-merged: sam3d, detany3d, 3d_mood, algorithm, la3d)
   - Unscored boxes: merges `v4_unify` + `v4_unify_algorithm` + `v4_unify_la3d`
5. **Builds scene tree** for hierarchical navigation
6. **Generates master index.json** with all image metadata

### Point Cloud Generation

```python
# Settings in prepare_data.py
DOWNSAMPLE_FACTOR = 2  # Keep every 2nd pixel → ~250k points per image

# Intrinsics scaling
# Camera params have intrinsics for image_size, but depth is at depth_size
# We scale intrinsics: fx' = fx * (depth_width / image_width)
```

---

## Coordinate Systems

### Camera Coordinates (OpenCV Convention)
All 3D data uses OpenCV camera coordinates:
- **X-axis**: Right
- **Y-axis**: Down  
- **Z-axis**: Forward (into scene)

### 3D Bounding Box Format (10D)
```
[x, y, z, w, h, l, qw, qx, qy, qz]
```
- `x, y, z`: Center position in camera coordinates (meters)
- `w, h, l`: Box dimensions (width, height, length in meters)
- `qw, qx, qy, qz`: Rotation quaternion

### Three.js Coordinate Transform
Three.js uses Y-up convention. We transform by negating Y and Z:
```javascript
// OpenCV to Three.js
const transformedCorners = corners.map(([x, y, z]) => [x, -y, -z]);
```

### Image Resolutions
| Type | Resolution | Notes |
|------|------------|-------|
| Original | ~640×480 | Original dataset images |
| SR (Step 0) | ~2560×1920 | 4x super-resolution |
| Depth (Step 1) | ~1024×768 | 1024 on longest edge |
| Visualization | SR resolution | Uses SR images and scaled intrinsics |

---

## Known Issues

### 1. Algorithm Boxes Were Misaligned (FIXED - Jan 2026)

**Status**: Fixed. Algorithm boxes are now included in the visualization.

**Root Cause**: The algorithm pipeline (`process.py`) had a bug where it loaded annotation masks at original resolution instead of SR resolution.

**Details**:
- Annotations are defined for original images (e.g., 640×394)
- The pipeline used `coco.annToMask(ann)` which returns masks at original resolution
- But depth/camera params are for SR images (e.g., 2560×1576)
- When the mask was resized to match depth, small objects got placed at wrong pixel locations

**Before fix** (Image 340175):
```
Annotation 10 (small book, 6×18 px):
  - detany3d: center=(-3.70, -1.67, 6.07)  ← Correct
  - algorithm: center=(0.11, 1.88, 3.66)   ← Wrong!
```

**After fix**:
```
Annotation 10 (small book, 6×18 px):
  - detany3d: center=(-3.70, -1.67, 6.07)  ← Correct
  - algorithm: center=(-3.75, -1.68, 6.06) ← Now aligned!
```

**Fix applied** (in algorithm pipeline `process.py`):
```python
# Changed from:
mask_2d_orig = coco.annToMask(ann)

# To:
mask_2d_orig = dataset_helper.get_annotation_mask(ann, image_id)
```

The `DatasetHelper.get_annotation_mask()` method properly rescales masks from original to SR resolution.

### 2. Scene Classification ID Collisions (FIXED)

**Issue**: Different datasets (COCO, Obj365) can have the same `image_id` for different images. Loading all classifications into a single dictionary caused overwriting.

**Fix**: Classifications are now keyed by `(dataset, image_id)` tuple.

### 3. Point Cloud Too Sparse (FIXED)

**Issue**: Original downsample factor of 10 resulted in only ~7k points per image.

**Fix**: Changed `DOWNSAMPLE_FACTOR` from 10 to 2, resulting in ~250k points.

---

## Troubleshooting History

### Problem: Bounding boxes misaligned with image
**Cause**: Visualization was using original low-res images but camera params were for SR images.
**Fix**: Updated to use SR images from `v4_sr` directory.

### Problem: 2D bounding boxes stretched/distorted
**Cause**: CSS was forcing canvas to fill container, distorting aspect ratio.
**Fix**: Changed canvas CSS to `max-width: 100%; max-height: 100%` instead of `width: 100%; height: 100%`.

### Problem: Score table shows all '-' in Explore section
**Cause**: Was trying to read VLM scores from unscored boxes.
**Fix**: Modified to look up scores from `appState.scoredBoxes` by matching annotation index and model.

### Problem: Point cloud viewer hard to navigate
**Cause**: Default Three.js OrbitControls settings not optimal.
**Fix**: Added right-click pan, dynamic zoom limits based on point cloud size, reset camera button.

### Problem: Algorithm toggle button not working
**Cause**: Code used `algorithm_regression` but data uses `algorithm`.
**Fix**: Updated `MODEL_PRIORITY`, `activeModels`, and HTML to use `algorithm`.

### Problem: Algorithm boxes misaligned (small objects especially)
**Cause**: Algorithm pipeline loaded masks at original resolution instead of SR resolution.
**Fix**: Changed `coco.annToMask(ann)` to `dataset_helper.get_annotation_mask(ann, image_id)` in `process.py`. Then re-ran:
1. Step 2 Algorithm (batch_process_v4.py)
2. Step 3 Scoring Algorithm
3. Step 3.5 Merge
4. Regenerate visualization data with `prepare_data.py`

---

## Technical Notes

### VLM Scoring (Step 3)
Each 3D box is scored on 6 metrics (0/1/2 scale each):
- `category`: Object category correctness
- `scale`: Size accuracy
- `translation`: Position accuracy
- `shape`: Aspect ratio accuracy
- `tilt`: Up/down orientation
- `rotation`: Yaw/heading accuracy

**Total score**: 0-11 (sum of all metrics, with category weighted)

### Model Priority Order
For tie-breaking when multiple models have the same score:
1. la3d (highest priority)
2. sam3d
3. algorithm
4. 3d_mood
5. detany3d

All 5 models now have proper VLM scores from the unified scoring pipeline.

### Browser Compatibility
- Tested on Chrome, Firefox, Safari
- Requires WebGL support for 3D viewer
- Uses ES6+ JavaScript features

---

## License
Internal use only - part of 3D Bounding Box Detection project.
