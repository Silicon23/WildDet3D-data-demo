# WildDet3D-Bench Visualization

Interactive visualization of 3D bounding boxes from the [WildDet3D-Bench](https://huggingface.co/datasets/Silicon23/WildDet3D-demo) dataset. Browse 2,196 images from COCO, LVIS, and Objects365 with projected 3D boxes and interactive point cloud views.

**[Live Demo](https://silicon23.github.io/WildDet3D-data-demo/)**

## Features

### Gallery
- 3-column image grid with pre-rendered 3D box overlays as thumbnails
- Filter by dataset split (COCO Val, LVIS Train, Obj365 Val)
- Search by image ID
- Collapsible scene category sidebar (indoor/outdoor hierarchy)
- Shuffle mode for random browsing
- 2,470 total images in the val set, 2,196 with valid 3D annotations

### Image Detail Page
Three synchronized visualizations for each image:

1. **2D Bounding Boxes** -- ground truth 2D boxes overlaid on the image
2. **3D Boxes Projected to 2D** -- human-selected 3D boxes projected onto the image using camera intrinsics
3. **Interactive 3D Point Cloud** -- navigate the reconstructed point cloud with 3D box wireframes

All boxes are colored by object category. The same category always maps to the same color across all images.

### Interaction
- **Click-to-hide** -- click any box in any view to hide that annotation across all three visualizations
- **Editable labels** -- annotation list panel at the bottom lets you rename categories and toggle visibility per annotation
- **Show/hide labels** -- toggle category name labels on the visualizations
- **Combined download** -- export the 3D projection and point cloud views side-by-side as a single PNG (1600x600)
- **Individual downloads** -- save any single view as PNG, or export a point cloud rotation video

### Controls (3D Viewer)
| Action | Control |
|--------|---------|
| Rotate | Left drag |
| Pan | Right drag |
| Zoom | Scroll |
| Reset view | Double-click |

## Architecture

Static HTML/CSS/JS site. No build step, no server-side processing. Data is served from HuggingFace.

```
visualization/
+-- index.html              # Gallery page
+-- image.html              # Detail page
+-- css/styles.css          # Dark theme styling
+-- js/
|   +-- app.js              # Main application logic
|   +-- overlay-renderer.js # 2D canvas overlay rendering
|   +-- three-viewer.js     # Three.js point cloud + bbox viewer
|   +-- scene-tree.js       # Collapsible scene hierarchy
+-- scripts/
|   +-- prepare_data.py     # Data preparation pipeline
|   +-- upload_hf.py        # Upload data to HuggingFace
```

### Data (hosted on HuggingFace)

All data is served from [`Silicon23/WildDet3D-demo`](https://huggingface.co/datasets/Silicon23/WildDet3D-demo):

| Directory | Description | Size |
|-----------|-------------|------|
| `data/index.json` | Master index with image metadata and scene tree | 880K |
| `data/boxes/` | Per-image JSON with 2D/3D boxes, categories, ignore flags | 36M |
| `data/images/` | Super-resolution images (4x upscaled) | 9.9G |
| `data/images_annotated/` | Pre-rendered thumbnails with 3D box overlays | 2.1G |
| `data/camera/` | Camera intrinsics per image | 25M |
| `data/pointclouds/` | PLY point clouds (~250k points each) | 16G |

## Data Source

The visualization displays the **WildDet3D-Bench** validation set from the WildDet3D 3D bounding box detection pipeline:

- **2,470 images** from COCO (424), LVIS (1,113), and Objects365 (933)
- **9,256 valid annotations** across 2,196 images (274 images have no valid boxes after filtering)
- Each annotation has exactly **one human-selected 3D bounding box** chosen from multiple algorithm candidates
- Annotations rated `unacceptable` by human reviewers or failing geometric validation are marked `ignore3D=1`

### 3D Box Format (10D)
```
[cx, cy, cz, w, h, l, qw, qx, qy, qz]
```
- `cx, cy, cz`: Center in camera coordinates (meters)
- `w, h, l`: Dimensions (meters)
- `qw, qx, qy, qz`: Rotation quaternion

### Coordinate System
- Camera coordinates: OpenCV convention (X-right, Y-down, Z-forward)
- 2D boxes: `[x1, y1, x2, y2]` in 4x super-resolution pixel coordinates
- Three.js transform: Y and Z negated for Y-up convention

## Data Preparation

To regenerate the data directory from source pipeline outputs:

```bash
python scripts/prepare_data.py --workers 64
```

This reads the val set JSON, copies SR images and camera parameters, generates point clouds from depth maps, renders annotated thumbnails, and builds the master index. Output goes to `data/` (~28GB).

To upload to HuggingFace:

```bash
python scripts/upload_hf.py
```

## Hosting

The site is hosted on GitHub Pages and serves data from HuggingFace. To run locally:

```bash
python -m http.server 8000
```

## License

Part of the WildDet3D project.
