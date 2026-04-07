#!/usr/bin/env python3
"""
Data Preparation Script for Human-Annotated Val Set Visualization

This script prepares all data needed for the static HTML visualization:
1. Copies SR RGB images
2. Copies camera parameters
3. Generates pointcloud PLY files from depth maps (with scale correction)
4. Extracts per-image bbox data from the val JSON
5. Builds the master index.json with scene hierarchy

Author: 3D Bounding Box Detection Team
Date: 2026-03-24
"""

import json
import os
import shutil
import ctypes
import colorsys
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# ============================================================================
# Configuration
# ============================================================================

BASE_DATA_PATH = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/single_frame_data"
EXPERIMENT_PATH = f"{BASE_DATA_PATH}/experiment"
HUMAN_DATA_PATH = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/human_interface/data"

# Input paths
# Note: _nocomp.json has unscaled box3d values, depth maps are also unscaled
VAL_JSON_PATH = f"{HUMAN_DATA_PATH}/v3_val/5_final/human_annotated_val_final_2026-03-25_nocomp.json"

# Scene classification paths - load all files to build complete lookup
# Note: LVIS uses COCO images, so LVIS images should look up in COCO classifications
# LVIS val images come from COCO train2017 (not val2017), so we load both train and val
# No V3Det in this dataset
SCENE_CLASSIFICATION_FILES = [
    # COCO (also covers LVIS since they share the same images)
    ("coco", f"{BASE_DATA_PATH}/../scene_background_diversity/coco/output/train/coco_train_classifications.jsonl"),
    ("coco", f"{BASE_DATA_PATH}/../scene_background_diversity/coco/output/val/coco_val_classifications.jsonl"),
    # Objects365
    ("obj365", f"{BASE_DATA_PATH}/../scene_background_diversity/output_scene_tags/classifications_full.jsonl"),
    ("obj365", f"{BASE_DATA_PATH}/../scene_background_diversity/val_output/obj365_val_classifications.jsonl"),
]

# Depth and camera paths
DEPTH_NEW_BASE = f"{EXPERIMENT_PATH}/v4_depth_new"  # Primary depth maps (unscaled)
DEPTH_OLD_BASE = f"{EXPERIMENT_PATH}/v4_depth"       # Fallback depth + camera intrinsics

# Super-resolution images (used for visualization - matches intrinsics and bbox coordinates)
SR_IMAGES_BASE = f"{EXPERIMENT_PATH}/v4_sr"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Pointcloud settings
DOWNSAMPLE_FACTOR = 2  # Keep every 2nd point (~250k points per image)


# ============================================================================
# Utility Functions
# ============================================================================

def load_jsonl(path):
    """Load JSONL file and return list of dicts."""
    data = []
    if not os.path.exists(path):
        print(f"Warning: JSONL file not found: {path}")
        return data
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def depth_to_pointcloud(depth_m, intrinsics, rgb=None, confidence=None,
                        downsample=DOWNSAMPLE_FACTOR,
                        image_size=None, depth_size=None,
                        confidence_threshold=128):
    """
    Convert depth map to 3D point cloud in camera coordinates.

    Args:
        depth_m: Depth map in meters (H, W)
        intrinsics: 3x3 camera intrinsic matrix (calibrated for image_size)
        rgb: Optional RGB image (H, W, 3)
        confidence: Optional confidence map (H, W), uint8 0-255
        downsample: Downsample factor
        image_size: [W, H] size that intrinsics are calibrated for
        depth_size: [W, H] actual depth map size (for scaling intrinsics)
        confidence_threshold: Filter out points with confidence below this

    Returns:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255) or None
    """
    H, W = depth_m.shape

    # Extract intrinsic parameters
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    # Scale intrinsics if depth size differs from image size
    if image_size is not None and depth_size is not None:
        scale_x = depth_size[0] / image_size[0]
        scale_y = depth_size[1] / image_size[1]
        fx = fx * scale_x
        fy = fy * scale_y
        cx = cx * scale_x
        cy = cy * scale_y

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Apply downsampling
    u = u[::downsample, ::downsample]
    v = v[::downsample, ::downsample]
    z = depth_m[::downsample, ::downsample]

    # Filter valid depth
    valid_mask = z > 0.1  # Minimum depth threshold

    # Filter by confidence map (remove sky/background)
    if confidence is not None:
        conf_downsampled = confidence[::downsample, ::downsample]
        valid_mask &= conf_downsampled >= confidence_threshold

    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = z[valid_mask]

    # Back-project to 3D
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy

    points = np.stack([x, y, z_valid], axis=-1)

    # Extract colors if RGB provided
    colors = None
    if rgb is not None:
        rgb_downsampled = rgb[::downsample, ::downsample]
        colors = rgb_downsampled[valid_mask]

    return points, colors


def depth_to_mesh_glb(depth_m, intrinsics, rgb, confidence=None,
                      downsample=DOWNSAMPLE_FACTOR,
                      image_size=None, depth_size=None,
                      confidence_threshold=128, edge_rtol=0.04):
    """
    Convert depth map to a textured triangle mesh (GLB), MoGe-2 style.

    Each pixel becomes a vertex; neighboring pixels form quads/triangles.
    Depth edges and low-confidence regions are removed.

    Returns:
        bytes: GLB file content, or None on failure
    """
    H, W = depth_m.shape

    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    if image_size is not None and depth_size is not None:
        scale_x = depth_size[0] / image_size[0]
        scale_y = depth_size[1] / image_size[1]
        fx, fy = fx * scale_x, fy * scale_y
        cx, cy = cx * scale_x, cy * scale_y

    # Downsample everything
    depth_ds = depth_m[::downsample, ::downsample]
    h, w = depth_ds.shape

    # Build vertex mask: valid depth + confidence
    mask = depth_ds > 0.1
    if confidence is not None:
        conf_ds = confidence[::downsample, ::downsample]
        mask &= conf_ds >= confidence_threshold

    # Remove depth edges (stretched triangles at object boundaries)
    # A pixel is an edge if max-min depth among its 3x3 neighbors exceeds rtol * depth
    from scipy.ndimage import maximum_filter, minimum_filter
    d_max = maximum_filter(depth_ds, size=3)
    d_min = minimum_filter(depth_ds, size=3)
    edge_mask = (d_max - d_min) > edge_rtol * depth_ds
    mask &= ~edge_mask

    # Build pixel grid → 3D vertices
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # Scale u,v back to original pixel coords for intrinsics
    u_orig = u * downsample
    v_orig = v * downsample

    z = depth_ds
    x = (u_orig - cx) * z / fx
    y = (v_orig - cy) * z / fy

    vertices = np.stack([x, y, z], axis=-1)  # (h, w, 3)

    # Build quad faces from grid: each (i,j) forms a quad with (i+1,j), (i+1,j+1), (i,j+1)
    # Only include quads where all 4 corners are valid
    quad_mask = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]

    # Vertex indices in flattened array
    idx = np.arange(h * w).reshape(h, w)
    i0 = idx[:-1, :-1][quad_mask]  # top-left
    i1 = idx[1:, :-1][quad_mask]   # bottom-left
    i2 = idx[1:, 1:][quad_mask]    # bottom-right
    i3 = idx[:-1, 1:][quad_mask]   # top-right

    # Two triangles per quad
    faces = np.column_stack([
        np.column_stack([i0, i1, i2]),
        np.column_stack([i0, i2, i3])
    ]).reshape(-1, 3)

    if len(faces) == 0:
        return None

    # Flatten vertices
    verts_flat = vertices.reshape(-1, 3)

    # Transform to OpenGL/Three.js coords: flip Y and Z
    verts_flat = verts_flat * [1, -1, -1]

    # UV coordinates: map each vertex to its image position
    uvs = np.stack([u / (w - 1), 1.0 - v / (h - 1)], axis=-1).reshape(-1, 2)

    # Vertex colors from downsampled RGB
    rgb_ds = np.array(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
    # Flatten to (h*w, 3), add alpha channel
    colors_flat = rgb_ds.reshape(-1, 3)
    alpha = np.full((colors_flat.shape[0], 1), 255, dtype=np.uint8)
    vertex_colors = np.hstack([colors_flat, alpha])

    mesh = trimesh.Trimesh(
        vertices=verts_flat,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False
    )

    return mesh.export(file_type='glb')


def save_ply(path, points, colors=None):
    """
    Save point cloud to PLY file (ASCII format).

    Args:
        path: Output path
        points: (N, 3) array
        colors: (N, 3) array of RGB values (0-255) or None
    """
    with open(path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def category_to_color(name):
    """
    Deterministic category name -> RGB color, matching the JS categoryToColor().
    Hash -> hue, fixed saturation=70%, lightness=60%.
    """
    h = 0
    for ch in name:
        h = ord(ch) + ((h << 5) - h)
        h = ctypes.c_int32(h).value  # simulate JS 32-bit signed int
    hue = ((h % 360) + 360) % 360
    # colorsys uses h in [0,1], l, s in [0,1]
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.60, 0.70)
    return (int(r * 255), int(g * 255), int(b * 255))


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])


def box3d_to_corners(box3d):
    """Convert 10D box [cx,cy,cz,w,h,l,qw,qx,qy,qz] to 8 corners."""
    cx, cy, cz, w, h, l, qw, qx, qy, qz = box3d
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    hw, hh, hl = w / 2, h / 2, l / 2
    local_corners = np.array([
        [-hw, -hh, -hl], [ hw, -hh, -hl], [ hw,  hh, -hl], [-hw,  hh, -hl],
        [-hw, -hh,  hl], [ hw, -hh,  hl], [ hw,  hh,  hl], [-hw,  hh,  hl]
    ])
    return (R @ local_corners.T).T + np.array([cx, cy, cz])


def project_3d_to_2d(points, intrinsics):
    """Project Nx3 points to Nx2 pixel coords using 3x3 intrinsics."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * points[:, 0] / points[:, 2] + cx
    v = fy * points[:, 1] / points[:, 2] + cy
    return np.stack([u, v], axis=1)


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
]


def render_annotated_image(sr_image_path, image_entry, intrinsics, output_path):
    """Render 2D and 3D bounding boxes onto the SR image, matching detail page style."""
    img = Image.open(sr_image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    boxes2d = image_entry.get('boxes2d', [])
    boxes3d = image_entry.get('boxes3d', [])
    categories = image_entry.get('categories', [])
    ignore3D = image_entry.get('ignore3D', [])

    # Match detail page: lineWidth=4 on ~900px canvas → scale to SR resolution
    line_w = max(4, round(4 * img.width / 900))

    for i in range(len(boxes3d)):
        if ignore3D[i] != 0:
            continue

        cat = categories[i] if i < len(categories) else f'object_{i}'
        color = category_to_color(cat)

        # Draw 3D box
        inner = boxes3d[i]
        if not inner or len(inner) == 0:
            continue
        box = inner[0]
        box3d = box.get('box3d')
        if not box3d:
            continue

        corners = box3d_to_corners(box3d)
        if np.any(corners[:, 2] <= 0.1):
            continue
        pts2d = project_3d_to_2d(corners, intrinsics)

        for a, b in BOX_EDGES:
            draw.line(
                [(pts2d[a, 0], pts2d[a, 1]), (pts2d[b, 0], pts2d[b, 1])],
                fill=color, width=line_w
            )

    img.save(output_path, quality=85)


def get_dataset_and_split(source):
    """
    Determine (dataset, split) for pipeline paths from the val JSON source field.

    Verified mapping:
      coco  -> coco/val   (COCO val2017 images)
      lvis  -> coco/train (LVIS val images come from COCO train2017)
      obj365 -> obj365/val
    """
    mapping = {
        "coco": ("coco", "val"),
        "lvis": ("coco", "train"),
        "obj365": ("obj365", "val"),
    }
    return mapping.get(source, (source, "val"))


# ============================================================================
# Processing Functions
# ============================================================================

# Module-level globals for parallel workers (set in main, inherited via fork)
_scene_by_dataset_and_id = {}
_output_dir = None


def process_single_image(image_entry):
    """
    Process a single image: copy files, generate pointcloud, render annotated image.

    Args:
        image_entry: Val JSON image entry dict

    Returns:
        dict with image metadata or None if failed
    """
    scene_by_dataset_and_id = _scene_by_dataset_and_id
    output_dir = _output_dir

    try:
        image_id = image_entry["image_id"]
        source = image_entry["source"]
        formatted_id = image_entry["formatted_id"]

        dataset, split = get_dataset_and_split(source)

        # Output paths
        out_subdir = f"{dataset}/{split}"
        img_out_dir = output_dir / "images" / out_subdir
        img_annot_dir = output_dir / "images_annotated" / out_subdir
        camera_out_dir = output_dir / "camera" / out_subdir
        pc_out_dir = output_dir / "pointclouds" / out_subdir

        img_out_dir.mkdir(parents=True, exist_ok=True)
        img_annot_dir.mkdir(parents=True, exist_ok=True)
        camera_out_dir.mkdir(parents=True, exist_ok=True)
        pc_out_dir.mkdir(parents=True, exist_ok=True)

        # Source paths
        sr_rgb_src = Path(SR_IMAGES_BASE) / dataset / split / "sr_images" / f"{formatted_id}.jpg"
        # Depth: try v4_depth_new first, fall back to v4_depth
        depth_src = Path(DEPTH_NEW_BASE) / dataset / split / "depth" / f"{formatted_id}_sr_1024_long.npy"
        if not depth_src.exists():
            depth_src = Path(DEPTH_OLD_BASE) / dataset / split / "depth" / f"{formatted_id}_sr_1024_long.npy"
        # Camera intrinsics always from v4_depth
        camera_src = Path(DEPTH_OLD_BASE) / dataset / split / "camera_parameters" / f"{formatted_id}.json"

        # Check if source files exist
        if not sr_rgb_src.exists():
            return None
        if not depth_src.exists():
            return None
        if not camera_src.exists():
            return None

        # Copy SR RGB image
        img_out_path = img_out_dir / f"{formatted_id}.jpg"
        if not img_out_path.exists():
            shutil.copy2(sr_rgb_src, img_out_path)

        # Copy camera params
        camera_out_path = camera_out_dir / f"{formatted_id}.json"
        if not camera_out_path.exists():
            shutil.copy2(camera_src, camera_out_path)

        # Load camera params for pointcloud generation
        with open(camera_src, 'r') as f:
            camera_params = json.load(f)
        intrinsics = np.array(camera_params["intrinsics"])
        image_size = camera_params.get("image_size")  # [W, H] intrinsics are calibrated for
        depth_size = camera_params.get("depth_size")   # [W, H] actual depth map size

        # Generate annotated image (3D boxes projected onto SR image)
        annot_out_path = img_annot_dir / f"{formatted_id}.jpg"
        if not annot_out_path.exists():
            render_annotated_image(sr_rgb_src, image_entry, intrinsics, annot_out_path)

        # Generate 3D mesh (GLB) — textured triangle mesh like MoGe-2
        mesh_out_path = pc_out_dir / f"{formatted_id}.glb"
        if not mesh_out_path.exists():
            # Load depth (already pre-scaled in pipeline)
            depth_mm = np.load(depth_src)
            depth_m = depth_mm.astype(np.float32) / 1000.0
            depth_h, depth_w = depth_m.shape

            # Load SR RGB for texture
            try:
                rgb_img = Image.open(sr_rgb_src).convert('RGB')
                rgb = np.array(rgb_img)
                if rgb.shape[0] != depth_h or rgb.shape[1] != depth_w:
                    rgb_img = rgb_img.resize((depth_w, depth_h), Image.BILINEAR)
                    rgb = np.array(rgb_img)
            except Exception as e:
                print(f"Error loading RGB for mesh: {e}")
                return None

            # Load confidence map from v4_depth (old) to filter sky/background
            confidence = None
            conf_src = Path(DEPTH_OLD_BASE) / dataset / split / "confidence" / f"{formatted_id}.png"
            if conf_src.exists():
                try:
                    conf_img = Image.open(conf_src).convert('L')
                    confidence = np.array(conf_img)
                    if confidence.shape[0] != depth_h or confidence.shape[1] != depth_w:
                        conf_img = conf_img.resize((depth_w, depth_h), Image.NEAREST)
                        confidence = np.array(conf_img)
                except Exception as e:
                    pass

            # Generate and save mesh
            glb_data = depth_to_mesh_glb(
                depth_m, intrinsics, rgb, confidence=confidence,
                image_size=image_size, depth_size=depth_size
            )
            if glb_data:
                with open(mesh_out_path, 'wb') as f:
                    f.write(glb_data)

        # Get scene classification using (dataset, image_id) key
        # LVIS uses COCO images, so look up in "coco" classifications
        lookup_dataset = "coco" if source == "lvis" else source
        scene_path = "unknown"

        if (lookup_dataset, image_id) in scene_by_dataset_and_id:
            scene_path = scene_by_dataset_and_id[(lookup_dataset, image_id)]

        # Get image dimensions
        width = image_entry.get("width", 0)
        height = image_entry.get("height", 0)

        # Count valid annotations
        num_annotations = len(image_entry.get("boxes2d", []))
        num_valid = sum(1 for ig in image_entry.get("ignore3D", []) if ig == 0)

        # Return metadata
        return {
            "image_id": image_id,
            "original_id": image_id,
            "source": source,
            "split": split,
            "dataset": dataset,
            "file_name": f"{out_subdir}/{formatted_id}.jpg",
            "scene_path": scene_path,
            "width": width,
            "height": height,
            "formatted_id": formatted_id,
            "num_annotations": num_annotations,
            "num_valid_boxes": num_valid,
        }

    except Exception as e:
        print(f"Error processing image {image_entry.get('image_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def write_box_data(val_images, output_dir):
    """
    Write per-image box data extracted from the val JSON.

    Each file contains: boxes2d, boxes3d (1 selected box per annotation),
    categories, ignore3D, ignore_reasons, human_quality.
    """
    boxes_out = output_dir / "boxes"
    boxes_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for img in tqdm(val_images, desc="Writing box data"):
        source = img['source']
        formatted_id = img['formatted_id']
        dataset, split = get_dataset_and_split(source)

        out_path = boxes_out / f"{dataset}_{split}_{formatted_id}.json"

        if not out_path.exists():
            box_data = {
                "image_id": img['image_id'],
                "source": source,
                "boxes2d": img['boxes2d'],
                "boxes3d": img['boxes3d'],
                "categories": img['categories'],
                "ignore3D": img['ignore3D'],
                "ignore_reasons": img.get('ignore_reasons', []),
                "human_quality": img.get('human_quality', []),
            }

            with open(out_path, 'w') as f:
                json.dump(box_data, f)
            written += 1

    print(f"Wrote {written} box data files")


def build_scene_tree(images_metadata):
    """Build hierarchical scene tree from image metadata."""
    tree = {"name": "root", "path": "", "children": [], "image_count": 0}
    node_map = {"": tree}

    for img in images_metadata:
        scene_path = img["scene_path"]
        if not scene_path or scene_path == "unknown":
            continue

        parts = scene_path.split("/")
        current_path = ""

        for part in parts:
            parent_path = current_path
            current_path = f"{current_path}/{part}" if current_path else part

            if current_path not in node_map:
                new_node = {
                    "name": part,
                    "path": current_path,
                    "children": [],
                    "image_count": 0
                }
                node_map[parent_path]["children"].append(new_node)
                node_map[current_path] = new_node

        # Increment count for leaf and all ancestors
        current_path = scene_path
        while current_path:
            if current_path in node_map:
                node_map[current_path]["image_count"] += 1
            if "/" in current_path:
                current_path = "/".join(current_path.split("/")[:-1])
            else:
                current_path = ""

        tree["image_count"] += 1

    return tree


def load_scene_classifications():
    """
    Load all scene classifications and build lookup by (dataset, image_id).

    Uses composite key to avoid ID collisions between datasets.

    Returns:
        dict: Mapping from (dataset, image_id) to scene_path
    """
    scene_by_dataset_and_id = {}

    for dataset, path in SCENE_CLASSIFICATION_FILES:
        if not os.path.exists(path):
            print(f"Warning: Scene classification file not found: {path}")
            continue

        classifications = load_jsonl(path)
        for cls in classifications:
            img_id = cls.get("image_id")
            scene_path = cls.get("selected_path", "unknown")
            if img_id is not None:
                scene_by_dataset_and_id[(dataset, img_id)] = scene_path

    print(f"Loaded {len(scene_by_dataset_and_id)} scene classifications")
    return scene_by_dataset_and_id


def main():
    parser = argparse.ArgumentParser(description="Prepare visualization data for human-annotated val set")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for testing)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print("=" * 60)
    print("Human-Annotated Val Set Visualization Data Preparation")
    print("=" * 60)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load val set annotations
    print("\n1. Loading human-annotated val set...")
    with open(VAL_JSON_PATH, 'r') as f:
        val_data = json.load(f)

    val_images = val_data["images"]
    total_annotations = sum(len(img.get("boxes2d", [])) for img in val_images)
    total_valid = sum(sum(1 for ig in img.get("ignore3D", []) if ig == 0) for img in val_images)

    print(f"   Loaded {len(val_images)} images, {total_annotations} annotations ({total_valid} valid)")

    if args.limit:
        val_images = val_images[:args.limit]
        print(f"   Limited to {len(val_images)} images for testing")

    # Load scene classifications
    print("\n2. Loading scene classifications...")
    scene_by_dataset_and_id = load_scene_classifications()

    # Set module-level globals for parallel workers (inherited via fork, no pickling)
    global _scene_by_dataset_and_id, _output_dir
    _scene_by_dataset_and_id = scene_by_dataset_and_id
    _output_dir = output_dir

    # Process images
    print(f"\n3. Processing {len(val_images)} images...")

    images_metadata = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_image, img): img["image_id"]
                       for img in val_images}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result:
                    images_metadata.append(result)
    else:
        for img in tqdm(val_images, desc="Processing"):
            result = process_single_image(img)
            if result:
                images_metadata.append(result)

    print(f"   Successfully processed {len(images_metadata)} images")

    # Filter out images with 0 valid boxes
    before_filter = len(images_metadata)
    images_metadata = [img for img in images_metadata if img["num_valid_boxes"] > 0]
    print(f"   Filtered to {len(images_metadata)} images with valid 3D boxes (removed {before_filter - len(images_metadata)})")

    # Build set of valid image IDs for filtering box data writes
    valid_image_ids = {img["image_id"] for img in images_metadata}

    # Write box data (only for images with valid boxes)
    print("\n4. Writing bounding box data...")
    write_box_data([img for img in val_images if img["image_id"] in valid_image_ids], output_dir)

    # Build scene tree
    print("\n5. Building scene tree...")
    scene_tree = build_scene_tree(images_metadata)

    # Build images_by_scene lookup
    images_by_scene = defaultdict(list)
    for img in images_metadata:
        scene_path = img["scene_path"]
        if scene_path and scene_path != "unknown":
            images_by_scene[scene_path].append(img["image_id"])

    # Check for bbox file availability
    for img in images_metadata:
        dataset = img["dataset"]
        split = img["split"]
        formatted_id = img["formatted_id"]

        boxes_path = output_dir / "boxes" / f"{dataset}_{split}_{formatted_id}.json"
        img["has_boxes"] = boxes_path.exists()

    # Build master index
    print("\n6. Building master index...")

    index = {
        "total_images_in_dataset": len(val_images),
        "total_images": len(images_metadata),
        "scene_tree": scene_tree,
        "images": images_metadata,
        "images_by_scene": dict(images_by_scene),
    }

    index_path = output_dir / "index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"   Saved index to {index_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images processed: {len(images_metadata)}")
    print(f"Images with boxes: {sum(1 for img in images_metadata if img['has_boxes'])}")
    print(f"Scene categories: {len(images_by_scene)}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
