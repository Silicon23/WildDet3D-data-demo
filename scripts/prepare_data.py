#!/usr/bin/env python3
"""
Data Preparation Script for Tinyval Visualization

This script prepares all data needed for the static HTML visualization:
1. Copies RGB images
2. Copies camera parameters
3. Generates pointcloud PLY files from depth maps
4. Copies bbox results (scored and unscored)
5. Builds the master index.json with scene hierarchy

Author: 3D Bounding Box Detection Team
Date: 2026-01-12
"""

import json
import os
import shutil
import numpy as np
from PIL import Image
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

# Input paths
TINYVAL_PATH = f"{BASE_DATA_PATH}/unified_datasets/unified_tinyval_5k.json"

# Scene classification paths - load all files to build complete lookup
# Note: LVIS uses COCO images, so LVIS images should look up in COCO classifications
# LVIS val images come from COCO train2017 (not val2017), so we load both train and val
# Each entry is (dataset_name, file_path) to avoid ID collisions between datasets
SCENE_CLASSIFICATION_FILES = [
    # COCO (also covers LVIS since they share the same images)
    ("coco", f"{BASE_DATA_PATH}/../scene_background_diversity/coco/output/train/coco_train_classifications.jsonl"),
    ("coco", f"{BASE_DATA_PATH}/../scene_background_diversity/coco/output/val/coco_val_classifications.jsonl"),
    # V3Det
    ("v3det", f"{BASE_DATA_PATH}/../scene_background_diversity/v3det/output/train/v3det_train_classifications.jsonl"),
    ("v3det", f"{BASE_DATA_PATH}/../scene_background_diversity/v3det/output/val/v3det_val_classifications.jsonl"),
    # Objects365
    ("obj365", f"{BASE_DATA_PATH}/../scene_background_diversity/output_scene_tags/classifications_full.jsonl"),
    ("obj365", f"{BASE_DATA_PATH}/../scene_background_diversity/val_output/obj365_val_classifications.jsonl"),
]

# Depth and camera paths
DEPTH_BASE = f"{EXPERIMENT_PATH}/v4_depth"

# Super Resolution images (Step 0 output) - these match the camera intrinsics
SR_IMAGES_BASE = f"{EXPERIMENT_PATH}/v4_sr"

# Super-resolution images (used for visualization - matches intrinsics and bbox coordinates)
SR_IMAGES_BASE = f"{EXPERIMENT_PATH}/v4_sr"

# Bbox result paths - use v4_score_merged_la3d (all 5 models merged: sam3d, detany3d, 3d_mood, algorithm, la3d)
SCORED_BOXES_BASE = f"{EXPERIMENT_PATH}/v4_score_merged_la3d"

# Unscored boxes - merged from all sources
UNSCORED_BOXES_BASE = f"{EXPERIMENT_PATH}/v4_unify"
UNSCORED_ALGORITHM_BASE = f"{EXPERIMENT_PATH}/v4_unify_algorithm"
UNSCORED_LA3D_BASE = f"{EXPERIMENT_PATH}/v4_unify_la3d"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Pointcloud settings
DOWNSAMPLE_FACTOR = 2  # Keep every 2nd point (was 10, now 25x denser)


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


def depth_to_pointcloud(depth_m, intrinsics, rgb=None, downsample=DOWNSAMPLE_FACTOR, 
                        image_size=None, depth_size=None):
    """
    Convert depth map to 3D point cloud in camera coordinates.
    
    Args:
        depth_m: Depth map in meters (H, W)
        intrinsics: 3x3 camera intrinsic matrix (calibrated for image_size)
        rgb: Optional RGB image (H, W, 3)
        downsample: Downsample factor
        image_size: [W, H] size that intrinsics are calibrated for
        depth_size: [W, H] actual depth map size (for scaling intrinsics)
        
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


def get_dataset_and_split(source, file_name):
    """
    Determine dataset and split from source and file_name.
    
    Args:
        source: Dataset source (coco, lvis, v3det, obj365)
        file_name: File path from annotation
        
    Returns:
        (dataset, split) tuple
    """
    # Map source to dataset name used in v4_depth paths
    dataset_map = {
        "coco": "coco",
        "lvis": "coco",  # LVIS uses COCO images
        "v3det": "v3det",
        "obj365": "obj365",
    }
    dataset = dataset_map.get(source, source)
    
    # Determine split from file_name
    if "train" in file_name.lower():
        split = "train"
    elif "val" in file_name.lower():
        split = "val"
    elif "test" in file_name.lower():
        split = "test"
    else:
        split = "val"  # Default
    
    return dataset, split


def get_original_image_id(image_entry):
    """
    Get the original image ID from the unified dataset entry.
    For looking up depth/camera files.
    """
    # The original_id field contains the original dataset's image ID
    return image_entry.get("original_id", image_entry["id"])


def format_image_id(image_id, dataset):
    """Format image ID for file lookup based on dataset."""
    return f"{image_id:012d}"


# ============================================================================
# Processing Functions
# ============================================================================

def process_single_image(args):
    """
    Process a single image: copy files, generate pointcloud.
    
    Args:
        args: Tuple of (image_entry, scene_classifications, output_dir)
        
    Returns:
        dict with image metadata or None if failed
    """
    image_entry, scene_by_dataset_and_id, output_dir = args
    
    try:
        image_id = image_entry["id"]
        source = image_entry["source"]
        file_name = image_entry["file_name"]
        original_id = get_original_image_id(image_entry)
        
        dataset, split = get_dataset_and_split(source, file_name)
        formatted_id = format_image_id(original_id, dataset)
        
        # Output paths
        out_subdir = f"{dataset}/{split}"
        img_out_dir = output_dir / "images" / out_subdir
        camera_out_dir = output_dir / "camera" / out_subdir
        pc_out_dir = output_dir / "pointclouds" / out_subdir
        
        img_out_dir.mkdir(parents=True, exist_ok=True)
        camera_out_dir.mkdir(parents=True, exist_ok=True)
        pc_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Source paths - use SR images which match intrinsics and bbox coordinates
        sr_rgb_src = Path(SR_IMAGES_BASE) / dataset / split / "sr_images" / f"{formatted_id}.jpg"
        depth_src = Path(DEPTH_BASE) / dataset / split / "depth" / f"{formatted_id}_sr_1024_long.npy"
        camera_src = Path(DEPTH_BASE) / dataset / split / "camera_parameters" / f"{formatted_id}.json"
        
        # Check if source files exist
        if not sr_rgb_src.exists():
            return None
        if not depth_src.exists():
            return None
        if not camera_src.exists():
            return None
        
        # Copy SR RGB image (already in correct coordinate system for bboxes)
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
        depth_size = camera_params.get("depth_size")  # [W, H] actual depth map size
        
        # Generate pointcloud
        pc_out_path = pc_out_dir / f"{formatted_id}.ply"
        if not pc_out_path.exists():
            # Load depth
            depth_mm = np.load(depth_src)
            depth_m = depth_mm.astype(np.float32) / 1000.0
            
            # Load SR RGB for colors (matches depth size)
            try:
                rgb_img = Image.open(sr_rgb_src).convert('RGB')
                rgb = np.array(rgb_img)
                # Resize RGB to match depth size if needed
                depth_h, depth_w = depth_m.shape
                if rgb.shape[0] != depth_h or rgb.shape[1] != depth_w:
                    rgb_img = rgb_img.resize((depth_w, depth_h), Image.BILINEAR)
                    rgb = np.array(rgb_img)
            except Exception as e:
                print(f"Error loading RGB for pointcloud: {e}")
                rgb = None
            
            # Generate and save pointcloud (scale intrinsics for depth size)
            points, colors = depth_to_pointcloud(
                depth_m, intrinsics, rgb, 
                image_size=image_size, depth_size=depth_size
            )
            save_ply(pc_out_path, points, colors)
        
        # Get scene classification using (dataset, image_id) key
        # LVIS uses COCO images, so look up in "coco" classifications
        lookup_dataset = "coco" if source == "lvis" else source
        scene_path = "unknown"
        
        # Try to find by original_id first, then image_id
        if (lookup_dataset, original_id) in scene_by_dataset_and_id:
            scene_path = scene_by_dataset_and_id[(lookup_dataset, original_id)]
        elif (lookup_dataset, image_id) in scene_by_dataset_and_id:
            scene_path = scene_by_dataset_and_id[(lookup_dataset, image_id)]
        
        # Get image dimensions
        width = image_entry.get("width", 0)
        height = image_entry.get("height", 0)
        
        # Return metadata
        return {
            "image_id": image_id,
            "original_id": original_id,
            "source": source,
            "split": split,
            "dataset": dataset,
            "file_name": f"{out_subdir}/{formatted_id}.jpg",
            "scene_path": scene_path,
            "width": width,
            "height": height,
            "formatted_id": formatted_id,
        }
        
    except Exception as e:
        print(f"Error processing image {image_entry.get('id', 'unknown')}: {e}")
        return None




def copy_bbox_results(tinyval_images, output_dir):
    """
    Copy bbox result files for all images in tinyval.
    
    Scored boxes come from v4_score_merged_la3d which already contains all 5 models:
    - sam3d, detany3d, 3d_mood, algorithm, la3d
    
    Unscored boxes are merged from v4_unify, v4_unify_algorithm, and v4_unify_la3d.
    """
    scored_out = output_dir / "boxes_scored"
    unscored_out = output_dir / "boxes_unscored"
    
    scored_out.mkdir(parents=True, exist_ok=True)
    unscored_out.mkdir(parents=True, exist_ok=True)
    
    copied_scored = 0
    copied_unscored = 0
    
    for img in tqdm(tinyval_images, desc="Copying bbox results"):
        source = img["source"]
        file_name = img["file_name"]
        original_id = get_original_image_id(img)
        dataset, split = get_dataset_and_split(source, file_name)
        formatted_id = format_image_id(original_id, dataset)
        
        # Scored boxes (step 3.5 merged with LA3D) - already contains all 5 models
        # Note: v4_score_merged_la3d uses original source names (lvis, not coco)
        scored_src = Path(SCORED_BOXES_BASE) / source / "val" / f"step30_result_{formatted_id}.json"
        scored_dst = scored_out / f"{dataset}_{split}_{formatted_id}.json"
        
        if not scored_dst.exists() and scored_src.exists():
            shutil.copy2(scored_src, scored_dst)
            copied_scored += 1
        
        # Unscored boxes (step 2.9) - merge from all 3 sources
        # Note: use source directly and val split (all tinyval data is in val/)
        unscored_dst = unscored_out / f"{dataset}_{split}_{formatted_id}.json"
        
        if not unscored_dst.exists():
            unscored_main = Path(UNSCORED_BOXES_BASE) / source / "val" / f"step29_result_{formatted_id}.json"
            unscored_algo = Path(UNSCORED_ALGORITHM_BASE) / source / "val" / f"step29_result_{formatted_id}.json"
            unscored_la3d = Path(UNSCORED_LA3D_BASE) / source / "val" / f"step29_result_{formatted_id}.json"
            
            merged_data = None
            
            # Load main unscored boxes (sam3d, detany3d, 3d_mood)
            if unscored_main.exists():
                with open(unscored_main, 'r') as f:
                    merged_data = json.load(f)
            
            # Merge algorithm boxes
            if unscored_algo.exists():
                with open(unscored_algo, 'r') as f:
                    algo_data = json.load(f)
                
                if merged_data is None:
                    merged_data = algo_data
                else:
                    # Merge algorithm boxes into main data
                    for i, algo_boxes in enumerate(algo_data.get("boxes3d", [])):
                        if i < len(merged_data.get("boxes3d", [])):
                            merged_data["boxes3d"][i].extend(algo_boxes)
                        else:
                            merged_data["boxes3d"].append(algo_boxes)
            
            # Merge LA3D boxes
            if unscored_la3d.exists():
                with open(unscored_la3d, 'r') as f:
                    la3d_data = json.load(f)
                
                if merged_data is None:
                    merged_data = la3d_data
                else:
                    # Merge LA3D boxes into merged data
                    for i, la3d_boxes in enumerate(la3d_data.get("boxes3d", [])):
                        if i < len(merged_data.get("boxes3d", [])):
                            merged_data["boxes3d"][i].extend(la3d_boxes)
                        else:
                            merged_data["boxes3d"].append(la3d_boxes)
            
            if merged_data is not None:
                with open(unscored_dst, 'w') as f:
                    json.dump(merged_data, f)
                copied_unscored += 1
    
    print(f"Copied {copied_scored} scored bbox files, {copied_unscored} unscored bbox files")


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
    
    Uses composite key to avoid ID collisions between datasets
    (e.g., COCO image 383384 vs Obj365 image 383384 are different images).
    
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
                # Key by (dataset, image_id) to avoid collisions
                scene_by_dataset_and_id[(dataset, img_id)] = scene_path
    
    print(f"Loaded {len(scene_by_dataset_and_id)} scene classifications")
    return scene_by_dataset_and_id


def main():
    parser = argparse.ArgumentParser(description="Prepare visualization data")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for testing)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tinyval Visualization Data Preparation")
    print("=" * 60)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Tinyval annotations
    print("\n1. Loading Tinyval annotations...")
    with open(TINYVAL_PATH, 'r') as f:
        tinyval = json.load(f)
    
    images = tinyval["images"]
    annotations = tinyval.get("annotations", [])
    categories = tinyval.get("categories", [])
    
    print(f"   Loaded {len(images)} images, {len(annotations)} annotations")
    
    if args.limit:
        images = images[:args.limit]
        print(f"   Limited to {len(images)} images for testing")
    
    # Load scene classifications
    print("\n2. Loading scene classifications...")
    scene_by_dataset_and_id = load_scene_classifications()
    
    # Build annotation count per image
    ann_count_by_image = defaultdict(int)
    for ann in annotations:
        ann_count_by_image[ann["image_id"]] += 1
    
    # Process images
    print(f"\n3. Processing {len(images)} images...")
    
    # Prepare arguments for parallel processing
    process_args = [(img, scene_by_dataset_and_id, OUTPUT_DIR) for img in images]
    
    images_metadata = []
    
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_image, arg): arg[0]["id"] 
                       for arg in process_args}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result:
                    result["num_annotations"] = ann_count_by_image.get(result["image_id"], 0)
                    images_metadata.append(result)
    else:
        for arg in tqdm(process_args, desc="Processing"):
            result = process_single_image(arg)
            if result:
                result["num_annotations"] = ann_count_by_image.get(result["image_id"], 0)
                images_metadata.append(result)
    
    print(f"   Successfully processed {len(images_metadata)} images")
    
    # Copy bbox results
    print("\n4. Copying bounding box results...")
    copy_bbox_results(images, OUTPUT_DIR)
    
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
        
        scored_path = OUTPUT_DIR / "boxes_scored" / f"{dataset}_{split}_{formatted_id}.json"
        unscored_path = OUTPUT_DIR / "boxes_unscored" / f"{dataset}_{split}_{formatted_id}.json"
        
        img["has_scored_boxes"] = scored_path.exists()
        img["has_unscored_boxes"] = unscored_path.exists()
    
    # Build master index
    print("\n6. Building master index...")
    
    # Build category lookup
    cat_lookup = {cat["id"]: cat["name"] for cat in categories}
    
    index = {
        "total_images": len(images_metadata),
        "scene_tree": scene_tree,
        "images": images_metadata,
        "images_by_scene": dict(images_by_scene),
        "categories": cat_lookup,
    }
    
    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"   Saved index to {index_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images processed: {len(images_metadata)}")
    print(f"Images with scored boxes: {sum(1 for img in images_metadata if img['has_scored_boxes'])}")
    print(f"Images with unscored boxes: {sum(1 for img in images_metadata if img['has_unscored_boxes'])}")
    print(f"Scene categories: {len(images_by_scene)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
