#!/usr/bin/env python3
"""Upload data/ to HuggingFace in separate commits per subdirectory."""

import os
from huggingface_hub import HfApi

REPO_ID = "Silicon23/WildDet3D-demo"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

api = HfApi(token=TOKEN)

# Upload order: smallest first
uploads = [
    ("index.json", False),       # single file
    ("boxes", True),             # 36M
    ("camera", True),            # 25M
    ("images_annotated", True),  # 2.1G
    ("images", True),            # 9.9G
    ("pointclouds", True),       # 16G
]

for name, is_dir in uploads:
    src = os.path.join(DATA_DIR, name)
    dest = f"data/{name}"

    if is_dir:
        print(f"\nUploading directory: {name} -> {dest}")
        api.upload_folder(
            folder_path=src,
            path_in_repo=dest,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add data/{name}",
        )
    else:
        print(f"\nUploading file: {name} -> {dest}")
        api.upload_file(
            path_or_fileobj=src,
            path_in_repo=dest,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add data/{name}",
        )
    print(f"  Done: {name}")

print("\nAll uploads complete!")
