"""
VGGT Feature Extractor for Geometry Query Distillation
=======================================================
Reads ScanNet posed image data, runs VGGT aggregator, and saves the
intermediate aggregated tokens as feature_3d.npz for offline distillation.

Output format (matches data_qwen.py expectation):
    {FEATURE_DIR}/{scene_id}/feature_3d.npz  →  key='feature', shape=(S, L, D)
        S = number of sampled frames
        L = spatial tokens per frame (e.g. 1036 = 28*37 for 518×518 input)
        D = VGGT embed_dim (1024)

Usage:
    # Single GPU, extract all scenes from training JSONs
    python scripts/extract_vggt_features.py \
        --vggt_path ./VGGT-1B \
        --data_root ./data/media \
        --json_dir ./data/train \
        --output_dir /public_datasets/VG-LLM/features/scannet \
        --max_frames 8 \
        --img_size 518 \
        --device cuda:0

    # Resume (skip already-done scenes)
    python scripts/extract_vggt_features.py ... --skip_existing

    # Multi-GPU parallel: split by rank
    CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/extract_vggt_features.py \
        --device cuda:0 --rank 0 --world_size 4 ...
"""

import argparse
import json
import os
import glob
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vggt_path", type=str, default="./VGGT-1B",
                   help="Path to VGGT checkpoint directory")
    p.add_argument("--data_root", type=str, default="./data/media",
                   help="Root of ScanNet posed images (contains scene*/ dirs)")
    p.add_argument("--json_dir", type=str, default="./data/train",
                   help="Dir containing *_train_*.json annotation files")
    p.add_argument("--output_dir", type=str,
                   default="/public_datasets/VG-LLM/features/scannet",
                   help="Output root: {output_dir}/{scene_id}/feature_3d.npz")
    p.add_argument("--max_frames", type=int, default=8,
                   help="Max frames to sample per scene (must match training config)")
    p.add_argument("--img_size", type=int, default=518,
                   help="VGGT input resolution (must be divisible by patch_size=14)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Number of frames per VGGT forward pass")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip scenes that already have feature_3d.npz")
    p.add_argument("--rank", type=int, default=0,
                   help="For manual data-parallel splitting")
    p.add_argument("--world_size", type=int, default=1,
                   help="Total number of parallel processes")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def collect_scene_ids(json_dir: str, data_root: str) -> list[str]:
    """Parse all training JSON files and collect unique scene IDs."""
    scene_ids = set()
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    print(f"Scanning {len(json_files)} JSON files for scene IDs...")
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        for item in data:
            # Try explicit scene_id field first
            sid = item.get("scene_id", None)
            if sid:
                scene_ids.add(sid)
                continue
            # Fallback: parse from image paths
            img_list = item.get("image", item.get("images", []))
            if not isinstance(img_list, list):
                img_list = [img_list]
            for img_path in img_list:
                if not isinstance(img_path, str):
                    continue
                for part in img_path.replace("\\", "/").split("/"):
                    if part.startswith("scene"):
                        scene_ids.add(part)
                        break

    # Verify scenes actually exist on disk
    valid = []
    for sid in sorted(scene_ids):
        scene_dir = os.path.join(data_root, "scannet", "posed_images", sid)
        if not os.path.exists(scene_dir):
            # Try alternative paths
            alt = os.path.join(data_root, sid)
            if os.path.exists(alt):
                valid.append(sid)
            else:
                # Try glob
                candidates = glob.glob(os.path.join(data_root, "**", sid), recursive=True)
                if candidates:
                    valid.append(sid)
                # else: silently skip missing scenes
        else:
            valid.append(sid)

    print(f"Found {len(scene_ids)} unique scene IDs, {len(valid)} exist on disk.")
    return valid


def find_scene_dir(scene_id: str, data_root: str) -> str | None:
    """Find the directory containing posed images for a scene."""
    candidates = [
        os.path.join(data_root, "scannet", "posed_images", scene_id),
        os.path.join(data_root, scene_id),
        os.path.join(data_root, "scannet", scene_id),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Glob fallback
    results = glob.glob(os.path.join(data_root, "**", scene_id), recursive=True)
    if results:
        return results[0]
    return None


def sample_frames(scene_dir: str, max_frames: int) -> list[str]:
    """Uniformly sample up to max_frames JPG files from a scene directory."""
    jpg_files = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
    if not jpg_files:
        jpg_files = sorted(glob.glob(os.path.join(scene_dir, "**", "*.jpg"), recursive=True))
    if not jpg_files:
        return []
    if len(jpg_files) <= max_frames:
        return jpg_files
    indices = np.linspace(0, len(jpg_files) - 1, max_frames, dtype=int)
    return [jpg_files[i] for i in indices]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def load_and_resize_image(path: str, img_size: int) -> torch.Tensor:
    """Load a JPEG, resize to (img_size, img_size), normalize to [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return tensor


# ---------------------------------------------------------------------------
# VGGT loading (aggregator only — no task heads needed)
# ---------------------------------------------------------------------------

def load_vggt_aggregator(vggt_path: str, device: torch.device):
    """Load VGGT and return just the aggregator to avoid running task heads."""
    # Add project src to path so imports work
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from qwen_vl.model.vggt.models.vggt import VGGT

    print(f"Loading VGGT from {vggt_path} ...")
    model = VGGT.from_pretrained(vggt_path)
    model = model.to(device).eval()

    # Disable task heads to save memory and time
    model.camera_head = None
    model.point_head = None
    model.depth_head = None
    model.track_head = None

    print("VGGT loaded (task heads disabled for feature extraction).")
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model,
    frame_paths: list[str],
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run VGGT aggregator on all frames and return the aggregated tokens.

    Returns:
        features: np.ndarray of shape (S, L, D)
            S = number of frames
            L = number of spatial tokens per frame
            D = VGGT embed_dim (1024)
    """
    # Load all frames into a (S, 3, H, W) tensor
    frames = torch.stack([load_and_resize_image(p, img_size) for p in frame_paths])  # (S,3,H,W)

    # VGGT aggregator expects (B, S, 3, H, W) — process all frames as one sequence
    # to preserve cross-frame attention (critical for 3D understanding)
    S = frames.shape[0]
    images = frames.unsqueeze(0).to(device, dtype=torch.bfloat16)  # (1, S, 3, H, W)

    # Run aggregator
    aggregated_tokens_list, patch_start_idx = model.aggregator(images)

    # aggregated_tokens_list[-1]: shape (1, S, L_total, D)
    # The aggregated tokens include both frame tokens and global tokens.
    # patch_start_idx tells us where patch (spatial) tokens start.
    tokens = aggregated_tokens_list[-1]  # (1, S, L_total, D) — use last layer
    B, Sf, L_total, D = tokens.shape

    # Extract only patch tokens (excluding camera/global tokens)
    patch_tokens = tokens[:, :, patch_start_idx:, :]  # (1, S, L_patch, D)

    # Convert to (S, L_patch, D) and return as float32 numpy
    feature = patch_tokens.squeeze(0).float().cpu().numpy()  # (S, L, D)
    return feature


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all scene IDs
    scene_ids = collect_scene_ids(args.json_dir, args.data_root)

    # Shard by rank for manual parallelism
    if args.world_size > 1:
        scene_ids = [s for i, s in enumerate(scene_ids) if i % args.world_size == args.rank]
        print(f"[Rank {args.rank}/{args.world_size}] Processing {len(scene_ids)} scenes")

    # Load VGGT (aggregator only)
    model = load_vggt_aggregator(args.vggt_path, device)

    # Process each scene
    success, skipped, failed = 0, 0, 0
    for scene_id in tqdm(scene_ids, desc="Extracting VGGT features"):
        out_path = os.path.join(args.output_dir, scene_id, "feature_3d.npz")

        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        # Find scene directory
        scene_dir = find_scene_dir(scene_id, args.data_root)
        if scene_dir is None:
            tqdm.write(f"[SKIP] {scene_id}: directory not found")
            failed += 1
            continue

        # Sample frames uniformly
        frame_paths = sample_frames(scene_dir, args.max_frames)
        if not frame_paths:
            tqdm.write(f"[SKIP] {scene_id}: no JPG frames found in {scene_dir}")
            failed += 1
            continue

        try:
            feature = extract_features(
                model, frame_paths,
                img_size=args.img_size,
                batch_size=args.batch_size,
                device=device,
            )

            # Save
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(out_path, feature=feature)
            success += 1

        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"[OOM] {scene_id}: reduce --max_frames or --batch_size")
            torch.cuda.empty_cache()
            failed += 1
        except Exception as e:
            tqdm.write(f"[ERROR] {scene_id}: {e}")
            failed += 1

    print(f"\nDone. success={success}, skipped={skipped}, failed={failed}")
    print(f"Features saved to: {args.output_dir}/{{scene_id}}/feature_3d.npz")


if __name__ == "__main__":
    main()
