"""
VGGT Feature Extractor for 3DRS-style Distillation
====================================================
Based on the official 3DRS extract_vggt_feature.py:
  https://github.com/Visual-AI/3DRS/blob/main/extract_vggt_feature.py

Reads ScanNet posed image data, runs VGGT aggregator, and saves the
aggregated tokens as vggt.npz for offline distillation.

Output format (matches data_3d.py `get_3d_features` expectation):
    {output_dir}/{scene_id}/vggt.npz
        key='feature':  shape (1, S, L_total, D)  — full aggregated tokens
        key='ps_idx':   int — patch_start_idx, used to slice out patch tokens

    data_3d.py loads as:
        data = np.load(path)
        start_idx = data['ps_idx']
        feature = data['feature'][:, :, start_idx:, :]   # → (1, S, L_patch, D)

Usage:
    # Single GPU
    python scripts/extract_vggt_features.py \\
        --vggt_path ./VGGT-1B \\
        --data_root data/scannet/posed_images \\
        --output_dir data/scannet/posed_images_3d_feature_vggt \\
        --num_frames 32 \\
        --device cuda:0

    # Resume (skip already-done scenes)
    python scripts/extract_vggt_features.py ... --skip_existing
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract VGGT features for 3DRS distillation")
    p.add_argument("--vggt_path", type=str, default="./VGGT-1B",
                   help="Path to VGGT checkpoint (model.pt or HF directory)")
    p.add_argument("--data_root", type=str, default="data/scannet/posed_images",
                   help="Root of ScanNet posed images (contains scene*/ dirs)")
    p.add_argument("--output_dir", type=str,
                   default="data/scannet/posed_images_3d_feature_vggt",
                   help="Output root: {output_dir}/{scene_id}/vggt.npz")
    p.add_argument("--num_frames", type=int, default=32,
                   help="Number of frames to uniformly sample per scene (3DRS default: 32)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip scenes that already have vggt.npz")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main — closely follows official 3DRS extract_vggt_feature.py
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device and dtype setup (from 3DRS official)
    device = args.device if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        dev_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        dtype = torch.bfloat16 if dev_capability[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # ---------------------------------------------------------------------------
    # Load VGGT model (support both model.pt and HF-style loading)
    # ---------------------------------------------------------------------------
    # Add project src to path so VGGT imports work
    project_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, os.path.join(project_root, "src"))

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    model_pt = os.path.join(args.vggt_path, "model.pt")
    if os.path.exists(model_pt):
        # 3DRS official loading: VGGT() + torch.load
        print(f"Loading VGGT from {model_pt} (3DRS official method) ...")
        model = VGGT()
        checkpoint = torch.load(model_pt, map_location=device)
        msg = model.load_state_dict(checkpoint)
        print(f"Loading status: {msg}")
    else:
        # HuggingFace-style loading
        print(f"Loading VGGT from {args.vggt_path} (HF from_pretrained) ...")
        model = VGGT.from_pretrained(args.vggt_path)

    model = model.to(device).eval()

    # ---------------------------------------------------------------------------
    # Iterate over each scene directory
    # ---------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    scene_list = sorted([
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ])
    print(f"Found {len(scene_list)} scene directories in {args.data_root}")

    num_frames_to_sample = args.num_frames
    success, skipped, failed = 0, 0, 0

    for scene in tqdm(scene_list, desc="Extracting VGGT features"):
        scene_dir = os.path.join(args.data_root, scene)

        # Output path
        scene_save_dir = os.path.join(args.output_dir, scene)
        save_path = os.path.join(scene_save_dir, "vggt.npz")

        if args.skip_existing and os.path.exists(save_path):
            skipped += 1
            continue

        # Get all jpg images, sort by filename, then uniformly sample
        file_names = sorted([f for f in os.listdir(scene_dir) if f.endswith('.jpg')])
        total_frames = len(file_names)
        if total_frames == 0:
            tqdm.write(f"[SKIP] {scene}: no JPG frames found")
            failed += 1
            continue

        sampled_indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)
        sampled_file_list = [os.path.join(scene_dir, file_names[i]) for i in sampled_indices]

        try:
            # Load and preprocess images using VGGT's official utility
            # Returns shape (N, 3, H, W)
            images = load_and_preprocess_images(sampled_file_list)
            images = images.to(device, non_blocking=True)
            images = images.to(dtype)
            # Add batch dimension → (1, num_frames, 3, H, W)
            images = images.unsqueeze(0)

            # Model inference — extract aggregated tokens
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                    aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Save aggregated_tokens_list[-1] and ps_idx
            # This is the EXACT same format as 3DRS official
            feature = aggregated_tokens_list[-1].cpu().numpy()
            ps_idx_np = ps_idx.cpu().numpy() if isinstance(ps_idx, torch.Tensor) else ps_idx

            os.makedirs(scene_save_dir, exist_ok=True)
            np.savez_compressed(save_path, feature=feature, ps_idx=ps_idx_np)
            success += 1

        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"[OOM] {scene}: reduce --num_frames (current: {num_frames_to_sample})")
            torch.cuda.empty_cache()
            failed += 1
        except Exception as e:
            tqdm.write(f"[ERROR] {scene}: {e}")
            failed += 1

    print(f"\nDone. success={success}, skipped={skipped}, failed={failed}")
    print(f"Features saved to: {args.output_dir}/{{scene_id}}/vggt.npz")


if __name__ == "__main__":
    main()
