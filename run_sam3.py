import argparse
import csv
import glob
import hashlib
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

import sam3
from matplotlib.colors import to_rgb
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_cxcywh_to_xywh, box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# Default values (used when no CLI arguments provided)
_prompt = ["person", "animal", "character"]
INPUT_FOLDER = "/home/thinz/git/_a2v_eval_data"


import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"


def get_cache_filename(remote_path: str) -> str:
    """
    Generate a stable cache filename from a remote path using hash.

    Args:
        remote_path: Remote path (manifold:// or oil://)

    Returns:
        Hash-based filename
    """
    # Create a hash of the remote path for stable filename
    path_hash = hashlib.sha256(remote_path.encode()).hexdigest()
    return path_hash


def download_from_manifold(manifold_path: str, local_path: str) -> None:
    """
    Download a file from manifold:// path to local filesystem using CLI.

    Args:
        manifold_path: Full manifold:// URL (e.g., manifold://bucket/tree/path/to/file)
        local_path: Local filesystem path where the file should be saved
    """
    try:
        # Use manifold CLI command
        result = subprocess.run(
            ["manifold", "get", manifold_path, local_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to download from manifold {manifold_path}: {e.stderr}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "manifold CLI command not found. Make sure you're running in an environment with manifold access."
        )


def is_local_path(path: str) -> bool:
    """
    Check if a path is a local filesystem path.

    Args:
        path: Path to check

    Returns:
        True if the path is local, False if it's a remote path
    """
    return not path.startswith("manifold://")


def download_file(remote_path: str, local_path: str) -> None:
    """
    Download a file from manifold:// path to local filesystem,
    or copy from local filesystem if it's already local.

    Args:
        remote_path: Full URL (manifold://) or local filesystem path
        local_path: Local filesystem path where the file should be saved
    """
    if is_local_path(remote_path):
        # It's a local file, just copy it to the cache location if different
        if os.path.abspath(remote_path) != os.path.abspath(local_path):
            import shutil

            shutil.copy2(remote_path, local_path)
            print(f"  Copied local file to cache: {remote_path} -> {local_path}")
    elif remote_path.startswith("manifold://"):
        download_from_manifold(remote_path, local_path)
    else:
        raise ValueError(
            f"Unsupported remote path format: {remote_path}. "
            f"Expected manifold:// or a local filesystem path"
        )


def plot_mask(mask, color="r", ax=None):
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)


def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two masks."""
    mask1_binary = mask1.squeeze() > 0.5
    mask2_binary = mask2.squeeze() > 0.5
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_mask_area(mask):
    """Compute the area (number of pixels) in a mask."""
    return (mask.squeeze() > 0.5).sum()


def select_best_masks(
    masks,
    scores,
    keyword_labels=None,
    top_k=2,
    iou_threshold=0.7,
    score_similarity_threshold=0.2,
):
    """
    Select top_k masks considering scores, overlap, and size.

    Args:
        masks: Array of masks
        scores: Array of scores
        keyword_labels: Array of keyword labels for each mask (optional)
        top_k: Number of masks to select
        iou_threshold: If IoU > this threshold, masks are considered to have too much overlap
        score_similarity_threshold: Score difference threshold for considering masks similar

    Returns:
        Indices of selected masks
    """
    if len(masks) <= top_k:
        return np.arange(len(masks))

    # Compute areas for all masks
    areas = np.array([compute_mask_area(mask) for mask in masks])

    # If we have more than 2 masks, filter out masks less than 30% of the biggest mask
    # within each keyword class
    original_indices = np.arange(len(masks))
    if len(masks) > 2 and keyword_labels is not None:
        # Apply size filtering within each keyword class
        mask_filter = np.zeros(len(masks), dtype=bool)

        unique_keywords = np.unique(keyword_labels)
        for keyword in unique_keywords:
            # Get masks for this keyword
            keyword_mask_indices = keyword_labels == keyword
            keyword_areas = areas[keyword_mask_indices]
            keyword_scores = scores[keyword_mask_indices]

            if len(keyword_areas) == 0:
                continue

            # Find the biggest mask for this keyword
            max_area_idx_local = keyword_areas.argmax()
            max_area = keyword_areas[max_area_idx_local]
            top_score = keyword_scores.max()
            biggest_mask_score = keyword_scores[max_area_idx_local]

            # Only filter by size if the biggest mask has a good score
            # (within threshold of the top scored mask for this keyword)
            if abs(biggest_mask_score - top_score) <= score_similarity_threshold:
                # Create a filter for this keyword: keep masks >= 30% of the biggest mask
                keyword_size_filter = keyword_areas >= (0.3 * max_area)

                # Map the local filter back to global indices
                global_keyword_indices = np.where(keyword_mask_indices)[0]
                mask_filter[global_keyword_indices[keyword_size_filter]] = True
            else:
                # If biggest mask doesn't have good score, keep all masks for this keyword
                mask_filter[keyword_mask_indices] = True

        # Apply filter
        masks = masks[mask_filter]
        scores = scores[mask_filter]
        areas = areas[mask_filter]
        if keyword_labels is not None:
            keyword_labels = keyword_labels[mask_filter]
        original_indices = original_indices[mask_filter]

        # Check again if we need to do selection
        if len(masks) <= top_k:
            return original_indices
    elif len(masks) > 2:
        # Fallback to old behavior if no keyword labels provided
        max_area_idx = areas.argmax()
        max_area = areas[max_area_idx]
        top_score = scores.max()
        biggest_mask_score = scores[max_area_idx]

        # Only filter by size if the biggest mask has a good score
        # (within threshold of the top scored mask)
        if abs(biggest_mask_score - top_score) <= score_similarity_threshold:
            mask_filter = areas >= (0.3 * max_area)

            # Apply filter
            masks = masks[mask_filter]
            scores = scores[mask_filter]
            areas = areas[mask_filter]
            original_indices = original_indices[mask_filter]

            # Check again if we need to do selection
            if len(masks) <= top_k:
                return original_indices

    # Start with indices sorted by score (descending)
    sorted_indices = np.argsort(scores)[::-1]

    # Always select the highest scoring mask first
    selected_indices = [sorted_indices[0]]

    # Select remaining masks
    for _ in range(top_k - 1):
        best_idx = None
        best_score_metric = -float("inf")

        for candidate_idx in sorted_indices:
            if candidate_idx in selected_indices:
                continue

            candidate_score = scores[candidate_idx]

            # Check overlap with already selected masks
            max_iou = 0.0
            for selected_idx in selected_indices:
                iou = compute_iou(masks[candidate_idx], masks[selected_idx])
                max_iou = max(max_iou, iou)

            # Skip candidates with too much overlap
            if max_iou > iou_threshold:
                continue

            # Use the candidate's score as the metric
            score_metric = candidate_score

            if score_metric > best_score_metric:
                best_score_metric = score_metric
                best_idx = candidate_idx

        if best_idx is not None:
            selected_indices.append(best_idx)

    # Map back to original indices
    return original_indices[selected_indices]


# Download checkpoint from Manifold if needed
checkpoint_path = f"{sam3_root}/checkpoints/sam3_v4.pt"
if not os.path.exists(checkpoint_path):
    print(f"Downloading checkpoint from Manifold to {checkpoint_path}...")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    import subprocess

    subprocess.run(
        [
            "manifold",
            "get",
            "sam3_ckpt_share/tree/video_ckpts/sam3_v4.pt",
            checkpoint_path,
        ],
        check=True,
    )
    print("Checkpoint downloaded successfully!")

# this step prints a few missing keys and unexpected keys -- this is OK and the model is correctly loaded
model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path)


def process_image(
    image_path, output_folder, processor, prompts, idx=None, mask_cache_dir=None
):
    """
    Process a single image with SAM3.

    Args:
        image_path: Path to the image file
        output_folder: Folder to save output
        processor: SAM3 processor instance
        prompts: List of text prompts for segmentation
        idx: Optional index for progress tracking
        mask_cache_dir: Optional directory to cache the final selected masks
    """
    if idx is not None:
        print(f"\n[{idx}] Processing: {image_path}")
    else:
        print(f"\nProcessing: {image_path}")

    # Load image
    image = Image.open(image_path)
    width, height = image.size

    # Run inference for each keyword separately and collect all results
    all_masks = []
    all_scores = []
    all_boxes = []
    all_keyword_labels = []

    for keyword in prompts:
        print(f"  Running inference for keyword: '{keyword}'")
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=keyword
        )

        # Collect results from this keyword
        if "masks" in inference_state and inference_state["masks"] is not None:
            masks = inference_state["masks"].cpu().float().numpy()
            scores = inference_state["scores"].cpu().float().numpy()
            boxes = inference_state["boxes"].cpu().float().numpy()

            mask_sizes = [compute_mask_area(mask) for mask in masks]
            print(f"    Found {len(masks)} masks for '{keyword}'")
            print(f"    Mask sizes: {mask_sizes}")

            all_masks.append(masks)
            all_scores.append(scores)
            all_boxes.append(boxes)
            # Track which keyword each mask belongs to
            all_keyword_labels.append(np.array([keyword] * len(masks)))

    # Concatenate all results
    keyword_labels = None
    if all_masks:
        all_masks = np.concatenate(all_masks, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_boxes = np.concatenate(all_boxes, axis=0)
        keyword_labels = np.concatenate(all_keyword_labels, axis=0)

        all_mask_sizes = [compute_mask_area(mask) for mask in all_masks]
        print(f"  Total masks from all keywords: {len(all_masks)}")
        print(f"  All scores: {all_scores}")
        print(f"  All mask sizes: {all_mask_sizes}")
        print(f"  Keyword labels: {keyword_labels}")

    # Visualize and save the output image with boxes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    # Plot masks if available
    if all_masks is not None and len(all_masks) > 0:
        masks = all_masks
        scores = all_scores

        # Select top 2 masks considering scores, overlap, and size
        if len(masks) > 2:
            top_indices = select_best_masks(
                masks,
                scores,
                keyword_labels=keyword_labels,
                top_k=2,
                iou_threshold=0.7,
                score_similarity_threshold=0.2,
            )
            masks = masks[top_indices]
            selected_scores = scores[top_indices]
            if keyword_labels is not None:
                selected_keywords = keyword_labels[top_indices]
            else:
                selected_keywords = None
            selected_mask_sizes = [compute_mask_area(mask) for mask in masks]
            print(f"  Selected top 2 masks with scores: {selected_scores}")
            print(f"  Selected mask sizes: {selected_mask_sizes}")
            if selected_keywords is not None:
                print(f"  Selected keywords: {selected_keywords}")

            # Debug info: show IoU between selected masks
            if len(masks) == 2:
                iou = compute_iou(masks[0], masks[1])
                print(f"  IoU between selected masks: {iou:.3f}")
        else:
            mask_sizes = [compute_mask_area(mask) for mask in masks]
            print(f"  Using all {len(masks)} masks")
            print(f"  Mask sizes: {mask_sizes}")

        # Cache the final selected masks if cache directory is provided
        if mask_cache_dir is not None:
            # Generate cache filename based on image path and prompts
            cache_key = f"{image_path}_{'_'.join(prompts)}"
            mask_cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
            mask_cache_path = os.path.join(mask_cache_dir, f"{mask_cache_hash}.npz")

            # Save masks and associated metadata
            np.savez(
                mask_cache_path,
                masks=masks,
                scores=selected_scores if len(all_masks) > 2 else scores,
                image_path=image_path,
                prompts=prompts,
            )
            print(f"  Cached masks to: {mask_cache_path}")

        # Define a list of distinct colors for each mask
        colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "cyan",
            "magenta",
            "orange",
            "purple",
            "pink",
            "lime",
        ]

        for i, mask in enumerate(masks):
            # mask has shape [1, H, W], squeeze to get [H, W]
            mask_2d = mask.squeeze()
            # Use a different color for each mask, cycling through the color list
            color = colors[i % len(colors)]
            plot_mask(mask_2d, color=color, ax=ax)

    # Save to output folder with same filename
    image_filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_filename)
    plt.savefig(output_path, bbox_inches="tight", dpi=150, pad_inches=0.1)
    print(f"  Saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 segmentation on images from folder or CSV"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default=None,
        help="Folder containing images to process (default: uses hardcoded INPUT_FOLDER)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV file with column 'image_manifold_path' containing image paths",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder for results (auto-generated if not specified)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Text prompts for segmentation (default: person, animal, character)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded files from manifold (default: ~/.cache/sam3_downloads)",
    )
    parser.add_argument(
        "--mask-cache-dir",
        type=str,
        default=None,
        help="Directory to cache the final selected masks (default: ~/.cache/sam3_masks)",
    )

    args = parser.parse_args()

    # Determine prompts
    prompts = args.prompts if args.prompts else _prompt

    # Create processor
    processor = Sam3Processor(model, confidence_threshold=0.5)

    if args.csv:
        # CSV mode: process images from CSV
        print(f"Processing images from CSV: {args.csv}")

        # Setup cache directory
        if args.cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "sam3_downloads"
            )
        else:
            cache_dir = args.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using cache directory: {cache_dir}")

        # Setup mask cache directory
        if args.mask_cache_dir is None:
            mask_cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "sam3_masks"
            )
        else:
            mask_cache_dir = args.mask_cache_dir
        os.makedirs(mask_cache_dir, exist_ok=True)
        print(f"Using mask cache directory: {mask_cache_dir}")

        # Read CSV
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if "image_manifold_path" not in rows[0]:
            raise ValueError("CSV must have 'image_manifold_path' column")

        print(f"Found {len(rows)} rows in CSV")

        # Create output folder
        if args.output_folder:
            output_folder = args.output_folder
        else:
            csv_basename = os.path.basename(args.csv).replace(".csv", "")
            output_folder = f"{csv_basename}_sam3_" + "_".join(prompts)
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder: {output_folder}")

        # Process each row
        for idx, row in enumerate(rows):
            image_remote_path = row["image_manifold_path"]

            # Create cache file path
            image_hash = get_cache_filename(image_remote_path)
            image_ext = os.path.splitext(image_remote_path)[1] or ".jpg"
            local_image_path = os.path.join(cache_dir, f"{image_hash}{image_ext}")

            # Download image if not already cached
            if os.path.exists(local_image_path):
                print(f"\n[{idx}/{len(rows)}] Using cached image: {local_image_path}")
            else:
                print(
                    f"\n[{idx}/{len(rows)}] Downloading image from: {image_remote_path}"
                )
                download_file(image_remote_path, local_image_path)

            # Process the image
            process_image(
                local_image_path,
                output_folder,
                processor,
                prompts,
                idx=idx,
                mask_cache_dir=mask_cache_dir,
            )

    else:
        # Folder mode: process images from input folder
        input_folder = args.input_folder if args.input_folder else INPUT_FOLDER
        print(f"Processing images from folder: {input_folder}")

        # Create output folder
        if args.output_folder:
            output_folder = args.output_folder
        else:
            output_folder = input_folder + "_sam3_" + "_".join(prompts)
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder: {output_folder}")

        # Find all images (png and jpg) in the input folder
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

        print(f"Found {len(image_paths)} images in {input_folder}")

        # Setup mask cache directory if needed
        mask_cache_dir = None
        if args.mask_cache_dir is not None:
            mask_cache_dir = args.mask_cache_dir
            os.makedirs(mask_cache_dir, exist_ok=True)
            print(f"Using mask cache directory: {mask_cache_dir}")

        # Process each image
        for idx, image_path in enumerate(image_paths):
            process_image(
                image_path,
                output_folder,
                processor,
                prompts,
                idx=idx,
                mask_cache_dir=mask_cache_dir,
            )

    print(f"\nProcessing complete! All outputs saved to {output_folder}")


if __name__ == "__main__":
    main()


# python /home/thinz/git/sam3/run_sam3.py --csv your_file.csv --cache-dir ~/.cache/sam3_downloads --prompts person animal character
