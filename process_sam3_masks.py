import argparse
import csv
import hashlib
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize


def get_cache_filename(remote_path: str) -> str:
    """
    Generate a stable cache filename from a remote path using hash.
    Matches the function in run_sam3.py.

    Args:
        remote_path: Remote path (manifold:// or oil://)

    Returns:
        Hash-based filename
    """
    path_hash = hashlib.sha256(remote_path.encode()).hexdigest()
    return path_hash


def get_mask_cache_filename(image_path: str, prompts: List[str]) -> str:
    """
    Generate the cache filename for a given image path and prompts.
    This matches the logic in run_sam3.py.

    Args:
        image_path: Path to the image file (local cached path)
        prompts: List of text prompts used for segmentation

    Returns:
        Hash-based filename (without extension)
    """
    cache_key = f"{image_path}_{'_'.join(prompts)}"
    mask_cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    return mask_cache_hash


def load_cached_masks(
    mask_cache_path: str,
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """
    Load cached masks from a .npz file.

    Args:
        mask_cache_path: Path to the cached mask file

    Returns:
        Tuple of (masks, scores, image_path, prompts)
    """
    if not os.path.exists(mask_cache_path):
        raise FileNotFoundError(f"Mask cache file not found: {mask_cache_path}")

    data = np.load(mask_cache_path, allow_pickle=True)
    masks = data["masks"]
    scores = data["scores"]
    image_path = str(data["image_path"])
    prompts = list(data["prompts"])

    return masks, scores, image_path, prompts


def plot_mask(mask, color="r", ax=None):
    """
    Plot a mask overlay on the current axes.

    Args:
        mask: 2D binary mask [H, W]
        color: Color for the mask overlay
        ax: Matplotlib axes to plot on (uses current axes if None)
    """
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Remove small disconnected areas from a mask, keeping components that are
    at least 10% the size of the largest component.

    Args:
        mask: Binary mask array [H, W] or [1, H, W]

    Returns:
        Cleaned mask with disconnected parts smaller than 10% of the largest component removed
    """
    # Squeeze to 2D if needed
    mask_2d = mask.squeeze()

    # Convert to binary (threshold at 0.5)
    binary_mask = (mask_2d > 0.5).astype(np.uint8)

    # Label connected components
    labeled_mask, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        # No components found, return empty mask
        return np.zeros_like(mask)

    # Find the largest component
    component_sizes = np.bincount(labeled_mask.ravel())
    # Skip background (label 0)
    component_sizes[0] = 0
    largest_component_size = component_sizes.max()

    # Calculate 10% threshold
    size_threshold = largest_component_size * 0.1

    # Create mask with all components that are >= 10% of the largest component
    cleaned_mask = np.zeros_like(mask_2d, dtype=np.float32)
    for label in range(1, num_features + 1):
        if component_sizes[label] >= size_threshold:
            cleaned_mask = np.logical_or(cleaned_mask, labeled_mask == label)

    cleaned_mask = cleaned_mask.astype(np.float32)

    # Restore original shape
    if len(mask.shape) == 3:
        cleaned_mask = cleaned_mask[np.newaxis, ...]

    return cleaned_mask


def find_separating_line(
    mask1: np.ndarray,
    mask2: np.ndarray,
    use_border_only: bool = True,
    max_border_pixels: int = 2000,
) -> Tuple[float, float, float, Dict]:
    """
    Find the optimal straight line that separates two masks.

    Optimization priorities:
    1. Minimize mask pixels on the wrong side of the line (misclassification)
    2. Minimize line intersection with masks (maximize distance from line to masks)

    Line is represented in parametric form: x*cos(theta) + y*sin(theta) = rho
    where theta is the angle (0 to pi) and rho is the distance from origin.

    Args:
        mask1: First binary mask [H, W] or [1, H, W]
        mask2: Second binary mask [H, W] or [1, H, W]
        use_border_only: If True, only use border pixels for distance calculations (much faster)
        max_border_pixels: Maximum number of border pixels to use (subsample if more)

    Returns:
        Tuple of (theta, rho, score, metrics_dict) where:
            - theta: angle of the line in radians (0 to pi)
            - rho: distance from origin
            - score: optimization score (lower is better - represents total cost)
            - metrics_dict: dictionary with detailed metrics
    """
    # Squeeze masks to 2D
    mask1_2d = (mask1.squeeze() > 0.5).astype(np.uint8)
    mask2_2d = (mask2.squeeze() > 0.5).astype(np.uint8)

    h, w = mask1_2d.shape

    # Get coordinates of ALL mask pixels (for final metrics)
    y1_all, x1_all = np.where(mask1_2d > 0)
    y2_all, x2_all = np.where(mask2_2d > 0)

    if len(x1_all) == 0 or len(x2_all) == 0:
        print("Warning: One or both masks are empty")
        return 0.0, 0.0, 0.0, {}

    # Compute centroids
    centroid1 = np.array([x1_all.mean(), y1_all.mean()])
    centroid2 = np.array([x2_all.mean(), y2_all.mean()])

    # Total number of mask pixels (for final metrics)
    total_pixels1 = len(x1_all)
    total_pixels2 = len(x2_all)

    # Extract border pixels for optimization (much faster)
    if use_border_only:
        # Get border pixels using morphological erosion
        # Border = original mask - eroded mask
        kernel = np.ones((3, 3), np.uint8)
        border1 = mask1_2d - cv2.erode(mask1_2d, kernel, iterations=1)
        border2 = mask2_2d - cv2.erode(mask2_2d, kernel, iterations=1)

        y1, x1 = np.where(border1 > 0)
        y2, x2 = np.where(border2 > 0)

        # Subsample if too many border pixels
        if len(x1) > max_border_pixels:
            indices = np.random.choice(len(x1), max_border_pixels, replace=False)
            x1, y1 = x1[indices], y1[indices]
        if len(x2) > max_border_pixels:
            indices = np.random.choice(len(x2), max_border_pixels, replace=False)
            x2, y2 = x2[indices], y2[indices]

        print(
            f"  Using border pixels: mask1={len(x1)}/{total_pixels1}, mask2={len(x2)}/{total_pixels2}"
        )
    else:
        # Use all pixels (slower but more accurate)
        x1, y1 = x1_all, y1_all
        x2, y2 = x2_all, y2_all
        print(f"  Using all pixels: mask1={len(x1)}, mask2={len(x2)}")

    def evaluate_line(params):
        """
        Evaluate the quality of a separating line.

        Returns cost (for minimization):
        1. Primary: number of misclassified pixels (pixels on wrong side)
        2. Secondary: negative minimum distance to masks (to maximize distance)

        The cost is structured as: misclassified_pixels + 0.001 * (1.0 / min_distance)
        This ensures misclassification is the dominant factor.
        """
        theta, rho = params

        # Line equation: x*cos(theta) + y*sin(theta) = rho
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Determine which side each centroid is on
        # Distance is signed: positive on one side, negative on the other
        dist_centroid1 = centroid1[0] * cos_theta + centroid1[1] * sin_theta - rho
        dist_centroid2 = centroid2[0] * cos_theta + centroid2[1] * sin_theta - rho

        # If centroids are on the same side, this is a very bad separation
        if dist_centroid1 * dist_centroid2 > 0:
            return 1e10  # Large penalty

        # Calculate signed distances for all pixels
        distances1 = x1 * cos_theta + y1 * sin_theta - rho
        distances2 = x2 * cos_theta + y2 * sin_theta - rho

        # Count misclassified pixels (pixels on the wrong side)
        # Mask1 should be on the same side as centroid1
        # Mask2 should be on the same side as centroid2
        if dist_centroid1 > 0:
            # mask1 should be on positive side, mask2 on negative side
            misclassified1 = np.sum(distances1 <= 0)
            misclassified2 = np.sum(distances2 >= 0)
            # Get pixels on correct side for distance calculation
            correct_distances1 = distances1[distances1 > 0]
            correct_distances2 = np.abs(distances2[distances2 < 0])
        else:
            # mask1 should be on negative side, mask2 on positive side
            misclassified1 = np.sum(distances1 >= 0)
            misclassified2 = np.sum(distances2 <= 0)
            # Get pixels on correct side for distance calculation
            correct_distances1 = np.abs(distances1[distances1 < 0])
            correct_distances2 = distances2[distances2 > 0]

        # Total misclassified pixels (PRIMARY CONSTRAINT)
        total_misclassified = misclassified1 + misclassified2

        # Minimum distance to masks on correct side (SECONDARY CONSTRAINT)
        # We want to maximize this, so we minimize 1/distance
        if len(correct_distances1) > 0 and len(correct_distances2) > 0:
            min_dist = min(np.min(correct_distances1), np.min(correct_distances2))
            # Add a small epsilon to avoid division by zero
            distance_cost = 1.0 / (min_dist + 0.1)
        else:
            # All pixels are misclassified - very bad
            distance_cost = 1e6

        # Combined cost: misclassification dominates, distance is secondary
        # Scale factor of 0.001 ensures misclassification is ~1000x more important
        cost = total_misclassified + 0.001 * distance_cost

        return cost

    # OPTIMIZED: Vectorized grid search for good initialization
    best_params = None
    best_cost = float("inf")

    # Try different angles (reduced from 72 to 36 for 2x speedup with minimal accuracy loss)
    num_angles = 36  # Every 5 degrees - good balance of speed and accuracy
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    # Precompute trigonometric values for all angles
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Precompute centroid projections for all angles
    # Shape: (num_angles,)
    centroid1_proj = centroid1[0] * cos_angles + centroid1[1] * sin_angles
    centroid2_proj = centroid2[0] * cos_angles + centroid2[1] * sin_angles

    # For each angle, compute rho range
    rho_mins = np.minimum(centroid1_proj, centroid2_proj)
    rho_maxs = np.maximum(centroid1_proj, centroid2_proj)

    # Try several rho values (reduced from 20 to 10 for 2x speedup)
    num_rho = 10

    # Vectorized evaluation for all angle-rho combinations
    for i, theta in enumerate(angles):
        # Generate rho values for this angle
        rhos = np.linspace(rho_mins[i], rho_maxs[i], num_rho)

        # Vectorized computation for all rho values at once
        cos_theta = cos_angles[i]
        sin_theta = sin_angles[i]

        # Compute centroid distances (scalar for this theta)
        dist_centroid1 = centroid1_proj[i] - rhos  # Shape: (num_rho,)
        dist_centroid2 = centroid2_proj[i] - rhos  # Shape: (num_rho,)

        # Filter out cases where centroids are on same side
        valid_mask = dist_centroid1 * dist_centroid2 < 0

        if not np.any(valid_mask):
            continue

        # Compute distances for all pixels (broadcast against rhos)
        # distances shape: (num_pixels, num_rho)
        distances1 = (
            x1[:, np.newaxis] * cos_theta + y1[:, np.newaxis] * sin_theta - rhos
        )
        distances2 = (
            x2[:, np.newaxis] * cos_theta + y2[:, np.newaxis] * sin_theta - rhos
        )

        # Vectorized misclassification counting for all valid rhos
        for j, rho in enumerate(rhos):
            if not valid_mask[j]:
                continue

            dist1 = distances1[:, j]
            dist2 = distances2[:, j]

            if dist_centroid1[j] > 0:
                misclassified1 = np.sum(dist1 <= 0)
                misclassified2 = np.sum(dist2 >= 0)
                correct_dist1 = dist1[dist1 > 0]
                correct_dist2 = np.abs(dist2[dist2 < 0])
            else:
                misclassified1 = np.sum(dist1 >= 0)
                misclassified2 = np.sum(dist2 <= 0)
                correct_dist1 = np.abs(dist1[dist1 < 0])
                correct_dist2 = dist2[dist2 > 0]

            total_misclassified = misclassified1 + misclassified2

            if len(correct_dist1) > 0 and len(correct_dist2) > 0:
                min_dist = min(np.min(correct_dist1), np.min(correct_dist2))
                distance_cost = 1.0 / (min_dist + 0.1)
            else:
                distance_cost = 1e6

            cost = total_misclassified + 0.001 * distance_cost

            if cost < best_cost:
                best_cost = cost
                best_params = [theta, rho]

    if best_params is None:
        print("Warning: Could not find initial separating line")
        return 0.0, 0.0, 0.0, {}

    # Refine with local optimization
    result = minimize(
        evaluate_line,
        best_params,
        method="Nelder-Mead",
        options={"xatol": 0.1, "fatol": 0.001, "maxiter": 500},
    )

    theta_opt, rho_opt = result.x
    cost_opt = result.fun

    # Normalize theta to [0, pi)
    theta_opt = theta_opt % np.pi

    # Calculate final metrics for reporting
    cos_theta = np.cos(theta_opt)
    sin_theta = np.sin(theta_opt)

    dist_centroid1 = centroid1[0] * cos_theta + centroid1[1] * sin_theta - rho_opt
    dist_centroid2 = centroid2[0] * cos_theta + centroid2[1] * sin_theta - rho_opt

    distances1 = x1 * cos_theta + y1 * sin_theta - rho_opt
    distances2 = x2 * cos_theta + y2 * sin_theta - rho_opt

    if dist_centroid1 > 0:
        misclassified1 = np.sum(distances1 <= 0)
        misclassified2 = np.sum(distances2 >= 0)
        correct_distances1 = distances1[distances1 > 0]
        correct_distances2 = np.abs(distances2[distances2 < 0])
    else:
        misclassified1 = np.sum(distances1 >= 0)
        misclassified2 = np.sum(distances2 <= 0)
        correct_distances1 = np.abs(distances1[distances1 < 0])
        correct_distances2 = distances2[distances2 > 0]

    total_misclassified = misclassified1 + misclassified2
    min_dist = (
        min(np.min(correct_distances1), np.min(correct_distances2))
        if len(correct_distances1) > 0 and len(correct_distances2) > 0
        else 0
    )

    metrics = {
        "misclassified_pixels": int(total_misclassified),
        "misclassified_mask1": int(misclassified1),
        "misclassified_mask2": int(misclassified2),
        "misclassified_percent": float(
            total_misclassified * 100.0 / (total_pixels1 + total_pixels2)
        ),
        "min_distance_to_masks": float(min_dist),
        "total_cost": float(cost_opt),
    }

    print(
        f"  Optimal separating line: theta={np.degrees(theta_opt):.2f}°, rho={rho_opt:.2f}"
    )
    print(
        f"    Misclassified pixels: {total_misclassified} ({metrics['misclassified_percent']:.2f}%)"
    )
    print(f"    Min distance to masks: {min_dist:.2f} pixels")

    return theta_opt, rho_opt, cost_opt, metrics


def draw_line_on_image(
    image: np.ndarray, theta: float, rho: float, color=(255, 0, 0), thickness: int = 2
) -> np.ndarray:
    """
    Draw a line on an image given parametric line parameters.

    Args:
        image: Image array [H, W, C] or [H, W]
        theta: Angle of the line in radians
        rho: Distance from origin
        color: RGB color tuple
        thickness: Line thickness in pixels

    Returns:
        Image with line drawn
    """
    import cv2

    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image_copy = (image * 255).astype(np.uint8)
    else:
        image_copy = image.copy()

    # Ensure 3 channels
    if len(image_copy.shape) == 2:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)

    h, w = image_copy.shape[:2]

    # Convert parametric form to two points
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Find intersection with image boundaries
    # Line equation: x*cos(theta) + y*sin(theta) = rho

    points = []

    # Check intersection with top edge (y=0)
    if abs(sin_theta) > 1e-6:
        x = rho / cos_theta
        if 0 <= x <= w:
            points.append((int(x), 0))

    # Check intersection with bottom edge (y=h)
    if abs(sin_theta) > 1e-6:
        x = (rho - h * sin_theta) / cos_theta
        if 0 <= x <= w:
            points.append((int(x), h - 1))

    # Check intersection with left edge (x=0)
    if abs(cos_theta) > 1e-6:
        y = rho / sin_theta
        if 0 <= y <= h:
            points.append((0, int(y)))

    # Check intersection with right edge (x=w)
    if abs(cos_theta) > 1e-6:
        y = (rho - w * cos_theta) / sin_theta
        if 0 <= y <= h:
            points.append((w - 1, int(y)))

    # Draw line if we have at least 2 intersection points
    if len(points) >= 2:
        cv2.line(image_copy, points[0], points[1], color, thickness)

    return image_copy


def visualize_masks(
    image: Image.Image,
    original_masks: np.ndarray,
    cleaned_masks: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    show_comparison: bool = True,
    separating_line: dict = None,
):
    """
    Visualize masks and save to file.

    Args:
        image: PIL Image object
        original_masks: Array of original masks [N, 1, H, W] (can be empty)
        cleaned_masks: Array of cleaned masks [N, 1, H, W] (can be empty)
        scores: Array of scores [N] (can be empty)
        output_path: Path to save the visualization
        show_comparison: If True, show both original and cleaned masks side by side
        separating_line: Optional dict with keys 'theta' and 'rho' for drawing separating line
    """
    # Define colors for masks
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

    # Check if we have masks
    has_masks = len(original_masks) > 0

    if show_comparison and has_masks:
        # Create side-by-side comparison
        num_cols = 3 if separating_line else 2
        fig, axes = plt.subplots(1, num_cols, figsize=(10 * num_cols, 8))

        # Original masks
        axes[0].imshow(image)
        axes[0].set_title("Original Masks (with disconnected parts)", fontsize=14)
        axes[0].axis("off")
        for i, mask in enumerate(original_masks):
            mask_2d = mask.squeeze()
            color = colors[i % len(colors)]
            plot_mask(mask_2d, color=color, ax=axes[0])

        # Cleaned masks
        axes[1].imshow(image)
        axes[1].set_title("Cleaned Masks (largest component only)", fontsize=14)
        axes[1].axis("off")
        for i, mask in enumerate(cleaned_masks):
            mask_2d = mask.squeeze()
            color = colors[i % len(colors)]
            plot_mask(mask_2d, color=color, ax=axes[1])

        # Add separating line visualization if provided
        if separating_line:
            img_with_line = draw_line_on_image(
                np.array(image),
                separating_line["theta"],
                separating_line["rho"],
                color=(0, 255, 0),
                thickness=3,
            )
            axes[2].imshow(img_with_line)

            # Build title with metrics
            title = f"Separating Line\n(angle={separating_line.get('theta_degrees', 0):.1f}°"
            if separating_line.get("is_default", False):
                title += ", default)"
            else:
                metrics = separating_line.get("metrics", {})
                misclassified_pct = metrics.get("misclassified_percent", 0)
                title += f", misclass={misclassified_pct:.1f}%)"

            axes[2].set_title(title, fontsize=14)
            axes[2].axis("off")
            for i, mask in enumerate(cleaned_masks):
                mask_2d = mask.squeeze()
                color = colors[i % len(colors)]
                plot_mask(mask_2d, color=color, ax=axes[2])

        plt.tight_layout()
    else:
        # Show only cleaned masks (and optionally separating line)
        # Or show only separating line if no masks
        num_cols = 2 if (separating_line and has_masks) else 1
        fig, axes = plt.subplots(1, num_cols, figsize=(12 * num_cols, 8))

        if num_cols == 1:
            axes = [axes]  # Make it iterable

        if separating_line and not has_masks:
            # Only show separating line (no masks)
            img_with_line = draw_line_on_image(
                np.array(image),
                separating_line["theta"],
                separating_line["rho"],
                color=(0, 255, 0),
                thickness=3,
            )
            axes[0].imshow(img_with_line)

            title = f"Default Separating Line\n(angle={separating_line.get('theta_degrees', 0):.1f}°, vertical line in middle)"
            axes[0].set_title(title, fontsize=14)
            axes[0].axis("off")
        else:
            # Show cleaned masks
            axes[0].imshow(image)
            axes[0].set_title("Cleaned Masks (largest component only)", fontsize=14)
            axes[0].axis("off")

            for i, mask in enumerate(cleaned_masks):
                mask_2d = mask.squeeze()
                color = colors[i % len(colors)]
                plot_mask(mask_2d, color=color, ax=axes[0])

            # Add separating line visualization if provided
            if separating_line:
                img_with_line = draw_line_on_image(
                    np.array(image),
                    separating_line["theta"],
                    separating_line["rho"],
                    color=(0, 255, 0),
                    thickness=3,
                )
                axes[1].imshow(img_with_line)

                # Build title with metrics
                title = f"Separating Line\n(angle={separating_line.get('theta_degrees', 0):.1f}°"
                if separating_line.get("is_default", False):
                    title += ", default)"
                else:
                    metrics = separating_line.get("metrics", {})
                    misclassified_pct = metrics.get("misclassified_percent", 0)
                    title += f", misclass={misclassified_pct:.1f}%)"

                axes[1].set_title(title, fontsize=14)
                axes[1].axis("off")
                for i, mask in enumerate(cleaned_masks):
                    mask_2d = mask.squeeze()
                    color = colors[i % len(colors)]
                    plot_mask(mask_2d, color=color, ax=axes[1])

        plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight", dpi=150, pad_inches=0.1)
    plt.close()


def process_masks_for_image(
    image_path: str,
    masks: np.ndarray = None,
    scores: np.ndarray = None,
    prompts: List[str] = None,
    local_image_path: str = None,
    output_folder: str = None,
    show_comparison: bool = True,
) -> Dict:
    """
    Process loaded masks for further analysis.
    The cached masks already contain only the top 2 masks selected by run_sam3.py.
    This function cleans them to keep only the largest connected component in each.

    If no masks are provided, creates a default vertical separating line in the middle.

    Args:
        image_path: Original image path (remote path)
        masks: Array of masks [N, 1, H, W] - already filtered to top 2 (None if no masks)
        scores: Array of scores [N] (None if no masks)
        prompts: List of prompts used
        local_image_path: Path to the local cached image file (optional, for visualization)
        output_folder: Folder to save visualizations (optional)
        show_comparison: If True, show both original and cleaned masks side by side

    Returns:
        Dictionary with processed mask information
    """
    # Handle case when no masks are provided
    if masks is None or len(masks) == 0:
        print("  No masks available, using default vertical line in middle")

        # Get image dimensions if possible
        image_width, image_height = None, None
        if local_image_path and os.path.exists(local_image_path):
            try:
                image = Image.open(local_image_path)
                image_width, image_height = image.size
            except Exception as e:
                print(f"  Warning: Could not load image for dimensions: {e}")

        # Default to common dimensions if image not available
        if image_width is None:
            image_width, image_height = 1024, 1024
            print(f"  Using default dimensions: {image_width}x{image_height}")

        # Create default vertical line in the middle
        # theta = π/2 (90 degrees, vertical line)
        # rho = width/2 (middle of the image)
        theta_default = np.pi / 2
        rho_default = image_width / 2

        result = {
            "image_path": image_path,
            "prompts": prompts or [],
            "num_masks": 0,
            "original_masks": np.array([]),
            "original_scores": np.array([]),
            "cleaned_masks": np.array([]),
            "cleaned_scores": np.array([]),
            "cleaned_mask_shapes": [],
            "cleaned_mask_areas": [],
            "separating_line": {
                "theta": theta_default,
                "rho": rho_default,
                "score": 0.0,
                "theta_degrees": np.degrees(theta_default),
                "is_default": True,
            },
        }

        print(
            f"  Default separating line: theta={np.degrees(theta_default):.2f}°, rho={rho_default:.2f} (vertical line in middle)"
        )
        return result

    # Clean each mask to keep only largest connected component
    cleaned_masks = []
    for i, mask in enumerate(masks):
        original_area = (mask.squeeze() > 0.5).sum()
        cleaned_mask = keep_largest_connected_component(mask)
        cleaned_area = (cleaned_mask.squeeze() > 0.5).sum()

        cleaned_masks.append(cleaned_mask)

        print(
            f"  Mask {i+1}: score={scores[i]:.4f}, "
            f"original_area={original_area}, "
            f"cleaned_area={cleaned_area}, "
            f"removed={original_area - cleaned_area} pixels"
        )

    cleaned_masks = np.array(cleaned_masks)

    result = {
        "image_path": image_path,
        "prompts": prompts,
        "num_masks": len(masks),
        "original_masks": masks,
        "original_scores": scores,
        "cleaned_masks": cleaned_masks,
        "cleaned_scores": scores,
        "cleaned_mask_shapes": [mask.shape for mask in cleaned_masks],
        "cleaned_mask_areas": [(mask.squeeze() > 0.5).sum() for mask in cleaned_masks],
    }

    # Find separating line if we have exactly 2 masks
    if len(cleaned_masks) == 2:
        print("  Finding optimal separating line...")
        theta, rho, cost, metrics = find_separating_line(
            cleaned_masks[0], cleaned_masks[1]
        )
        result["separating_line"] = {
            "theta": theta,
            "rho": rho,
            "cost": cost,
            "theta_degrees": np.degrees(theta),
            "metrics": metrics,
        }

    print(f"  Image: {image_path}")
    print(
        f"  Number of masks: {result['num_masks']} (already filtered to top 2 in run_sam3.py)"
    )
    print(f"  All masks cleaned to keep only largest connected component")

    # Visualize if requested
    if output_folder is not None and local_image_path is not None:
        if os.path.exists(local_image_path):
            try:
                # Load the image
                image = Image.open(local_image_path)

                # Generate output filename
                image_hash = hashlib.sha256(image_path.encode()).hexdigest()[:16]
                output_filename = f"{image_hash}_cleaned_masks.png"
                output_path = os.path.join(output_folder, output_filename)

                # Visualize and save (pass separating line if available)
                separating_line = result.get("separating_line", None)
                visualize_masks(
                    image,
                    masks,
                    cleaned_masks,
                    scores,
                    output_path,
                    show_comparison=show_comparison,
                    separating_line=separating_line,
                )
                print(f"  Saved visualization to: {output_path}")
                result["visualization_path"] = output_path
            except Exception as e:
                print(f"  Warning: Failed to create visualization: {e}")
        else:
            print(f"  Warning: Local image not found at {local_image_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Load and process cached SAM3 masks from CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file with column 'image_manifold_path' containing image paths",
    )
    parser.add_argument(
        "--mask-cache-dir",
        type=str,
        required=True,
        help="Directory containing cached mask files",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["person", "animal", "character"],
        help="Text prompts used for segmentation (must match those used in run_sam3.py)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory where images were cached during run_sam3.py (default: ~/.cache/sam3_downloads)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder for visualizations (default: no visualization)",
    )
    parser.add_argument(
        "--show-comparison",
        action="store_true",
        default=False,
        help="Show side-by-side comparison of original and cleaned masks (default: only show cleaned)",
    )

    args = parser.parse_args()

    print(f"Loading masks from cache directory: {args.mask_cache_dir}")
    print(f"Using prompts: {args.prompts}")

    # Setup cache directory (same as run_sam3.py)
    if args.cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sam3_downloads")
    else:
        cache_dir = args.cache_dir
    print(f"Using image cache directory: {cache_dir}")

    # Setup output folder for visualizations
    output_folder = None
    if args.output_folder is not None:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        print(f"Visualizations will be saved to: {output_folder}")
        print(f"Show comparison: {args.show_comparison}")

    # Read CSV
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if "image_manifold_path" not in rows[0]:
        raise ValueError("CSV must have 'image_manifold_path' column")

    print(f"Found {len(rows)} rows in CSV\n")

    # Process each row
    all_results = []
    missing_masks = []

    for idx, row in enumerate(rows):
        image_remote_path = row["image_manifold_path"]

        # Reconstruct the local cached image path (same logic as run_sam3.py)
        image_hash = get_cache_filename(image_remote_path)
        image_ext = os.path.splitext(image_remote_path)[1] or ".jpg"
        local_image_path = os.path.join(cache_dir, f"{image_hash}{image_ext}")

        # Generate mask cache filename based on local image path + prompts
        mask_cache_hash = get_mask_cache_filename(local_image_path, args.prompts)
        mask_cache_path = os.path.join(args.mask_cache_dir, f"{mask_cache_hash}.npz")

        print(f"\n[{idx}/{len(rows)}] Processing: {image_remote_path}")

        try:
            # Try to load cached masks
            if os.path.exists(mask_cache_path):
                masks, scores, cached_image_path, prompts = load_cached_masks(
                    mask_cache_path
                )
            else:
                print(f"  Warning: Mask not found, will use default separating line")
                print(f"  Expected at: {mask_cache_path}")
                masks, scores = None, None
                missing_masks.append(image_remote_path)

            # Process the masks (or create default line if no masks)
            result = process_masks_for_image(
                image_remote_path,
                masks,
                scores,
                args.prompts,
                local_image_path=local_image_path,
                output_folder=output_folder,
                show_comparison=args.show_comparison,
            )
            all_results.append(result)

        except Exception as e:
            print(f"  Error processing image: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"  Total rows in CSV: {len(rows)}")
    print(f"  Successfully loaded: {len(all_results)}")
    print(f"  Missing masks: {len(missing_masks)}")

    if missing_masks:
        print(f"\nMissing masks for:")
        for path in missing_masks[:10]:  # Show first 10
            print(f"  - {path}")
        if len(missing_masks) > 10:
            print(f"  ... and {len(missing_masks) - 10} more")

    # TODO: Add your further processing here
    # The all_results list contains dictionaries with masks, scores, and metadata
    # You can now perform additional analysis, visualization, or export

    return all_results


if __name__ == "__main__":
    main()


# Example usage:
# python /home/thinz/git/sam3/process_sam3_masks.py --csv your_file.csv --mask-cache-dir ./my_mask_cache --prompts person animal character
