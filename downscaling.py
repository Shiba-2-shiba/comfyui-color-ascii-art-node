# downscaling.py
import logging
import torch
import torch.nn.functional as F
from typing import Tuple

# Attempt to import kornia, warn if unavailable
try:
    from kornia.color import rgb_to_lab, lab_to_rgb
    _kornia_available = True
except ImportError:
    _kornia_available = False
    # Define dummy functions if kornia is not available to avoid import errors later
    # These will raise an error if actually called.
    def rgb_to_lab(x):
        raise ImportError("Kornia library is required for contrast downscaling but not installed.")
    def lab_to_rgb(x):
        raise ImportError("Kornia library is required for contrast downscaling but not installed.")

# --- Logger Setup ---
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3.Downscaling")
logger.setLevel(logging.INFO) # Default, will be adjusted by main node
if not logger.hasHandlers(): # Add basic handler if needed
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Lanczos Resampling Functions (from lanczos-py.txt) ---

def lanczos_kernel(x: torch.Tensor, a: int = 3) -> torch.Tensor:
    """Compute the Lanczos kernel."""
    # Handle the special case where x = 0
    zero_mask = x.abs() < 1e-7
    x = torch.where(zero_mask, torch.ones_like(x) * 1e-7, x) # Avoid division by zero

    # Compute the Lanczos kernel
    px = torch.pi * x
    kernel = torch.where(
        x.abs() < a,
        (torch.sin(px) * torch.sin(px / a)) / (px * px / a), # Corrected formula division
        torch.zeros_like(x),
    )

    # Set the kernel value to 1 at x = 0
    kernel = torch.where(zero_mask, torch.ones_like(kernel), kernel)
    return kernel

def compute_weights_and_indices(
    in_size: int,
    out_size: int,
    scale: float,
    a: int = 3,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cpu", # Default to CPU, can be changed
    support_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Compute weights and indices for Lanczos resampling."""
    # Compute coordinates in the input space, centered
    coord = (torch.arange(out_size, dtype=dtype, device=device) + 0.5) / scale - 0.5

    # Scale the support window based on the scaling factor
    effective_a = float(a)
    if scale < 1.0:  # Downsampling case
        effective_a = a * support_scaling / scale

    # Compute integer boundaries for indices
    left = torch.floor(coord - effective_a + 0.5).long()
    right = torch.ceil(coord + effective_a + 0.5).long()

    # Clamp boundaries to valid input range
    left = torch.clamp(left, 0, in_size)
    right = torch.clamp(right, 0, in_size)

    # Calculate max number of indices needed per output pixel
    max_indices_per_pixel = (right - left).max().item()
    if max_indices_per_pixel <= 0:
         # Handle edge case where window is empty or invalid
         logger.warning("Lanczos window calculation resulted in zero indices. Returning identity-like sparse matrix.")
         # Create sparse identity matrix components
         indices = torch.stack([torch.arange(out_size, device=device), torch.arange(out_size, device=device)], dim=0).long()
         weights = torch.ones(out_size, dtype=dtype, device=device)
         size = torch.Size((out_size, in_size))
         return weights, indices, size


    # Generate indices within the boundaries for each output pixel
    # Shape: [out_size, max_indices_per_pixel]
    expanded_indices = left[:, None] + torch.arange(max_indices_per_pixel, device=device)[None, :]

    # Create mask for valid indices within the calculated range [left, right) and input bounds [0, in_size)
    valid_mask = (expanded_indices < right[:, None]) & (expanded_indices >= 0) & (expanded_indices < in_size)

    # Compute weights using the Lanczos kernel relative to the coordinate
    # x shape: [out_size, max_indices_per_pixel]
    x = expanded_indices.to(dtype) - coord[:, None]
    weights = lanczos_kernel(x, a)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

    # Normalize weights per output pixel
    sum_weights = weights.sum(dim=1, keepdim=True)
    # Avoid division by zero for pixels with no valid contributing input pixels
    weights = torch.where(sum_weights.abs() > 1e-7, weights / sum_weights, torch.zeros_like(weights))

    # --- Prepare for sparse tensor ---
    # Get row indices (output indices) corresponding to each weight/column index
    out_coord_indices = torch.arange(out_size, device=device)[:, None].expand(-1, max_indices_per_pixel)

    # Filter out zero weights and corresponding indices
    relevant_mask = weights.abs() > 1e-7
    final_weights = weights[relevant_mask]
    final_out_indices = out_coord_indices[relevant_mask] # Row indices for sparse tensor
    final_in_indices = expanded_indices[relevant_mask]   # Column indices for sparse tensor

    # Stack indices for sparse tensor [2, num_non_zero]
    final_indices = torch.stack([final_out_indices, final_in_indices], dim=0)
    size = torch.Size((out_size, in_size)) # Size of the sparse matrix

    return final_weights, final_indices, size


def lanczos_resize(
    x: torch.Tensor, size: tuple[int, int], a: int = 3, support_scaling: float = 1.5
) -> torch.Tensor:
    """Resize images using Lanczos resampling."""
    if x is None:
        raise ValueError("Input tensor 'x' cannot be None")
    B, C, H, W = x.shape
    H_out, W_out = size
    dtype, device = x.dtype, x.device

    if H == H_out and W == W_out:
        return x

    logger.debug(f"Applying Lanczos resize from ({W}x{H}) to ({W_out}x{H_out}) with a={a}")

    try:
        # Compute scaling factors
        h_scale = H_out / H
        w_scale = W_out / W

        # Resize height first
        if H != H_out:
            weights_h, indices_h, size_h = compute_weights_and_indices(
                H, H_out, h_scale, a, dtype, device, support_scaling
            )
            # Create sparse matrix: W_h [H_out, H]
            W_h = torch.sparse_coo_tensor(indices_h, weights_h, size_h, device=device).to(dtype)
            # Reshape input: [B*C, H, W]
            x_reshaped = x.reshape(B * C, H, W)
            # Apply matrix multiplication: W_h @ x_reshaped -> [B*C, H_out, W]
            x = W_h @ x_reshaped
            # Reshape back: [B, C, H_out, W]
            x = x.reshape(B, C, H_out, W)

        # Resize width
        if W != W_out:
            weights_w, indices_w, size_w = compute_weights_and_indices(
                W, W_out, w_scale, a, dtype, device, support_scaling
            )
            # Create sparse matrix: W_w [W_out, W]
            W_w = torch.sparse_coo_tensor(indices_w, weights_w, size_w, device=device).to(dtype)
            # Reshape input: [B*C, H_out, W]
            x_reshaped = x.reshape(B * C, H_out, W)
            # Apply matrix multiplication (transpose W_w for right multiplication):
            # x_reshaped @ W_w.T -> [B*C, H_out, W_out]
            x = x_reshaped @ W_w.T # Using transpose property: (A @ B).T = B.T @ A.T
            # Reshape back: [B, C, H_out, W_out]
            x = x.reshape(B, C, H_out, W_out)

        return torch.clamp(x, 0, 1)  # Ensure output is in [0, 1]

    except Exception as e:
        logger.error(f"Error during Lanczos resize: {e}", exc_info=True)
        # Fallback to simpler resize if Lanczos fails? Or re-raise?
        # For now, re-raise
        raise RuntimeError("Lanczos resize failed.") from e


# --- Contrast-Based Downscaling Functions (from contrast_based-py.txt) ---

def find_pixel_luminance(chunk: torch.Tensor) -> torch.Tensor:
    """Helper function for contrast_downscale."""
    # chunk shape: [B, N, patch_area] (L channel)
    mid_idx = chunk.shape[2] // 2 # Approximate middle index
    mid = chunk[:, :, mid_idx].unsqueeze(2) # Pixel near center
    med = chunk.median(dim=2, keepdim=True).values # Median luminance in patch
    mu = chunk.mean(dim=2, keepdim=True) # Mean luminance in patch
    maxi = chunk.max(dim=2, keepdim=True).values # Max luminance in patch
    mini = chunk.min(dim=2, keepdim=True).values # Min luminance in patch

    # Default to the middle pixel's luminance
    out = mid.clone()
    # Condition 1: If median is below mean AND the range above median is larger than below, use minimum (preserve dark)
    cond1 = (med < mu) & ((maxi - med) > (med - mini))
    # Condition 2: If median is above mean AND the range below median is larger than above, use maximum (preserve bright)
    cond2 = (med > mu) & ((maxi - med) < (med - mini))

    out = torch.where(cond1, mini, out)
    out = torch.where(cond2, maxi, out)
    # Return shape: [B, N, 1]
    return out


def contrast_downscale(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Contrast-based downscaling using LAB color space."""
    if not _kornia_available:
        raise ImportError("Kornia library is required for contrast downscaling. Please install it (`pip install kornia`).")
    if img is None:
        raise ValueError("Input tensor 'img' cannot be None")

    N, C, H, W = img.shape
    if H % patch_size != 0 or W % patch_size != 0:
        # Pad if dimensions are not divisible by patch_size
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        img = F.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='replicate')
        N, C, H, W = img.shape
        logger.debug(f"Padded image for contrast downscale to {W}x{H}")

    out_h = H // patch_size
    out_w = W // patch_size

    logger.debug(f"Applying Contrast downscale with patch_size={patch_size} to target size ({out_w}x{out_h})")

    try:
        lab = rgb_to_lab(img)  # [B, 3, H, W]
        L, A, B_ch = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3] # Renamed B to B_ch to avoid conflict

        # Unfold channels into patches [B, C*patch_area, L] where L = out_h*out_w
        patches_l = F.unfold(L, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        patches_a = F.unfold(A, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        patches_b = F.unfold(B_ch, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

        # Reshape to [B, L, patch_area]
        patches_l = patches_l.transpose(1, 2)
        patches_a = patches_a.transpose(1, 2)
        patches_b = patches_b.transpose(1, 2)

        # Process luminance using the contrast logic
        result_l = find_pixel_luminance(patches_l) # Shape: [B, L, 1]

        # Compute median for A and B channels (less sensitive to outliers)
        result_a = patches_a.median(dim=2, keepdim=True).values # Shape: [B, L, 1]
        result_b = patches_b.median(dim=2, keepdim=True).values # Shape: [B, L, 1]

        # Concatenate results [B, L, 3]
        out_lab_flat = torch.cat([result_l, result_a, result_b], dim=2)

        # Reshape back to image format [B, 3, out_h, out_w]
        out_lab = out_lab_flat.transpose(1, 2).reshape(N, C, out_h, out_w)

        out_rgb = lab_to_rgb(out_lab)  # [B, 3, out_h, out_w]
        return torch.clamp(out_rgb, 0, 1)

    except Exception as e:
        logger.error(f"Error during contrast-based downscaling: {e}", exc_info=True)
        # Fallback or re-raise
        raise RuntimeError("Contrast-based downscaling failed.") from e
