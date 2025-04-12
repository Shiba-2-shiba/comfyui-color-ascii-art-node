# ascii_utils.py
import os
import logging
from typing import List, Tuple, Optional # Added Optional
import numpy as np
import torch
from PIL import Image # Removed ImageFont, ImageDraw as they are not used here
import scipy.ndimage # Keep for gaussian_filter

# --- Logger Setup ---
# This module's logger (intended to be configured by the main node)
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3.Utils")
# Set a default level; will be overridden by the main node's setup
logger.setLevel(logging.INFO)
# Add a basic handler if none exists (useful for standalone testing)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent duplicate logs if root logger has handler

def setup_logging(log_level_str: str, base_logger_name="ComfyUI.ASCIIArtNodeV3"):
    """Sets the logger level for the base logger and its children."""
    level = logging.INFO # Default level
    log_level_upper = log_level_str.upper() # Ensure case-insensitivity

    if log_level_upper == "DEBUG":
        level = logging.DEBUG
    elif log_level_upper == "INFO":
        level = logging.INFO
    elif log_level_upper == "WARNING":
        level = logging.WARNING
    elif log_level_upper == "ERROR":
        level = logging.ERROR
    elif log_level_upper == "NONE":
        # Set level higher than CRITICAL to disable logging
        level = logging.CRITICAL + 1
    else:
        # Default to INFO if an invalid string is provided
        logger.warning(f"Invalid log level string '{log_level_str}', defaulting to INFO.")
        level = logging.INFO

    # Get the base logger instance
    base_logger = logging.getLogger(base_logger_name)
    # Set the level for the base logger; child loggers will inherit this
    base_logger.setLevel(level)

    # Ensure the base logger has a handler (important if running outside ComfyUI's setup)
    if not base_logger.hasHandlers():
         # Add a basic console handler if none are configured
         handler = logging.StreamHandler()
         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
         handler.setFormatter(formatter)
         base_logger.addHandler(handler)
         # Typically, custom node loggers shouldn't propagate to avoid duplicate logs in ComfyUI
         base_logger.propagate = False

    # Log the level that was set
    logger.info(f"Log level set to: {logging.getLevelName(level)} ({level}) for logger tree starting at '{base_logger_name}'")


def tensor_to_pil(tensor_image: torch.Tensor) -> Image.Image:
    """Converts a ComfyUI IMAGE tensor (B, H, W, C) to a PIL Image (RGB)."""
    # Input validation
    if tensor_image is None:
        raise ValueError("Input tensor_image cannot be None")
    if not isinstance(tensor_image, torch.Tensor):
        raise TypeError(f"Expected input to be a torch.Tensor, got {type(tensor_image)}")
    if tensor_image.dim() != 4:
         raise ValueError(f"Expected input tensor to have 4 dimensions (B, H, W, C), got {tensor_image.dim()}")
    if tensor_image.shape[0] == 0:
         raise ValueError("Input tensor batch size is 0.")
    if tensor_image.shape[0] > 1:
        # Warn if batch size > 1, as we only process the first image
        logger.warning(f"Input tensor batch size is {tensor_image.shape[0]}, using only the first image.")

    # Process the first image in the batch
    image_np = tensor_image[0].cpu().float().numpy() # Select first image, ensure float for scaling
    # Denormalize from [0, 1] to [0, 255] and convert to uint8
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    # Check channel dimension (should be last)
    if image_np.shape[-1] != 3:
         raise ValueError(f"Expected tensor C dimension (last) to be 3 (RGB), got {image_np.shape[-1]}")

    # NumPy shape is (H, W, C), PIL expects (W, H) for size but array is HxW
    pil_image = Image.fromarray(image_np, mode='RGB')
    logger.debug(f"Converted tensor (shape: {tensor_image.shape}) to PIL Image (size: {pil_image.size}, mode: {pil_image.mode})")
    return pil_image

def mask_tensor_to_pil(tensor_mask: torch.Tensor, target_size: Tuple[int, int]) -> Image.Image:
    """Converts a ComfyUI MASK tensor (B, 1, H, W) or (B, H, W) to a PIL Image (L) resized."""
    # Input validation
    if tensor_mask is None:
        raise ValueError("Input tensor_mask cannot be None")
    if not isinstance(tensor_mask, torch.Tensor):
        raise TypeError(f"Expected mask input to be a torch.Tensor, got {type(tensor_mask)}")
    if target_size is None or len(target_size) != 2:
         raise ValueError(f"Invalid target_size provided: {target_size}. Expected (width, height).")

    original_shape = tensor_mask.shape
    processed_shape = None # To store shape after dimension reduction

    # Handle different mask dimensions (B, C, H, W), (B, H, W), (H, W)
    if tensor_mask.dim() == 4: # B, C, H, W (expected C=1)
        if tensor_mask.shape[1] != 1:
             logger.warning(f"Input mask tensor has {tensor_mask.shape[1]} channels, expected 1. Using the first channel.")
        # Select first image in batch, first channel
        mask_np = tensor_mask[0, 0].cpu().float().numpy() # Shape (H, W)
        processed_shape = mask_np.shape
    elif tensor_mask.dim() == 3: # B, H, W
        # Select first image in batch
        mask_np = tensor_mask[0].cpu().float().numpy() # Shape (H, W)
        processed_shape = mask_np.shape
    elif tensor_mask.dim() == 2: # H, W (assuming batch size 1 was implicitly handled)
        mask_np = tensor_mask.cpu().float().numpy()
        processed_shape = mask_np.shape
    else:
        raise ValueError(f"Expected mask tensor to have 2, 3, or 4 dimensions, got {tensor_mask.dim()}")

    # Check batch size warning if applicable
    if tensor_mask.dim() > 2 and original_shape[0] > 1:
        logger.warning(f"Input mask tensor batch size is {original_shape[0]}, using only the first mask.")

    # Denormalize from [0, 1] to [0, 255] and convert to uint8
    mask_np = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode='L') # Create PIL image in Luminance mode

    # Resize to match the target image size (W, H) if necessary
    # Pillow resize takes (width, height)
    if mask_pil.size != target_size:
        try:
            mask_pil_resized = mask_pil.resize(target_size, Image.Resampling.LANCZOS) # Use high-quality resampling
            logger.debug(f"Converted mask tensor (original shape: {original_shape}, processed shape: {processed_shape}) to PIL Image, resized from {mask_pil.size} to {target_size}")
        except Exception as resize_e:
             logger.error(f"Failed to resize mask from {mask_pil.size} to {target_size}: {resize_e}", exc_info=True)
             # Fallback or re-raise? Let's re-raise for now.
             raise RuntimeError("Mask resizing failed.") from resize_e
    else:
        mask_pil_resized = mask_pil
        logger.debug(f"Converted mask tensor (original shape: {original_shape}, processed shape: {processed_shape}) to PIL Image (size: {mask_pil_resized.size}, mode: {mask_pil_resized.mode})")

    return mask_pil_resized

def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image (RGB) back to a ComfyUI IMAGE tensor (B, H, W, C)."""
    # Input validation
    if pil_image is None:
        raise ValueError("Input pil_image cannot be None")
    if not isinstance(pil_image, Image.Image):
         raise TypeError(f"Expected input to be a PIL Image, got {type(pil_image)}")

    # Ensure image is in RGB mode
    if pil_image.mode != 'RGB':
        logger.warning(f"Input PIL image mode is {pil_image.mode}, converting to RGB.")
        try:
            pil_image = pil_image.convert('RGB')
        except Exception as convert_e:
             logger.error(f"Failed to convert PIL image to RGB: {convert_e}", exc_info=True)
             raise RuntimeError("PIL to RGB conversion failed.") from convert_e

    # Convert PIL image to NumPy array and normalize to [0, 1] float32
    image_np = np.array(pil_image).astype(np.float32) / 255.0 # Shape (H, W, C)
    # Add batch dimension using None slicing
    tensor_image = torch.from_numpy(image_np)[None, ...] # Shape (1, H, W, C)
    logger.debug(f"Converted PIL Image (size: {pil_image.size}) back to tensor (shape: {tensor_image.shape})")
    return tensor_image


def load_custom_characters(file_path: str) -> List[str]:
    """Loads ASCII character sets from a file, one set per non-empty, non-comment line."""
    logger.debug(f"Attempting to load character sets from: {file_path}")
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Character set file not found: {file_path}")

    sets = []
    try:
        # Open file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            line_number = 0
            for line in file:
                line_number += 1
                # Remove leading/trailing whitespace
                line = line.strip()
                # Skip empty lines and lines starting with '#' (comments)
                if not line or line.startswith('#'):
                    continue

                # Optional: Remove potential set labels like "SetN:"
                if ':' in line:
                    parts = line.split(':', 1)
                    # Take the part after the first colon as the character set
                    chars = parts[1].strip()
                    if not chars:
                         logger.warning(f"Ignoring line {line_number} with label but no characters: {line}")
                         continue
                    sets.append(chars)
                else:
                    # If no colon, the whole line is treated as a character set
                    sets.append(line)

        # Check if any sets were actually loaded
        if not sets:
             raise ValueError(f"No valid character sets found in the file: {file_path}. Ensure lines are not empty or only comments.")
        logger.debug(f"Successfully loaded {len(sets)} character sets.")
        return sets
    except FileNotFoundError: # Should be caught by isfile check, but good practice
        logger.error(f"File not found during open operation: {file_path}", exc_info=True)
        raise # Re-raise the specific error
    except Exception as e:
        # Catch other potential errors during file reading or processing
        logger.error(f"Error reading or parsing character file {file_path}: {e}", exc_info=True)
        # Re-raise a more generic error indicating failure
        raise RuntimeError(f"Failed to load or parse character file '{file_path}'. Check file content and encoding.") from e

def calculate_edge_info(mask_pil: Image.Image, edge_factor: float, target_size_wh: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Calculates an edge intensity map from a mask using Gaussian blur.
    The map indicates proximity to the mask edge (0.5 value).
    Returns a NumPy array (H', W') scaled to target_size_wh, values 0.0-1.0 (1.0 at edge boundary).
    """
    if mask_pil is None:
        logger.debug("No mask provided for edge info calculation.")
        return None
    if edge_factor <= 0:
        logger.debug("Edge factor is <= 0, skipping edge info calculation.")
        return None # No edge effect if factor is zero or negative

    logger.debug(f"Calculating edge info (factor: {edge_factor}) for target size {target_size_wh}")
    try:
        # Ensure mask is in Luminance mode and convert to NumPy array [0.0, 1.0]
        mask_np_hw = np.array(mask_pil.convert('L')).astype(np.float32) / 255.0 # Shape (H, W)

        # Apply Gaussian filter to the mask to get smooth transitions
        # Sigma determines the width of the transition area (edge influence)
        # Ensure sigma is positive
        sigma = max(0.1, edge_factor)
        blurred_mask_np = scipy.ndimage.gaussian_filter(mask_np_hw, sigma=sigma)
        logger.debug(f"Blurred mask for edge info (sigma={sigma}), shape: {blurred_mask_np.shape}, min: {blurred_mask_np.min():.3f}, max: {blurred_mask_np.max():.3f}")

        # Calculate edge intensity: map values near 0.5 to 1.0 (edge), values near 0 or 1 to 0.0 (center)
        # Formula: abs(value - 0.5) * 2.0 scales the distance from 0.5 to the range [0, 1]
        edge_intensity = np.abs(blurred_mask_np - 0.5) * 2.0
        # Clip values to ensure they stay within the [0, 1] range
        edge_intensity = np.clip(edge_intensity, 0.0, 1.0)

        # Convert edge intensity map back to PIL for resizing
        edge_info_pil = Image.fromarray((edge_intensity * 255).astype(np.uint8), mode='L')

        # Resize this edge intensity map to match the target dimensions (e.g., pixelated grid size)
        if edge_info_pil.size != target_size_wh:
            edge_info_pil_resized = edge_info_pil.resize(target_size_wh, Image.Resampling.LANCZOS) # Use high-quality resize
            logger.debug(f"Resized edge info map from {edge_info_pil.size} to {target_size_wh}")
        else:
            edge_info_pil_resized = edge_info_pil
            logger.debug(f"Edge info map already at target size {target_size_wh}")


        # Final edge_info as NumPy array (H', W'), float values [0.0, 1.0]
        edge_info_np = np.array(edge_info_pil_resized).astype(np.float32) / 255.0
        logger.debug(f"Edge info calculated, shape: {edge_info_np.shape}, min: {edge_info_np.min():.3f}, max: {edge_info_np.max():.3f}")
        return edge_info_np

    except Exception as e:
        logger.error(f"Failed to calculate edge info: {e}", exc_info=True)
        return None # Return None if calculation fails to allow fallback

def apply_mask_blending(original_pil: Image.Image,
                        overlay_pil: Image.Image,
                        mask_pil: Image.Image,
                        mask_blend_radius: float) -> Image.Image:
    """
    Applies the overlay image onto the original using a mask with optional Gaussian blur blending.
    """
    if original_pil is None or overlay_pil is None or mask_pil is None:
         raise ValueError("Inputs for mask blending (original, overlay, mask) cannot be None.")

    logger.debug(f"Applying mask blending with blend radius: {mask_blend_radius}")
    try:
        original_size = original_pil.size
        # Ensure overlay matches original size (important for pasting/compositing)
        if overlay_pil.size != original_size:
             logger.warning(f"Overlay size {overlay_pil.size} differs from original {original_size}. Resizing overlay.")
             overlay_pil = overlay_pil.resize(original_size, Image.Resampling.LANCZOS)

        # Convert images to RGBA for alpha compositing (preserves transparency)
        original_rgba = original_pil.convert('RGBA')
        overlay_rgba = overlay_pil.convert('RGBA')

        # Ensure mask is L mode and correct size for use as alpha channel
        blend_mask_pil = mask_pil.convert('L')
        if blend_mask_pil.size != original_size:
            logger.warning(f"Mask size {blend_mask_pil.size} differs from original {original_size}. Resizing mask.")
            blend_mask_pil = blend_mask_pil.resize(original_size, Image.Resampling.LANCZOS)

        # Apply Gaussian blur to the mask for smooth blending if radius > 0
        if mask_blend_radius > 0:
            logger.debug(f"Applying Gaussian blur to mask for blending (radius ~ sigma={max(0.1, mask_blend_radius / 2.0)})")
            # Convert mask to NumPy array [0.0, 1.0] for filtering
            mask_np_hw = np.array(blend_mask_pil).astype(np.float32) / 255.0
            # Use scipy.ndimage.gaussian_filter for blurring
            # Sigma is approximated from the radius; ensure it's positive
            sigma_blend = max(0.1, mask_blend_radius / 2.0)
            blurred_mask_np = scipy.ndimage.gaussian_filter(mask_np_hw, sigma=sigma_blend)
            # Clip result to [0, 1] and convert back to PIL Image (L)
            blurred_mask_np = np.clip(blurred_mask_np, 0.0, 1.0)
            blend_mask_pil = Image.fromarray((blurred_mask_np * 255).astype(np.uint8), mode='L')
            logger.debug("Mask blurred for blending using scipy.")
        else:
            logger.debug("Using original mask for blending (no blur).")

        # --- Alpha Compositing ---
        # Create a transparent canvas matching the original size
        # final_image_rgba = Image.new('RGBA', original_size, (0, 0, 0, 0))
        # Paste the background (original image) onto the canvas
        # final_image_rgba.paste(original_rgba, (0, 0))

        # Apply the (potentially blurred) mask as the alpha channel to the overlay image
        overlay_with_alpha = overlay_rgba.copy()
        overlay_with_alpha.putalpha(blend_mask_pil)

        # Composite the overlay (with its new alpha) onto the original background
        # Image.alpha_composite requires both images to be RGBA
        final_image_rgba = Image.alpha_composite(original_rgba, overlay_with_alpha)

        # Convert final composite image back to RGB for output
        final_image_rgb = final_image_rgba.convert('RGB')
        logger.info("Mask applied successfully using alpha compositing.")
        return final_image_rgb

    except Exception as e:
        logger.error(f"Error applying mask blending: {e}", exc_info=True)
        # Fallback strategy: Return the original overlay image without masking? Or original?
        # Returning the overlay might be less surprising if masking fails.
        logger.warning("Mask application failed, returning original overlay image.")
        # Ensure the fallback is also RGB
        return overlay_pil.convert('RGB')
