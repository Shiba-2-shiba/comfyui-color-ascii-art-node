# ascii_utils.py (Modified for Batch Processing)
import os
import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
from PIL import Image
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
    log_level_upper = log_level_str.upper() # Ensure case-insensitivity [cite: 120]

    if log_level_upper == "DEBUG": # [cite: 121]
        level = logging.DEBUG
    elif log_level_upper == "INFO": # [cite: 121]
        level = logging.INFO
    elif log_level_upper == "WARNING": # [cite: 121]
        level = logging.WARNING
    elif log_level_upper == "ERROR": # [cite: 121]
        level = logging.ERROR
    elif log_level_upper == "NONE": # [cite: 121]
        # Set level higher than CRITICAL to disable logging
        level = logging.CRITICAL + 1 # [cite: 121]
    else:
        # Default to INFO if an invalid string is provided
        logger.warning(f"Invalid log level string '{log_level_str}', defaulting to INFO.") # [cite: 122]
        level = logging.INFO # [cite: 122]

    # Get the base logger instance
    base_logger = logging.getLogger(base_logger_name) # [cite: 122]
    # Set the level for the base logger; child loggers will inherit this
    base_logger.setLevel(level) # [cite: 123]

    # Ensure the base logger has a handler (important if running outside ComfyUI's setup)
    if not base_logger.hasHandlers(): # [cite: 123]
         # Add a basic console handler if none are configured
         handler = logging.StreamHandler() # [cite: 123]
         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # [cite: 123]
         handler.setFormatter(formatter) # [cite: 123]
         base_logger.addHandler(handler) # [cite: 123]
         # Typically, custom node loggers shouldn't propagate to avoid duplicate logs in ComfyUI
         base_logger.propagate = False # [cite: 124]

    # Log the level that was set
    logger.info(f"Log level set to: {logging.getLevelName(level)} ({level}) for logger tree starting at '{base_logger_name}'") # [cite: 124]


def tensor_to_pil(tensor_image: torch.Tensor) -> List[Image.Image]: # Modified return type
    """Converts a ComfyUI IMAGE tensor (B, H, W, C) to a list of PIL Images (RGB)."""
    # Input validation
    if tensor_image is None: # [cite: 124]
        raise ValueError("Input tensor_image cannot be None") # [cite: 125]
    if not isinstance(tensor_image, torch.Tensor): # [cite: 125]
        raise TypeError(f"Expected input to be a torch.Tensor, got {type(tensor_image)}") # [cite: 125]
    if tensor_image.dim() != 4: # [cite: 125]
         raise ValueError(f"Expected input tensor to have 4 dimensions (B, H, W, C), got {tensor_image.dim()}") # [cite: 125]
    if tensor_image.shape[0] == 0: # [cite: 125]
         raise ValueError("Input tensor batch size is 0.") # [cite: 125]

    batch_size = tensor_image.shape[0]
    pil_images = []

    logger.debug(f"Converting tensor (shape: {tensor_image.shape}) to list of {batch_size} PIL Images...")

    for i in range(batch_size):
        # Process each image in the batch
        image_np = tensor_image[i].cpu().float().numpy() # Select i-th image, ensure float for scaling
        # Denormalize from [0, 1] to [0, 255] and convert to uint8
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8) # [cite: 126]

        # Check channel dimension (should be last)
        if image_np.shape[-1] != 3: # [cite: 126]
             raise ValueError(f"Expected tensor C dimension (last) to be 3 (RGB) for batch item {i}, got {image_np.shape[-1]}") # [cite: 127]

        # NumPy shape is (H, W, C), PIL expects (W, H) for size but array is HxW
        try:
            pil_image = Image.fromarray(image_np, mode='RGB') # [cite: 127]
            pil_images.append(pil_image)
        except Exception as e:
            logger.error(f"Failed to convert batch item {i} to PIL Image: {e}", exc_info=True)
            # Decide how to handle errors: skip, return partial list, or raise? Re-raising for now.
            raise RuntimeError(f"Failed to convert batch item {i} to PIL Image.") from e

    logger.debug(f"Successfully converted tensor to {len(pil_images)} PIL Images.")
    return pil_images

def mask_tensor_to_pil(tensor_mask: torch.Tensor, target_size: Tuple[int, int]) -> Image.Image:
    """
    Converts a ComfyUI MASK tensor (B, 1, H, W) or (B, H, W) to a PIL Image (L) resized.
    NOTE: This function currently only processes the FIRST mask in the batch.
          If batch support for masks is needed, this function requires similar modifications
          to tensor_to_pil and pil_to_tensor.
    """
    # Input validation
    if tensor_mask is None: # [cite: 127]
        raise ValueError("Input tensor_mask cannot be None") # [cite: 128]
    if not isinstance(tensor_mask, torch.Tensor): # [cite: 128]
        raise TypeError(f"Expected mask input to be a torch.Tensor, got {type(tensor_mask)}") # [cite: 128]
    if target_size is None or len(target_size) != 2: # [cite: 128]
         raise ValueError(f"Invalid target_size provided: {target_size}. Expected (width, height).") # [cite: 128]

    original_shape = tensor_mask.shape # [cite: 129]
    processed_shape = None # To store shape after dimension reduction # [cite: 129]

    # Handle different mask dimensions (B, C, H, W), (B, H, W), (H, W)
    if tensor_mask.dim() == 4: # B, C, H, W (expected C=1) # [cite: 129]
        if tensor_mask.shape[1] != 1: # [cite: 129]
             logger.warning(f"Input mask tensor has {tensor_mask.shape[1]} channels, expected 1. Using the first channel.") # [cite: 129]
        # Select first image in batch, first channel
        mask_np = tensor_mask[0, 0].cpu().float().numpy() # Shape (H, W) # [cite: 129]
        processed_shape = mask_np.shape # [cite: 130]
    elif tensor_mask.dim() == 3: # B, H, W # [cite: 130]
        # Select first image in batch
        mask_np = tensor_mask[0].cpu().float().numpy() # Shape (H, W) # [cite: 130]
        processed_shape = mask_np.shape # [cite: 130]
    elif tensor_mask.dim() == 2: # H, W (assuming batch size 1 was implicitly handled) # [cite: 130]
        mask_np = tensor_mask.cpu().float().numpy() # [cite: 131]
        processed_shape = mask_np.shape # [cite: 131]
    else:
        raise ValueError(f"Expected mask tensor to have 2, 3, or 4 dimensions, got {tensor_mask.dim()}") # [cite: 131]

    # Check batch size warning if applicable
    if tensor_mask.dim() > 2 and original_shape[0] > 1: # [cite: 131]
        logger.warning(f"Input mask tensor batch size is {original_shape[0]}, using only the first mask.") # [cite: 131]

    # Denormalize from [0, 1] to [0, 255] and convert to uint8
    mask_np = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8) # [cite: 131]
    mask_pil = Image.fromarray(mask_np, mode='L') # Create PIL image in Luminance mode # [cite: 132]

    # Resize to match the target image size (W, H) if necessary
    # Pillow resize takes (width, height)
    if mask_pil.size != target_size: # [cite: 132]
        try:
            mask_pil_resized = mask_pil.resize(target_size, Image.Resampling.LANCZOS) # Use high-quality resampling # [cite: 132]
            logger.debug(f"Converted mask tensor (original shape: {original_shape}, processed shape: {processed_shape}) to PIL Image, resized from {mask_pil.size} to {target_size}") # [cite: 132]
        except Exception as resize_e: # [cite: 133]
             logger.error(f"Failed to resize mask from {mask_pil.size} to {target_size}: {resize_e}", exc_info=True) # [cite: 133]
             # Fallback or re-raise? Let's re-raise for now.
             raise RuntimeError("Mask resizing failed.") from resize_e # [cite: 134]
    else:
        mask_pil_resized = mask_pil # [cite: 134]
        logger.debug(f"Converted mask tensor (original shape: {original_shape}, processed shape: {processed_shape}) to PIL Image (size: {mask_pil_resized.size}, mode: {mask_pil_resized.mode})") # [cite: 134]

    return mask_pil_resized

def pil_to_tensor(pil_images: List[Image.Image]) -> torch.Tensor: # Modified input type
    """Converts a list of PIL Images (RGB) back to a ComfyUI IMAGE tensor (B, H, W, C)."""
    # Input validation
    if not isinstance(pil_images, list) or not pil_images:
        raise ValueError("Input pil_images must be a non-empty list.")
    if not all(isinstance(img, Image.Image) for img in pil_images): # [cite: 135]
         raise TypeError(f"Expected input to be a list of PIL Images, found other types.") # [cite: 135]

    tensor_list = []
    logger.debug(f"Converting list of {len(pil_images)} PIL Images back to tensor...")

    for i, pil_image in enumerate(pil_images):
        # Ensure image is in RGB mode
        if pil_image.mode != 'RGB': # [cite: 135]
            logger.warning(f"Input PIL image {i} mode is {pil_image.mode}, converting to RGB.") # [cite: 135]
            try:
                pil_image = pil_image.convert('RGB') # [cite: 135]
            except Exception as convert_e: # [cite: 135]
                logger.error(f"Failed to convert PIL image {i} to RGB: {convert_e}", exc_info=True) # [cite: 136]
                raise RuntimeError(f"PIL to RGB conversion failed for image {i}.") from convert_e # [cite: 136]

        # Convert PIL image to NumPy array and normalize to [0, 1] float32
        try:
            image_np = np.array(pil_image).astype(np.float32) / 255.0 # Shape (H, W, C) # [cite: 136]
            # Convert NumPy array to Tensor for this single image (H, W, C)
            single_tensor = torch.from_numpy(image_np)
            tensor_list.append(single_tensor)
        except Exception as e:
            logger.error(f"Failed to convert PIL image {i} to tensor: {e}", exc_info=True)
            raise RuntimeError(f"Failed to convert PIL image {i} to tensor.") from e

    # Stack the list of tensors along the batch dimension (dim=0)
    try:
        output_tensor = torch.stack(tensor_list, dim=0) # Shape (B, H, W, C)
        logger.debug(f"Successfully converted PIL list to tensor (shape: {output_tensor.shape})") # [cite: 136]
        return output_tensor # [cite: 137]
    except Exception as e:
        logger.error(f"Failed to stack tensors: {e}", exc_info=True)
        # This might happen if images have different dimensions
        raise RuntimeError("Failed to stack tensors into a batch. Ensure all input PIL images have the same dimensions.") from e


def load_custom_characters(file_path: str) -> List[str]: # [cite: 137]
    """Loads ASCII character sets from a file, one set per non-empty, non-comment line."""
    logger.debug(f"Attempting to load character sets from: {file_path}") # [cite: 137]
    # Check if file exists
    if not os.path.isfile(file_path): # [cite: 137]
        raise FileNotFoundError(f"Character set file not found: {file_path}") # [cite: 137]

    sets = [] # [cite: 137]
    try:
        # Open file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file: # [cite: 137]
            line_number = 0 # [cite: 138]
            for line in file: # [cite: 138]
                line_number += 1 # [cite: 138]
                # Remove leading/trailing whitespace
                line = line.strip() # [cite: 138]
                # Skip empty lines and lines starting with '#' (comments)
                if not line or line.startswith('#'): # [cite: 138]
                    continue # [cite: 139]

                # Optional: Remove potential set labels like "SetN:"
                if ':' in line: # [cite: 139]
                    parts = line.split(':', 1) # [cite: 140]
                    # Take the part after the first colon as the character set
                    chars = parts[1].strip() # [cite: 140]
                    if not chars: # [cite: 140]
                         logger.warning(f"Ignoring line {line_number} with label but no characters: {line}") # [cite: 141]
                         continue # [cite: 141]
                    sets.append(chars) # [cite: 141]
                else:
                    # If no colon, the whole line is treated as a character set
                    sets.append(line) # [cite: 142]

        # Check if any sets were actually loaded
        if not sets: # [cite: 142]
             raise ValueError(f"No valid character sets found in the file: {file_path}. Ensure lines are not empty or only comments.") # [cite: 142]
        logger.debug(f"Successfully loaded {len(sets)} character sets.") # [cite: 143]
        return sets # [cite: 143]
    except FileNotFoundError: # Should be caught by isfile check, but good practice # [cite: 143]
        logger.error(f"File not found during open operation: {file_path}", exc_info=True) # [cite: 143]
        raise # Re-raise the specific error # [cite: 143]
    except Exception as e: # [cite: 143]
        # Catch other potential errors during file reading or processing
        logger.error(f"Error reading or parsing character file {file_path}: {e}", exc_info=True) # [cite: 144]
        # Re-raise a more generic error indicating failure
        raise RuntimeError(f"Failed to load or parse character file '{file_path}'. Check file content and encoding.") from e # [cite: 144]

def calculate_edge_info(mask_pil: Image.Image, edge_factor: float, target_size_wh: Tuple[int, int]) -> Optional[np.ndarray]: # [cite: 144]
    """
    Calculates an edge intensity map from a mask using Gaussian blur.
    The map indicates proximity to the mask edge (0.5 value).
    Returns a NumPy array (H', W') scaled to target_size_wh, values 0.0-1.0 (1.0 at edge boundary).
    """ # [cite: 146]
    if mask_pil is None: # [cite: 147]
        logger.debug("No mask provided for edge info calculation.") # [cite: 147]
        return None # [cite: 147]
    if edge_factor <= 0: # [cite: 147]
        logger.debug("Edge factor is <= 0, skipping edge info calculation.") # [cite: 147]
        return None # No edge effect if factor is zero or negative # [cite: 147]

    logger.debug(f"Calculating edge info (factor: {edge_factor}) for target size {target_size_wh}") # [cite: 147]
    try:
        # Ensure mask is in Luminance mode and convert to NumPy array [0.0, 1.0]
        mask_np_hw = np.array(mask_pil.convert('L')).astype(np.float32) / 255.0 # Shape (H, W) # [cite: 148]

        # Apply Gaussian filter to the mask to get smooth transitions
        # Sigma determines the width of the transition area (edge influence)
        # Ensure sigma is positive
        sigma = max(0.1, edge_factor) # [cite: 148]
        blurred_mask_np = scipy.ndimage.gaussian_filter(mask_np_hw, sigma=sigma) # [cite: 148]
        logger.debug(f"Blurred mask for edge info (sigma={sigma}), shape: {blurred_mask_np.shape}, min: {blurred_mask_np.min():.3f}, max: {blurred_mask_np.max():.3f}") # [cite: 149]

        # Calculate edge intensity: map values near 0.5 to 1.0 (edge), values near 0 or 1 to 0.0 (center)
        # Formula: abs(value - 0.5) * 2.0 scales the distance from 0.5 to the range [0, 1]
        edge_intensity = np.abs(blurred_mask_np - 0.5) * 2.0 # [cite: 149]
        # Clip values to ensure they stay within the [0, 1] range
        edge_intensity = np.clip(edge_intensity, 0.0, 1.0) # [cite: 150]

        # Convert edge intensity map back to PIL for resizing
        edge_info_pil = Image.fromarray((edge_intensity * 255).astype(np.uint8), mode='L') # [cite: 150]

        # Resize this edge intensity map to match the target dimensions (e.g., pixelated grid size)
        if edge_info_pil.size != target_size_wh: # [cite: 150]
            edge_info_pil_resized = edge_info_pil.resize(target_size_wh, Image.Resampling.LANCZOS) # Use high-quality resize # [cite: 150]
            logger.debug(f"Resized edge info map from {edge_info_pil.size} to {target_size_wh}") # [cite: 151]
        else:
            edge_info_pil_resized = edge_info_pil # [cite: 151]
            logger.debug(f"Edge info map already at target size {target_size_wh}") # [cite: 151]


        # Final edge_info as NumPy array (H', W'), float values [0.0, 1.0]
        edge_info_np = np.array(edge_info_pil_resized).astype(np.float32) / 255.0 # [cite: 151]
        logger.debug(f"Edge info calculated, shape: {edge_info_np.shape}, min: {edge_info_np.min():.3f}, max: {edge_info_np.max():.3f}") # [cite: 151]
        return edge_info_np # [cite: 152]

    except Exception as e: # [cite: 152]
        logger.error(f"Failed to calculate edge info: {e}", exc_info=True) # [cite: 152]
        return None # Return None if calculation fails to allow fallback # [cite: 152]

def apply_mask_blending(original_pil: Image.Image,
                        overlay_pil: Image.Image,
                        mask_pil: Image.Image,
                        mask_blend_radius: float) -> Image.Image: # [cite: 153]
    """
    Applies the overlay image onto the original using a mask with optional Gaussian blur blending.
    """ # [cite: 153]
    if original_pil is None or overlay_pil is None or mask_pil is None: # [cite: 154]
         raise ValueError("Inputs for mask blending (original, overlay, mask) cannot be None.") # [cite: 154]

    logger.debug(f"Applying mask blending with blend radius: {mask_blend_radius}") # [cite: 154]
    try:
        original_size = original_pil.size # [cite: 154]
        # Ensure overlay matches original size (important for pasting/compositing)
        if overlay_pil.size != original_size: # [cite: 154]
             logger.warning(f"Overlay size {overlay_pil.size} differs from original {original_size}. Resizing overlay.") # [cite: 154]
             overlay_pil = overlay_pil.resize(original_size, Image.Resampling.LANCZOS) # [cite: 155]

        # Convert images to RGBA for alpha compositing (preserves transparency)
        original_rgba = original_pil.convert('RGBA') # [cite: 155]
        overlay_rgba = overlay_pil.convert('RGBA') # [cite: 155]

        # Ensure mask is L mode and correct size for use as alpha channel
        blend_mask_pil = mask_pil.convert('L') # [cite: 155]
        if blend_mask_pil.size != original_size: # [cite: 155]
            logger.warning(f"Mask size {blend_mask_pil.size} differs from original {original_size}. Resizing mask.") # [cite: 156]
            blend_mask_pil = blend_mask_pil.resize(original_size, Image.Resampling.LANCZOS) # [cite: 157]

        # Apply Gaussian blur to the mask for smooth blending if radius > 0
        if mask_blend_radius > 0: # [cite: 157]
            logger.debug(f"Applying Gaussian blur to mask for blending (radius ~ sigma={max(0.1, mask_blend_radius / 2.0)})") # [cite: 157]
            # Convert mask to NumPy array [0.0, 1.0] for filtering
            mask_np_hw = np.array(blend_mask_pil).astype(np.float32) / 255.0 # [cite: 158]
            # Use scipy.ndimage.gaussian_filter for blurring
            # Sigma is approximated from the radius; ensure it's positive
            sigma_blend = max(0.1, mask_blend_radius / 2.0) # [cite: 158]
            blurred_mask_np = scipy.ndimage.gaussian_filter(mask_np_hw, sigma=sigma_blend) # [cite: 158]
            # Clip result to [0, 1] and convert back to PIL Image (L)
            blurred_mask_np = np.clip(blurred_mask_np, 0.0, 1.0) # [cite: 159]
            blend_mask_pil = Image.fromarray((blurred_mask_np * 255).astype(np.uint8), mode='L') # [cite: 159]
            logger.debug("Mask blurred for blending using scipy.") # [cite: 159]
        else:
            logger.debug("Using original mask for blending (no blur).") # [cite: 159]

        # --- Alpha Compositing ---
        # Create a transparent canvas matching the original size
        # final_image_rgba = Image.new('RGBA', original_size, (0, 0, 0, 0)) # [cite: 160]
        # Paste the background (original image) onto the canvas
        # final_image_rgba.paste(original_rgba, (0, 0)) # [cite: 160]

        # Apply the (potentially blurred) mask as the alpha channel to the overlay image
        overlay_with_alpha = overlay_rgba.copy() # [cite: 160]
        overlay_with_alpha.putalpha(blend_mask_pil) # [cite: 160]

        # Composite the overlay (with its new alpha) onto the original background
        # Image.alpha_composite requires both images to be RGBA
        final_image_rgba = Image.alpha_composite(original_rgba, overlay_with_alpha) # [cite: 161]

        # Convert final composite image back to RGB for output
        final_image_rgb = final_image_rgba.convert('RGB') # [cite: 161]
        logger.info("Mask applied successfully using alpha compositing.") # [cite: 161]
        return final_image_rgb # [cite: 161]

    except Exception as e: # [cite: 161]
        logger.error(f"Error applying mask blending: {e}", exc_info=True) # [cite: 162]
        # Fallback strategy: Return the original overlay image without masking? Or original?
        # Returning the overlay might be less surprising if masking fails. # [cite: 163]
        logger.warning("Mask application failed, returning original overlay image.") # [cite: 164]
        # Ensure the fallback is also RGB
        return overlay_pil.convert('RGB') # [cite: 164]
