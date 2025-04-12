# pixelation.py
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast # For mixed precision if available
from PIL import Image, ImageEnhance # Keep ImageEnhance for fallback or comparison
from typing import Tuple

# --- Module Imports ---
# Use relative imports assuming files are in the same directory/package
try:
    from .sharpening import unsharp_mask
    from .downscaling import lanczos_resize, contrast_downscale
except ImportError:
    # Fallback to direct imports if relative fail (e.g., running script directly)
    print("Warning: Relative imports failed in pixelation.py. Trying direct imports.")
    # Ensure the current directory is in the Python path for direct imports to work
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    try:
        from sharpening import unsharp_mask
        from downscaling import lanczos_resize, contrast_downscale
    except ImportError as direct_e:
         print(f"Error: Direct imports also failed in pixelation.py. Error: {direct_e}")
         # Re-raise or define dummy functions if needed for basic script execution
         raise ImportError("Could not import sharpening/downscaling modules.")


# --- Logger Setup ---
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3.Pixelation")
# Default level, will be adjusted by main node's setup_logging call
logger.setLevel(logging.INFO)
# Add a basic handler if none exists (e.g., when testing standalone)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Avoid duplicate logs in ComfyUI

def pixelate_image(
    image: Image.Image, # Input PIL image
    pixel_size: int, # The size of the square 'pixel' blocks
    aspect_ratio_correction: float, # Correction factor for character aspect ratio
    brightness: float, # Brightness adjustment factor (1.0 = no change)
    contrast: float, # Contrast adjustment factor (1.0 = no change)
    device: torch.device, # PyTorch device (CPU or CUDA)
    sharpen_mode: str = "None", # Sharpening mode ('None', 'unsharp')
    sharpen_amount: float = 1.0, # Sharpening strength
    sharpen_threshold: float = 0.0, # Sharpening threshold
    downscale_mode: str = "nearest" # Downscaling method
    ) -> Image.Image: # Returns the pixelated PIL image
    """
    Applies optional sharpening, brightness/contrast adjustments, and pixelates
    the image using various downscaling methods.
    """
    # --- Input Validation ---
    if image is None:
        raise ValueError("Input PIL image cannot be None")
    if pixel_size < 1:
        raise ValueError("Pixel size must be at least 1")
    if aspect_ratio_correction <= 0:
         logger.warning(f"Aspect ratio correction ({aspect_ratio_correction}) should be positive. Clamping to 0.01.")
         aspect_ratio_correction = 0.01

    logger.info(f"Pixelation process started: downscale_mode='{downscale_mode}', sharpen_mode='{sharpen_mode}'")

    # --- 1. Convert PIL to Tensor ---
    # Ensure image is RGB, convert to NumPy array [0, 1], then to Tensor [B, C, H, W]
    try:
        # Ensure RGB format before converting
        image_rgb = image.convert('RGB')
        image_np = np.array(image_rgb).astype(np.float32) / 255.0 # H, W, C
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0) # B, C, H, W
        image_tensor = image_tensor.to(device) # Move tensor to the designated device
        original_height, original_width = image_tensor.shape[2], image_tensor.shape[3]
        logger.debug(f"Input image converted to tensor, shape: {image_tensor.shape} on device: {device}")
    except Exception as e:
        logger.error(f"Failed to convert PIL image to tensor: {e}", exc_info=True)
        raise RuntimeError("Image to Tensor conversion failed.") from e

    # --- 2. Optional Sharpening ---
    # Apply sharpening before other adjustments if requested
    if sharpen_mode == "unsharp":
        # Basic check for sharpening amount
        if sharpen_amount > 0:
            try:
                logger.debug(f"Applying unsharp mask: amount={sharpen_amount}, threshold={sharpen_threshold}")
                # Call the imported unsharp_mask function
                # Using default kernel_size=5, sigma=1.0 as reasonable defaults
                image_tensor = unsharp_mask(
                    image_tensor,
                    kernel_size=5, # Can be made configurable later if needed
                    sigma=1.0,     # Can be made configurable later if needed
                    amount=sharpen_amount,
                    threshold=sharpen_threshold
                )
                logger.debug("Unsharp mask applied successfully.")
            except Exception as e:
                logger.error(f"Failed to apply unsharp mask: {e}", exc_info=True)
                # Decide whether to continue without sharpening or raise error
                logger.warning("Continuing pixelation without sharpening due to error.")
        else:
             logger.debug("Sharpen amount is 0, skipping unsharp mask.")
    elif sharpen_mode != "None":
        # Warn if an unsupported mode is selected
        logger.warning(f"Unsupported sharpen_mode '{sharpen_mode}', skipping sharpening.")

    # --- 3. Brightness & Contrast Adjustment (on Tensor) ---
    # Applying adjustments directly on the tensor
    try:
        # Apply contrast first: Scale values around the midpoint (0.5)
        # Clamp immediately to prevent values going far out of range [0, 1]
        if contrast != 1.0:
             logger.debug(f"Applying contrast ({contrast}) on tensor")
             # Ensure midpoint subtraction doesn't cause issues with specific dtypes if needed
             image_tensor = torch.clamp((image_tensor - 0.5) * contrast + 0.5, 0.0, 1.0)

        # Apply brightness: Additive shift, then clamp
        if brightness != 1.0:
             logger.debug(f"Applying brightness ({brightness}) on tensor")
             image_tensor = torch.clamp(image_tensor + (brightness - 1.0), 0.0, 1.0)

        logger.debug("Brightness/contrast adjustments applied.")
    except Exception as e:
        logger.error(f"Error applying brightness/contrast on tensor: {e}", exc_info=True)
        # Let the main node handle this failure if needed, but raise here
        raise RuntimeError("Failed to apply brightness/contrast adjustments.") from e


    # --- 4. Downscaling (Pixelation) ---
    logger.debug(f"Pixelating with pixel_size={pixel_size}, aspect_correction={aspect_ratio_correction}, mode='{downscale_mode}'")

    # Calculate target grid size (number of pixel blocks) based on pixel_size and aspect ratio correction
    target_pixelated_width = original_width // pixel_size
    # effective_pixel_height adjusts for non-square character cells
    effective_pixel_height = pixel_size * aspect_ratio_correction
    target_pixelated_height = int(original_height / effective_pixel_height)

    # Ensure calculated dimensions are at least 1x1
    target_pixelated_width = max(1, target_pixelated_width)
    target_pixelated_height = max(1, target_pixelated_height)
    # Target size tuple in (Height, Width) format for PyTorch functions
    target_size_hw: Tuple[int, int] = (target_pixelated_height, target_pixelated_width)
    logger.debug(f"Original dims (WxH): ({original_width}, {original_height}), Target pixelated grid dims (HxW): {target_size_hw}")

    try:
        pixelated_tensor = None # Initialize variable
        # Use autocast for potential mixed-precision performance gains on CUDA
        # Enable based on device type
        use_amp = (device.type == 'cuda')
        with autocast(enabled=use_amp):
            # Select the downscaling method based on the input parameter
            if downscale_mode in ["nearest", "bilinear", "bicubic", "area"]:
                # Use PyTorch's built-in interpolation methods
                logger.debug(f"Using F.interpolate with mode='{downscale_mode}'")
                # 'area' mode is generally good for downsampling
                # 'nearest' gives the classic blocky pixel look
                # 'nearest-exact' can be better than 'nearest' if available
                if downscale_mode == "nearest":
                     try:
                         # Try using nearest-exact for potentially better alignment
                         pixelated_tensor = F.interpolate(image_tensor, size=target_size_hw, mode='nearest-exact')
                     except (NotImplementedError, TypeError): # Catch potential errors if mode not supported
                         logger.warning("`nearest-exact` mode not supported or failed, using standard `nearest`.")
                         pixelated_tensor = F.interpolate(image_tensor, size=target_size_hw, mode='nearest')
                else:
                    # Use align_corners=False for area, bilinear, bicubic unless specific behavior is needed
                    align_corners = False if downscale_mode != 'area' else None # area doesn't use align_corners
                    pixelated_tensor = F.interpolate(image_tensor, size=target_size_hw, mode=downscale_mode, align_corners=align_corners)

            elif downscale_mode == "lanczos":
                # Use the custom Lanczos resize function
                logger.debug("Using Lanczos resize")
                pixelated_tensor = lanczos_resize(image_tensor, size=target_size_hw)

            elif downscale_mode == "contrast":
                # Use the contrast-based downscaling method
                logger.debug("Using Contrast-based downscale")
                # This method uses patch_size, which corresponds to our pixel_size
                pixelated_tensor = contrast_downscale(image_tensor, patch_size=pixel_size)
                # Important: Check if the output size matches the calculated target size
                # contrast_downscale might handle padding differently. Resize if needed.
                if pixelated_tensor.shape[2:] != target_size_hw:
                     logger.warning(f"Contrast downscale output size {pixelated_tensor.shape[2:]} differs from target {target_size_hw}. Resizing to target using bilinear.")
                     # Resize to the calculated target size using a simple, fast method
                     pixelated_tensor = F.interpolate(pixelated_tensor, size=target_size_hw, mode='bilinear', align_corners=False)

            else:
                # Fallback for unknown modes
                logger.error(f"Unsupported downscale_mode: '{downscale_mode}'. Falling back to 'nearest'.")
                pixelated_tensor = F.interpolate(image_tensor, size=target_size_hw, mode='nearest')

        # Log the shape after downscaling
        logger.debug(f"Downscaled tensor shape: {pixelated_tensor.shape}")

        # --- 5. Convert Tensor back to PIL ---
        # Permute back to H, W, C; move to CPU; denormalize [0, 1] -> [0, 255]; convert type
        pixelated_np = pixelated_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        # Clip values just in case any operation went slightly out of [0, 1] bounds
        pixelated_np = np.clip(pixelated_np * 255.0, 0, 255).astype(np.uint8)
        # Create PIL Image from the NumPy array
        pixelated_pil = Image.fromarray(pixelated_np, mode='RGB')
        logger.info(f"Pixelation complete, output PIL image size: {pixelated_pil.size}")
        return pixelated_pil

    except ImportError as e:
         # Specifically catch ImportError if kornia is needed but missing
         logger.error(f"ImportError during pixelation (likely kornia missing for 'contrast' mode): {e}")
         # Re-raise the specific error to be caught by the main node
         raise ImportError("Missing required library (likely kornia for 'contrast' mode). Please install it.") from e
    except Exception as e:
        # Catch any other errors during the downscaling process
        logger.error(f"Error during downscaling/pixelation step: {e}", exc_info=True)
        # Re-raise a generic error
        raise RuntimeError("Pixelation failed.") from e
