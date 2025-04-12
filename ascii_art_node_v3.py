# ascii_art_node_v3.py (Main Node File - Updated with Color Matching)
import os
import random
import logging
import numpy as np
import torch
from typing import Optional, Tuple

# --- ComfyUI Specific Imports ---
try:
    from folder_paths import get_filename_list, get_full_path
except ImportError:
    print("Warning: ComfyUI folder_paths not found. Using dummy functions.")
    def get_filename_list(dir_name):
        try:
            if dir_name == "font":
                return [f for f in os.listdir('.') if f.lower().endswith(('.ttf', '.otf'))] or ["dummy_font.ttf"]
            else:
                return [f"dummy_{dir_name}_1.txt"]
        except Exception:
            return [f"dummy_{dir_name}_error.txt"]
    def get_full_path(dir_name, filename):
        return os.path.abspath(filename)

# --- Module Imports ---
try:
    from .ascii_utils import (setup_logging, tensor_to_pil, mask_tensor_to_pil,
                             pil_to_tensor, load_custom_characters,
                             calculate_edge_info, apply_mask_blending)
    from .pixelation import pixelate_image
    from .ascii_drawing import create_ascii_art
    # <<< Added Import for Color Matching >>>
    from .colormatch import apply_color_match
except ImportError as e:
    print(f"Warning: Relative imports failed in main node. Trying direct imports. Error: {e}")
    import sys
    sys.path.append(os.path.dirname(__file__))
    try:
        from ascii_utils import (setup_logging, tensor_to_pil, mask_tensor_to_pil,
                                 pil_to_tensor, load_custom_characters,
                                 calculate_edge_info, apply_mask_blending)
        from pixelation import pixelate_image
        from ascii_drawing import create_ascii_art
        # <<< Added Import for Color Matching (Fallback) >>>
        from colormatch import apply_color_match
    except ImportError as direct_e:
         print(f"Error: Direct imports also failed. Ensure modules are in the correct path. Error: {direct_e}")
         raise ImportError("Could not import necessary modules. Check file structure and paths.") from e


# --- Logger Setup ---
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


class ASCIIArtNodeV3:
    """
    Custom ComfyUI node for generating colorful ASCII art.
    Includes options for sharpening, various pixelation methods, and color matching.
    """
    DOWNSCALE_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos", "contrast"]
    SHARPEN_MODES = ["None", "unsharp"]
    # <<< Added Color Match Methods >>>
    COLOR_MATCH_METHODS = ['mkl', 'hm', 'reinhard', 'idt', 'hm-mkl-hm'] # Common methods from color-matcher

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input parameters for the ComfyUI node interface."""
        try:
            font_list = get_filename_list("font")
            if not font_list:
                 logger.warning("No fonts found in ComfyUI's fonts directory. Please add .ttf or .otf files.")
                 font_list = ["font_not_found.ttf"]
        except Exception as e:
            logger.error(f"Could not list fonts from ComfyUI directory: {e}", exc_info=True)
            font_list = ["error_loading_font.ttf"]

        inputs = {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "downscale_mode": (cls.DOWNSCALE_MODES, {"default": "area"}),
                "aspect_ratio_correction": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 10.0, "step": 0.05}),
                "font_name": (font_list, ),
                "font_size_min": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "font_size_max": ("INT", {"default": 16, "min": 1, "max": 200, "step": 1}),
                "ascii_chars_filename": ("STRING", {"default": "set4.txt"}),
                "char_selection_mode": (["brightness", "hue", "saturation", "luminance_hue"], {"default": "luminance_hue"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "sharpen_mode": (cls.SHARPEN_MODES, {"default": "None"}),
                "sharpen_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "sharpen_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "mask": ("MASK",),
                "mask_blend_radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "mask_edge_adjustment": (["None", "SmallerChars", "LowerDensity"], {"default": "None"}),
                "mask_edge_factor": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                # <<< Added Color Match Inputs >>>
                "enable_color_match": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "color_match_method": (cls.COLOR_MATCH_METHODS, {"default": "mkl"}),
                # <<< --- >>>
                "log_level": (["DEBUG", "INFO", "WARNING", "ERROR", "NONE"], {"default": "INFO"}),
            }
        }
        return inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_ascii_art"
    CATEGORY = "Image Processing/ASCII Art"

    def generate_ascii_art(self,
                           image: torch.Tensor,
                           pixel_size: int,
                           downscale_mode: str,
                           aspect_ratio_correction: float,
                           font_name: str,
                           font_size_min: int,
                           font_size_max: int,
                           ascii_chars_filename: str,
                           char_selection_mode: str,
                           seed: int,
                           # Optional parameters
                           sharpen_mode: str = "None",
                           sharpen_amount: float = 1.0,
                           sharpen_threshold: float = 0.0,
                           brightness: float = 1.0,
                           contrast: float = 1.0,
                           mask: Optional[torch.Tensor] = None,
                           mask_blend_radius: float = 0.0,
                           mask_edge_adjustment: str = "None",
                           mask_edge_factor: float = 1.0,
                           # <<< Added Color Match Parameters >>>
                           enable_color_match: bool = False,
                           color_match_method: str = 'mkl',
                           # <<< --- >>>
                           log_level: str = "INFO"):
        """Main function called by ComfyUI to generate the ASCII art image."""
        # --- 1. Setup Phase ---
        setup_logging(log_level, "ComfyUI.ASCIIArtNodeV3")
        logger.info("Starting ASCII Art Generation V3 (Refactored w/ Options + ColorMatch)")
        # Log key parameters
        logger.debug(f"--- Input Parameters ---")
        logger.debug(f"  Pixel Size: {pixel_size}, Downscale Mode: {downscale_mode}")
        logger.debug(f"  Aspect Correction: {aspect_ratio_correction}")
        logger.debug(f"  Font: {font_name}, Size Range: ({font_size_min}-{font_size_max})")
        logger.debug(f"  Chars File: {ascii_chars_filename}, Char Mode: {char_selection_mode}")
        logger.debug(f"  Sharpen: {sharpen_mode} (Amount: {sharpen_amount}, Threshold: {sharpen_threshold})")
        logger.debug(f"  Brightness: {brightness}, Contrast: {contrast}")
        logger.debug(f"  Seed: {seed}")
        logger.debug(f"  Mask Present: {mask is not None}, Blend Radius: {mask_blend_radius}")
        logger.debug(f"  Mask Edge Adjust: {mask_edge_adjustment}, Edge Factor: {mask_edge_factor}")
        # <<< Log Color Match Parameters >>>
        logger.debug(f"  Color Match Enabled: {enable_color_match}, Method: {color_match_method}")
        logger.debug(f"------------------------")

        # Validate font sizes
        if font_size_min > font_size_max:
            logger.warning(f"font_size_min ({font_size_min}) > font_size_max ({font_size_max}). Swapping them.")
            font_size_min, font_size_max = font_size_max, font_size_min
        font_size_min = max(1, font_size_min)
        font_size_max = max(font_size_min, font_size_max)

        # Determine processing device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.debug(f"Using device: {device}")

        # --- 2. Input Processing Phase ---
        try:
            pil_image = tensor_to_pil(image)
            original_size = pil_image.size
            logger.debug(f"Input image converted to PIL, size: {original_size}")
        except Exception as e:
            logger.error(f"Failed to convert input image tensor to PIL: {e}", exc_info=True)
            return (torch.zeros_like(image),) # Return blank tensor on failure

        # Locate font file
        try:
            font_path = get_full_path("font", font_name)
            if not font_path or not os.path.isfile(font_path):
                 node_dir = os.path.dirname(__file__)
                 potential_path = os.path.join(node_dir, font_name)
                 if os.path.isfile(potential_path):
                     font_path = potential_path
                     logger.warning(f"Font '{font_name}' not found via ComfyUI paths, using relative path: {font_path}")
                 else:
                    raise FileNotFoundError(f"Font file '{font_name}' not found in ComfyUI/models/font or relative to the node.")
            logger.debug(f"Using font file: {font_path}")
        except Exception as e:
            logger.error(f"Error resolving font path for '{font_name}': {e}", exc_info=True)
            raise ValueError(f"Could not load font '{font_name}'. Check name and ensure it's in the 'ComfyUI/models/font' directory.") from e

        # Load ASCII characters
        try:
            base_dir = os.path.dirname(__file__)
            ascii_chars_file_path = os.path.join(base_dir, ascii_chars_filename)
            if not os.path.isfile(ascii_chars_file_path):
                 logger.warning(f"ASCII chars file '{ascii_chars_filename}' not found relative to node. Ensure it exists.")
                 if not os.path.isfile(ascii_chars_file_path):
                   raise FileNotFoundError(f"ASCII chars file not found at '{ascii_chars_file_path}'.")

            ascii_sets = load_custom_characters(ascii_chars_file_path)
            logger.debug(f"Loaded {len(ascii_sets)} ASCII character sets from: {ascii_chars_file_path}")
            if not ascii_sets:
                 raise ValueError("No valid character sets found in the loaded file.")
        except Exception as e:
            logger.error(f"Failed to load ASCII characters from '{ascii_chars_filename}': {e}", exc_info=True)
            raise ValueError(f"Could not load or parse ASCII characters file '{ascii_chars_filename}'. Check path and format.") from e

        # --- 3. Seeding and Randomization ---
        random.seed(seed)
        np_seed = seed % (2**32)
        np.random.seed(np_seed)
        # torch.manual_seed(seed)
        # if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
        logger.debug(f"Seeding Python random with {seed}, NumPy random with {np_seed}")
        chosen_set = random.choice(ascii_sets)
        logger.debug(f"Selected character set (length {len(chosen_set)}): {chosen_set[:30]}...")

        # --- 4. Image Pixelation Step ---
        try:
            logger.info(f"Pixelating image using mode: '{downscale_mode}'...")
            pixelated_image = pixelate_image(
                image=pil_image,
                pixel_size=pixel_size,
                aspect_ratio_correction=aspect_ratio_correction,
                brightness=brightness,
                contrast=contrast,
                device=device,
                sharpen_mode=sharpen_mode,
                sharpen_amount=sharpen_amount,
                sharpen_threshold=sharpen_threshold,
                downscale_mode=downscale_mode
            )
            logger.debug(f"Pixelated image generated, size: {pixelated_image.size}")
            if pixelated_image.size[0] == 0 or pixelated_image.size[1] == 0:
                 raise ValueError("Pixelation resulted in a zero-dimension image. Check pixel_size and aspect ratio.")
        except ImportError as e:
             logger.error(f"ImportError during pixelation: {e}. 'contrast' mode requires 'kornia'.")
             raise ImportError("Pixelation failed. The 'contrast' downscale mode requires the 'kornia' library. Please install it (`pip install kornia`) or choose a different downscale mode.") from e
        except Exception as e:
            logger.error(f"Error during pixelation step: {e}", exc_info=True)
            raise ValueError("Pixelation process failed.") from e

        # --- 4.5. Color Matching Step (Conditional) <<< NEW STEP >>> ---
        image_for_ascii_drawing = pixelated_image # Default: use the direct pixelated image
        if enable_color_match:
            logger.info(f"Applying color matching (method: {color_match_method})...")
            try:
                # Call the color match function from colormatch module
                # Pass original PIL image as source, pixelated PIL image as target
                matched_pixelated_image = apply_color_match(
                    source_image_pil=pil_image,
                    target_image_pil=pixelated_image,
                    method=color_match_method
                )
                # Check if matching actually returned a modified image (apply_color_match returns target on error)
                if matched_pixelated_image is not pixelated_image:
                    logger.info("Color matching applied successfully.")
                    image_for_ascii_drawing = matched_pixelated_image # Use matched image for drawing
                    logger.debug(f"Using color-matched image for ASCII drawing, size: {image_for_ascii_drawing.size}")
                else:
                    # apply_color_match returned the original target, likely due to error or missing library
                    logger.warning("Color matching did not modify the image (possibly skipped due to error/missing library). Using original pixelated image.")
                    # image_for_ascii_drawing remains pixelated_image
            except Exception as cm_e:
                # Catch any unexpected errors during the call itself
                logger.error(f"Unexpected error calling apply_color_match: {cm_e}", exc_info=True)
                logger.warning("Proceeding with original pixelated image due to color matching error.")
                # image_for_ascii_drawing remains pixelated_image
        else:
            logger.info("Color matching is disabled, using original pixelated image colors.")

        # --- 5. Mask Processing (Conditional) ---
        mask_pil = None
        edge_info_np = None
        if mask is not None:
            try:
                logger.info("Processing mask input...")
                mask_pil = mask_tensor_to_pil(mask, original_size)
                logger.debug(f"Mask converted to PIL, size: {mask_pil.size}")

                if mask_edge_adjustment != "None":
                    logger.debug(f"Calculating edge info for mask adjustment: mode='{mask_edge_adjustment}', factor={mask_edge_factor}")
                    # Use the size of the image that will be used for ASCII drawing as target
                    target_edge_info_size_wh = image_for_ascii_drawing.size
                    edge_info_np = calculate_edge_info(mask_pil, mask_edge_factor, target_edge_info_size_wh)
                    if edge_info_np is None:
                        logger.warning("Failed to calculate edge info, disabling mask edge adjustment.")
                        mask_edge_adjustment = "None"
                    else:
                         logger.debug(f"Edge info calculated, shape: {edge_info_np.shape}")
            except Exception as e:
                logger.error(f"Error processing mask or calculating edge info: {e}", exc_info=True)
                logger.warning("Disabling mask features due to processing error.")
                mask = None
                mask_pil = None
                edge_info_np = None
                mask_edge_adjustment = "None"

        # --- 6. ASCII Art Generation Step ---
        try:
            logger.info("Generating ASCII art representation...")
            # <<< Pass the potentially color-matched image >>>
            ascii_image = create_ascii_art(
                image=image_for_ascii_drawing, # Use the (potentially) color-matched image
                ascii_chars=chosen_set,
                font_path=font_path,
                font_size_min=font_size_min,
                font_size_max=font_size_max,
                original_size=original_size,
                char_selection_mode=char_selection_mode,
                mask_edge_adjustment=mask_edge_adjustment,
                edge_info=edge_info_np
            )
            logger.debug(f"Generated ASCII art PIL image, size: {ascii_image.size}")
        except Exception as e:
            logger.error(f"Error during ASCII art creation: {e}", exc_info=True)
            raise ValueError("Failed to create ASCII art representation.") from e

        # --- 7. Mask Application and Blending (Conditional) ---
        if mask is not None and mask_pil is not None:
            try:
                logger.info("Applying mask blending...")
                final_image = apply_mask_blending(
                    original_pil=pil_image,
                    overlay_pil=ascii_image,
                    mask_pil=mask_pil,
                    mask_blend_radius=mask_blend_radius
                )
                logger.debug("Mask blending applied.")
            except Exception as e:
                logger.error(f"Error applying mask blending: {e}", exc_info=True)
                logger.warning("Mask blending failed, returning unmasked ASCII art as fallback.")
                final_image = ascii_image # Fallback to unmasked ASCII art
        else:
            final_image = ascii_image # No mask, use the full ASCII art
            logger.info("No mask applied or mask processing failed, using full ASCII art.")

        # --- 8. Output Conversion ---
        try:
            output_tensor = pil_to_tensor(final_image)
            logger.debug(f"Final image converted back to tensor, shape: {output_tensor.shape}")
            logger.info("ASCII Art Generation V3 (Refactored w/ Options + ColorMatch) Finished Successfully.")
        except Exception as e:
             logger.error(f"Failed to convert final PIL image to tensor: {e}", exc_info=True)
             return (torch.zeros_like(image),) # Return blank tensor on failure

        # Return the result as a tuple
        return (output_tensor,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "ASCIIArtNodeV3": ASCIIArtNodeV3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ASCIIArtNodeV3": "ASCII Art Generator V3 (Ref+Opts+CM)" # Updated display name
}