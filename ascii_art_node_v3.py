# ascii_art_node_v3.py (Main Node File - Modified for Batch Processing and Dynamic Charsets)
import os
import random
import logging
import numpy as np
import torch
from typing import Optional, Tuple, List
from PIL import Image

# --- ComfyUI Specific Imports ---
try:
    from folder_paths import get_filename_list, get_full_path
except ImportError:
    print("Warning: ComfyUI folder_paths not found. Using dummy functions.")
    # Dummy functions as provided in the original script
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
    from .colormatch import apply_color_match
    from .charset_generator import generate_dynamic_charset ### ADDED ###
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
        from colormatch import apply_color_match
        from charset_generator import generate_dynamic_charset ### ADDED ###
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
    Includes options for sharpening, various pixelation methods, color matching,
    batch processing, and dynamic charset generation.
    """
    DOWNSCALE_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos", "contrast"]
    SHARPEN_MODES = ["None", "unsharp"]
    COLOR_MATCH_METHODS = ['mkl', 'hm', 'reinhard', 'idt', 'hm-mkl-hm']

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
                "char_selection_mode": (["brightness", "hue", "saturation", "luminance_hue"], {"default": "luminance_hue"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                ### ADDED: Charset generation options ###
                "charset_source": (["File", "Dynamic"], {"default": "File"}),
                "ascii_chars_filename": ("STRING", {"default": "set4.txt"}),
                "dynamic_chars_to_test": ("STRING", {
                    "multiline": True,
                    "default": R"""!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ """
                }),
                "dynamic_sigma": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                ### END ADDED ###
                "sharpen_mode": (cls.SHARPEN_MODES, {"default": "None"}),
                "sharpen_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "sharpen_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "mask": ("MASK",),
                "mask_blend_radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "mask_edge_adjustment": (["None", "SmallerChars", "LowerDensity"], {"default": "None"}),
                "mask_edge_factor": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "enable_color_match": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "color_match_method": (cls.COLOR_MATCH_METHODS, {"default": "mkl"}),
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
                           char_selection_mode: str,
                           seed: int,
                           # Optional parameters
                           charset_source: str = "File", ### ADDED ###
                           ascii_chars_filename: str = "set4.txt",
                           dynamic_chars_to_test: str = "", ### ADDED ###
                           dynamic_sigma: float = 1.5, ### ADDED ###
                           sharpen_mode: str = "None",
                           sharpen_amount: float = 1.0,
                           sharpen_threshold: float = 0.0,
                           brightness: float = 1.0,
                           contrast: float = 1.0,
                           mask: Optional[torch.Tensor] = None,
                           mask_blend_radius: float = 0.0,
                           mask_edge_adjustment: str = "None",
                           mask_edge_factor: float = 1.0,
                           enable_color_match: bool = False,
                           color_match_method: str = 'mkl',
                           log_level: str = "INFO"):
        """Main function called by ComfyUI to generate the ASCII art image for a batch."""
        # --- 1. Setup Phase (Done Once) ---
        setup_logging(log_level, "ComfyUI.ASCIIArtNodeV3")
        logger.info(f"Starting ASCII Art Generation V3 for {image.shape[0]} image(s)")
        # Log key parameters
        logger.debug(f"--- Input Parameters ---")
        logger.debug(f"  Pixel Size: {pixel_size}, Downscale Mode: {downscale_mode}")
        logger.debug(f"  Aspect Correction: {aspect_ratio_correction}")
        logger.debug(f"  Font: {font_name}, Size Range: ({font_size_min}-{font_size_max})")
        ### MODIFIED: Logging for charset source ###
        logger.debug(f"  Charset Source: {charset_source}")
        if charset_source == 'File':
            logger.debug(f"    File: {ascii_chars_filename}")
        else:
            logger.debug(f"    Sigma: {dynamic_sigma}, Chars to Test: {len(dynamic_chars_to_test)}")
        logger.debug(f"  Char Selection Mode: {char_selection_mode}")
        ### END MODIFIED ###
        logger.debug(f"  Sharpen: {sharpen_mode} (Amount: {sharpen_amount}, Threshold: {sharpen_threshold})")
        logger.debug(f"  Brightness: {brightness}, Contrast: {contrast}")
        logger.debug(f"  Seed: {seed}")
        logger.debug(f"  Mask Present: {mask is not None}, Blend Radius: {mask_blend_radius}")
        logger.debug(f"  Mask Edge Adjust: {mask_edge_adjustment}, Edge Factor: {mask_edge_factor}")
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
        else:
            device = torch.device("cpu")
        logger.debug(f"Using device: {device}")

        # --- 2. Load Resources (Done Once) ---
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

        ### MODIFIED: Load or generate ASCII characters based on source ###
        ascii_sets = []
        try:
            if charset_source == "File":
                logger.info(f"Loading character sets from file: {ascii_chars_filename}")
                base_dir = os.path.dirname(__file__)
                ascii_chars_file_path = os.path.join(base_dir, ascii_chars_filename)
                if not os.path.isfile(ascii_chars_file_path):
                   raise FileNotFoundError(f"ASCII chars file not found at '{ascii_chars_file_path}'.")
                ascii_sets = load_custom_characters(ascii_chars_file_path)
                logger.debug(f"Loaded {len(ascii_sets)} ASCII character sets from file.")
            
            elif charset_source == "Dynamic":
                logger.info("Dynamically generating character set...")
                # Use font_size_max for higher precision in density calculation
                generated_set = generate_dynamic_charset(
                    font_path=font_path,
                    font_size=font_size_max,
                    characters_to_test=dynamic_chars_to_test,
                    sigma=dynamic_sigma
                )
                if generated_set:
                    # Dynamic generation produces a single set
                    ascii_sets = [generated_set]
                logger.debug(f"Dynamically generated a character set with {len(generated_set)} characters.")

            if not ascii_sets:
                raise ValueError("No valid character sets were loaded or generated.")
        
        except Exception as e:
            logger.error(f"Failed to prepare ASCII characters: {e}", exc_info=True)
            raise ValueError(f"Could not prepare ASCII characters. Check source ('{charset_source}') and parameters.") from e
        ### END MODIFIED ###

        # --- 3. Process Input Tensor to PIL List ---
        try:
            pil_images = tensor_to_pil(image)
            batch_size = len(pil_images)
            if batch_size == 0:
                raise ValueError("tensor_to_pil returned an empty list.")
            logger.info(f"Converted input tensor to a list of {batch_size} PIL images.")
        except Exception as e:
            logger.error(f"Failed to convert input image tensor to PIL list: {e}", exc_info=True)
            return (torch.zeros_like(image),)

        # --- 4. Process Mask (Once, applied to all images in batch) ---
        mask_pil = None
        if mask is not None:
            try:
                original_size = pil_images[0].size
                logger.info("Processing mask input (using first mask for the entire batch)...")
                mask_pil = mask_tensor_to_pil(mask, original_size)
                logger.debug(f"Mask converted to PIL, size: {mask_pil.size}")
            except Exception as e:
                logger.error(f"Error processing mask: {e}", exc_info=True)
                logger.warning("Disabling mask features for the batch due to processing error.")
                mask = None
                mask_pil = None

        # --- 5. Per-Image Processing Loop ---
        final_images_list = []

        for i, pil_image in enumerate(pil_images):
            logger.info(f"--- Processing Batch Image {i+1}/{batch_size} ---")
            current_original_size = pil_image.size
            logger.debug(f"  Input PIL size: {current_original_size}")

            # --- 5.1 Seeding and Character Set Selection ---
            random.seed(seed)
            np_seed = (seed + i) % (2**32)
            np.random.seed(np_seed)
            logger.debug(f"  Seeding for image {i+1}: Python random with {seed}, NumPy random with {np_seed}")
            
            ### MODIFIED: Character set selection logic ###
            if charset_source == "File":
                # Randomly choose one of the sets from the file
                chosen_set = random.choice(ascii_sets)
                logger.debug(f"  Randomly selected a character set from file (length {len(chosen_set)}).")
            else: # Dynamic source
                # Use the single, dynamically generated set
                chosen_set = ascii_sets[0]
                logger.debug(f"  Using the dynamically generated character set (length {len(chosen_set)}).")
            logger.debug(f"  Using set: {chosen_set[:40]}...")
            ### END MODIFIED ###
            
            # ... (The rest of the processing loop remains the same) ...

            # --- 5.2 Image Pixelation Step ---
            try:
                logger.info(f"  Pixelating image {i+1} using mode: '{downscale_mode}'...")
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
                logger.debug(f"  Pixelated image {i+1} generated, size: {pixelated_image.size}")
                if pixelated_image.size[0] == 0 or pixelated_image.size[1] == 0:
                     raise ValueError("Pixelation resulted in a zero-dimension image. Check pixel_size and aspect ratio.")
            except ImportError as e:
                 logger.error(f"ImportError during pixelation for image {i+1}: {e}. 'contrast' mode requires 'kornia'.")
                 raise ImportError("Pixelation failed. The 'contrast' downscale mode requires the 'kornia' library. Please install it (`pip install kornia`) or choose a different downscale mode.") from e
            except Exception as e:
                logger.error(f"Error during pixelation step for image {i+1}: {e}", exc_info=True)
                raise ValueError(f"Pixelation process failed for image {i+1}.") from e

            # --- 5.3 Color Matching Step (Conditional) ---
            image_for_ascii_drawing = pixelated_image
            if enable_color_match:
                logger.info(f"  Applying color matching to image {i+1} (method: {color_match_method})...")
                try:
                    matched_pixelated_image = apply_color_match(
                        source_image_pil=pil_image,
                        target_image_pil=pixelated_image,
                        method=color_match_method
                    )
                    if matched_pixelated_image is not pixelated_image:
                        logger.info(f"  Color matching applied successfully for image {i+1}.")
                        image_for_ascii_drawing = matched_pixelated_image
                        logger.debug(f"  Using color-matched image for ASCII drawing, size: {image_for_ascii_drawing.size}")
                    else:
                        logger.warning(f"  Color matching did not modify image {i+1} (skipped/error). Using original pixelated image.")
                except Exception as cm_e:
                    logger.error(f"Unexpected error calling apply_color_match for image {i+1}: {cm_e}", exc_info=True)
                    logger.warning(f"  Proceeding with original pixelated image {i+1} due to color matching error.")
            else:
                logger.info(f"  Color matching is disabled for image {i+1}, using original pixelated colors.")

            # --- 5.4 Calculate Edge Info (if mask enabled) ---
            current_edge_info_np = None
            if mask_pil is not None and mask_edge_adjustment != "None":
                try:
                    target_edge_info_size_wh = image_for_ascii_drawing.size
                    logger.debug(f"  Calculating edge info for image {i+1} using target size {target_edge_info_size_wh}")
                    current_edge_info_np = calculate_edge_info(mask_pil, mask_edge_factor, target_edge_info_size_wh)
                    if current_edge_info_np is None:
                        logger.warning(f"  Failed to calculate edge info for image {i+1}, disabling mask edge adjustment for this image.")
                    else:
                         logger.debug(f"  Edge info calculated for image {i+1}, shape: {current_edge_info_np.shape}")
                except Exception as e:
                     logger.error(f"  Error calculating edge info for image {i+1}: {e}", exc_info=True)
                     logger.warning("  Disabling mask edge adjustment for this image due to error.")
                     current_edge_info_np = None

            # --- 5.5 ASCII Art Generation Step ---
            try:
                logger.info(f"  Generating ASCII art representation for image {i+1}...")
                ascii_image = create_ascii_art(
                    image=image_for_ascii_drawing,
                    ascii_chars=chosen_set,
                    font_path=font_path,
                    font_size_min=font_size_min,
                    font_size_max=font_size_max,
                    original_size=current_original_size,
                    char_selection_mode=char_selection_mode,
                    mask_edge_adjustment=mask_edge_adjustment,
                    edge_info=current_edge_info_np
                )
                logger.debug(f"  Generated ASCII art PIL image {i+1}, size: {ascii_image.size}")
            except Exception as e:
                logger.error(f"Error during ASCII art creation for image {i+1}: {e}", exc_info=True)
                raise ValueError(f"Failed to create ASCII art representation for image {i+1}.") from e

            # --- 5.6 Mask Application and Blending (Conditional) ---
            final_image_for_batch = None
            if mask is not None and mask_pil is not None:
                try:
                    logger.info(f"  Applying mask blending for image {i+1}...")
                    final_image_for_batch = apply_mask_blending(
                        original_pil=pil_image,
                        overlay_pil=ascii_image,
                        mask_pil=mask_pil,
                        mask_blend_radius=mask_blend_radius
                    )
                    logger.debug(f"  Mask blending applied for image {i+1}.")
                except Exception as e:
                    logger.error(f"Error applying mask blending for image {i+1}: {e}", exc_info=True)
                    logger.warning(f"  Mask blending failed for image {i+1}, using unmasked ASCII art as fallback.")
                    final_image_for_batch = ascii_image
            else:
                final_image_for_batch = ascii_image
                logger.info(f"  No mask applied or mask processing failed for image {i+1}.")

            # --- 5.7 Add processed image to list ---
            if final_image_for_batch:
                 final_images_list.append(final_image_for_batch)
            else:
                 raise RuntimeError(f"Processing failed to produce a final image for batch item {i+1}.")


        # --- 6. Output Conversion (After Loop) ---
        if not final_images_list:
            logger.error("No images were successfully processed in the batch.")
            return (torch.zeros_like(image),)

        try:
            output_tensor = pil_to_tensor(final_images_list)
            logger.debug(f"Final batch of {len(final_images_list)} images converted back to tensor, shape: {output_tensor.shape}")
            logger.info("ASCII Art Generation V3 Finished Successfully.")
        except Exception as e:
             logger.error(f"Failed to convert final PIL image list to tensor: {e}", exc_info=True)
             return (torch.zeros_like(image),)

        return (output_tensor,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "ASCIIArtNodeV3": ASCIIArtNodeV3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ASCIIArtNodeV3": "ASCII Art Generator V3" ### MODIFIED: Simplified name ###
}
