# ascii_drawing.py (Modified for HSV Character Selection + Dark Area Font Size)
import os
import random
import logging
from typing import List, Optional, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Logger Setup ---
# This module's logger (configured by the main node)
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3.Drawing")
# Set a default level; will be overridden by the main node's setup
logger.setLevel(logging.INFO)
# Add a basic handler if none exists (useful for standalone testing)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Avoid duplicate logs

# --- Font Caching ---
# Module-level cache for loaded font objects
_font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
# Stores which sizes have been successfully cached for a given font path
_font_cache_details: Dict[str, List[int]] = {}

def _get_font(font_path: str, size: int) -> Optional[ImageFont.FreeTypeFont]:
    """
    Retrieves a font of a specific size from the cache or loads and caches it.
    Returns the font object or None if loading fails.
    """
    size = max(1, int(size)) # Ensure size is at least 1
    key = (font_path, size)

    if key in _font_cache:
        return _font_cache[key]
    else:
        try:
            if not os.path.isfile(font_path):
                 logger.error(f"Font file not found at path: {font_path}")
                 return None

            font = ImageFont.truetype(font_path, size)
            _font_cache[key] = font
            if font_path not in _font_cache_details:
                _font_cache_details[font_path] = []
            if size not in _font_cache_details[font_path]:
                _font_cache_details[font_path].append(size)
            logger.debug(f"Cached font: '{os.path.basename(font_path)}' size {size}")
            return font
        except OSError as e:
            logger.error(f"OS error loading font '{font_path}' at size {size}: {e}", exc_info=False) # exc_info=False for brevity
            return None
        except Exception as e:
            logger.error(f"Failed to load font '{font_path}' at size {size}: {e}", exc_info=True)
            return None

def _get_nearest_cached_font(font_path: str, desired_size: int, min_size: int, max_size: int) -> Optional[ImageFont.FreeTypeFont]:
    """
    Finds the best matching font for the desired size. Tries exact match first,
    then falls back to nearest cached or minimum size.
    """
    clamped_size = max(1, max(min_size, min(max_size, int(desired_size))))

    # 1. Try exact match
    font = _get_font(font_path, clamped_size)
    if font:
        return font

    # 2. Fallback: Find nearest *already cached* size
    logger.warning(f"Could not load font '{os.path.basename(font_path)}' at exact size {clamped_size}. Trying nearest cached size.")
    if font_path in _font_cache_details and _font_cache_details[font_path]:
        cached_sizes = _font_cache_details[font_path]
        # Find the cached size closest to the clamped desired size
        nearest_size = min(cached_sizes, key=lambda s: abs(s - clamped_size))
        logger.debug(f"Found nearest cached size: {nearest_size} for desired {clamped_size}")
        font = _get_font(font_path, nearest_size) # Retrieve from cache/load
        if font:
            return font
        else:
             # This case should ideally not happen if _get_font worked before for this size
             logger.error(f"Failed to retrieve supposedly cached font size {nearest_size} for {font_path}.")

    # 3. Last Resort: Try loading the minimum allowed size
    logger.warning(f"No suitable cached font found for '{os.path.basename(font_path)}' near size {clamped_size}. Attempting to load minimum size {min_size}.")
    font = _get_font(font_path, min_size)
    if font:
        return font

    logger.error(f"Completely failed to load any usable font size for '{font_path}' between {min_size}-{max_size}.")
    return None


def create_ascii_art(
    image: Image.Image, # Input: The *pixelated* PIL image
    ascii_chars: str, # String containing the characters to use
    font_path: str, # Path to the font file
    font_size_min: int, # Minimum font size for mapping
    font_size_max: int, # Maximum font size for mapping
    original_size: Tuple[int, int], # Target output size (Width, Height)
    char_selection_mode: str, # Method to map pixel value to character index
    mask_edge_adjustment: str, # How characters behave near mask edges
    edge_info: Optional[np.ndarray] = None, # Optional edge map
    ) -> Image.Image: # Output: The final ASCII art PIL image
    """
    Creates the final ASCII art image by drawing characters onto a canvas.
    Uses HSV color space components for character selection modes (hue, saturation, etc.).
    Adjusts font size for very dark areas to improve visibility.
    """
    # --- Initial Setup ---
    output_width, output_height = original_size
    logger.debug(f"Creating ASCII art canvas with final size: {output_width}x{output_height}")
    # Initialize with a white background (or could be configurable)
    ascii_image = Image.new('RGB', (output_width, output_height), (255, 255, 255))
    draw = ImageDraw.Draw(ascii_image)

    # --- Font Pre-loading ---
    # Try to load min and max sizes early to catch font errors sooner
    font_min_obj = _get_font(font_path, font_size_min)
    font_max_obj = _get_font(font_path, font_size_max)
    if not font_min_obj or not font_max_obj:
        # If even min/max fail, raise error immediately
        raise ValueError(f"Failed to load required font '{os.path.basename(font_path)}' at min/max sizes ({font_size_min}/{font_size_max}). Cannot create ASCII art.")
    logger.debug(f"Successfully pre-loaded/cached min ({font_size_min}) and max ({font_size_max}) font sizes.")


    # --- Prepare Data from Pixelated Image ---
    pixelated_width, pixelated_height = image.size
    if pixelated_width <= 0 or pixelated_height <= 0:
        logger.error(f"Pixelated image has invalid dimensions: {pixelated_width}x{pixelated_height}. Cannot draw ASCII art.")
        return ascii_image # Return blank canvas

    try:
        # === HSV Conversion and Channel Separation ===
        # Convert the image to HSV *first*
        image_hsv = image.convert('HSV')
        # Then convert to NumPy array, values will be 0-255
        image_np_hsv = np.array(image_hsv)
        # Normalize HSV values to range [0, 1]
        image_np_hsv_float = image_np_hsv.astype(np.float32) / 255.0
        # Separate H, S, V channels
        hue_values = image_np_hsv_float[:, :, 0]        # H channel (0-1)
        saturation_values = image_np_hsv_float[:, :, 1] # S channel (0-1)
        value_values = image_np_hsv_float[:, :, 2]      # V channel (Brightness/Value, 0-1)
        logger.debug("Converted pixelated image to HSV and normalized channels.")

        # Keep RGB version for color picking later (for draw.text fill)
        image_np_rgb = np.array(image.convert('RGB')) # Shape (H', W', C)
        # === End HSV Section ===

    except Exception as e:
        logger.error(f"Failed to convert pixelated image to NumPy array (RGB/HSV): {e}", exc_info=True)
        raise RuntimeError("Could not process pixelated image data.") from e

    # --- Calculate Character Selection Values & Luminance (for font size) ---
    logger.debug(f"Calculating character selection values based on mode: '{char_selection_mode}'")
    selection_values = None # Shape (H', W'), range [0, 1] -> Used for character index
    luminance_values = None # Shape (H', W'), range [0, 1] -> Used for font size logic

    try:
        # === Character Selection Logic ===
        if char_selection_mode == "brightness":
            selection_values = value_values # Use Value from HSV as brightness
            logger.debug("Using V (Value) from HSV for character selection.")
        elif char_selection_mode == "hue":
            selection_values = hue_values
            logger.debug("Using H (Hue) from HSV for character selection.")
        elif char_selection_mode == "saturation":
            selection_values = saturation_values
            logger.debug("Using S (Saturation) from HSV for character selection.")
        elif char_selection_mode == "luminance_hue":
            # Combine Value (Luminance proxy) and Hue. Adjust weights as needed.
            weight_v = 0.7 # Weight for Value/Brightness
            weight_h = 0.3 # Weight for Hue
            selection_values = (value_values * weight_v) + (hue_values * weight_h)
            # Ensure the combined value is still normalized (clip just in case)
            selection_values = np.clip(selection_values, 0.0, 1.0)
            logger.debug(f"Using weighted V ({weight_v}) and H ({weight_h}) from HSV for character selection.")
            # Warning log about fallback removed here
        else: # Fallback for unknown modes
            logger.warning(f"Unknown char_selection_mode '{char_selection_mode}', defaulting to brightness (V from HSV).")
            selection_values = value_values
        # === End Character Selection Logic ===

        # Log min/max of the FINAL selection values used for character index
        if selection_values is not None:
             logger.debug(f"Selection values calculated, shape: {selection_values.shape}, min: {selection_values.min():.3f}, max: {selection_values.max():.3f}")
        else:
             # This should not happen with the logic above, but check just in case
             raise ValueError("Failed to calculate selection_values.")

        # === Luminance Calculation (for font size adjustment) ===
        # Keep standard luminance calculation based on RGB for font sizing logic
        try:
            # Use the previously obtained RGB numpy array
            image_np_rgb_float = image_np_rgb.astype(np.float32) / 255.0
            # Standard Luminance formula (Rec. 601)
            luminance_values = (image_np_rgb_float[:, :, 0] * 0.299 +
                                image_np_rgb_float[:, :, 1] * 0.587 +
                                image_np_rgb_float[:, :, 2] * 0.114)
            luminance_values = np.clip(luminance_values, 0.0, 1.0) # Ensure range [0, 1]
            logger.debug(f"Luminance values (for font size) calculated from RGB, shape: {luminance_values.shape}, min: {luminance_values.min():.3f}, max: {luminance_values.max():.3f}")
        except Exception as lum_e:
             logger.error(f"Failed to calculate luminance from RGB for font size: {lum_e}", exc_info=True)
             # Fallback: Use Value from HSV if RGB luminance calculation fails
             logger.warning("Falling back to using V (Value) from HSV for font size logic due to luminance calculation error.")
             if value_values is not None:
                 luminance_values = value_values # Use value_values as fallback
             else:
                 # Critical fallback if V is also somehow None
                 logger.error("Cannot determine luminance or fallback value for font sizing. Using constant value 0.5.")
                 luminance_values = np.full((pixelated_height, pixelated_width), 0.5, dtype=np.float32)
        # === End Luminance Calculation ===


    except Exception as e:
        logger.error(f"Failed to calculate character selection/luminance values: {e}", exc_info=True)
        raise ValueError("Could not calculate character selection/luminance values.") from e


    # --- Draw Characters Loop ---
    logger.info("Drawing characters onto the canvas...")
    scale_x = output_width / pixelated_width
    scale_y = output_height / pixelated_height
    logger.debug(f"Scaling factors: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")

    num_chars_drawn = 0
    num_chars_skipped_density = 0
    num_chars_skipped_font = 0
    # Track actual min/max sizes used after adjustments
    min_final_size_used, max_final_size_used = font_size_max + 1, font_size_min - 1

    # Validate edge_info shape against the pixelated grid dimensions (H', W')
    edge_info_validated = edge_info
    if edge_info is not None and edge_info.shape != (pixelated_height, pixelated_width):
        logger.error(f"Edge info shape {edge_info.shape} does not match pixelated image shape {(pixelated_height, pixelated_width)}. Disabling edge adjustment.")
        mask_edge_adjustment = "None"
        edge_info_validated = None

    # Iterate through the grid of the *pixelated* image (H', W')
    for y in range(pixelated_height):
        for x in range(pixelated_width):
            # --- Get Data for Current Cell ---
            pixel_color_tuple = tuple(image_np_rgb[y, x]) # Color for the character (from RGB)
            select_val = selection_values[y, x]          # Value to select character (from H, S, V, or combo)
            current_luminance = luminance_values[y, x]   # Luminance for font size logic (from RGB or V fallback)

            # --- Determine Character ---
            # Map selection value [0, 1] to character index [0, len-1]
            char_index = int(select_val * (len(ascii_chars) - 1) + 0.5) # Add 0.5 for rounding
            # Clamp index to valid range
            char_index = max(0, min(len(ascii_chars) - 1, char_index))
            ascii_char = ascii_chars[char_index]

            # --- Font Size Calculation (Including Dark Area Logic) ---
            # Calculate base size using luminance normally
            base_font_size_from_luminance = font_size_min + current_luminance * (font_size_max - font_size_min)

            # Define a threshold for 'dark' colors (adjust 0.0-1.0 as needed)
            darkness_threshold = 0.3 # Example: Below 30% luminance is considered dark

            if current_luminance < darkness_threshold:
                # If the area is very dark, use a larger font size to fill the space
                # Using font_size_max directly for maximum visibility in dark areas.
                base_font_size = font_size_max
                # Optional: log this event only once or periodically to avoid spam
                # logger.debug(f"Dark area ({x},{y}), L={current_luminance:.2f}, using max font size {base_font_size}")
            else:
                # Otherwise, use the size calculated from luminance
                base_font_size = base_font_size_from_luminance

            # Ensure base_font_size is within bounds and integer
            base_font_size = int(np.clip(base_font_size, font_size_min, font_size_max))
            # --- End Font Size Calculation (Base) ---


            # --- Apply Mask Edge Adjustment to Font Size/Density ---
            final_font_size = base_font_size # Start with the potentially adjusted base size
            draw_probability = 1.0          # Probability to draw the character

            if mask_edge_adjustment != "None" and edge_info_validated is not None:
                # edge_info_validated has values [0, 1], 1 means closer to edge
                adjustment_factor = edge_info_validated[y, x]

                if mask_edge_adjustment == "SmallerChars":
                    # Interpolate between min size (at edge) and base size (away from edge)
                    adjusted_size = font_size_min * adjustment_factor + base_font_size * (1.0 - adjustment_factor)
                    # Ensure adjusted size is within bounds [1, base_font_size] and integer
                    final_font_size = max(1, min(base_font_size, int(adjusted_size + 0.5)))
                elif mask_edge_adjustment == "LowerDensity":
                    # Reduce draw probability closer to the edge
                    draw_probability = 1.0 - adjustment_factor

            # --- Draw Decision & Font Retrieval ---
            if random.random() < draw_probability:
                # Try to get the font for the final calculated size
                font = _get_nearest_cached_font(font_path, final_font_size, font_size_min, font_size_max)

                if font is None:
                    # Log and skip if font loading failed completely for this cell
                    if num_chars_skipped_font < 10: # Log first few failures
                       logger.error(f"Skipping character at ({x},{y}) due to font loading failure for size near {final_font_size}.")
                    elif num_chars_skipped_font == 10:
                       logger.error("Further font loading errors for this run will be suppressed.")
                    num_chars_skipped_font += 1
                    continue # Skip drawing this character

                # Track the actual range of font sizes used
                min_final_size_used = min(min_final_size_used, final_font_size)
                max_final_size_used = max(max_final_size_used, final_font_size)

                # --- Calculate Draw Position ---
                # Map grid coordinates (x, y) to canvas coordinates (top-left corner)
                draw_x = int(x * scale_x)
                draw_y = int(y * scale_y)

                # --- Draw Character ---
                # Draw the character with its determined color and font/size
                try:
                    # Using anchor='lt' aligns the top-left of the text bounding box
                    # If Pillow version is old, it might not support anchor
                    draw.text((draw_x, draw_y), ascii_char, font=font, fill=pixel_color_tuple, anchor='lt')
                except TypeError:
                     # Fallback for older Pillow versions without anchor
                     # logger.warning("Pillow version might not support 'anchor'. Drawing without it.", exc_info=False) # Reduce logging noise
                     try:
                         draw.text((draw_x, draw_y), ascii_char, font=font, fill=pixel_color_tuple)
                     except Exception as draw_e_fallback:
                         if num_chars_skipped_font < 10:
                              logger.error(f"Error drawing text (fallback) at ({draw_x},{draw_y}): {draw_e_fallback}", exc_info=False)
                         num_chars_skipped_font += 1
                         continue
                except Exception as draw_e:
                     # Catch other potential drawing errors (e.g., font doesn't support char)
                     if num_chars_skipped_font < 10:
                          logger.error(f"Error drawing text at ({draw_x},{draw_y}): {draw_e}", exc_info=False)
                     elif num_chars_skipped_font == 10:
                          logger.error("Further text drawing errors for this run will be suppressed.")
                     num_chars_skipped_font += 1
                     continue

                # Update statistics
                num_chars_drawn += 1
            else:
                # Character skipped due to LowerDensity adjustment
                num_chars_skipped_density += 1

    # --- Final Logging ---
    total_cells = pixelated_width * pixelated_height
    if num_chars_drawn > 0:
         logger.info(f"Drawing complete. Drawn: {num_chars_drawn}/{total_cells}, Skipped (Density): {num_chars_skipped_density}, Skipped (Font/Draw Error): {num_chars_skipped_font}.")
         # Report the actual range of font sizes requested *after* dark area/edge adjustments
         logger.debug(f"Actual font size range used (after adjustments): {min_final_size_used}-{max_final_size_used} (Requested range: {font_size_min}-{font_size_max})")
    else:
         # Handle case where nothing was drawn
         logger.warning(f"Drawing complete, but NO characters were drawn ({total_cells} cells total). Skipped (Density): {num_chars_skipped_density}, Skipped (Font/Draw Error): {num_chars_skipped_font}. Check parameters, mask, and font.")

    # Return the final PIL image
    return ascii_image