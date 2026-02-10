import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import Optional, List, Tuple
from comfy_api.latest import io
from .ascii_drawing import _get_font
from .ascii_utils import tensor_to_pil, pil_to_tensor, load_custom_characters, calculate_edge_info
from .pixelation import pixelate_image
from .colormatch import apply_color_match
from .charset_generator import generate_dynamic_charset
from folder_paths import get_full_path, get_filename_list
import os

logger = logging.getLogger("ComfyUI.ASCIIArtCustomFont")

class ASCIIArtCustomFont(io.ComfyNode):
    DOWNSCALE_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos", "contrast"]
    COLOR_MATCH_METHODS = ['mkl', 'hm', 'reinhard', 'idt', 'hm-mkl-hm']
    CHAR_SELECTION_MODES = ["brightness", "hue", "saturation", "luminance_hue"]
    CHARSET_SOURCES = ["File", "Dynamic"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        try:
            font_list = get_filename_list("font")
            if not font_list:
                font_list = ["font_not_found.ttf"]
        except Exception:
            font_list = ["error_loading_font.ttf"]

        return io.Schema(
            node_id="ASCIIArtCustomFont",
            display_name="ASCII Art Custom Font (High-Res)",
            category="Image Processing/ASCII Art",
            inputs=[
                io.Image.Input(id="image"),
                io.Int.Input(id="pixel_size", default=20, min=1, max=200, step=1, label="Grid (Pixel) Size"),
                io.Float.Input(id="resolution_scale", default=4.0, min=1.0, max=16.0, step=0.1, label="Resolution Scale"),
                io.Combo.Input(id="downscale_mode", options=cls.DOWNSCALE_MODES, default="area"),
                io.Float.Input(id="aspect_ratio_correction", default=0.75, min=0.1, max=10.0, step=0.05),
                io.Combo.Input(id="font_name", options=font_list),
                io.Int.Input(id="font_size_min", default=8, min=1, max=100, step=1),
                io.Int.Input(id="font_size_max", default=16, min=1, max=200, step=1),
                io.Combo.Input(id="char_selection_mode", options=cls.CHAR_SELECTION_MODES, default="luminance_hue"),
                io.Int.Input(id="seed", default=0, min=0, max=0xffffffffffffffff),
                
                # Optionals
                io.Combo.Input(id="charset_source", options=cls.CHARSET_SOURCES, default="File", optional=True),
                io.String.Input(id="ascii_chars_filename", default="set4.txt", optional=True),
                io.String.Input(id="dynamic_chars_to_test", multiline=True, default=R"""!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ """, optional=True),
                io.Float.Input(id="dynamic_sigma", default=1.5, min=0.1, max=10.0, step=0.1, optional=True),
                io.Float.Input(id="brightness", default=1.0, min=0.0, max=5.0, step=0.05, optional=True),
                io.Float.Input(id="contrast", default=1.0, min=0.0, max=5.0, step=0.05, optional=True),
                io.Boolean.Input(id="enable_color_match", default=False, optional=True),
                io.Combo.Input(id="color_match_method", options=cls.COLOR_MATCH_METHODS, default="mkl", optional=True),
            ],
            outputs=[
                io.Image.Output(id="output_image", display_name="IMAGE"),
            ]
        )

    @classmethod
    def execute(cls,
                image: torch.Tensor,
                pixel_size: int,
                resolution_scale: float,
                downscale_mode: str,
                aspect_ratio_correction: float,
                font_name: str,
                font_size_min: int,
                font_size_max: int,
                char_selection_mode: str,
                seed: int,
                charset_source: str = "File",
                ascii_chars_filename: str = "set4.txt",
                dynamic_chars_to_test: str = "",
                dynamic_sigma: float = 1.5,
                brightness: float = 1.0,
                contrast: float = 1.0,
                enable_color_match: bool = False,
                color_match_method: str = 'mkl') -> io.NodeOutput:

        logger.info(f"Starting High-Res ASCII Art Generation (Scale: {resolution_scale}x)")
        
        # 1. Setup Resources
        try:
            font_path = get_full_path("font", font_name)
            if not font_path or not os.path.isfile(font_path):
                 raise FileNotFoundError(f"Font file '{font_name}' not found.")
        except Exception as e:
            raise ValueError(f"Could not load font '{font_name}'.") from e

        ascii_sets = []
        if charset_source == "File":
            base_dir = os.path.dirname(__file__)
            ascii_chars_file_path = os.path.join(base_dir, ascii_chars_filename)
            ascii_sets = load_custom_characters(ascii_chars_file_path)
        elif charset_source == "Dynamic":
            # Generate charset based on the UN-SCALED font size logic for consistency with selection
            generated_set = generate_dynamic_charset(font_path, font_size_max, dynamic_chars_to_test, dynamic_sigma)
            if generated_set: ascii_sets = [generated_set]
        
        if not ascii_sets: raise ValueError("No valid character sets loaded.")

        pil_images = tensor_to_pil(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_images_list = []

        for i, pil_image in enumerate(pil_images):
            # 2. Pixelation (Grid Analysis)
            # We pixelate to determines the grid (which characters go where)
            # This happens at the logical grid resolution
            pixelated_image = pixelate_image(
                image=pil_image, pixel_size=pixel_size, aspect_ratio_correction=aspect_ratio_correction,
                brightness=brightness, contrast=contrast, device=device, downscale_mode=downscale_mode
            )
            
            # Color matching (optional)
            if enable_color_match:
                # pixelated_image now has the target colors
                pixelated_image = apply_color_match(pil_image, pixelated_image, color_match_method)

            # 3. High-Res Drawing
            # Calculate output dimensions
            orig_w, orig_h = pil_image.size
            out_w = int(orig_w * resolution_scale)
            out_h = int(orig_h * resolution_scale)
            
            # Create High-Res Canvas
            ascii_image = Image.new('RGB', (out_w, out_h), (255, 255, 255))
            draw = ImageDraw.Draw(ascii_image)

            # Grid dimensions from pixelated image
            grid_w, grid_h = pixelated_image.size
            if grid_w == 0 or grid_h == 0:
                final_images_list.append(ascii_image)
                continue

            # Calculate High-Res Cell Size
            cell_w = out_w / grid_w
            cell_h = out_h / grid_h

            # Convert pixelated image to numpy for fast access
            image_np = np.array(pixelated_image) # RGB
            
            # Pre-calculate selection values (similar to original node)
            if char_selection_mode == "brightness":
                selection_vals = np.array(pixelated_image.convert('HSV'))[:,:,2] / 255.0
            elif char_selection_mode == "hue":
                selection_vals = np.array(pixelated_image.convert('HSV'))[:,:,0] / 255.0
            elif char_selection_mode == "saturation":
                selection_vals = np.array(pixelated_image.convert('HSV'))[:,:,1] / 255.0
            else: # luminance_hue or default
                hsv = np.array(pixelated_image.convert('HSV')) / 255.0
                selection_vals = (hsv[:,:,2] * 0.7) + (hsv[:,:,0] * 0.3)
            
            # Font Scaling
            scaled_font_min = int(font_size_min * resolution_scale)
            scaled_font_max = int(font_size_max * resolution_scale)
            
            import random
            random.seed(seed + i)
            chosen_set = random.choice(ascii_sets)
            
            # Glyph Cache (Simple Dictionary)
            # Key: (size, char), Value: (bitmap_image, offset_x, offset_y)
            glyph_cache = {}
            # Font Object Cache
            scaled_font_cache = {}

            for y in range(grid_h):
                for x in range(grid_w):
                    # Data for cell
                    color = tuple(image_np[y, x])
                    val = selection_vals[y, x]
                    
                    # Select Char
                    char_idx = int(val * (len(chosen_set) - 1) + 0.5)
                    char_idx = max(0, min(len(chosen_set) - 1, char_idx))
                    char = chosen_set[char_idx]

                    # Select Font Size (Simple luminance based)
                    # For high-res, we just scale the min/max logic
                    lum = sum(color) / (3 * 255.0)
                    size = scaled_font_min + lum * (scaled_font_max - scaled_font_min)
                    size = int(size)

                    
                    # 1. Retrieve/Load Font Object
                    font = scaled_font_cache.get(size)
                    if font is None:
                        try:
                            font = ImageFont.truetype(font_path, size)
                            scaled_font_cache[size] = font
                        except Exception:
                            # Cache failure as None to avoid retry loop overload? 
                            # For now just skip
                            continue

                    if font is None:
                        continue

                    # 2. Retrieve/Generate Glyph Bitmap
                    glyph_key = (size, char)
                    glyph_data = glyph_cache.get(glyph_key)
                    
                    if glyph_data is None:
                        # Render new glyph
                        try:
                            # Get bounding box
                            bbox = font.getbbox(char)
                            if bbox is None:
                                # For whitespace or empty rendering
                                glyph_cache[glyph_key] = (None, 0, 0)
                                continue
                            
                            l, t, r, b = bbox
                            w, h = r - l, b - t
                            offset_x, offset_y = l, t
                            
                            w = max(1, w)
                            h = max(1, h)
                            mask_img = Image.new('L', (w, h), 0)
                            m_draw = ImageDraw.Draw(mask_img)
                            
                            m_draw.text((-offset_x, -offset_y), char, font=font, fill=255)
                            
                            glyph_data = (mask_img, offset_x, offset_y)
                            glyph_cache[glyph_key] = glyph_data
                            
                        except Exception:
                            glyph_cache[glyph_key] = (None, 0, 0)
                            continue
                    
                    # Unpack carefully
                    if glyph_data is None: 
                        continue # Should not happen unless logic err
                        
                    mask_img, offset_x, offset_y = glyph_data
                    
                    if mask_img is not None:
                        # Re-calculate paste position based on logic
                        paste_x = int(x * cell_w) + offset_x
                        paste_y = int(y * cell_h) + offset_y
                        
                        try:
                            sub_w, sub_h = mask_img.size
                            if paste_x >= out_w or paste_y >= out_h or paste_x + sub_w <= 0 or paste_y + sub_h <= 0:
                                continue
                                
                            ascii_image.paste(color, (paste_x, paste_y), mask_img)
                        except Exception:
                            pass
            
            final_images_list.append(ascii_image)

        output_tensor = pil_to_tensor(final_images_list)
        return io.NodeOutput(output_tensor)
