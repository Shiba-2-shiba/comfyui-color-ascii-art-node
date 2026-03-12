import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from comfy_api.latest import io

try:
    from folder_paths import (
        exists_annotated_filepath,
        get_filename_list,
        get_full_path,
    )
except ImportError:
    def get_filename_list(dir_name):
        if dir_name == "font":
            return [f for f in os.listdir(".") if f.lower().endswith((".ttf", ".otf"))] or ["dummy_font.ttf"]
        return []

    def get_full_path(dir_name, filename):
        return os.path.abspath(filename)

    def exists_annotated_filepath(filename):
        return os.path.isfile(os.path.abspath(filename))

try:
    from .ascii_drawing import _get_font
    from .ascii_utils import pil_to_tensor, tensor_to_pil
    from .pixelation import pixelate_image
    from .text_utils import (
        build_text_cells,
        calculate_text_grid,
        inspect_font_support,
        load_text_file,
        make_layout_report,
        resolve_input_text_file,
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from ascii_drawing import _get_font
    from ascii_utils import pil_to_tensor, tensor_to_pil
    from pixelation import pixelate_image
    from text_utils import (
        build_text_cells,
        calculate_text_grid,
        inspect_font_support,
        load_text_file,
        make_layout_report,
        resolve_input_text_file,
    )


logger = logging.getLogger("ComfyUI.ASCIINovelTextArt")


def _create_background_canvas(
    original_image: Image.Image,
    pixelated_image: Image.Image,
    out_size: Tuple[int, int],
    background_mode: str,
) -> Image.Image:
    if background_mode == "white":
        return Image.new("RGB", out_size, (255, 255, 255))

    if background_mode == "sampled_average":
        image_np = np.array(pixelated_image)
        avg_color = tuple(int(value) for value in image_np.reshape(-1, 3).mean(axis=0))
        return Image.new("RGB", out_size, avg_color)

    if background_mode == "source_image":
        return original_image.resize(out_size, Image.Resampling.LANCZOS).convert("RGB")

    raise ValueError(f"Unsupported background_mode: {background_mode}")


def _background_luminance(color: Tuple[int, int, int]) -> float:
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _sample_luminance(color: Tuple[int, int, int]) -> float:
    r, g, b = color
    return ((r * 0.299) + (g * 0.587) + (b * 0.114)) / 255.0


def _compute_font_scale(sampled_color: Tuple[int, int, int], adaptive_font_strength: float) -> float:
    if adaptive_font_strength <= 0:
        return 1.0

    luminance = _sample_luminance(sampled_color)
    min_scale = max(0.55, 1.0 - (adaptive_font_strength * 0.45))
    max_scale = 1.0 + (adaptive_font_strength * 0.8)
    return min_scale + ((1.0 - luminance) * (max_scale - min_scale))


def _resolve_draw_color(
    char_color_mode: str,
    sampled_color: Tuple[int, int, int],
    background_color: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    if char_color_mode == "sampled_color":
        return sampled_color
    if char_color_mode == "grayscale":
        gray = int(round(sum(sampled_color) / 3.0))
        return (gray, gray, gray)
    if char_color_mode == "black":
        return (0, 0, 0)
    if char_color_mode == "knockout_white":
        luminance = _background_luminance(background_color)
        if luminance >= 235:
            tint_value = max(160, min(210, int(round(210 - (luminance - 235) * 2.0))))
            return (tint_value, tint_value, tint_value)
        return (255, 255, 255)
    raise ValueError(f"Unsupported char_color_mode: {char_color_mode}")


def _render_text_grid(
    base_canvas: Image.Image,
    source_canvas: Image.Image,
    pixelated_image: Image.Image,
    text_cells: List[str],
    font_path: str,
    font_size: int,
    resolution_scale: float,
    char_color_mode: str,
    text_opacity: float,
    blend_strength: float,
    adaptive_font_strength: float,
) -> Image.Image:
    out_w, out_h = base_canvas.size
    grid_w, grid_h = pixelated_image.size
    if grid_w <= 0 or grid_h <= 0:
        return base_canvas

    cell_w = out_w / grid_w
    cell_h = out_h / grid_h
    image_np = np.array(pixelated_image)
    base_rgba = base_canvas.convert("RGBA")
    text_layer = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    scaled_font_size = max(1, int(font_size * resolution_scale))
    glyph_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image, int, int]] = {}
    cell_index = 0
    opacity_alpha = max(0, min(255, int(round(text_opacity * 255.0))))
    blend_strength = max(0.0, min(1.0, blend_strength))

    for y in range(grid_h):
        for x in range(grid_w):
            if cell_index >= len(text_cells):
                composited = Image.alpha_composite(base_rgba, text_layer).convert("RGB")
                return Image.blend(source_canvas, composited, blend_strength)

            char = text_cells[cell_index]
            cell_index += 1

            if char in ("\n", "\r"):
                continue

            sampled_color = tuple(int(value) for value in image_np[y, x])
            dynamic_font_size = max(
                1,
                int(round(scaled_font_size * _compute_font_scale(sampled_color, adaptive_font_strength))),
            )
            glyph_key = (char, dynamic_font_size)
            glyph_data = glyph_cache.get(glyph_key)
            if glyph_data is None:
                font = _get_font(font_path, dynamic_font_size)
                if font is None:
                    raise ValueError(f"Could not load font at size {dynamic_font_size}: {font_path}")
                try:
                    bbox = font.getbbox(char)
                except Exception:
                    glyph_cache[glyph_key] = (None, None, 0, 0)
                    continue

                if bbox is None:
                    glyph_cache[glyph_key] = (None, None, 0, 0)
                    continue

                left, top, right, bottom = bbox
                width = max(1, right - left)
                height = max(1, bottom - top)
                mask_img = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask_img)
                draw.text((-left, -top), char, font=font, fill=255)
                alpha_mask = mask_img if opacity_alpha >= 255 else mask_img.point(
                    lambda value: (value * opacity_alpha) // 255
                )
                glyph_data = (mask_img, alpha_mask, left, top)
                glyph_cache[glyph_key] = glyph_data

            mask_img, alpha_mask, offset_x, offset_y = glyph_data
            if mask_img is None or alpha_mask is None:
                continue

            paste_x = int(x * cell_w) + offset_x
            paste_y = int(y * cell_h) + offset_y
            center_x = min(out_w - 1, max(0, int((x + 0.5) * cell_w)))
            center_y = min(out_h - 1, max(0, int((y + 0.5) * cell_h)))
            background_color = base_canvas.getpixel((center_x, center_y))
            draw_color = _resolve_draw_color(char_color_mode, sampled_color, background_color)
            draw_rgba = (*draw_color, opacity_alpha)

            try:
                text_layer.paste(draw_rgba, (paste_x, paste_y), alpha_mask)
            except Exception:
                logger.debug("Skipping glyph paste failure at (%s, %s)", x, y, exc_info=True)

    composited = Image.alpha_composite(base_rgba, text_layer).convert("RGB")
    return Image.blend(source_canvas, composited, blend_strength)


class ASCIINovelTextArt(io.ComfyNode):
    DOWNSCALE_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos", "contrast"]
    SHARPEN_MODES = ["None", "unsharp"]
    TEXT_ENCODINGS = ["utf-8", "utf-8-sig", "cp932", "shift_jis", "auto"]
    NEWLINE_MODES = ["remove", "space", "preserve"]
    TEXT_SHORTAGE_MODES = ["error", "loop", "truncate_blank"]
    CHAR_COLOR_MODES = ["sampled_color", "grayscale", "black", "knockout_white"]
    BACKGROUND_MODES = ["white", "sampled_average", "source_image"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        try:
            font_list = get_filename_list("font")
            if not font_list:
                font_list = ["font_not_found.ttf"]
        except Exception:
            font_list = ["error_loading_font.ttf"]

        return io.Schema(
            node_id="ASCIINovelTextArt",
            display_name="ASCII Novel Text Art",
            category="Image Processing/ASCII Art",
            description="Render ASCII art from an image and an uploaded text file, consuming the text sequentially across the grid.",
            search_aliases=["novel ascii", "text ascii art", "txt ascii art", "ascii novel"],
            essentials_category="Image Tools/ASCII Art",
            inputs=[
                io.Image.Input(id="image"),
                io.Int.Input(id="pixel_size", default=20, min=1, max=200, step=1),
                io.Float.Input(id="resolution_scale", default=4.0, min=1.0, max=16.0, step=0.1),
                io.Combo.Input(id="downscale_mode", options=cls.DOWNSCALE_MODES, default="area"),
                io.Float.Input(id="aspect_ratio_correction", default=0.75, min=0.1, max=10.0, step=0.05),
                io.Combo.Input(id="font_name", options=font_list),
                io.Int.Input(id="font_size", default=12, min=1, max=300, step=1),
                io.Float.Input(id="text_opacity", default=0.55, min=0.05, max=1.0, step=0.05),
                io.Float.Input(id="blend_strength", default=0.8, min=0.0, max=1.0, step=0.05),
                io.Float.Input(id="adaptive_font_strength", default=0.35, min=0.0, max=1.0, step=0.05),
                io.String.Input(
                    id="text_file_path",
                    default="",
                    placeholder="Drop a .txt file below or enter a filename from input/",
                    tooltip="TXT filename stored in ComfyUI input files. Use the upload area below this field.",
                ),
                io.Combo.Input(id="char_color_mode", options=cls.CHAR_COLOR_MODES, default="sampled_color", optional=True),
                io.Combo.Input(id="background_mode", options=cls.BACKGROUND_MODES, default="white", optional=True),
                io.Combo.Input(id="text_encoding", options=cls.TEXT_ENCODINGS, default="utf-8", optional=True),
                io.Combo.Input(id="newline_mode", options=cls.NEWLINE_MODES, default="remove", optional=True),
                io.Combo.Input(id="text_shortage_mode", options=cls.TEXT_SHORTAGE_MODES, default="loop", optional=True),
                io.Combo.Input(id="sharpen_mode", options=cls.SHARPEN_MODES, default="None", optional=True),
                io.Float.Input(id="sharpen_amount", default=1.0, min=0.0, max=5.0, step=0.1, optional=True),
                io.Float.Input(id="sharpen_threshold", default=0.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Float.Input(id="brightness", default=1.0, min=0.0, max=5.0, step=0.05, optional=True),
                io.Float.Input(id="contrast", default=1.0, min=0.0, max=5.0, step=0.05, optional=True),
                io.Int.Input(id="glyph_check_limit", default=5000, min=1, max=100000, step=1, optional=True),
            ],
            outputs=[
                io.Image.Output(id="output_image", display_name="IMAGE"),
                io.Int.Output(id="used_chars"),
                io.Int.Output(id="required_chars"),
                io.String.Output(id="report_text"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        pixel_size: int,
        resolution_scale: float,
        downscale_mode: str,
        aspect_ratio_correction: float,
        font_name: str,
        font_size: int,
        text_file_path: str,
        text_opacity: float = 0.55,
        blend_strength: float = 0.8,
        adaptive_font_strength: float = 0.35,
        char_color_mode: str = "sampled_color",
        background_mode: str = "white",
        text_encoding: str = "utf-8",
        newline_mode: str = "remove",
        text_shortage_mode: str = "loop",
        sharpen_mode: str = "None",
        sharpen_amount: float = 1.0,
        sharpen_threshold: float = 0.0,
        brightness: float = 1.0,
        contrast: float = 1.0,
        glyph_check_limit: int = 5000,
    ) -> io.NodeOutput:
        pil_images = tensor_to_pil(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        font_path = get_full_path("font", font_name)
        if not font_path or not os.path.isfile(font_path):
            raise FileNotFoundError(f"Font file '{font_name}' not found.")

        resolved_text_path = resolve_input_text_file(text_file_path)
        text_content, selected_encoding = load_text_file(
            resolved_text_path,
            encoding=text_encoding,
            newline_mode=newline_mode,
        )
        font_report = inspect_font_support(text_content, font_path, font_size, limit=glyph_check_limit)

        rendered_images: List[Image.Image] = []
        used_chars = 0
        required_chars = 0
        grid_width = 0
        grid_height = 0
        render_width = 0
        render_height = 0
        warnings: List[str] = []

        for pil_image in pil_images:
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
                downscale_mode=downscale_mode,
            )

            theoretical_grid_width, theoretical_grid_height, _ = calculate_text_grid(
                pil_image.size,
                pixel_size,
                aspect_ratio_correction,
            )
            grid_width, grid_height = pixelated_image.size
            required_chars = grid_width * grid_height
            render_width = int(pil_image.size[0] * resolution_scale)
            render_height = int(pil_image.size[1] * resolution_scale)
            if (grid_width, grid_height) != (theoretical_grid_width, theoretical_grid_height):
                warnings.append("grid_adjusted_to_actual_pixelation")

            text_cells, consumed_chars = build_text_cells(
                text=text_content,
                required_chars=required_chars,
                grid_width=grid_width,
                shortage_mode=text_shortage_mode,
                preserve_newlines=(newline_mode == "preserve"),
            )
            used_chars = consumed_chars

            base_canvas = _create_background_canvas(
                pil_image,
                pixelated_image,
                (render_width, render_height),
                background_mode,
            )
            source_canvas = pil_image.resize((render_width, render_height), Image.Resampling.LANCZOS).convert("RGB")
            rendered_image = _render_text_grid(
                base_canvas=base_canvas,
                source_canvas=source_canvas,
                pixelated_image=pixelated_image,
                text_cells=text_cells,
                font_path=font_path,
                font_size=font_size,
                resolution_scale=resolution_scale,
                char_color_mode=char_color_mode,
                text_opacity=text_opacity,
                blend_strength=blend_strength,
                adaptive_font_strength=adaptive_font_strength,
            )
            rendered_images.append(rendered_image)

        report_text = make_layout_report(
            file_path=text_file_path,
            selected_encoding=selected_encoding,
            grid_width=grid_width,
            grid_height=grid_height,
            required_chars=required_chars,
            usable_chars=len(text_content),
            render_width=render_width,
            render_height=render_height,
            glyph_supported_ratio=font_report["glyph_supported_ratio"],
            unsupported_chars=font_report["unsupported_chars"],
            warnings=list(dict.fromkeys(warnings)),
        )
        report_text += f"\nused_chars={used_chars}"
        report_text += f"\nchar_color_mode={char_color_mode}"
        report_text += f"\nbackground_mode={background_mode}"
        report_text += f"\ntext_opacity={text_opacity:.2f}"
        report_text += f"\nblend_strength={blend_strength:.2f}"
        report_text += f"\nadaptive_font_strength={adaptive_font_strength:.2f}"

        output_tensor = pil_to_tensor(rendered_images)
        return io.NodeOutput(output_tensor, used_chars, required_chars, report_text)

    @classmethod
    def VALIDATE_INPUTS(cls, text_file_path=None, **_kwargs):
        if not text_file_path:
            return "Text file is required."
        if not text_file_path.lower().endswith(".txt"):
            return f"Only .txt files are supported: {text_file_path}"
        if not exists_annotated_filepath(text_file_path):
            return f"Invalid text file: {text_file_path}"
        return True
