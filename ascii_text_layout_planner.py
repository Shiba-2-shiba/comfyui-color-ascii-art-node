import logging
import os

import torch
from comfy_api.latest import io

try:
    from folder_paths import get_filename_list, get_full_path
except ImportError:
    def get_filename_list(dir_name):
        if dir_name == "font":
            return [f for f in os.listdir(".") if f.lower().endswith((".ttf", ".otf"))] or ["dummy_font.ttf"]
        return []

    def get_full_path(dir_name, filename):
        return os.path.abspath(filename)

try:
    from .ascii_utils import tensor_to_pil
    from .text_utils import (
        calculate_text_grid,
        inspect_font_support,
        load_text_file,
        make_layout_report,
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from ascii_utils import tensor_to_pil
    from text_utils import (
        calculate_text_grid,
        inspect_font_support,
        load_text_file,
        make_layout_report,
    )


logger = logging.getLogger("ComfyUI.ASCIIText.LayoutPlanner")


class ASCIITextLayoutPlanner(io.ComfyNode):
    TEXT_ENCODINGS = ["utf-8", "utf-8-sig", "cp932", "shift_jis", "auto"]
    NEWLINE_MODES = ["remove", "space", "preserve"]
    TEXT_SHORTAGE_MODES = ["error", "loop", "truncate_blank"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        try:
            font_list = get_filename_list("font")
            if not font_list:
                font_list = ["font_not_found.ttf"]
        except Exception:
            font_list = ["error_loading_font.ttf"]

        return io.Schema(
            node_id="ASCIITextLayoutPlanner",
            display_name="ASCII Text Layout Planner",
            category="Image Processing/ASCII Art",
            inputs=[
                io.Image.Input(id="image"),
                io.Int.Input(id="pixel_size", default=20, min=1, max=200, step=1),
                io.Float.Input(id="aspect_ratio_correction", default=0.75, min=0.1, max=10.0, step=0.05),
                io.Float.Input(id="resolution_scale", default=4.0, min=1.0, max=16.0, step=0.1),
                io.Combo.Input(id="font_name", options=font_list),
                io.Int.Input(id="font_size", default=12, min=1, max=300, step=1),
                io.String.Input(id="text_file_path", default=""),
                io.Combo.Input(id="text_encoding", options=cls.TEXT_ENCODINGS, default="utf-8", optional=True),
                io.Combo.Input(id="newline_mode", options=cls.NEWLINE_MODES, default="remove", optional=True),
                io.Combo.Input(id="text_shortage_mode", options=cls.TEXT_SHORTAGE_MODES, default="error", optional=True),
                io.Int.Input(id="glyph_check_limit", default=5000, min=1, max=100000, step=1, optional=True),
            ],
            outputs=[
                io.Int.Output(id="required_chars"),
                io.Int.Output(id="usable_chars"),
                io.Int.Output(id="grid_width"),
                io.Int.Output(id="grid_height"),
                io.Int.Output(id="render_width"),
                io.Int.Output(id="render_height"),
                io.Float.Output(id="glyph_supported_ratio"),
                io.String.Output(id="unsupported_chars_preview"),
                io.String.Output(id="report_text"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        pixel_size: int,
        aspect_ratio_correction: float,
        resolution_scale: float,
        font_name: str,
        font_size: int,
        text_file_path: str,
        text_encoding: str = "utf-8",
        newline_mode: str = "remove",
        text_shortage_mode: str = "error",
        glyph_check_limit: int = 5000,
    ) -> io.NodeOutput:
        pil_images = tensor_to_pil(image)
        target_image = pil_images[0]
        image_width, image_height = target_image.size

        font_path = get_full_path("font", font_name)
        if not font_path or not os.path.isfile(font_path):
            raise FileNotFoundError(f"Font file '{font_name}' not found.")

        text_content, selected_encoding = load_text_file(
            text_file_path,
            encoding=text_encoding,
            newline_mode=newline_mode,
        )

        grid_width, grid_height, required_chars = calculate_text_grid(
            target_image.size,
            pixel_size,
            aspect_ratio_correction,
        )
        render_width = int(image_width * resolution_scale)
        render_height = int(image_height * resolution_scale)

        font_report = inspect_font_support(
            text_content,
            font_path,
            font_size,
            limit=glyph_check_limit,
        )

        warnings = []
        usable_chars = len(text_content)
        if len(pil_images) > 1:
            warnings.append("batch_input_detected:first_image_used_for_layout")
        if text_shortage_mode == "error" and usable_chars < required_chars:
            warnings.append("text_shortage_expected")
        if font_report["unsupported_chars"]:
            warnings.append("unsupported_glyphs_detected")

        unsupported_preview = ", ".join(font_report["unsupported_chars"][:12])
        report_text = make_layout_report(
            file_path=text_file_path,
            selected_encoding=selected_encoding,
            grid_width=grid_width,
            grid_height=grid_height,
            required_chars=required_chars,
            usable_chars=usable_chars,
            render_width=render_width,
            render_height=render_height,
            glyph_supported_ratio=font_report["glyph_supported_ratio"],
            unsupported_chars=font_report["unsupported_chars"],
            warnings=warnings,
        )

        return io.NodeOutput(
            required_chars,
            usable_chars,
            grid_width,
            grid_height,
            render_width,
            render_height,
            float(font_report["glyph_supported_ratio"]),
            unsupported_preview,
            report_text,
        )
