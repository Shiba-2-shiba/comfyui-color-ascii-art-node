import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from comfy_api.latest import io

try:
    from folder_paths import (
        exists_annotated_filepath,
        filter_files_content_types,
        get_annotated_filepath,
        get_filename_list,
        get_full_path,
        get_input_directory,
    )
except ImportError:
    def get_filename_list(dir_name):
        if dir_name == "font":
            return [f for f in os.listdir(".") if f.lower().endswith((".ttf", ".otf"))] or ["dummy_font.ttf"]
        return []

    def get_full_path(dir_name, filename):
        return os.path.abspath(filename)

    def get_input_directory():
        return os.getcwd()

    def get_annotated_filepath(filename):
        return os.path.abspath(filename)

    def exists_annotated_filepath(filename):
        return os.path.isfile(get_annotated_filepath(filename))

    def filter_files_content_types(files, _content_types):
        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
        return [f for f in files if f.lower().endswith(image_exts)]

try:
    from .ascii_utils import pil_to_tensor
    from .text_utils import (
        calculate_text_grid,
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from ascii_utils import pil_to_tensor
    from text_utils import (
        calculate_text_grid,
    )


logger = logging.getLogger("ComfyUI.ASCIIText.LayoutPlanner")


def _list_input_images() -> list[str]:
    input_dir = get_input_directory()
    files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]
    try:
        files = filter_files_content_types(files, ["image"])
    except Exception:
        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
        files = [f for f in files if f.lower().endswith(image_exts)]
    return sorted(files)


def _load_input_image(image_name: str) -> torch.Tensor:
    image_path = get_annotated_filepath(image_name)
    output_images = []
    width = None
    height = None

    with Image.open(image_path) as img:
        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame)
            if frame.mode == "I":
                frame = frame.point(lambda value: value * (1 / 255))
            rgb_frame = frame.convert("RGB")

            if width is None:
                width, height = rgb_frame.size
            if rgb_frame.size != (width, height):
                continue

            output_images.append(rgb_frame)
            if img.format == "MPO":
                break

    if not output_images:
        raise ValueError(f"Could not load image: {image_name}")

    return pil_to_tensor(output_images)


class ASCIITextLayoutPlanner(io.ComfyNode):
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
            description="Load an input image and preview the character grid and output size for ASCII text layouts.",
            search_aliases=["ascii planner", "text layout planner", "load image planner", "ascii grid planner"],
            essentials_category="Image Tools/ASCII Art",
            inputs=[
                io.Combo.Input(
                    id="image",
                    options=_list_input_images(),
                    upload=io.UploadType.image,
                    tooltip="Drag and drop an image file, like the core Load Image node.",
                ),
                io.Int.Input(id="pixel_size", default=20, min=1, max=200, step=1),
                io.Float.Input(id="aspect_ratio_correction", default=0.75, min=0.1, max=10.0, step=0.05),
                io.Float.Input(id="resolution_scale", default=4.0, min=1.0, max=16.0, step=0.1),
                io.Combo.Input(id="font_name", options=font_list),
                io.Int.Input(id="font_size", default=12, min=1, max=300, step=1),
            ],
            outputs=[
                io.Image.Output(id="output_image", display_name="IMAGE"),
                io.Int.Output(id="required_chars"),
                io.Int.Output(id="grid_width"),
                io.Int.Output(id="grid_height"),
                io.Int.Output(id="render_width"),
                io.Int.Output(id="render_height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: str,
        pixel_size: int,
        aspect_ratio_correction: float,
        resolution_scale: float,
        font_name: str,
        font_size: int,
    ) -> io.NodeOutput:
        font_path = get_full_path("font", font_name)
        if not font_path or not os.path.isfile(font_path):
            raise FileNotFoundError(f"Font file '{font_name}' not found.")

        image_tensor = _load_input_image(image)
        image_np = image_tensor[0].cpu().numpy()
        image_height, image_width = image_np.shape[0], image_np.shape[1]

        grid_width, grid_height, required_chars = calculate_text_grid(
            (image_width, image_height),
            pixel_size,
            aspect_ratio_correction,
        )
        render_width = int(image_width * resolution_scale)
        render_height = int(image_height * resolution_scale)

        return io.NodeOutput(
            image_tensor,
            required_chars,
            grid_width,
            grid_height,
            render_width,
            render_height,
        )

    @classmethod
    def VALIDATE_INPUTS(cls, image, **_kwargs):
        if not exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
