# ascii_art_node_v3.py (Main Node File - Refactored for V3 Schema)
import os
import random
import logging
import numpy as np
import torch
from typing import Optional, Tuple, List

# --- V3 Schema Imports ---
# V3スキーマで必須となるio, ComfyExtension, overrideをインポートします
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override

# --- ComfyUI Specific Imports ---
try:
    # `get_filename_list`はV3でもUIのドロップダウンリストを動的に生成するために使用します
    from folder_paths import get_filename_list, get_full_path
except ImportError:
    print("Warning: ComfyUI folder_paths not found. Using dummy functions.")
    # Dummy functions for standalone testing
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

# --- Module Imports (Helper scripts) ---
# これらのヘルパースクリプトはV3化による変更は不要です
try:
    from .ascii_utils import (setup_logging, tensor_to_pil, mask_tensor_to_pil,
                              pil_to_tensor, load_custom_characters,
                              calculate_edge_info, apply_mask_blending)
    from .pixelation import pixelate_image
    from .ascii_drawing import create_ascii_art
    from .colormatch import apply_color_match
    from .charset_generator import generate_dynamic_charset
except ImportError as e:
    # Fallback for standalone execution or path issues
    print(f"Warning: Relative imports failed in main node. Trying direct imports. Error: {e}")
    # (Error handling for imports remains the same)
    import sys
    sys.path.append(os.path.dirname(__file__))
    from ascii_utils import (setup_logging, tensor_to_pil, mask_tensor_to_pil,
                             pil_to_tensor, load_custom_characters,
                             calculate_edge_info, apply_mask_blending)
    from pixelation import pixelate_image
    from ascii_drawing import create_ascii_art
    from colormatch import apply_color_match
    from charset_generator import generate_dynamic_charset


# --- Logger Setup ---
# ロガー設定は変更ありません
logger = logging.getLogger("ComfyUI.ASCIIArtNodeV3")
# (Logger setup remains the same)


# ▼▼▼【原則】V1クラスをio.ComfyNodeを継承するように変更 ▼▼▼
class ASCIIArtNodeV3(io.ComfyNode):
    # V1の定数をクラス属性として保持しておくと、define_schema内で参照できて便利です
    DOWNSCALE_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos", "contrast"]
    SHARPEN_MODES = ["None", "unsharp"]
    COLOR_MATCH_METHODS = ['mkl', 'hm', 'reinhard', 'idt', 'hm-mkl-hm']
    CHAR_SELECTION_MODES = ["brightness", "hue", "saturation", "luminance_hue"]
    MASK_EDGE_ADJUSTMENT_MODES = ["None", "SmallerChars", "LowerDensity"]
    LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "NONE"]
    CHARSET_SOURCES = ["File", "Dynamic"]

    # ▼▼▼【原則1】V1のINPUT_TYPESをdefine_schemaメソッドに変換 ▼▼▼
    @classmethod
    def define_schema(cls) -> io.Schema:
        # フォントリストの取得ロジックはdefine_schema内に移動します
        try:
            font_list = get_filename_list("font")
            if not font_list:
                 logger.warning("No fonts found in ComfyUI's fonts directory. Please add .ttf or .otf files.")
                 font_list = ["font_not_found.ttf"]
        except Exception as e:
            logger.error(f"Could not list fonts from ComfyUI directory: {e}", exc_info=True)
            font_list = ["error_loading_font.ttf"]

        return io.Schema(
            # node_idはクラス名と一致させるのが慣例です
            node_id="ASCIIArtNodeV3",
            # 表示名を定義します
            display_name="ASCII Art Generator V3",
            # V1のCATEGORYをここに記述します
            category="Image Processing/ASCII Art",
            
            # ▼▼▼【原則2, 3】V1のINPUT_TYPESの各項目をioクラスにマッピング ▼▼▼
            inputs=[
                # requiredセクション
                io.Image.Input(id="image"),
                io.Int.Input(id="pixel_size", default=20, min=1, max=200, step=1),
                io.Combo.Input(id="downscale_mode", options=cls.DOWNSCALE_MODES, default="area"),
                io.Float.Input(id="aspect_ratio_correction", default=0.75, min=0.1, max=10.0, step=0.05),
                io.Combo.Input(id="font_name", options=font_list),
                io.Int.Input(id="font_size_min", default=8, min=1, max=100, step=1),
                io.Int.Input(id="font_size_max", default=16, min=1, max=200, step=1),
                io.Combo.Input(id="char_selection_mode", options=cls.CHAR_SELECTION_MODES, default="luminance_hue"),
                io.Int.Input(id="seed", default=0, min=0, max=0xffffffffffffffff),

                # optionalセクション
                # optional=Trueに設定することで、V1のoptionalと同じ挙動になります
                io.Combo.Input(id="charset_source", options=cls.CHARSET_SOURCES, default="File", optional=True),
                io.String.Input(id="ascii_chars_filename", default="set4.txt", optional=True),
                io.String.Input(
                    id="dynamic_chars_to_test",
                    multiline=True,
                    default=R"""!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ """,
                    optional=True
                ),
                io.Float.Input(id="dynamic_sigma", default=1.5, min=0.1, max=10.0, step=0.1, optional=True),
                io.Combo.Input(id="sharpen_mode", options=cls.SHARPEN_MODES, default="None", optional=True),
                io.Float.Input(id="sharpen_amount", default=1.0, min=0.0, max=5.0, step=0.1, optional=True),
                io.Float.Input(id="sharpen_threshold", default=0.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Float.Input(id="brightness", default=1.0, min=0.0, max=5.0, step=0.05, optional=True),
                io.Float.Input(id="contrast", default=1.0, min=0.0, max=5.0, step=0.05, optional=True),
                io.Mask.Input(id="mask", optional=True),
                io.Float.Input(id="mask_blend_radius", default=5.0, min=0.0, max=100.0, step=0.1, optional=True),
                io.Combo.Input(id="mask_edge_adjustment", options=cls.MASK_EDGE_ADJUSTMENT_MODES, default="None", optional=True),
                io.Float.Input(id="mask_edge_factor", default=3.0, min=0.0, max=20.0, step=0.1, optional=True),
                io.Boolean.Input(id="enable_color_match", default=False, label_on="Enabled", label_off="Disabled", optional=True),
                io.Combo.Input(id="color_match_method", options=cls.COLOR_MATCH_METHODS, default="mkl", optional=True),
                io.Combo.Input(id="log_level", options=cls.LOG_LEVELS, default="INFO", optional=True),
            ],
            
            # ▼▼▼【原則1, 2】V1のRETURN_TYPESを変換 ▼▼▼
            outputs=[
                io.Image.Output(id="output_image", display_name="IMAGE"),
            ]
        )

    # ▼▼▼【原則4】V1のFUNCTIONメソッドを'@classmethod def execute'にリネーム ▼▼▼
    # 引数名はinputsのidと完全に一致させ、型ヒントを追記します
    @classmethod
    def execute(cls,
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
                charset_source: str = "File",
                ascii_chars_filename: str = "set4.txt",
                dynamic_chars_to_test: str = "",
                dynamic_sigma: float = 1.5,
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
                log_level: str = "INFO") -> io.NodeOutput:
        
        # --- ここから下のコア処理ロジックは一切変更不要です ---
        # --- Core processing logic below requires no changes for V3 migration ---

        # 1. Setup Phase
        setup_logging(log_level, "ComfyUI.ASCIIArtNodeV3")
        logger.info(f"Starting ASCII Art Generation V3 for {image.shape[0]} image(s)")
        
        # (Parameter logging remains the same)
        logger.debug(f"--- Input Parameters ---")
        logger.debug(f"  Pixel Size: {pixel_size}, Downscale Mode: {downscale_mode}")
        logger.debug(f"  Font: {font_name}, Size Range: ({font_size_min}-{font_size_max})")
        # ( ... all other logging ... )
        logger.debug(f"------------------------")

        if font_size_min > font_size_max:
            logger.warning(f"font_size_min ({font_size_min}) > font_size_max ({font_size_max}). Swapping them.")
            font_size_min, font_size_max = font_size_max, font_size_min
        font_size_min = max(1, font_size_min)
        font_size_max = max(font_size_min, font_size_max)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Load Resources
        try:
            font_path = get_full_path("font", font_name)
            if not font_path or not os.path.isfile(font_path):
                 raise FileNotFoundError(f"Font file '{font_name}' not found.")
            logger.debug(f"Using font file: {font_path}")
        except Exception as e:
            logger.error(f"Error resolving font path for '{font_name}': {e}", exc_info=True)
            raise ValueError(f"Could not load font '{font_name}'.") from e

        ascii_sets = []
        try:
            if charset_source == "File":
                base_dir = os.path.dirname(__file__)
                ascii_chars_file_path = os.path.join(base_dir, ascii_chars_filename)
                if not os.path.isfile(ascii_chars_file_path):
                   raise FileNotFoundError(f"ASCII chars file not found at '{ascii_chars_file_path}'.")
                ascii_sets = load_custom_characters(ascii_chars_file_path)
            elif charset_source == "Dynamic":
                generated_set = generate_dynamic_charset(
                    font_path=font_path, font_size=font_size_max,
                    characters_to_test=dynamic_chars_to_test, sigma=dynamic_sigma
                )
                if generated_set: ascii_sets = [generated_set]
            if not ascii_sets: raise ValueError("No valid character sets were loaded or generated.")
        except Exception as e:
            logger.error(f"Failed to prepare ASCII characters: {e}", exc_info=True)
            raise

        # 3. Process Input Tensor to PIL List
        pil_images = tensor_to_pil(image)
        
        # 4. Process Mask
        mask_pil = None
        if mask is not None:
            mask_pil = mask_tensor_to_pil(mask, pil_images[0].size)

        # 5. Per-Image Processing Loop
        final_images_list = []
        for i, pil_image in enumerate(pil_images):
            logger.info(f"--- Processing Batch Image {i+1}/{len(pil_images)} ---")
            current_original_size = pil_image.size
            
            random.seed(seed + i)
            np.random.seed((seed + i) % (2**32))
            chosen_set = random.choice(ascii_sets)

            pixelated_image = pixelate_image(
                image=pil_image, pixel_size=pixel_size, aspect_ratio_correction=aspect_ratio_correction,
                brightness=brightness, contrast=contrast, device=device, sharpen_mode=sharpen_mode,
                sharpen_amount=sharpen_amount, sharpen_threshold=sharpen_threshold, downscale_mode=downscale_mode
            )

            image_for_ascii_drawing = pixelated_image
            if enable_color_match:
                image_for_ascii_drawing = apply_color_match(
                    source_image_pil=pil_image, target_image_pil=pixelated_image, method=color_match_method
                )

            current_edge_info_np = None
            if mask_pil is not None and mask_edge_adjustment != "None":
                current_edge_info_np = calculate_edge_info(mask_pil, mask_edge_factor, image_for_ascii_drawing.size)
            
            ascii_image = create_ascii_art(
                image=image_for_ascii_drawing, ascii_chars=chosen_set, font_path=font_path,
                font_size_min=font_size_min, font_size_max=font_size_max,
                original_size=current_original_size, char_selection_mode=char_selection_mode,
                mask_edge_adjustment=mask_edge_adjustment, edge_info=current_edge_info_np
            )

            final_image_for_batch = ascii_image
            if mask_pil is not None:
                final_image_for_batch = apply_mask_blending(
                    original_pil=pil_image, overlay_pil=ascii_image,
                    mask_pil=mask_pil, mask_blend_radius=mask_blend_radius
                )
            
            final_images_list.append(final_image_for_batch)

        # 6. Output Conversion
        output_tensor = pil_to_tensor(final_images_list)
        
        # ▼▼▼【原則5】返り値の厳格化 ▼▼▼
        # V1のタプル形式 `(output_tensor,)` から、io.NodeOutputオブジェクトに変更します
        # キーワード引数 `image` は、outputsで定義したidと一致させます
        return io.NodeOutput(output_tensor)


# --- V1-style Registration ---
# Even for V3 nodes, this mapping is crucial for the ComfyUI loader to find the node.
# The io.ComfyNode class handles the backward compatibility automatically.
#NODE_CLASS_MAPPINGS = {
#    "ASCIIArtNodeV3": ASCIIArtNodeV3
#}

#NODE_DISPLAY_NAME_MAPPINGS = {
#    "ASCIIArtNodeV3": "ASCII Art Generator V3"
#}

