import logging
import os
from typing import Dict, List, Tuple

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from .ascii_drawing import _get_font
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from ascii_drawing import _get_font


logger = logging.getLogger("ComfyUI.ASCIIText.Utils")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


SUPPORTED_ENCODINGS = ("utf-8", "utf-8-sig", "cp932", "shift_jis")
NEWLINE_MODES = ("remove", "space", "preserve")
TEXT_SHORTAGE_MODES = ("error", "loop", "truncate_blank")


def list_input_text_files() -> List[str]:
    if folder_paths is None:
        return sorted(
            f for f in os.listdir(".")
            if os.path.isfile(f) and f.lower().endswith(".txt")
        )

    input_dir = folder_paths.get_input_directory()
    return sorted(
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".txt")
    )


def resolve_input_text_file(text_file: str) -> str:
    if not text_file:
        raise ValueError("text_file is empty.")
    if not text_file.lower().endswith(".txt"):
        raise ValueError(f"Only .txt files are supported: {text_file}")

    if folder_paths is None:
        if os.path.isfile(text_file):
            return text_file
        candidate = os.path.abspath(text_file)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Text file not found: {text_file}")

    if not folder_paths.exists_annotated_filepath(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")

    resolved = folder_paths.get_annotated_filepath(text_file)
    if not resolved.lower().endswith(".txt"):
        raise ValueError(f"Only .txt files are supported: {text_file}")
    return resolved


def normalize_text_stream(text: str, newline_mode: str) -> str:
    if newline_mode not in NEWLINE_MODES:
        raise ValueError(f"Unsupported newline_mode: {newline_mode}")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    normalized = normalized.replace("\t", " ")

    if newline_mode == "remove":
        normalized = normalized.replace("\n", "")
    elif newline_mode == "space":
        normalized = normalized.replace("\n", " ")

    return normalized


def load_text_file(file_path: str, encoding: str = "utf-8", newline_mode: str = "remove") -> Tuple[str, str]:
    if not file_path:
        raise ValueError("text_file_path is empty.")

    resolved_path = file_path
    if not os.path.isfile(resolved_path):
        resolved_path = resolve_input_text_file(file_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    encodings_to_try = SUPPORTED_ENCODINGS if encoding == "auto" else (encoding,)
    last_error = None
    loaded_text = None
    selected_encoding = encodings_to_try[0]

    for candidate in encodings_to_try:
        try:
            with open(resolved_path, "r", encoding=candidate) as file:
                loaded_text = file.read()
            selected_encoding = candidate
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if loaded_text is None:
        if last_error:
            raise RuntimeError(
                f"Failed to decode text file '{resolved_path}' with encoding '{encoding}'."
            ) from last_error
        raise RuntimeError(f"Failed to read text file '{resolved_path}'.")

    return normalize_text_stream(loaded_text, newline_mode), selected_encoding


def calculate_text_grid(image_size: Tuple[int, int], pixel_size: int, aspect_ratio_correction: float) -> Tuple[int, int, int]:
    if pixel_size < 1:
        raise ValueError("pixel_size must be at least 1.")
    if aspect_ratio_correction <= 0:
        raise ValueError("aspect_ratio_correction must be positive.")

    image_width, image_height = image_size
    grid_width = max(1, image_width // pixel_size)
    effective_pixel_height = pixel_size * aspect_ratio_correction
    grid_height = max(1, int(image_height / effective_pixel_height))
    required_chars = grid_width * grid_height
    return grid_width, grid_height, required_chars


def _is_renderable_char(char: str, font_path: str, font_size: int) -> bool:
    if char in ("", "\n", "\r"):
        return True
    if char.isspace():
        return True

    font = _get_font(font_path, font_size)
    if font is None:
        return False

    try:
        bbox = font.getbbox(char)
    except Exception:
        return False

    if bbox is None:
        return False

    left, top, right, bottom = bbox
    return (right - left) > 0 and (bottom - top) > 0


def inspect_font_support(text: str, font_path: str, font_size: int, limit: int = 5000) -> Dict[str, object]:
    sample_text = text[: max(0, limit)]
    checked_chars = 0
    supported_chars = 0
    unsupported_preview: List[str] = []
    seen_unsupported = set()

    for char in sample_text:
        checked_chars += 1
        if _is_renderable_char(char, font_path, font_size):
            supported_chars += 1
            continue

        if char not in seen_unsupported:
            unsupported_preview.append(repr(char)[1:-1])
            seen_unsupported.add(char)

    ratio = 1.0 if checked_chars == 0 else supported_chars / checked_chars
    return {
        "checked_chars": checked_chars,
        "supported_chars": supported_chars,
        "unsupported_chars": unsupported_preview,
        "glyph_supported_ratio": ratio,
    }


def build_text_cells(
    text: str,
    required_chars: int,
    grid_width: int,
    shortage_mode: str = "error",
    preserve_newlines: bool = False,
) -> Tuple[List[str], int]:
    if shortage_mode not in TEXT_SHORTAGE_MODES:
        raise ValueError(f"Unsupported text_shortage_mode: {shortage_mode}")
    if required_chars < 0:
        raise ValueError("required_chars must be non-negative.")
    if grid_width < 1:
        raise ValueError("grid_width must be at least 1.")

    if required_chars == 0:
        return [], 0

    if not text:
        if shortage_mode == "error":
            raise ValueError("Text content is empty after preprocessing.")
        if shortage_mode == "truncate_blank":
            return [" "] * required_chars, 0
        raise ValueError("Loop mode cannot be used with empty text.")

    cells: List[str] = []
    text_index = 0
    consumed_chars = 0
    consecutive_newlines = 0

    while len(cells) < required_chars:
        if text_index >= len(text):
            if shortage_mode == "error":
                raise ValueError(
                    f"Text is too short for the current layout. Required: {required_chars}, available: {len(text)}."
                )
            if shortage_mode == "truncate_blank":
                cells.extend([" "] * (required_chars - len(cells)))
                break
            text_index = 0

        char = text[text_index]
        text_index += 1
        consumed_chars += 1

        if preserve_newlines and char == "\n":
            consecutive_newlines += 1
            if consecutive_newlines > len(text) * 2:
                raise ValueError("Text contains too many newlines to build a stable layout.")
            current_column = len(cells) % grid_width
            pad_count = grid_width - current_column if current_column != 0 else grid_width
            cells.extend([" "] * min(pad_count, required_chars - len(cells)))
            continue

        consecutive_newlines = 0
        cells.append(char)

    return cells[:required_chars], consumed_chars


def make_layout_report(
    file_path: str,
    selected_encoding: str,
    grid_width: int,
    grid_height: int,
    required_chars: int,
    usable_chars: int,
    render_width: int,
    render_height: int,
    glyph_supported_ratio: float,
    unsupported_chars: List[str],
    warnings: List[str],
) -> str:
    lines = [
        f"text_file_path={file_path}",
        f"text_encoding={selected_encoding}",
        f"grid={grid_width}x{grid_height}",
        f"required_chars={required_chars}",
        f"usable_chars={usable_chars}",
        f"render_size={render_width}x{render_height}",
        f"glyph_supported_ratio={glyph_supported_ratio:.3f}",
    ]

    if unsupported_chars:
        preview = ", ".join(unsupported_chars[:12])
        lines.append(f"unsupported_chars_preview={preview}")

    if warnings:
        lines.append("warnings=" + " | ".join(warnings))

    return "\n".join(lines)
