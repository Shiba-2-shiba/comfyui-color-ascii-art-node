# __init__.py (V3 comfy_entrypoint)
# 
# V3の作法に則り、ComfyExtensionを定義し、
# comfy_entrypoint関数からそのインスタンスを返すように変更します。

# 1. 必要なV3モジュールと、登録したいノードクラスをインポートします
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from aiohttp import web
from server import PromptServer
from .ascii_art_node_v3 import ASCIIArtNodeV3
from .ascii_art_custom_font import ASCIIArtCustomFont
from .ascii_text_layout_planner import ASCIITextLayoutPlanner
from .ascii_novel_text_art import ASCIINovelTextArt

WEB_DIRECTORY = "web"

# 2. ComfyExtensionを継承したクラスを作成します
class ASCIIArtExtensionV3(ComfyExtension):
    # get_node_listメソッドで、登録したいノードクラスのリストを返します
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ASCIIArtNodeV3, ASCIIArtCustomFont, ASCIITextLayoutPlanner, ASCIINovelTextArt]

# 3. comfy_entrypointという名前の非同期関数を定義し、
#    上で作成したExtensionクラスのインスタンスを返します
async def comfy_entrypoint() -> ASCIIArtExtensionV3:
    return ASCIIArtExtensionV3()


# --- Font Directory Registration ---
# この部分はノードの機能に必要なので、そのまま残します
from folder_paths import folder_names_and_paths, get_input_directory
import os
import re

if "font" not in folder_names_and_paths:
    base_path = os.path.dirname(os.path.realpath(__file__))
    font_dir = os.path.join(base_path, "font")
    if os.path.isdir(font_dir):
        folder_names_and_paths["font"] = ([font_dir], {".ttf", ".otf"})
        print(f"ASCII Art Node: Registered font directory: {font_dir}")
    else:
        print(f"ASCII Art Node: Font directory not found at {font_dir}, skipping registration.")


def _allocate_uploaded_text_path(filename: str) -> tuple[str, str]:
    base_name = os.path.basename(filename or "uploaded.txt")
    name_root, extension = os.path.splitext(base_name)
    if extension.lower() != ".txt":
        raise ValueError("Only .txt files are supported.")

    safe_root = re.sub(r"[^A-Za-z0-9._-]+", "_", name_root).strip("._") or "uploaded"
    candidate_name = f"{safe_root}{extension.lower()}"
    input_dir = get_input_directory()
    candidate_path = os.path.join(input_dir, candidate_name)
    suffix = 1

    while os.path.exists(candidate_path):
        candidate_name = f"{safe_root}_{suffix}{extension.lower()}"
        candidate_path = os.path.join(input_dir, candidate_name)
        suffix += 1

    return candidate_name, candidate_path


if not globals().get("_ASCII_TEXT_UPLOAD_ROUTE_REGISTERED"):
    _ASCII_TEXT_UPLOAD_ROUTE_REGISTERED = True

    @PromptServer.instance.routes.post("/asci/upload-text")
    async def upload_ascii_text(request):
        data = await request.post()
        upload = data.get("file")
        if upload is None or not getattr(upload, "filename", None):
            return web.json_response({"error": "Missing text file upload."}, status=400)

        try:
            output_name, output_path = _allocate_uploaded_text_path(upload.filename)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        file_bytes = upload.file.read()
        if not file_bytes:
            return web.json_response({"error": "Uploaded text file is empty."}, status=400)

        with open(output_path, "wb") as output_file:
            output_file.write(file_bytes)

        return web.json_response({"filename": output_name})

print("Loaded ASCII Art Custom Nodes (V3 Entrypoint)")
