# __init__.py (V3 comfy_entrypoint)
# 
# V3の作法に則り、ComfyExtensionを定義し、
# comfy_entrypoint関数からそのインスタンスを返すように変更します。

# 1. 必要なV3モジュールと、登録したいノードクラスをインポートします
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from .ascii_art_node_v3 import ASCIIArtNodeV3
from .ascii_art_custom_font import ASCIIArtCustomFont
from .ascii_text_layout_planner import ASCIITextLayoutPlanner
from .ascii_novel_text_art import ASCIINovelTextArt

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
from folder_paths import folder_names_and_paths
import os

if "font" not in folder_names_and_paths:
    base_path = os.path.dirname(os.path.realpath(__file__))
    font_dir = os.path.join(base_path, "font")
    if os.path.isdir(font_dir):
        folder_names_and_paths["font"] = ([font_dir], {".ttf", ".otf"})
        print(f"ASCII Art Node: Registered font directory: {font_dir}")
    else:
        print(f"ASCII Art Node: Font directory not found at {font_dir}, skipping registration.")

print("Loaded ASCII Art Custom Nodes (V3 Entrypoint)")
