from .ascii_art_node import ASCIIArtNode
from .ascii_art_node_v2 import ASCIIArtNodev2
from .ascii_art_single_font_node import ASCIIArtSinglefontNode

from folder_paths import folder_names_and_paths
import os

# ノードクラスのマッピングを設定
NODE_CLASS_MAPPINGS = {
    "ASCIIArtNode": ASCIIArtNode,          # ここにASCIIArtNodeを追加
    "ASCIIArtNodev2": ASCIIArtNodev2,       # 既存のASCIIArtNodev2を維持
    "ASCIIArtSinglefontNode": ASCIIArtSinglefontNode
}

# font フォルダがすでに登録されているか確認
if "font" not in folder_names_and_paths:
    base_path = os.path.dirname(os.path.realpath(__file__))
    font_dir = os.path.join(base_path, "font")
    
    # folder_names_and_paths に font フォルダを追加
    folder_names_and_paths["font"] = ([font_dir], {".ttf", ".otf"})
