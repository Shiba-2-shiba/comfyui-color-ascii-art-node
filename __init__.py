# __init__.py (Refactored for V3 Schema)

# V3では、ノード登録は各ノードファイルのcomfy_entrypointで行われるため、
# このファイルでNODE_CLASS_MAPPINGSやNODE_DISPLAY_NAME_MAPPINGSを定義する必要はなくなりました。
# したがって、これらのマッピングに関連するコードはすべて削除します。

# Import folder_paths for font directory registration
from folder_paths import folder_names_and_paths
import os

# --- Font Directory Registration ---
# フォントディレクトリをComfyUIに登録するこの部分は、V3でも引き続き必要です。
# This part remains necessary in V3 to register the font directory with ComfyUI.
if "font" not in folder_names_and_paths:
    # Get the directory where this __init__.py file is located
    base_path = os.path.dirname(os.path.realpath(__file__))
    # Construct the path to the 'font' subdirectory
    font_dir = os.path.join(base_path, "font")

    # Add the font directory path and allowed extensions to ComfyUI's folder paths
    # This allows ComfyUI to find .ttf and .otf files in the 'font' subdirectory
    if os.path.isdir(font_dir):
        folder_names_and_paths["font"] = ([font_dir], {".ttf", ".otf"})
        print(f"ASCII Art Node: Registered font directory: {font_dir}")
    else:
        print(f"ASCII Art Node: Font directory not found at {font_dir}, skipping registration.")

# Optional: Indicate successful loading of the custom node package
print("Loaded ASCII Art Custom Nodes (V3 Schema)")

# __all__も不要になったため削除します。
# __all__ is no longer needed and has been removed.
