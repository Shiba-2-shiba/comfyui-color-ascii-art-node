# __init__.py (Final V3 Compatible)
# This file imports the node class mappings from the node file
# and exposes them to ComfyUI. This is the standard and most reliable way.

# 1. Import the node class mappings from the node file.
#    The WEB_DIRECTORY is also important for any web assets.
from .ascii_art_node_v3 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 2. Expose the mappings to ComfyUI.
#    This allows the loader to find the nodes.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


# --- Font Directory Registration ---
# This part remains necessary to make fonts available in the node's dropdown.
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

print("Loaded ASCII Art Custom Nodes (V3 Compatible)")
