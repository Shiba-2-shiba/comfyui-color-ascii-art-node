# Import necessary classes from node files
from .ascii_art_node_v3 import ASCIIArtNodeV3 # Import the new V3 class

# Import folder_paths for font directory registration
from folder_paths import folder_names_and_paths
import os

# --- Node Class Mappings ---
# Map node class names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "ASCIIArtNodeV3": ASCIIArtNodeV3              # Add the new V3 node
}

# --- Node Display Name Mappings ---
# Map node class names to user-friendly display names for the ComfyUI menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "ASCIIArtNodeV3": "ASCII Art Generator V3"         # Display name from V3 file
}

# --- Font Directory Registration ---
# Ensure the 'font' directory is registered for font selection dropdowns
# This part remains unchanged from your original __init__.py
if "font" not in folder_names_and_paths:
    # Get the directory where this __init__.py file is located
    base_path = os.path.dirname(os.path.realpath(__file__))
    # Construct the path to the 'font' subdirectory
    font_dir = os.path.join(base_path, "font")

    # Add the font directory path and allowed extensions to ComfyUI's folder paths
    # This allows ComfyUI to find .ttf and .otf files in the 'font' subdirectory
    folder_names_and_paths["font"] = ([font_dir], {".ttf", ".otf"})
    print(f"Registered font directory: {font_dir}") # Optional: Log registration

# Optional: Indicate successful loading of the custom node package
print("Loaded ASCII Art Custom Nodes (including V3)")

# __all__ is optional but good practice, listing exposed mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
