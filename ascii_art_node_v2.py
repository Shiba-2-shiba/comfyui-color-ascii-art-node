import os
import random
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import torch
from typing import List
from folder_paths import get_filename_list, get_full_path  # Adjusted import path

class CustomNode:
    pass

class ASCIIArtNodev2(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pixel_size": ("INT", {"default": 6, "min": 1, "max": 100}),
                "font_size_min": ("INT", {"default": 20, "min": 1, "max": 100}),
                "aspect_ratio_correction": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 10.0}),
                "font_name": (get_filename_list("font"), {"tooltip": "Select a font from the font directory"}),  # Modified to use the font list
                "ascii_chars_filename": ("STRING", {"default": "ascii_custom_characters.txt"}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_ascii_art"

    def generate_ascii_art(self, image, pixel_size: int, font_size_min: int, aspect_ratio_correction: float, font_name: str, ascii_chars_filename: str, brightness: float, contrast: float):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        font_path = get_full_path("font", font_name)  # Use the full path function for font
        ascii_chars_file_path = os.path.join(os.path.dirname(__file__), ascii_chars_filename)

        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='RGB')

        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: Expected a torch.Tensor, PIL.Image, or NumPy array, but got {type(image)}")

        ascii_sets = self.load_custom_characters(ascii_chars_file_path)
        chosen_set = self.choose_random_set(ascii_sets)
        pixelated_image = self.pixelate_image(image, pixel_size, aspect_ratio_correction, brightness, contrast)
        ascii_image = self.create_ascii_art(pixelated_image, chosen_set, font_path, font_size_min, image.size, device)

        ascii_image_np = np.array(ascii_image) / 255.0
        ascii_image_tensor = torch.tensor(ascii_image_np).permute(2, 0, 1).unsqueeze(0)

        ascii_image_tensor = ascii_image_tensor.permute(0, 2, 3, 1)
        return (ascii_image_tensor, )

    def load_custom_characters(self, file_path: str) -> List[str]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        sets = [line.strip().split(': ')[1] for line in lines if line.startswith('Set')]
        return sets
    
    def choose_random_set(self, sets: List[str]) -> str:
        return random.choice(sets)
    
    def pixelate_image(self, image: Image.Image, pixel_size: int, aspect_ratio_correction: float, brightness: float, contrast: float) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

        new_width = image.width // pixel_size
        new_height = int(image.height // (pixel_size * aspect_ratio_correction))
        small = image.resize((new_width, new_height), resample=Image.NEAREST)
        return small
    
    def create_ascii_art(self, image: Image.Image, ascii_chars: str, font_path: str, font_size_min: int, original_size: tuple, device: torch.device) -> Image.Image:
        width, height = original_size

        ascii_image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(ascii_image)

        font_cache = {size: ImageFont.truetype(font_path, size) for size in range(font_size_min, font_size_min * 4 + 1, font_size_min)}

        scale_x = width / image.width
        scale_y = height / image.height

        # Convert image to numpy array and transfer to GPU
        image_np = np.array(image)
        image_torch = torch.tensor(image_np, device=device).float()  # Convert to float to avoid dtype issues

        # Convert pixel values to brightness ratios
        brightness_values = image_torch.mean(dim=2) / 255.0
        
        # Decide on font sizes based on brightness
        font_sizes = torch.where(brightness_values > 0.5, 
                                 torch.tensor(font_size_min * 2, device=device), 
                                 torch.tensor(font_size_min, device=device))
        
        # Batch processing for faster execution
        ascii_chars_torch = torch.tensor([ord(c) for c in ascii_chars], device=device)
        brightness_scaled = (brightness_values * (len(ascii_chars) - 1)).long()
        ascii_indices = ascii_chars_torch[brightness_scaled]

        font_sizes_np = font_sizes.cpu().numpy()

        for y in range(image.height):
            for x in range(image.width):
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                
                pixel_color = tuple(image_np[y, x])
                ascii_char = chr(ascii_indices[y, x].item())

                font_size = font_sizes_np[y, x]
                font = font_cache[font_size]

                draw.text((scaled_x, scaled_y), ascii_char, font=font, fill=pixel_color)

        return ascii_image