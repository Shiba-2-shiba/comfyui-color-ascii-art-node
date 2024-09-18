import os
import random
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import List
from folder_paths import get_filename_list, get_full_path

class CustomNode:
    pass

class ASCIIArtNodev2(CustomNode):
    font_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {"default": 20, "min": 1, "max": 100}),
                "font_size_min": ("INT", {"default": 20, "min": 1, "max": 100}),
                "aspect_ratio_correction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "font_name": (get_filename_list("font"), {"tooltip": "Select a font from the font directory"}),
                "ascii_chars_filename": ("STRING", {"default": "set1.txt"}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0})
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_ascii_art"

    def generate_ascii_art(self, image, pixel_size: int, font_size_min: int, aspect_ratio_correction: float, font_name: str, ascii_chars_filename: str, brightness: float, contrast: float, mask=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        font_path = get_full_path("font", font_name)
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

        if mask is not None:
            mask = mask.squeeze().cpu().numpy()
            mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask = mask.resize(image.size, Image.LANCZOS)
            ascii_image = ascii_image.convert('RGBA')
            final_image = Image.new('RGBA', image.size)
            final_image.paste(image.convert('RGBA'), (0, 0), Image.fromarray(255 - np.array(mask), mode='L'))
            final_image.paste(ascii_image, (0, 0), mask)
        else:
            final_image = ascii_image

        final_image_np = np.array(final_image) / 255.0
        final_image_tensor = torch.tensor(final_image_np).permute(2, 0, 1).unsqueeze(0)

        final_image_tensor = final_image_tensor.permute(0, 2, 3, 1)
        return (final_image_tensor,)

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

        with autocast():
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float().to('cuda' if torch.cuda.is_available() else 'cpu') / 255.0
            new_width = image_tensor.shape[2] // pixel_size
            new_height = int(image_tensor.shape[1] // (pixel_size * aspect_ratio_correction))
            image_tensor_resized = F.interpolate(image_tensor.unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze(0)
            return Image.fromarray((image_tensor_resized.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    def create_ascii_art(self, image: Image.Image, ascii_chars: str, font_path: str, font_size_min: int, original_size: tuple, device: torch.device) -> Image.Image:
        width, height = original_size
        ascii_image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(ascii_image)

        # Adjust font caching to handle a wider range of sizes
        if font_path not in self.font_cache:
            self.font_cache[font_path] = {size: ImageFont.truetype(font_path, size) for size in range(font_size_min, font_size_min * 4 + 1, font_size_min)}

        font_cache = self.font_cache[font_path]

        scale_x = width / image.width
        scale_y = height / image.height

        image_np = np.array(image)
        brightness_values = image_np.mean(axis=2) / 255.0
        font_sizes = np.where(brightness_values > 0.5, font_size_min * 2, font_size_min)

        # Ensure all font sizes are in cache, otherwise fallback to nearest size
        batch_size = 20
        for y in range(0, image.height, batch_size):  # Process rows in batches
            for x in range(0, image.width, batch_size):  # Process columns in batches
                for dy in range(min(batch_size, image.height - y)):  # Ensure no pixels are left out
                    for dx in range(min(batch_size, image.width - x)):  # Ensure no pixels are left out
                        scaled_x = int((x + dx) * scale_x)
                        scaled_y = int((y + dy) * scale_y)

                        pixel_color = tuple(image_np[y + dy, x + dx])
                        ascii_char = ascii_chars[int(brightness_values[y + dy, x + dx] * (len(ascii_chars) - 1))]
                        font_size = font_sizes[y + dy, x + dx]
                        
                        # Ensure we use the nearest available font size if the requested size is not cached
                        nearest_size = min(font_cache.keys(), key=lambda s: abs(s - font_size))
                        font = font_cache[nearest_size]

                        draw.text((scaled_x, scaled_y), ascii_char, font=font, fill=pixel_color)

        return ascii_image
