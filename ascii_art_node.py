"""
@author: Shiba-2-shiba
@title: Colorful ASCII Art Node
@nickname: ColorASCII
@description: This node generates colorful ASCII art using custom character sets and fonts.
"""
import os
import random
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import torch
from typing import List
from folder_paths import get_filename_list, get_full_path

class CustomNode:
    pass

class ASCIIArtNode(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pixel_size": ("INT", {"default": 20, "min": 1, "max": 100}),
                "font_size_min": ("INT", {"default": 20, "min": 1, "max": 100}),
                "aspect_ratio_correction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "font_name": (get_filename_list("font"), {"tooltip": "Select a font from the font directory"}),
                "ascii_chars_filename": ("STRING", {"default": "set1.txt"}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_ascii_art"

    def generate_ascii_art(self, image, pixel_size: int, font_size_min: int, aspect_ratio_correction: float, font_name: str, ascii_chars_filename: str, brightness: float, contrast: float, mask=None):
        base_path = os.path.dirname(os.path.abspath(__file__))
        font_path = get_full_path("font", font_name)
        ascii_chars_file_path = os.path.join(base_path, ascii_chars_filename)

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
        ascii_image = self.create_ascii_art(pixelated_image, chosen_set, font_path, font_size_min, image.size)

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
        return (final_image_tensor, )

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
    
    def create_ascii_art(self, image: Image.Image, ascii_chars: str, font_path: str, font_size_min: int, original_size: tuple) -> Image.Image:
        width, height = original_size

        ascii_image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(ascii_image)

        font_sizes = [font_size_min, font_size_min * 2, font_size_min * 3]
        font_cache = {size: ImageFont.truetype(font_path, size) for size in font_sizes}

        scale_x = width / image.width
        scale_y = height / image.height

        used_positions = set()

        for y in range(image.height):
            for x in range(image.width):
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)

                if (scaled_x, scaled_y) in used_positions:
                    continue
                
                pixel_color = image.getpixel((x, y))
                ascii_char_r = ascii_chars[pixel_color[0] * len(ascii_chars) // 256]
                ascii_char_g = ascii_chars[pixel_color[1] * len(ascii_chars) // 256]
                ascii_char_b = ascii_chars[pixel_color[2] * len(ascii_chars) // 256]

                ascii_char = f"{ascii_char_r}{ascii_char_g}{ascii_char_b}"
                
                font_size = random.choice(font_sizes)
                monospace_font = font_cache[font_size]
                
                draw.text((scaled_x, scaled_y), ascii_char, font=monospace_font, fill=pixel_color)
                
                used_positions.add((scaled_x, scaled_y))

        return ascii_image
