#!/usr/bin/env python3
"""
Generate an icon for DJ Music Analyzer application
This creates a simple icon.png file that can be used by the application
"""

import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def generate_icon():
    """Generate a simple icon.png file for the application"""
    # Create a simple icon
    icon = Image.new('RGBA', (128, 128), color=(74, 123, 188, 255))  # Blue background
    draw = ImageDraw.Draw(icon)
    
    # Draw simple waveform
    for i in range(20, 108, 4):
        amplitude = 20 + int(abs(30 * math.sin(i/10)))
        draw.line([(i, 64-amplitude), (i, 64+amplitude)], fill=(255, 255, 255, 255), width=3)
    
    # Draw DJ text
    try:
        font = ImageFont.truetype("Arial", 24)
    except Exception:
        try:
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
    
    draw.text((45, 40), "DJ", fill=(255, 255, 255, 255), font=font)
    
    # Save the icon
    icon_path = Path(__file__).parent / "icon.png"
    icon.save(icon_path)
    print(f"Icon saved to {icon_path}")
    return icon_path


if __name__ == "__main__":
    generate_icon()
