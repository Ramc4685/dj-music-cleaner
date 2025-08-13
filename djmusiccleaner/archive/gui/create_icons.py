"""
Simple icon creation script to generate icons for PyInstaller packaging
This creates a basic text-based icon for demonstration purposes
"""
import os
from PIL import Image, ImageDraw, ImageFont

def create_icon(size=512):
    # Create a new image with white background
    img = Image.new('RGB', (size, size), color=(53, 78, 200))
    d = ImageDraw.Draw(img)
    
    try:
        # Try to use a built-in font
        font = ImageFont.truetype("Arial.ttf", size=size//4)
    except:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Draw text
    d.text((size//6, size//3), "DJ\nMUSIC", fill=(255, 255, 255), font=font)
    
    return img

def main():
    # Create output directory if it doesn't exist
    os.makedirs("gui_app/resources", exist_ok=True)
    
    # Create icon
    icon = create_icon()
    
    # Save as PNG
    icon.save("gui_app/resources/icon.png")
    
    # Save as ICO for Windows
    icon.save("gui_app/resources/icon.ico")
    
    # Mac OS icons use .icns format, but we'll save as PNG for simplicity
    # In a real application, you would convert to .icns format
    icon.save("gui_app/resources/icon.icns")
    
    print("Icons created successfully!")

if __name__ == "__main__":
    main()
