#!/usr/bin/env python3
"""
Convert multiple JPEG images into a single PDF document.

Usage:
    python convert_images_to_pdf.py image1.jpg image2.jpg image3.jpg image4.jpg -o output.pdf

Or with all JPEGs in a folder:
    python convert_images_to_pdf.py *.jpg -o output.pdf
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def add_caption_to_image(img, caption_text, caption_height=80, bg_color='white', text_color='black'):
    """
    Add a caption below an image.

    Args:
        img: PIL Image object
        caption_text: Text to display as caption
        caption_height: Height of caption area in pixels (default: 80)
        bg_color: Background color for caption area (default: 'white')
        text_color: Text color for caption (default: 'black')

    Returns:
        New PIL Image with caption added
    """
    # Create new image with extra space for caption
    width, height = img.size
    new_height = height + caption_height
    new_img = Image.new('RGB', (width, new_height), bg_color)

    # Paste original image at top
    new_img.paste(img, (0, 0))

    # Add caption text
    draw = ImageDraw.Draw(new_img)

    # Try to use a nice font, fall back to default if not available
    try:
        # Try to load a system font (adjust size based on caption height)
        font_size = int(caption_height * 0.4)  # 40% of caption height
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            # Try alternative common fonts
            font_size = int(caption_height * 0.4)
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            # Fall back to default font
            font = ImageFont.load_default()

    # Calculate text position (centered horizontally, centered vertically in caption area)
    bbox = draw.textbbox((0, 0), caption_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = (width - text_width) // 2
    text_y = height + (caption_height - text_height) // 2

    # Draw the text
    draw.text((text_x, text_y), caption_text, fill=text_color, font=font)

    return new_img


def convert_images_to_pdf(image_paths, output_pdf, sort_images=True, add_captions=True):
    """
    Convert multiple images to a single PDF file.

    Args:
        image_paths: List of paths to image files
        output_pdf: Path for the output PDF file
        sort_images: Whether to sort images alphabetically (default: True)
        add_captions: Whether to add filename as caption under each image (default: True)
    """
    # Convert to Path objects and filter valid images
    valid_images = []
    for img_path in image_paths:
        p = Path(img_path)
        if p.exists() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            valid_images.append(p)
        else:
            print(f"WARNING: Skipping invalid/missing file: {img_path}")

    if not valid_images:
        print("ERROR: No valid images found!")
        return False

    # Sort alphabetically if requested
    if sort_images:
        valid_images.sort()

    print(f"Converting {len(valid_images)} images to PDF...")
    if add_captions:
        print("   Adding filenames as captions...")

    # Open all images and convert to RGB (required for PDF)
    images = []
    for img_path in valid_images:
        try:
            img = Image.open(img_path)
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Add caption with filename
            if add_captions:
                img = add_caption_to_image(img, img_path.name)

            images.append(img)
            print(f"  [OK] Loaded: {img_path.name} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"  [FAIL] Failed to load {img_path.name}: {e}")

    if not images:
        print("ERROR: No images could be loaded!")
        return False

    # Save as PDF
    try:
        # First image as base, rest as additional pages
        images[0].save(
            output_pdf,
            save_all=True,
            append_images=images[1:],
            resolution=100.0,
            quality=95,
            optimize=False
        )

        output_size = Path(output_pdf).stat().st_size / (1024 * 1024)  # MB
        print(f"\nSUCCESS: PDF created successfully!")
        print(f"   Output: {output_pdf}")
        print(f"   Pages: {len(images)}")
        print(f"   Size: {output_size:.2f} MB")
        return True

    except Exception as e:
        print(f"ERROR: Failed to create PDF: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert multiple JPEG images into a single PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert 4 specific images
  python convert_images_to_pdf.py img1.jpg img2.jpg img3.jpg img4.jpg -o document.pdf

  # Convert all JPEGs in current folder
  python convert_images_to_pdf.py *.jpg -o document.pdf

  # Keep original order (don't sort)
  python convert_images_to_pdf.py img1.jpg img2.jpg -o doc.pdf --no-sort
        """
    )

    parser.add_argument(
        'images',
        nargs='+',
        help='Image files to convert (JPEG/PNG)'
    )

    parser.add_argument(
        '-o', '--output',
        default='output.pdf',
        help='Output PDF filename (default: output.pdf)'
    )

    parser.add_argument(
        '--no-sort',
        action='store_true',
        help='Do not sort images alphabetically'
    )

    parser.add_argument(
        '--no-captions',
        action='store_true',
        help='Do not add filenames as captions under images'
    )

    args = parser.parse_args()

    # Convert images to PDF
    success = convert_images_to_pdf(
        args.images,
        args.output,
        sort_images=not args.no_sort,
        add_captions=not args.no_captions
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
