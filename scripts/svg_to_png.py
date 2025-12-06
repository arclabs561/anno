#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cairosvg>=2.7.0",
# ]
# ///

"""Convert SVG files to PNG."""

import sys
from pathlib import Path

try:
    import cairosvg
except ImportError:
    print("Error: Install cairosvg with: uv pip install cairosvg")
    sys.exit(1)

def convert_svg_to_png(svg_path: Path, png_path: Path, size: int = 1024):
    """Convert SVG to PNG."""
    try:
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=size,
            output_height=size
        )
        print(f"✅ {svg_path.name} → {png_path.name}")
        return True
    except Exception as e:
        print(f"❌ Error converting {svg_path.name}: {e}")
        return False

def main():
    assets_dir = Path(__file__).parent.parent / "assets" / "logo_variations"
    
    if not assets_dir.exists():
        print(f"Error: Directory not found: {assets_dir}")
        sys.exit(1)
    
    svg_files = list(assets_dir.glob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {assets_dir}")
        sys.exit(0)
    
    print(f"Converting {len(svg_files)} SVG files to PNG...\n")
    
    converted = 0
    for svg_path in svg_files:
        png_path = svg_path.with_suffix('.png')
        if convert_svg_to_png(svg_path, png_path):
            converted += 1
    
    print(f"\n📊 Converted {converted}/{len(svg_files)} files")

if __name__ == "__main__":
    main()

