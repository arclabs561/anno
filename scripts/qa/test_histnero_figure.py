"""Check the published raster as well as the SVG geometry used by the figure build."""

from pathlib import Path
import unittest

from PIL import Image, ImageChops


class HistneroFigureTest(unittest.TestCase):
    def test_png_keeps_whitespace_around_all_content(self):
        png = (
            Path(__file__).resolve().parents[2]
            / "docs/figures/histnero-person-errors.png"
        )
        with Image.open(png) as image:
            rgb = image.convert("RGB")
        ink = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
        bounds = ink.getbbox()
        self.assertIsNotNone(bounds)
        left, top, right, bottom = bounds
        # The layout uses 12 pt side margins at 326 pt wide. Allow for glyph
        # overhang and raster rounding while catching text clipped by the canvas.
        minimum = rgb.width * 0.02
        self.assertGreaterEqual(
            min(left, top, rgb.width - right, rgb.height - bottom), minimum
        )


if __name__ == "__main__":
    unittest.main()
