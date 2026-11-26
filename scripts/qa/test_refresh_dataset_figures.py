"""Distinguish changed pixels from changes to the PNG encoder or CSV line endings."""

import importlib.util
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

same_result = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "refresh_dataset_figures.py")
)["same_result"]


class PublishedFigureChecks(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("PIL"), "pixel comparison requires Pillow"
    )
    def test_png_compares_pixels_even_when_alpha_is_unchanged(self):
        from PIL import Image

        with TemporaryDirectory() as temp:
            built, existing = (
                Path(temp) / f"{name}.png" for name in ("built", "existing")
            )
            picture = Image.new("RGBA", (16, 16), "white")
            picture.save(built, compress_level=0)
            picture.save(existing, compress_level=9)
            self.assertNotEqual(built.read_bytes(), existing.read_bytes())
            self.assertTrue(same_result(built, existing))
            picture.putpixel((3, 6), (0, 0, 0, 255))
            picture.save(existing)
            self.assertFalse(same_result(built, existing))
            existing.write_bytes(b"truncated")
            self.assertFalse(same_result(built, existing))

    def test_csv_ignores_line_endings_and_detects_changed_counts(self):
        with TemporaryDirectory() as temp:
            built, existing = (
                Path(temp) / f"{name}.csv" for name in ("built", "existing")
            )
            built.write_bytes(b"count\n3\n")
            existing.write_bytes(b"count\r\n3\r\n")
            self.assertTrue(same_result(built, existing))
            existing.write_bytes(b"count\n4\n")
            self.assertFalse(same_result(built, existing))


if __name__ == "__main__":
    unittest.main()
