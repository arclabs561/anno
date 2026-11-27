"""Check the native figure publication path and distinguish rendered changes from encoding."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refresh_dataset_figures as figures


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
            self.assertTrue(figures.same_result(built, existing))
            picture.putpixel((3, 6), (0, 0, 0, 255))
            picture.save(existing)
            self.assertFalse(figures.same_result(built, existing))
            existing.write_bytes(b"truncated")
            self.assertFalse(figures.same_result(built, existing))

    def test_text_outputs_ignore_line_endings_and_detect_changes(self):
        with TemporaryDirectory() as temp:
            for suffix, contents in (
                ("csv", b"count\n3\n"),
                ("svg", b"<svg>\n<desc>3</desc>\n</svg>\n"),
            ):
                with self.subTest(suffix=suffix):
                    built, existing = (
                        Path(temp) / f"{name}.{suffix}"
                        for name in ("built", "existing")
                    )
                    built.write_bytes(contents)
                    existing.write_bytes(contents.replace(b"\n", b"\r\n"))
                    self.assertTrue(figures.same_result(built, existing))
                    existing.write_bytes(contents.replace(b"3", b"4"))
                    self.assertFalse(figures.same_result(built, existing))

    @unittest.skipUnless(
        all(importlib.util.find_spec(dep) for dep in ["matplotlib", "resvg_py", "PIL"]),
        "native chart previews require the figure build dependencies",
    )
    def test_native_studies_share_one_build_and_export_their_own_svg_png_and_input(
        self,
    ):
        from PIL import Image

        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "figures"
            output.mkdir()
            cli = root / "anno"
            sources = [root / f"{study}-tradeoff.json" for study in ("left", "right")]
            for source in sources:
                source.write_text(json.dumps({"title": source.stem}), encoding="utf-8")

            def draw(command, *, check):
                binary, viz, kind, source, option, dest = command
                self.assertEqual(
                    (binary, viz, kind, option), (cli, "viz", "tradeoff", "--output")
                )
                self.assertTrue(check)
                title = json.loads(source.read_text(encoding="utf-8"))["title"]
                dest.write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="30">'
                    f'<text x="4" y="21" font-family="DejaVu Sans">{title}</text></svg>',
                    encoding="utf-8",
                )

            with patch.object(figures, "build_anno", return_value=cli) as build:
                with patch.object(
                    figures.subprocess, "run", side_effect=draw
                ) as render:
                    figures.render_tradeoffs(sources, output, anno=None)
                build.assert_called_once_with()
                self.assertEqual(render.call_count, len(sources))

            for source in sources:
                with self.subTest(source=source.name):
                    self.assertEqual(
                        source.read_bytes(), (output / source.name).read_bytes()
                    )
                    svg = output / source.with_suffix(".svg").name
                    self.assertIn(source.stem, svg.read_text(encoding="utf-8"))
                    with Image.open(svg.with_suffix(".png")) as preview:
                        self.assertEqual(preview.size, (120, 30))
                        self.assertIsNotNone(
                            preview.getbbox(), "preview dropped the SVG text"
                        )

            # A study borrowing the name of a prior generator must not replace its image.
            with self.assertRaisesRegex(ValueError, "More than one figure"):
                figures.render_tradeoffs(sources, output, anno=cli)


if __name__ == "__main__":
    unittest.main()
