"""Check the native figure publication path and distinguish rendered changes from encoding."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figure_catalog
import refresh_dataset_figures as figures


class PublishedFigureChecks(unittest.TestCase):
    def test_export_contains_only_figure_catalog_products(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dataset-coverage.md").write_text("aggregate")
            output = root / "public"
            allowed = {"panel.png", "panel.svg", figures.DATASET_REFERENCE, figures.CATALOG_NAME}

            def render(directory, config, anno):
                directory.mkdir()
                for name in allowed | {"source.cache", "debug-predictions.json", "model.onnx"}:
                    (directory / name).write_bytes(b"fixture")
                return []

            with (
                patch.object(figures, "EVAL", root),
                patch.object(figures, "SUMMARIES", ()),
                patch.object(figures, "render", render),
                patch.object(figures, "coverage_text", return_value="aggregate"),
                patch.object(sys, "argv", ["render", "--output", str(output)]),
            ):
                figures.main()
            self.assertEqual({p.name for p in output.iterdir()}, allowed)

    def test_catalog_reports_a_named_study_whose_input_was_removed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "figure-studies.csv").write_text(
                "study,datasets,guide\nheldout-paired,WikiGold,guide.md\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Missing native inputs.*heldout-paired"):
                figure_catalog.registered_studies(root, [])

    def test_registered_inputs_reject_unreviewed_metadata_and_non_native_fields(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "review-paired.json"
            native = {"schema": "anno.paired/v1", "rows": [{"observations": 4}]}
            typed = subprocess.CompletedProcess([], 0, stdout=json.dumps(native).encode())
            source.write_text(json.dumps({**native, "metadata": {"case": "local only"}}))
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                figures.public_report(source, "", "anno")
            source.write_text(json.dumps({
                **native, "rows": [{"observations": 4, "case": "local only"}],
            }))
            with (
                patch.object(figures.subprocess, "run", return_value=typed),
                self.assertRaisesRegex(ValueError, "native fields"),
            ):
                figures.public_report(source, "", "anno")
            source.write_text(
                '{"schema":"anno.paired/v1","title":"local only","title":"Public"}'
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                figures.public_report(source, "", "anno")

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
                ("html", b"<html>\n<p>3</p>\n</html>\n"),
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
                    existing.write_bytes(b"\xff")
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
            sources = [
                root / f"{study}.json"
                for study in (
                    "left-tradeoff",
                    "right-tradeoff",
                    "genia-span-overlap",
                    "wikigold-entity-errors",
                    "comparison-paired",
                )
            ]
            for source in sources:
                view = source.stem.rsplit("-", 1)[1]
                source.write_text(
                    json.dumps({"schema": f"anno.{view}/v1", "title": source.stem}),
                    encoding="utf-8",
                )
            (root / "figure-studies.csv").write_text(
                "study,datasets,guide\n" + "".join(f"{p.stem},Sample,guide.md\n" for p in sources)
            )
            scratch = root / "local-review-errors.json"
            scratch.write_text(
                json.dumps({"schema": "anno.errors/v1", "metadata": {"case": "local only"}})
            )
            (root / "other-measurements.json").touch()
            selected = figure_catalog.registered_studies(
                root, figures.native_paths(root, "json")
            )
            self.assertEqual(list(selected), sorted(sources))

            def draw(command, **options):
                if command[-2:] == ["--format", "json"]:
                    self.assertEqual(command[:2], [cli, "viz"])
                    self.assertIn(command[2], figures.NATIVE_VIEWS)
                    return subprocess.CompletedProcess(command, 0, stdout=options["input"])
                binary, viz, kind, source, option, dest = command
                self.assertEqual(
                    (binary, viz, kind, option),
                    (cli, "viz", "render", "--output"),
                )
                self.assertTrue(options["check"])
                title = json.loads(source.read_text(encoding="utf-8"))["title"]
                if dest.suffix == ".html":
                    dest.write_text(f"<html><title>{title}</title></html>", encoding="utf-8")
                    return
                dest.write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="30">'
                    f'<text x="4" y="21" font-family="DejaVu Sans">{title}</text></svg>',
                    encoding="utf-8",
                )

            with patch.object(figures, "build_anno", return_value=cli) as build:
                with patch.object(
                    figures.subprocess, "run", side_effect=draw
                ) as render:
                    figures.render_native(selected, output, anno=None)
                build.assert_called_once_with()
                self.assertEqual(render.call_count, 3 * len(sources))
            self.assertFalse((output / scratch.name).exists())
            (root / "dataset-ci-survey.json").write_text('{"day_utc":"2026-01-01"}')
            with patch.object(figure_catalog, "datasets", return_value=[]):
                figure_catalog.write_catalog(root, output, selected)
            catalog = (output / figure_catalog.NAME).read_text()
            self.assertNotIn("local only", catalog)
            self.assertEqual(
                [study["id"] for study in json.loads(catalog)["studies"]],
                [source.stem for source in sorted(sources)],
            )

            for source in sources:
                with self.subTest(source=source.name):
                    self.assertEqual(
                        source.read_bytes(), (output / source.name).read_bytes()
                    )
                    svg = output / source.with_suffix(".svg").name
                    self.assertIn(source.stem, svg.read_text(encoding="utf-8"))
                    self.assertIn(source.stem, svg.with_suffix(".html").read_text(encoding="utf-8"))
                    with Image.open(svg.with_suffix(".png")) as preview:
                        self.assertEqual(preview.size, (120, 30))
                        self.assertIsNotNone(
                            preview.getbbox(), "preview dropped the SVG text"
                        )

            # A study borrowing the name of a prior generator must not replace its image.
            with self.assertRaisesRegex(ValueError, "More than one figure"):
                figures.render_native(selected, output, anno=cli)


if __name__ == "__main__":
    unittest.main()
