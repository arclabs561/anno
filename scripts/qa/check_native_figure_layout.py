# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright==1.62.0"]
# ///
"""Check exported Anno SVG and HTML figures in an offline system Chrome."""

import argparse
import csv
from collections import Counter, defaultdict
import json
import shutil
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refresh_dataset_figures import native_paths

# The document has 16 px of padding on each side. The wider case embeds a
# 680 px figure, matching the gallery. The narrower one exercises mobile CSS.
WIDTHS = (390, 712)
EMBED = """<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:16px}svg{display:block;width:100%;height:auto}</style>"""

SVG_LAYOUT = """() => {
    const svg = document.querySelector('svg');
    const frame = svg.getBoundingClientRect();
    const problems = [];
    if (!svg.hasAttribute('viewBox') || !frame.height || !svg.querySelector('text')) {
        problems.push('missing responsive chart or text labels');
    }
    const named = (svg.getAttribute('aria-labelledby') || '').trim().split(/ +/);
    if (svg.getAttribute('role') !== 'img' ||
        !named.some(id => document.getElementById(id)?.textContent.trim())) {
        problems.push('chart has no accessible name');
    }
    const placed = [];
    for (const label of svg.querySelectorAll('text')) {
        const rect = label.getBoundingClientRect();
        // Allow one CSS pixel for subpixel font and viewport rounding.
        if (rect.left < frame.left - 1 || rect.right > frame.right + 1 ||
            rect.top < frame.top - 1 || rect.bottom > frame.bottom + 1) {
            problems.push('label clipped by viewBox: ' + label.textContent.trim().slice(0, 90));
        }
        // Multi-line text can have a bounding box with intentional empty space.
        if (!rect.width || !rect.height || label.querySelector('tspan')) continue;
        for (const prior of placed) {
            const a = prior.box;
            const w = Math.min(rect.right, a.right) - Math.max(rect.left, a.left);
            const h = Math.min(rect.bottom, a.bottom) - Math.max(rect.top, a.top);
            if (w > 1 && h > .7 * Math.min(rect.height, a.height)) {
                problems.push('labels overlap: ' + prior.text + ' / ' + label.textContent.trim());
            }
        }
        placed.push({box: rect, text: label.textContent.trim()});
    }
    return problems;
}"""

HTML_LAYOUT = """({kind, labels}) => {
    const problems = [];
    const visible = element => !!element?.getBoundingClientRect().height;
    if (!document.querySelector('meta[name="viewport"]')) problems.push('missing mobile viewport');
    for (const disclosure of document.querySelectorAll('details')) disclosure.open = true;
    if (document.documentElement.scrollWidth > innerWidth + 1) problems.push('page scrolls sideways');
    const tables = [...document.querySelectorAll('table')].filter(visible);
    if (!tables.length || tables.some(table => !table.caption?.textContent.trim())) {
        problems.push('missing visible data table or caption');
    }
    if (kind === 'paired') {
        const sections = [...document.querySelectorAll('section')];
        for (const label of labels) {
            const section = sections.find(s => s.querySelector('h2')?.textContent === label);
            const table = section?.querySelector('table');
            const columns = table?.querySelectorAll('thead th[scope="col"]');
            const rows = [...(table?.querySelectorAll('tbody tr') || [])];
            if (!visible(table) || table.getBoundingClientRect().right > innerWidth + 1 ||
                columns.length < 3 || rows.length === 0 ||
                rows.some(row => !row.querySelector('th[scope="row"]') || !row.querySelector('td'))) {
                problems.push('paired cohort missing its accessible measurements: ' + label);
            }
        }
    }
    if (kind === 'overlap') {
        for (const cell of document.querySelectorAll('td,thead th')) {
            const header = cell.closest('thead');
            if (header && header.getBoundingClientRect().width <= 1) continue;
            const word = cell.textContent.trim();
            if (!/^[a-zA-Z]+$|^[0-9,.]+%?$/.test(word) || !visible(cell)) continue;
            const range = document.createRange();
            range.selectNodeContents(cell);
            if (range.getClientRects().length > 1) problems.push('word or number splits across lines: ' + word);
        }
    }
    if (['tradeoff', 'overlap'].includes(kind) && tables.some(table => {
        const frame = table.parentElement.closest('.scroll');
        return frame && frame.scrollWidth > frame.clientWidth + 1;
    })) problems.push('data table needs horizontal scrolling');
    if (kind === 'tradeoff' && innerWidth <= 390) {
        const rows = tables.flatMap(table => [...table.querySelectorAll('tbody tr')]);
        if (!rows.length) problems.push('missing queue settings');
        for (const row of rows) {
            const cells = [...row.querySelectorAll('td')];
            const rowBox = row.getBoundingClientRect();
            if (!visible(row.querySelector('th[scope="row"]')) || !cells.length ||
                cells.some(cell => {
                    const box = cell.getBoundingClientRect();
                    const before = getComputedStyle(cell, '::before');
                    return !cell.dataset.label || !visible(cell) ||
                        ['none', 'normal', '""'].includes(before.content) ||
                        box.left < -1 || box.right > innerWidth + 1;
                }) || cells.every(cell => Math.abs(cell.getBoundingClientRect().top -
                    cells[0].getBoundingClientRect().top) < 1) || rowBox.width > innerWidth) {
                problems.push('queue setting needs labeled, stacked counts within the viewport: ' +
                    row.querySelector('th')?.textContent);
            }
        }
    }
    return problems;
}"""


def check_explorer(page, directory, width):
    """Exercise the offline filters against the published receipts, including a cross-row false positive."""
    with (directory / "dataset-evaluation-coverage.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        outcomes = list(csv.DictReader(file))
    with (directory / "dataset-density-units.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        units = Counter(row["id"] for row in csv.DictReader(file))
    ids = {r["id"] for r in outcomes}
    by_backend, by_status, same_record = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
    )
    for row in outcomes:
        by_status[row["status"]].add(row["id"])
        for backend in {row["backend"], *row["reported_backend"].split(" | ")} - {
            "",
            "unreported",
        }:
            by_backend[backend].add(row["id"])
            same_record[backend, row["status"]].add(row["id"])
    if not by_backend or not units:
        return ["missing backend outcomes or partition observations"]
    # Prefer a pair where a dataset-level AND would wrongly admit an ID.
    # A refreshed survey can use another state when every load succeeded.
    states = sorted(by_status, key=lambda s: (s != "Load rejected", s))
    pair = next(
        (
            (b, s)
            for s in states
            for b in sorted(by_backend)
            if (by_backend[b] & by_status[s]) - same_record[b, s]
        ),
        None,
    )
    backend, state = pair or next(
        key for key, ids_in_pair in same_record.items() if ids_in_pair
    )
    page.set_content((directory / "dataset-explorer.html").read_text(encoding="utf-8"))
    visible = page.locator("details.dataset:not([hidden])")
    issues = []
    if visible.count() != len(ids):
        issues.append("initial dataset count disagrees with the evaluation CSV")
    page.get_by_label("Backend", exact=True).select_option(backend)
    backend_matches = set(visible.evaluate_all("nodes => nodes.map(n => n.dataset.id)"))
    if backend_matches != by_backend[backend]:
        issues.append("backend filter disagrees with the evaluation CSV")
    page.get_by_label("Outcome", exact=True).select_option(state)
    shown = set(visible.evaluate_all("nodes => nodes.map(n => n.dataset.id)"))
    if shown != same_record[backend, state]:
        issues.append("backend and outcome filters matched different records")
    page.get_by_role("button", name="Clear filters").click()
    page.wait_for_function(
        "n => document.querySelectorAll('details.dataset:not([hidden])').length === n",
        arg=len(ids),
    )
    if page.locator("#count").inner_text() != f"{len(ids)} of {len(ids)} datasets":
        issues.append("reset did not restore the announced number of datasets")
    if width == WIDTHS[0]:
        ident, expected = units.most_common(1)[0]
        page.get_by_label("Dataset or declared task").fill(ident)
        if visible.count() >= len(ids):
            issues.append(f"search did not narrow the dataset list for {ident}")
        dataset = page.locator(f'details.dataset[data-id="{ident}"]')
        dataset.locator(":scope > summary").click()
        dataset.get_by_text("Measured files and partitions", exact=True).click()
        partitions = dataset.locator("details[open] tbody tr")
        if partitions.count() != expected or not partitions.first.is_visible():
            issues.append(f"{ident} drilldown is missing accessible partition rows")
        if page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"):
            issues.append(f"{ident} drilldown scrolls the page sideways")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Figure refresh output directory")
    parser.add_argument(
        "--chrome",
        default=shutil.which("google-chrome") or shutil.which("chromium"),
        help="System Google Chrome or Chromium executable",
    )
    parser.add_argument(
        "--require-explorer",
        action="store_true",
        help="Require the searchable dataset reference and its CSVs in a full figure build",
    )
    args = parser.parse_args()
    if not args.chrome:
        parser.error("Install system Chrome or specify --chrome PATH.")
    inputs = native_paths(args.directory, "json")
    if not inputs:
        parser.error(f"No native figure inputs in {args.directory}.")

    explorer = args.directory / "dataset-explorer.html"
    if args.require_explorer and not explorer.is_file():
        parser.error(f"Missing dataset explorer in {args.directory}.")
    if explorer.is_file():
        for name in ("dataset-evaluation-coverage.csv", "dataset-density-units.csv"):
            if not (args.directory / name).is_file():
                parser.error(f"Missing explorer evidence {name} in {args.directory}.")
    failures = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chrome)
        for width in WIDTHS:
            page = browser.new_page(
                viewport={"width": width, "height": 900}, offline=True
            )
            for source in inputs:
                kind = source.stem.rsplit("-", 1)[1]
                labels = [
                    r["label"] for r in json.loads(source.read_text()).get("rows", [])
                ]
                for suffix in (".svg", ".html"):
                    path = source.with_suffix(suffix)
                    markup = path.read_text(encoding="utf-8")
                    if suffix == ".svg":
                        markup = EMBED + markup[markup.index("<svg") :]
                    page.set_content(markup)
                    page.evaluate("document.fonts.ready")
                    if suffix == ".svg":
                        issues = page.evaluate(SVG_LAYOUT)
                    else:
                        issues = page.evaluate(
                            HTML_LAYOUT, {"kind": kind, "labels": labels}
                        )
                    failures.extend(
                        f"{path.name} at {width}px: {issue}" for issue in issues
                    )
            if explorer.is_file():
                failures.extend(
                    f"{explorer.name} at {width}px: {issue}"
                    for issue in check_explorer(page, args.directory, width)
                )
            page.close()
        browser.close()
    if failures:
        raise SystemExit("\n".join(failures))
    sizes = ", ".join(f"{width}px" for width in WIDTHS)
    extra = " and the dataset explorer" if explorer.is_file() else ""
    print(f"Checked {len(inputs)} native SVG/HTML pairs{extra} at {sizes}.")


if __name__ == "__main__":
    main()
