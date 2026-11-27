# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reopen persisted Anno caption links in a headless Chromium editor and seek a local recording."""

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import tempfile
import wave

from caption_links import map_clusters, transcript

HERE = Path(__file__).resolve().parent


def replay(browser):
    cues = [
        {"id": "a", "text": "🎙 e\u0301", "start_ms": 100, "end_ms": 400},
        {"id": "b", "text": "Nu", "start_ms": 700, "end_ms": 900},
    ]
    text, positions = transcript(cues)
    model_result = {"id": "captions", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "clusters": [[[2, 7]]]}
    receipt = map_clusters(text, positions, {"results": [model_result]})
    cases = [
        ("original", cues, [["a", "e\u0301", 0.1, 0.4], ["b", "Nu", 0.7, 0.9]]),
        ("retimed", [dict(cues[0], start_ms=500, end_ms=650), cues[1]],
         [["a", "e\u0301", 0.5, 0.65], ["b", "Nu", 0.7, 0.9]]),
        ("edited", [cues[0], dict(cues[1], text="Nu!")], "The caption transcript has changed"),
        ("replaced", [cues[0], dict(cues[1], id="replacement")], "The caption cue layout has changed"),
    ]
    check_playback(browser, receipt, "e\u0301\nNu", cases)


def check_playback(browser, receipt, quote, cases):
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as playwright:
        directory = Path(tmp)
        # A short silent recording covers the authored cue times in both fixtures.
        with wave.open(str(directory / "recording.wav"), "wb") as wav:
            wav.setparams((1, 1, 8000, 0, "NONE", ""))
            wav.writeframes(bytes([128]) * (20 * 8000))
        (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

        for scenario, current, expected in cases:
            (directory / "request.json").write_text(json.dumps({"cues": current, "quote": quote}), encoding="utf-8")
            app = playwright.chromium.launch(executable_path=browser)
            try:
                page = app.new_page()
                # Serve only the five test assets through Playwright’s in-process route.
                origin = "http://localhost/"
                resources = {origin + name: HERE / name for name in ("test_playback.html", "playback.js")}
                resources.update({origin + name: directory / name for name in
                                  ("request.json", "receipt.json", "recording.wav")})

                def serve(route):
                    asset = resources.get(route.request.url)
                    if asset is None:
                        route.abort()
                    else:
                        route.fulfill(body=asset.read_bytes(), content_type=mimetypes.guess_type(asset.name)[0])

                page.route("**/*", serve)
                page.goto(origin + "test_playback.html", wait_until="domcontentloaded")
                page.wait_for_function('document.querySelector("#result").textContent !== "pending"')
                observed = page.locator("#result").text_content()
            finally:
                app.close()
            if isinstance(expected, list) and observed.startswith("["):
                observed = json.loads(observed)
            if observed != expected:
                raise AssertionError(f"{scenario}: expected {expected!r}, got {observed!r}")
            print(f"{scenario}: {observed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("browser", help="Path to Chrome or Chromium")
    replay(parser.parse_args().browser)
