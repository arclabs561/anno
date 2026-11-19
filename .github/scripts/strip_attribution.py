# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown-it-py==4.2.0"]
# ///
"""Filter a local Markdown draft: uv run .github/scripts/strip_attribution.py < draft.md

--check-event emits a read-only Actions warning for the owner's PR or review.
"""

import argparse
import io
import json
import os
import re
import sys

from markdown_it import MarkdownIt

SESSION = r"https://claude\.ai/code/session_[A-Za-z0-9_-]+"
FOOTER = re.compile(
    r"_?(?:🤖 )?Generated (?:by|with) (?:Claude Code|\[Claude Code\]"
    r"\(https://(?:claude\.ai/code|www\.anthropic\.com/claude-code)/?\))\.?_?"
    rf"|{SESSION}|<{SESSION}>|Claude-Session: {SESSION}",
    re.IGNORECASE,
)


def strip_footer(body: str) -> str:
    """Remove only trailing top-level paragraphs made entirely of known footer lines."""
    lines = io.StringIO(body, newline="").readlines()
    blocks = MarkdownIt("commonmark").enable("table").parse(body)
    # Raw HTML can contain Markdown blocks, e.g. inside an unclosed <blockquote>.
    # Keep those drafts intact. Closed HTML comments are common in PR templates.
    for block in blocks:
        for token in [block, *(block.children or [])]:
            if token.type in ("html_block", "html_inline") and not re.fullmatch(
                r"\s*<!--(?:(?!-->).)*-->\s*", token.content, re.DOTALL
            ):
                return body

    start = len(lines)
    found = False
    for block in reversed(blocks):
        if block.level != 0 or block.map is None:
            continue
        first, end = block.map
        if any(line.strip(" \t\r\n") for line in lines[end:start]):
            break  # Reference definitions, for example, have no block token.
        source = [line.rstrip("\r\n") for line in lines[first:end]]
        if block.type == "paragraph_open" and all(FOOTER.fullmatch(line) for line in source):
            start = first
            found = True
        elif found and block.type == "hr" and source == ["---"]:
            start = first
            break
        else:
            break

    if not found:
        return body
    while start and not lines[start - 1].strip(" \t\r\n"):
        start -= 1
    return "".join(lines[:start])


def owner_body(event_name: str, event: dict) -> str:
    owner = event.get("repository", {}).get("owner", {}).get("id")
    if not owner or event.get("sender", {}).get("id") != owner:
        return ""
    key = {"pull_request_target": "pull_request", "pull_request_review": "review"}.get(event_name)
    item = event.get(key, {})
    if item.get("user", {}).get("id") == owner:
        return item.get("body") or ""
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-event", action="store_true")
    if parser.parse_args().check_event:
        with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as stream:
            body = owner_body(os.environ["GITHUB_EVENT_NAME"], json.load(stream))
        if body and strip_footer(body) != body:
            print("::warning::This draft ends with an attribution footer. Review the current text before editing.")
    else:
        body = sys.stdin.buffer.read().decode("utf-8")
        sys.stdout.buffer.write(strip_footer(body).encode("utf-8"))


if __name__ == "__main__":
    main()
