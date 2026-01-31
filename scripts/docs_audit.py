#!/usr/bin/env python3
"""
Lightweight docs hygiene checks.

Goals:
- Fast, offline, deterministic.
- Catches the common ways docs drift back into noise:
  - broken internal markdown links
  - broken internal markdown anchors (#section links)
  - stale path references (docs/design/, docs/research/, docs/reference/)
  - stale docs path mentions (plain text "docs/.../X.md" references)
  - stale crate name references (anno-cli)

Intentionally:
- Skips docs/archive/ (historical + cursor-ignored).
- Does not validate external URLs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def strip_fenced_code_blocks(text: str) -> str:
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def iter_tracked_text_files(repo_root: Path) -> list[Path]:
    """
    Return git-tracked files with relevant extensions.

    Using `git ls-files` avoids accidentally scanning local-only dirs like `.venv/`,
    downloaded artifacts, or editor caches.
    """
    exts = {".md", ".rs", ".toml", ".sh", ".py"}

    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    paths = [p for p in proc.stdout.split(b"\x00") if p]

    out: list[Path] = []
    for raw in paths:
        rel = Path(raw.decode("utf-8", errors="replace"))
        p = (repo_root / rel).resolve()

        # Skip historical/archival areas. These intentionally preserve old paths/names.
        if rel.parts and rel.parts[0] == "archive":
            continue
        if rel.parts[:2] == ("docs", "archive"):
            continue

        if p.suffix.lower() not in exts:
            continue

        # Avoid self-matching on the patterns this script contains.
        if rel == Path("scripts/docs_audit.py"):
            continue

        out.append(p)

    return out


def check_stale_doc_path_mentions(repo_root: Path) -> int:
    """
    Catch drift where docs mention a `docs/.../*.md` path that no longer exists.

    This complements markdown link checking because many references are plain text,
    not clickable links.
    """
    # Conservative: check markdown files in docs/ + README.md.
    md_paths: list[Path] = []
    for p in iter_tracked_text_files(repo_root):
        rel = p.relative_to(repo_root)
        if p.suffix.lower() != ".md":
            continue
        if rel == Path("README.md") or (rel.parts and rel.parts[0] == "docs"):
            md_paths.append(p)

    # Match common "docs/..." paths ending in .md (avoid grabbing trailing punctuation).
    path_re = re.compile(r"(docs/[A-Za-z0-9_./-]+\.md)(?![A-Za-z0-9_/.-])")

    missing: list[tuple[Path, str]] = []
    for p in md_paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        stripped = strip_fenced_code_blocks(text)
        for m in path_re.finditer(stripped):
            rel = m.group(1)
            target = (repo_root / rel).resolve()
            if not target.exists():
                missing.append((p.relative_to(repo_root), rel))

    if not missing:
        print("OK: no stale docs/*.md path mentions found.")
        return 0

    print(f"FAIL: found {len(missing)} stale docs path mentions (showing all):")
    for src, rel in missing:
        print(f"- {src}: mentions {rel} (missing)")
    return 1


def check_banned_strings(repo_root: Path) -> int:
    banned = [
        # Old doc layout.
        r"\bdocs/design/",
        r"\bdocs/research/",
        r"\bdocs/reference/",
        # Old pre-split paths.
        r"\bcrates/anno/cli/",
        r"\bcrates/anno/eval/",
    ]
    patterns = [(s, re.compile(s)) for s in banned]

    offenders: list[tuple[Path, str]] = []
    for p in iter_tracked_text_files(repo_root):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for raw, pat in patterns:
            if pat.search(text):
                offenders.append((p.relative_to(repo_root), raw))

    if not offenders:
        print("OK: no banned stale references found.")
        return 0

    print(f"FAIL: found {len(offenders)} stale references (showing all):")
    for path, raw in offenders:
        print(f"- {path}: matches {raw!r}")
    return 1


def run_docs_links(repo_root: Path) -> int:
    script = repo_root / "scripts" / "check_docs_links.py"
    if not script.exists():
        print("FAIL: scripts/check_docs_links.py not found.", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(script)]
    proc = subprocess.run(cmd, cwd=repo_root)
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Docs hygiene checks (fast, offline).")
    ap.add_argument(
        "--skip-links",
        action="store_true",
        help="Skip markdown link checking (only run stale-reference scan).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    rc = 0
    if not args.skip_links:
        rc = max(rc, run_docs_links(repo_root))
    rc = max(rc, check_stale_doc_path_mentions(repo_root))
    rc = max(rc, check_banned_strings(repo_root))

    if rc == 0:
        print("OK: docs audit passed.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())


