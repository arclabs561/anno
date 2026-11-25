#!/usr/bin/env python3
"""
Fast internal link check for docs markdown.

- Checks relative markdown links in the README, docs and repository skills
- Ignores external URLs and `mailto:`
- Strips fenced code blocks to reduce false positives
- Skips `docs/archive/` (intentionally historical + cursor-ignored)
"""

from __future__ import annotations

import re
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


def github_anchor_slug(text: str) -> str:
    """
    Best-effort GitHub-style heading slug.

    Notes:
    - We intentionally keep this simple (offline, deterministic).
    - It should be "good enough" for validating common `(#anchor)` links in our docs.
    """
    # Drop inline code + basic HTML tags.
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()
    out: list[str] = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace() or ch in {"-", "_"}:
            out.append("-")
        else:
            # Drop punctuation/symbols.
            continue
    slug = "".join(out)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def extract_github_anchors(md_text: str) -> set[str]:
    """
    Extract GitHub-style heading anchors for a markdown document.

    Handles duplicates by appending `-1`, `-2`, ... like GitHub.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in md_text.splitlines():
        m = heading_re.match(line)
        if not m:
            continue
        heading = m.group(2).strip()
        if not heading:
            continue
        base = github_anchor_slug(heading)
        if not base:
            continue
        n = seen.get(base, 0)
        anchor = base if n == 0 else f"{base}-{n}"
        seen[base] = n + 1
        anchors.add(anchor)
    return anchors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs_root = repo_root / "docs"
    readme_path = repo_root / "README.md"

    if not docs_root.exists():
        print("ERROR: docs/ directory not found.", file=sys.stderr)
        return 2

    link_re = re.compile(r"\]\(([^)]+)\)")
    broken_files: list[tuple[Path, str, Path]] = []
    broken_anchors: list[tuple[Path, str, Path, str]] = []
    checked = 0
    anchor_checked = 0
    anchor_cache: dict[Path, set[str]] = {}

    md_paths: list[Path] = list(docs_root.rglob("*.md"))
    md_paths.extend((repo_root / ".agents" / "skills").rglob("*.md"))
    if readme_path.exists():
        md_paths.append(readme_path)

    for md_path in md_paths:
        # Skip docs/archive/* only (historical by design).
        try:
            rel_to_docs = md_path.relative_to(docs_root)
        except Exception:
            rel_to_docs = None
        if rel_to_docs is not None and rel_to_docs.parts and rel_to_docs.parts[0] == "archive":
            continue

        text = md_path.read_text(encoding="utf-8", errors="replace")
        stripped = strip_fenced_code_blocks(text)
        anchors_here = extract_github_anchors(stripped)
        anchor_cache[md_path.resolve()] = anchors_here

        for raw in link_re.findall(stripped):
            link = raw.strip()
            if not link:
                continue

            # External links (or mailto) are out of scope for this script.
            if "://" in link or link.startswith("mailto:"):
                continue

            # Trim <...> wrapper (valid Markdown syntax for URLs/paths).
            if link.startswith("<") and link.endswith(">"):
                link = link[1:-1].strip()

            # Split fragment/query.
            link_path, frag = (link.split("#", 1) + [""])[:2]
            link_path = link_path.split("?", 1)[0].strip()
            frag = frag.split("?", 1)[0].strip()
            if frag.startswith("#"):
                frag = frag[1:]

            # Pure anchor link, same file.
            if not link_path and frag:
                anchor_checked += 1
                if frag not in anchors_here:
                    broken_anchors.append((md_path.relative_to(repo_root), link, md_path.resolve(), frag))
                continue

            if not link_path:
                continue

            target = (md_path.parent / link_path).resolve()
            checked += 1
            if not target.exists():
                broken_files.append((md_path.relative_to(repo_root), link, target))
                continue

            # Validate fragment against target markdown headings, if applicable.
            if frag and target.suffix.lower() == ".md":
                anchor_checked += 1
                target_abs = target.resolve()
                if target_abs not in anchor_cache:
                    try:
                        target_text = target.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        # If we can't read, treat as broken anchor to be safe.
                        broken_anchors.append((md_path.relative_to(repo_root), link, target_abs, frag))
                        continue
                    target_stripped = strip_fenced_code_blocks(target_text)
                    anchor_cache[target_abs] = extract_github_anchors(target_stripped)
                if frag not in anchor_cache[target_abs]:
                    broken_anchors.append((md_path.relative_to(repo_root), link, target_abs, frag))

    scope = "docs/ (excluding docs/archive/), .agents/skills/"
    if readme_path.exists():
        scope += " and README.md"
    print(f"Checked {checked} relative links and {anchor_checked} anchors under {scope}")
    if not broken_files and not broken_anchors:
        print("OK: no broken relative markdown links or anchors found.")
        return 0

    if broken_files:
        print(f"Broken file links: {len(broken_files)}")
        for src, link, resolved in broken_files:
            try:
                resolved_rel = resolved.relative_to(repo_root)
            except Exception:
                resolved_rel = resolved
            print(f"- {src}: ({link}) -> {resolved_rel}")

    if broken_anchors:
        print(f"Broken anchors: {len(broken_anchors)}")
        for src, link, resolved, frag in broken_anchors:
            try:
                resolved_rel = resolved.relative_to(repo_root)
            except Exception:
                resolved_rel = resolved
            print(f"- {src}: ({link}) -> {resolved_rel}  (missing #{frag})")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())


