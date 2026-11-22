#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
QA random URL testing for anno extract.

Picks random URLs from a pool of multilingual sources (Wikipedia, Wikimedia,
RSS feeds, etc.) and runs `anno extract --url` on each, producing a markdown
report with entity counts, timing, and error details.

Usage:
    uv run scripts/qa_random_urls.py [OPTIONS]

Examples:
    uv run scripts/qa_random_urls.py --count 3 --no-rss --seed 42
    uv run scripts/qa_random_urls.py --count 10 --backends stacked,bert-onnx
    uv run scripts/qa_random_urls.py --category wikipedia --lang ja,zh,ar
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError

# ---------------------------------------------------------------------------
# Source pool
#
# Design: maximize diversity along 5 axes -- language, writing system, domain,
# HTML structure, and content era/style.  Sources are bucketed into categories
# so callers can --category filter for fast smoke tests (wikipedia only) or
# full cross-domain runs (all).  All Wikimedia sources use the universal
# Special:Random path, which works on every language edition (localized paths
# like Spezial:Zufällige_Seite cause 404s with automated fetchers).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    name: str
    url: str
    category: str  # wikipedia, wikimedia, rss, other
    lang: str      # ISO 639-1 or "multi"
    script: str    # Latin, Cyrillic, CJK, Arabic, Devanagari, etc.


# Language -> script mapping for Wikipedia/Wikimedia sources
_LANG_SCRIPT: dict[str, str] = {
    # Major languages (Latin script)
    "en": "Latin", "de": "Latin", "fr": "Latin", "es": "Latin", "it": "Latin",
    "pt": "Latin", "nl": "Latin", "pl": "Latin", "sv": "Latin", "da": "Latin",
    "no": "Latin", "fi": "Latin", "cs": "Latin", "ro": "Latin", "hu": "Latin",
    "ca": "Latin", "tr": "Latin", "id": "Latin", "ms": "Latin", "vi": "Latin",
    "tl": "Latin", "sw": "Latin", "eu": "Latin",
    # Cyrillic
    "ru": "Cyrillic", "uk": "Cyrillic", "sr": "Cyrillic", "bg": "Cyrillic",
    "kk": "Cyrillic", "ky": "Cyrillic", "tg": "Cyrillic", "tt": "Cyrillic",
    "cv": "Cyrillic", "os": "Cyrillic", "ab": "Cyrillic", "cu": "Cyrillic",
    # CJK
    "ja": "CJK", "zh": "CJK", "ko": "CJK",
    # Arabic script
    "ar": "Arabic", "fa": "Arabic", "ur": "Arabic", "ug": "Arabic",
    # Indic scripts
    "hi": "Devanagari", "mr": "Devanagari", "ne": "Devanagari", "sa": "Devanagari",
    "pi": "Devanagari",
    "th": "Thai", "lo": "Lao", "my": "Myanmar", "km": "Khmer", "si": "Sinhala",
    "ka": "Georgian", "hy": "Armenian", "el": "Greek", "he": "Hebrew",
    "am": "Ethiopic", "ta": "Tamil", "bn": "Bengali", "te": "Telugu",
    "bo": "Tibetan",
    # Indigenous Americas (Latin-based orthographies)
    "nah": "Latin", "qu": "Latin", "ay": "Latin", "gn": "Latin",
    "chr": "Cherokee", "cr": "CanSyllabics", "iu": "CanSyllabics",
    "cdo": "Latin",
    # African (Latin-based orthographies)
    "zu": "Latin", "xh": "Latin", "sn": "Latin", "yo": "Latin",
    "ig": "Latin", "ha": "Latin", "rw": "Latin", "lg": "Latin",
    # Pacific / Oceanian (Latin-based)
    "mi": "Latin", "haw": "Latin", "ty": "Latin", "sm": "Latin", "fj": "Latin",
    # Conlangs (Latin-based)
    "eo": "Latin", "ia": "Latin", "io": "Latin", "vo": "Latin", "jbo": "Latin",
    # Classical / dead (Latin-based orthographies or native script)
    "la": "Latin", "ang": "Latin", "got": "Gothic", "arc": "Syriac",
}


def _wp(lang: str) -> Source:
    """Wikipedia source shorthand."""
    return Source(f"Wikipedia ({lang})", f"https://{lang}.wikipedia.org/wiki/Special:Random",
                  "wikipedia", lang, _LANG_SCRIPT.get(lang, "Latin"))


def _wm(project: str, label: str, lang: str) -> Source:
    """Wikimedia sibling source shorthand."""
    return Source(f"{label} ({lang})", f"https://{lang}.{project}.org/wiki/Special:Random",
                  "wikimedia", lang, _LANG_SCRIPT.get(lang, "Latin"))


# Wikipedia Special:Random -- 45 languages (all use universal Special:Random path)
WIKIPEDIA_SOURCES = [_wp(lang) for lang in [
    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "da", "no", "fi",
    "cs", "ro", "hu", "ca", "tr", "id", "ms", "vi", "tl", "sw", "eu",
    "ru", "uk", "sr", "bg",
    "ja", "zh", "ko",
    "ar", "fa", "ur",
    "hi", "mr", "ne",
    "th", "ka", "hy", "el", "he", "am", "ta", "bn", "te",
]]

# Wikimedia siblings -- all use universal Special:Random path
_WM_PROJECTS: list[tuple[str, str, list[str]]] = [
    ("wiktionary", "Wiktionary", ["en", "fr", "de", "ja", "ru", "zh", "ko", "ar"]),
    ("wikinews", "Wikinews", ["en", "fr", "de", "ja", "ru", "ar"]),
    ("wikiquote", "Wikiquote", ["en", "fr", "de", "ja", "ru"]),
    ("wikivoyage", "Wikivoyage", ["en", "de", "fr", "zh", "ru"]),
    ("wikisource", "Wikisource", ["en", "fr", "de", "zh", "ru", "ar", "ja"]),
    ("wikibooks", "Wikibooks", ["en", "de", "ja"]),
]
WIKIMEDIA_SOURCES = [_wm(proj, label, lang)
                     for proj, label, langs in _WM_PROJECTS for lang in langs]

# Conlang and classical language Wikipedia editions
CONLANG_SOURCES = [
    Source(f"Wikipedia ({lang})", f"https://{lang}.wikipedia.org/wiki/Special:Random",
           "conlang", lang, _LANG_SCRIPT.get(lang, "Latin"))
    for lang in [
    # Constructed languages
    "eo",   # Esperanto
    "ia",   # Interlingua
    "io",   # Ido
    "vo",   # Volapuk
    "jbo",  # Lojban
    # Classical / dead / liturgical
    "la",   # Latin
    "ang",  # Old English
    "cu",   # Old Church Slavonic
    "sa",   # Sanskrit
    "pi",   # Pali
    # Indigenous Americas
    "nah",  # Nahuatl
    "qu",   # Quechua
    "ay",   # Aymara
    "gn",   # Guarani
    "chr",  # Cherokee (Syllabary script)
    "cr",   # Cree (Canadian Syllabics)
    "iu",   # Inuktitut (Canadian Syllabics)
    # African
    "zu",   # Zulu
    "xh",   # Xhosa
    "sn",   # Shona
    "yo",   # Yoruba
    "ig",   # Igbo
    "ha",   # Hausa
    "rw",   # Kinyarwanda
    "lg",   # Luganda
    # Pacific / Oceanian
    "mi",   # Maori
    "haw",  # Hawaiian
    "ty",   # Tahitian
    "sm",   # Samoan
    "fj",   # Fijian
    # Central/South/SE Asian minority
    "my",   # Burmese (Myanmar script)
    "km",   # Khmer
    "kk",   # Kazakh
    "ky",   # Kyrgyz
    "tg",   # Tajik
    "tt",   # Tatar
    "cv",   # Chuvash
    "os",   # Ossetian (Caucasus)
    "ab",   # Abkhaz (Caucasus)
    # Chinese dialect
    "cdo",  # Min Dong
]]

# Non-wiki random endpoints -- diverse domains, styles, eras
OTHER_SOURCES = [
    Source("Simple English Wikipedia", "https://simple.wikipedia.org/wiki/Special:Random", "other", "en", "Latin"),
    Source("Wikidata", "https://www.wikidata.org/wiki/Special:Random", "other", "multi", "Latin"),
    Source("Wikimedia Commons", "https://commons.wikimedia.org/wiki/Special:Random", "other", "multi", "Latin"),
    # Encyclopedias & knowledge bases
    Source("Stanford Encyclopedia of Philosophy", "https://plato.stanford.edu/cgi-bin/encyclopedia/random", "other", "en", "Latin"),
    Source("RationalWiki", "https://rationalwiki.org/wiki/Special:Random", "other", "en", "Latin"),
    Source("Citizendium", "https://en.citizendium.org/wiki/Special:Random", "other", "en", "Latin"),
    Source("Scholarpedia", "http://www.scholarpedia.org/article/Special:Random", "other", "en", "Latin"),
    Source("wikiHow", "https://www.wikihow.com/Special:Randomizer", "other", "en", "Latin"),
    # Literary / historical
    Source("Project Gutenberg (random)", "https://www.gutenberg.org/ebooks/search/?sort_order=random", "other", "en", "Latin"),
    Source("OpenLibrary (random)", "https://openlibrary.org/random", "other", "en", "Latin"),
    # Places & travel
    Source("Atlas Obscura (random)", "https://www.atlasobscura.com/random", "other", "en", "Latin"),
    # Sports (tabular data, player bios, team stats)
    Source("Baseball Reference (random)", "https://www.baseball-reference.com/rand.fcgi", "other", "en", "Latin"),
    Source("Basketball Reference (random)", "https://www.basketball-reference.com/rand.fcgi", "other", "en", "Latin"),
    Source("Hockey Reference (random)", "https://www.hockey-reference.com/rand.fcgi", "other", "en", "Latin"),
    # Pop culture / fiction (different HTML, fictional entities)
    Source("TVTropes (random)", "https://tvtropes.org/pmwiki/randomitem.php", "other", "en", "Latin"),
    Source("Wookieepedia (Star Wars)", "https://starwars.fandom.com/wiki/Special:Random", "other", "en", "Latin"),
    Source("Memory Alpha (Star Trek)", "https://memory-alpha.fandom.com/wiki/Special:Random", "other", "en", "Latin"),
]

# RSS feed sources -- parsed at runtime
RSS_SOURCES = [
    # English news
    Source("BBC News", "http://feeds.bbci.co.uk/news/rss.xml", "rss", "en", "Latin"),
    Source("BBC Science", "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml", "rss", "en", "Latin"),
    Source("NPR News", "https://feeds.npr.org/1001/rss.xml", "rss", "en", "Latin"),
    Source("Al Jazeera English", "https://www.aljazeera.com/xml/rss/all.xml", "rss", "en", "Latin"),
    # Multilingual broadcasters
    Source("BBC Arabic", "http://feeds.bbci.co.uk/arabic/rss.xml", "rss", "ar", "Arabic"),
    Source("BBC Russian", "http://feeds.bbci.co.uk/russian/rss.xml", "rss", "ru", "Cyrillic"),
    Source("DW German", "https://rss.dw.com/rdf/rss-de-all", "rss", "de", "Latin"),
    Source("DW English", "https://rss.dw.com/rdf/rss-en-all", "rss", "en", "Latin"),
    Source("DW Spanish", "https://rss.dw.com/rdf/rss-sp-all", "rss", "es", "Latin"),
    Source("France24 French", "https://www.france24.com/fr/rss", "rss", "fr", "Latin"),
    Source("France24 English", "https://www.france24.com/en/rss", "rss", "en", "Latin"),
    Source("France24 Arabic", "https://www.france24.com/ar/rss", "rss", "ar", "Arabic"),
    Source("NHK World", "https://www3.nhk.or.jp/rss/news/cat0.xml", "rss", "ja", "CJK"),
    # Global / multilingual
    Source("Global Voices", "https://globalvoices.org/feed/", "rss", "en", "Latin"),
    Source("UN News (peace)", "https://news.un.org/feed/subscribe/en/news/topic/peace-and-security/feed/rss.xml", "rss", "en", "Latin"),
    Source("UN News (health)", "https://news.un.org/feed/subscribe/en/news/topic/health/feed/rss.xml", "rss", "en", "Latin"),
    # Tech / academic
    Source("ArXiv CS.CL", "https://rss.arxiv.org/rss/cs.CL", "rss", "en", "Latin"),
    Source("ArXiv CS.AI", "https://rss.arxiv.org/rss/cs.AI", "rss", "en", "Latin"),
    Source("Hacker News", "https://hnrss.org/frontpage", "rss", "en", "Latin"),
    Source("Lobsters", "https://lobste.rs/rss", "rss", "en", "Latin"),
]

# Reference-hop sources -- these resolve via API at runtime, not direct URLs
# Category "refhop" triggers Wikipedia reference extraction
REFHOP_SOURCES = [
    Source("Wiki refhop (en)", "https://en.wikipedia.org/", "refhop", "en", "Latin"),
    Source("Wiki refhop (de)", "https://de.wikipedia.org/", "refhop", "de", "Latin"),
    Source("Wiki refhop (fr)", "https://fr.wikipedia.org/", "refhop", "fr", "Latin"),
    Source("Wiki refhop (es)", "https://es.wikipedia.org/", "refhop", "es", "Latin"),
    Source("Wiki refhop (ja)", "https://ja.wikipedia.org/", "refhop", "ja", "CJK"),
    Source("Wiki refhop (ru)", "https://ru.wikipedia.org/", "refhop", "ru", "Cyrillic"),
    Source("Wiki refhop (zh)", "https://zh.wikipedia.org/", "refhop", "zh", "CJK"),
    Source("Wiki refhop (ar)", "https://ar.wikipedia.org/", "refhop", "ar", "Arabic"),
    Source("Wiki refhop (pt)", "https://pt.wikipedia.org/", "refhop", "pt", "Latin"),
    Source("Wiki refhop (ko)", "https://ko.wikipedia.org/", "refhop", "ko", "CJK"),
]

DOCUMENT_SOURCES = [
    Source("RFC 3339", "https://www.rfc-editor.org/rfc/rfc3339.html", "documents", "en", "Latin"),
    Source("W3C bidi controls", "https://www.w3.org/International/questions/qa-bidi-unicode-controls", "documents", "en", "Latin"),
    Source("Night on the Galactic Railroad", "https://www.aozora.gr.jp/cards/000081/files/456_15050.html", "documents", "ja", "CJK"),
]

ALL_SOURCES = (WIKIPEDIA_SOURCES + CONLANG_SOURCES + WIKIMEDIA_SOURCES
               + OTHER_SOURCES + RSS_SOURCES + REFHOP_SOURCES + DOCUMENT_SOURCES)


# ---------------------------------------------------------------------------
# RSS helper
# ---------------------------------------------------------------------------

def fetch_random_rss_link(feed_url: str, timeout: int = 15) -> Optional[str]:
    """Fetch an RSS/Atom feed and return one random entry link."""
    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": "anno-qa/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        root = ET.fromstring(data)

        links: list[str] = []

        # RSS 2.0: channel/item/link
        for item in root.iter("item"):
            link_el = item.find("link")
            if link_el is not None and link_el.text:
                links.append(link_el.text.strip())

        # Atom: entry/link[@href]
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            for link_el in entry.findall("atom:link", ns):
                href = link_el.get("href")
                if href:
                    links.append(href.strip())
            # Also try without namespace (some feeds)
            for link_el in entry.findall("link"):
                href = link_el.get("href") or (link_el.text or "").strip()
                if href:
                    links.append(href)

        # RDF: rdf:item/link
        for item in root.iter("{http://purl.org/rss/1.0/}item"):
            link_el = item.find("{http://purl.org/rss/1.0/}link")
            if link_el is not None and link_el.text:
                links.append(link_el.text.strip())

        if not links:
            return None
        return random.choice(links)

    except (HTTPError, URLError, ET.ParseError, TimeoutError, OSError):
        return None


def fetch_wiki_refhop(wiki_base: str, timeout: int = 15) -> Optional[str]:
    """Get a random Wikipedia article, then pick one of its external references."""
    try:
        lang = wiki_base.rstrip("/").split("//")[1].split(".")[0]
        api = f"https://{lang}.wikipedia.org/w/api.php"

        # Step 1: get a random article title
        req = urllib.request.Request(
            f"{api}?action=query&list=random&rnlimit=1&rnnamespace=0&format=json",
            headers={"User-Agent": "anno-qa/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        title = data["query"]["random"][0]["title"]

        # Step 2: get external links from that article
        encoded_title = urllib.request.quote(title)
        req2 = urllib.request.Request(
            f"{api}?action=query&titles={encoded_title}&prop=extlinks&ellimit=50&format=json",
            headers={"User-Agent": "anno-qa/1.0"},
        )
        with urllib.request.urlopen(req2, timeout=timeout) as resp:
            data2 = json.loads(resp.read())

        pages = data2["query"]["pages"]
        for page in pages.values():
            extlinks = page.get("extlinks", [])
            links = []
            for el in extlinks:
                url = el.get("*", el.get("url", ""))
                if url.startswith("http"):
                    links.append(url)
            if links:
                return random.choice(links)

        # No external links -- fall back to the article itself.  This still
        # exercises anno on a random Wikipedia page, just without the domain hop.
        return f"https://{lang}.wikipedia.org/wiki/{encoded_title}"

    except (HTTPError, URLError, KeyError, TimeoutError, OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Anno runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    source: Source
    url: str
    status: str  # OK, error, timeout
    entity_count: int = 0
    type_counts: dict[str, int] = field(default_factory=dict)
    elapsed_secs: float = 0.0
    error_msg: str = ""
    stderr_tail: str = ""
    backend: str = ""
    reported_model: str = ""


def resolve_url(source: Source) -> Optional[str]:
    """Resolve the actual URL to test."""
    if source.category == "rss":
        return fetch_random_rss_link(source.url)
    if source.category == "refhop":
        return fetch_wiki_refhop(source.url)
    return source.url


def validate_extraction(data: object) -> None:
    """Check the extract JSON contract for a nonempty document. Zero matches are valid."""
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
        raise ValueError("missing entities array")
    prov = data.get("provenance")
    if not isinstance(prov, dict) or not isinstance(prov.get("model"), str) or not prov["model"].strip():
        raise ValueError("missing provenance model")
    size = prov.get("text_chars")
    if type(size) is not int or size <= 0:
        raise ValueError("expected a positive provenance.text_chars")
    entities = data["entities"]
    if type(prov.get("entity_count")) is not int or prov["entity_count"] != len(entities):
        raise ValueError("provenance.entity_count disagrees with entities")
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(f"entity {i} is not an object")
        for key in ("type", "text"):
            if not isinstance(entity.get(key), str) or not entity[key].strip():
                raise ValueError(f"entity {i} has an empty {key}")
        start, end = entity.get("start"), entity.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= size:
            raise ValueError(f"entity {i} has invalid character offsets")


def run_anno(
    anno_bin: str,
    url: str,
    backend: str,
    timeout: int,
    verbose: bool,
) -> RunResult:
    """Run anno extract on a URL and parse the result."""
    # placeholder -- filled by caller
    result = RunResult(
        source=Source("", "", "", "", ""),
        url=url,
        backend=backend,
        status="error",
    )

    cmd = [
        anno_bin, "extract",
        "--url", url,
        "--model", backend,
        "--format", "json",
        "--detect-lang",
    ]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        result.elapsed_secs = elapsed

        if verbose:
            if proc.stderr:
                print(f"  stderr: {proc.stderr[:500]}", file=sys.stderr)

        if proc.returncode != 0:
            result.status = "error"
            result.error_msg = f"exit code {proc.returncode}"
            result.stderr_tail = (proc.stderr or "")[-500:]
            return result

        # Parse JSON output
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            result.status = "error"
            result.error_msg = f"JSON parse error: {e}"
            return result

        try:
            validate_extraction(data)
        except ValueError as e:
            result.error_msg = f"invalid extraction: {e}"
            return result
        result.status = "OK"
        result.reported_model = data["provenance"]["model"]
        result.entity_count = len(data["entities"])
        result.type_counts = dict(Counter(ent["type"] for ent in data["entities"]))

        return result

    except subprocess.TimeoutExpired:
        result.elapsed_secs = time.monotonic() - t0
        result.status = "timeout"
        result.error_msg = f"timeout after {timeout}s"
        return result
    except FileNotFoundError:
        result.status = "error"
        result.error_msg = f"anno binary not found: {anno_bin}"
        return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: list[RunResult],
    seed: Optional[int],
    backends: list[str],
    start_time: datetime,
) -> str:
    lines: list[str] = []
    ok_count = sum(1 for r in results if r.status == "OK")
    err_count = len(results) - ok_count
    all_langs = sorted(set(r.source.lang for r in results))
    all_scripts = sorted(set(r.source.script for r in results))

    total_types: Counter[str] = Counter()
    total_entities = 0
    for r in results:
        total_entities += r.entity_count
        total_types.update(r.type_counts)

    seed_str = str(seed) if seed is not None else "random"
    backend_str = ", ".join(backends)

    lines.append("# QA Random URL Report")
    lines.append(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')} | Seed: {seed_str} | Backend: {backend_str}")
    lines.append("")
    lines.append("The seed selects sources. Live pages may change between requests, including between backends.")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Successful checks: {ok_count}/{len(results)}")
    lines.append(f"- Failures: {err_count}" + (f" ({', '.join(r.status for r in results if r.status != 'OK')})" if err_count else ""))
    lines.append(f"- Languages: {', '.join(all_langs)}")
    lines.append(f"- Scripts: {', '.join(all_scripts)}")
    type_summary = ", ".join(f"{t}: {c}" for t, c in total_types.most_common())
    lines.append(f"- Entities found: {total_entities} total ({type_summary})")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("| # | Source | URL | Lang | Script | Requested / reported model | Entities | Time | Status |")
    lines.append("|---|--------|-----|------|--------|----------------------------|----------|------|--------|")

    for i, r in enumerate(results, 1):
        url_display = r.url
        url_display = url_display.replace("|", "%7C")
        time_str = f"{r.elapsed_secs:.1f}s"
        ent_str = str(r.entity_count) if r.status == "OK" else "-"
        status_icon = "OK" if r.status == "OK" else f"**{r.status}**"
        lines.append(
            f"| {i} | {r.source.name} | {url_display} | {r.source.lang} | {r.source.script} | {r.backend or '—'} / {r.reported_model or '—'} | {ent_str} | {time_str} | {status_icon} |"
        )

    lines.append("")

    # Type breakdown
    if total_types:
        lines.append("## Entity Type Breakdown")
        lines.append("")
        lines.append("| Type | Count | % |")
        lines.append("|------|-------|---|")
        for t, c in total_types.most_common():
            pct = (c / total_entities * 100) if total_entities > 0 else 0
            lines.append(f"| {t} | {c} | {pct:.1f}% |")
        lines.append("")

    # Errors section
    errors = [(i, r) for i, r in enumerate(results, 1) if r.status != "OK"]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for i, r in errors:
            lines.append(f"### [{i}] {r.source.name}")
            lines.append(f"- URL: {r.url}")
            lines.append(f"- Error: {r.error_msg}")
            if r.stderr_tail:
                lines.append(f"- stderr (tail): `{r.stderr_tail[:200]}`")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QA random URL testing for anno extract",
    )
    p.add_argument("--count", "-n", type=int, default=10, help="Number of URLs to test (default: 10)")
    p.add_argument("--backends", "-b", default="stacked", help="Comma-separated backends (default: stacked)")
    p.add_argument("--timeout", type=int, default=60, help="Per-URL timeout in seconds (default: 60)")
    p.add_argument("--output", "-o", help="Report file path (default: reports/qa-random-<timestamp>.md)")
    p.add_argument("--category", "-c", default="all",
                   choices=["wikipedia", "conlang", "wikimedia", "rss", "other", "refhop", "documents", "all"],
                   help="Filter source category (default: all)")
    p.add_argument("--lang", help="Filter by language codes (comma-separated)")
    p.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    p.add_argument("--no-rss", action="store_true", help="Skip RSS and refhop sources (faster, no feed/API fetching)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show full anno output")
    p.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1.0)")
    p.add_argument("--self-test", action="store_true", help="Validate source pool integrity (no anno binary needed)")
    p.add_argument("--dry-run", action="store_true", help="Show selected sources and resolved URLs without running anno")
    return p.parse_args()


def self_test() -> int:
    """Validate source pool integrity without running anno."""
    errors: list[str] = []

    # Check all sources have required fields
    for s in ALL_SOURCES:
        if not s.name:
            errors.append(f"Source with url={s.url} has empty name")
        if not s.url:
            errors.append(f"Source {s.name} has empty url")
        if s.category not in ("wikipedia", "conlang", "wikimedia", "rss", "other", "refhop", "documents"):
            errors.append(f"Source {s.name} has unknown category: {s.category}")
        if not s.lang:
            errors.append(f"Source {s.name} has empty lang")
        if not s.script:
            errors.append(f"Source {s.name} has empty script")

    # Check no duplicate URLs (within same category)
    seen: dict[str, str] = {}
    for s in ALL_SOURCES:
        key = s.url
        if key in seen:
            errors.append(f"Duplicate URL: {s.name} and {seen[key]} share {key}")
        seen[key] = s.name

    # Check Wikipedia/conlang sources use Special:Random
    for s in ALL_SOURCES:
        if s.category in ("wikipedia", "conlang", "wikimedia"):
            if "Special:Random" not in s.url:
                errors.append(f"Source {s.name} should use Special:Random but has: {s.url}")

    # Check RSS sources use http(s)
    for s in ALL_SOURCES:
        if s.category == "rss" and not s.url.startswith("http"):
            errors.append(f"RSS source {s.name} has non-http URL: {s.url}")

    # Check script mapping covers all languages
    all_langs = {s.lang for s in ALL_SOURCES if s.lang != "multi"}
    unmapped = all_langs - set(_LANG_SCRIPT.keys())
    if unmapped:
        errors.append(f"Languages without script mapping (defaulting to Latin): {sorted(unmapped)}")

    # Summary stats
    categories = Counter(s.category for s in ALL_SOURCES)
    scripts = sorted(set(s.script for s in ALL_SOURCES))
    langs = sorted(set(s.lang for s in ALL_SOURCES))

    print(f"Source pool: {len(ALL_SOURCES)} total")
    print(f"Categories: {dict(categories.most_common())}")
    print(f"Languages ({len(langs)}): {', '.join(langs)}")
    print(f"Scripts ({len(scripts)}): {', '.join(scripts)}")

    # Check diversity invariants
    if len(langs) < 40:
        errors.append(f"Language diversity too low: {len(langs)} < 40")
    if len(scripts) < 10:
        errors.append(f"Script diversity too low: {len(scripts)} < 10")
    if len(ALL_SOURCES) < 100:
        errors.append(f"Source pool too small: {len(ALL_SOURCES)} < 100")

    # Category balance: each non-refhop category should have >= 5 sources
    for cat in ("wikipedia", "conlang", "wikimedia", "other", "rss"):
        if categories.get(cat, 0) < 5:
            errors.append(f"Category {cat} has too few sources: {categories.get(cat, 0)} < 5")

    # Regression: report generation doesn't crash with synthetic data
    try:
        synthetic = [
            RunResult(source=ALL_SOURCES[0], url="https://example.com", status="OK",
                      entity_count=5, type_counts={"PER": 3, "ORG": 2}, elapsed_secs=1.0),
            RunResult(source=ALL_SOURCES[1], url="https://example.org", status="error",
                      error_msg="test error", stderr_tail="stderr sample"),
            RunResult(source=ALL_SOURCES[2], url="https://example.net", status="timeout",
                      error_msg="timeout after 60s"),
        ]
        report = generate_report(synthetic, seed=42, backends=["stacked"],
                                 start_time=datetime.now(timezone.utc))
        assert "QA Random URL Report" in report, "report missing header"
        assert "## Errors" in report, "report missing errors section"
        assert "## Results" in report, "report missing results table"
        print("Report generation: OK (synthetic data)")
    except Exception as e:
        errors.append(f"Report generation crashed: {e}")

    if errors:
        print(f"\nFAILED: {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK: All invariants pass")
    return 0


def main() -> int:
    args = parse_args()

    if args.self_test:
        return self_test()

    start_time = datetime.now(timezone.utc)
    anno_bin = os.environ.get("ANNO", "./target/release/anno")
    backends = [b.strip() for b in args.backends.split(",")]

    if args.count < 1 or not all(backends):
        print("error: expected a positive count and a nonempty backend name", file=sys.stderr)
        return 1

    # Seed
    if args.seed is not None:
        random.seed(args.seed)

    # Filter sources
    pool = list(ALL_SOURCES)
    if args.no_rss:
        pool = [s for s in pool if s.category not in ("rss", "refhop")]
    if args.category != "all":
        pool = [s for s in pool if s.category == args.category]
    if args.lang:
        lang_set = {l.strip() for l in args.lang.split(",")}
        pool = [s for s in pool if s.lang in lang_set]

    if not pool:
        print("error: No sources match the given filters.", file=sys.stderr)
        return 1

    # Sample
    count = min(args.count, len(pool))
    selected = random.sample(pool, count)

    print(f"QA random URL test: {count} URLs, backend(s): {', '.join(backends)}, seed: {args.seed}")
    print(f"Source pool: {len(pool)} sources after filters")
    print()

    # Dry-run: show selection and resolve URLs, but don't run anno
    if args.dry_run:
        for i, source in enumerate(selected, 1):
            url = resolve_url(source)
            resolved = url or "(resolve failed)"
            print(f"  [{i}/{count}] {source.name} ({source.lang}/{source.script}) [{source.category}]")
            if url != source.url:
                print(f"           feed: {source.url}")
            print(f"           url:  {resolved}")
        return 0

    all_results: list[RunResult] = []

    for i, source in enumerate(selected, 1):
        # Resolve URL (RSS/refhop sources need a network fetch to get the actual article URL)
        url = resolve_url(source)
        if url is None:
            print(f"  [{i}/{count}] {source.name}: URL resolution failed")
            r = RunResult(source=source, url=source.url, status="error",
                          error_msg="URL resolution failed")
            all_results.append(r)
            continue

        for backend in backends:
            print(f"  [{i}/{count}] {source.name} ({source.lang}/{source.script}) -> {backend}...", end=" ", flush=True)
            result = run_anno(anno_bin, url, backend, args.timeout, args.verbose)
            result.source = source
            result.url = url  # resolved URL

            if result.status == "OK":
                top_types = ", ".join(f"{t}:{c}" for t, c in sorted(result.type_counts.items(), key=lambda x: -x[1])[:3])
                print(f"{result.entity_count} entities ({top_types}) in {result.elapsed_secs:.1f}s")
            else:
                print(f"{result.status}: {result.error_msg}")

            all_results.append(result)

        # Rate limiting
        if i < count and args.delay > 0:
            time.sleep(args.delay)

    # Generate report
    report = generate_report(all_results, args.seed, backends, start_time)

    # Output
    if args.output:
        out_path = Path(args.output)
    else:
        ts = start_time.strftime("%Y%m%d-%H%M%S")
        out_path = Path("qa/reports") / f"qa-random-{ts}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")

    # Summary
    ok_count = sum(1 for r in all_results if r.status == "OK")
    total = len(all_results)
    print(f"Done: {ok_count}/{total} OK")

    return 0 if ok_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
