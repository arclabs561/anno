#!/usr/bin/env python3
"""Replay eLife edition-cut author queries using the receipt, or verify the XML."""
import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
import urllib.request
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/elifesciences/elife-article-xml"
HREF = "{http://www.w3.org/1999/xlink}href"
PERMITTED = {
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
}
FIELDS = ("prefix", "given-names", "surname", "suffix")


def require(test, reason):
    if not test:
        raise ValueError(reason)


def clean(node):
    return unicodedata.normalize("NFC", " ".join("".join(node.itertext()).split()))


def byline(meta):
    authors = []
    for group in meta.findall("contrib-group"):
        for contributor in group.findall("contrib[@contrib-type='author']"):
            names = [x for x in contributor if x.tag in
                     ("name", "collab", "string-name", "name-alternatives", "anonymous")]
            if len(names) != 1:
                return None
            name = names[0]
            if name.tag == "name":
                if len({x.tag for x in name}) != len(name) or any(x.tag not in FIELDS for x in name):
                    return None
                values = [(field, clean(name.find(field))) for field in FIELDS if name.find(field) is not None]
            elif name.tag in ("collab", "string-name") and not len(name):
                values = [(name.tag, clean(name))]
            else:
                return None
            if not values or any(not value for _, value in values):
                return None
            authors.append(values)
    if not authors:
        return None
    literal = json.dumps(authors, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"count": len(authors), "sha256": sha256(literal).hexdigest()}


def xml_record(data, article, target=None):
    meta = ET.fromstring(data).find("./front/article-meta")
    require(meta is not None, "missing article metadata")
    doi = meta.findtext("article-id[@pub-id-type='doi']", "")
    require(doi.casefold() == f"10.7554/elife.{article}", "work DOI mismatch")
    licenses = [x.get(HREF) for x in meta.findall("permissions/license")]
    require(len(licenses) == 1 and licenses[0] in PERMITTED, "unrecognized license")
    record = {"xml_sha256": sha256(data).hexdigest(), "license": licenses[0]}
    if target is None:
        record["byline"] = byline(meta)
    else:
        links = {x.get(HREF, "").casefold() for x in meta.findall("related-article[@related-article-type='corrected-article']")}
        require(f"10.7554/elife.{target}" in links, "notice omits the target link")
    return record


def cuts(index, target, day):
    releases = [(x["versionDate"][:10], x["version"]) for x in index["versions"]
                if x.get("doi", "").casefold() == f"10.7554/elife.{target}" and "versionDate" in x and "version" in x]
    previous = sorted(x for x in releases if x[0] < day)
    same_day = [x for x in releases if x[0] == day]
    return (previous[-1] if previous else None), (same_day[0] if len(same_day) == 1 else None)


def cached(cache, name, url, fetch):
    path = cache / name
    if fetch and not path.exists():
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read()
        path.write_bytes(data)
    return path.read_bytes()


def check_case(case, revision, cache, fetch):
    notice, target, day = case["notice"], case["target"], case["day"]
    index = cached(cache, f"{target}.versions.json", f"https://api.elifesciences.org/articles/{target}/versions", fetch)
    before, after = cuts(json.loads(index), target, day)
    for key, cut in [("before", before), ("after", after)]:
        source = case[key]
        if source:
            require(cut == (source["day"], source["version"]), f"{notice}: {key} edition")

    for key in ("notice_source", "before", "after", "reference"):
        source = case[key]
        if source is None:
            continue
        is_notice = key == "notice_source"
        article = notice if is_notice else target
        file = f"elife-{article}-v{source['version']}.xml"
        name = ("historic-" if key == "reference" else "") + file
        url = f"{RAW}/{source.get('revision', revision)}/articles/{file}"
        summary = xml_record(cached(cache, name, url, fetch), article, target if is_notice else None)
        for field, value in summary.items():
            require(value == source[field], f"{notice}: {key} {field}")


def check_registry(cache, sample):
    rows = {}
    offset = 0
    while True:
        page = json.loads((cache / f"year_{offset}.json").read_bytes())["message"]
        require(page["items"], "empty registry page before registry end")
        for row in page["items"]:
            doi = row["DOI"].casefold()
            require(doi not in rows, "registry paging repeated a DOI")
            rows[doi] = row
        offset += len(page["items"])
        if offset >= page["total-results"]:
            break
    require(len(rows) == sample["registry_works"], "registry snapshot size changed")
    candidates = []
    for doi, row in rows.items():
        targets = [r["DOI"].casefold() for r in row.get("update-to", []) if r["type"].casefold() == "correction"]
        title = (row.get("title") or [""])[0]
        if targets or re.match(r"^(correction|erratum|corrigendum)\b", title, flags=re.IGNORECASE):
            require(len(targets) == 1, f"review missing or ambiguous target for {doi}")
            candidates.append({"day": date(*row["published"]["date-parts"][0]).isoformat(),
                               "notice": doi.removeprefix("10.7554/elife."),
                               "target": targets[0].removeprefix("10.7554/elife.")})
    candidates.sort(key=lambda r: (r["day"], r["notice"]))
    require(candidates == sample["eligible"], "eligible correction frame changed")


def evaluate(sample):
    registry, cases = sample["eligible"], sample["cases"]
    eligible = [(r["day"], r["notice"], r["target"]) for r in registry]
    require(eligible == sorted(eligible) and len({r[1] for r in eligible}) == len(eligible), "registry order or duplicates")
    require([(c["day"], c["notice"], c["target"]) for c in cases] == eligible[:12], "first twelve notices")
    tallies = {"queries": len(cases), "works": len({c["target"] for c in cases}), "verified_reference": 0}
    for arm in ["mutable", "edition"]:
        for metric in ["answered", "scorable", "errors"]:
            tallies[f"{arm}_{metric}"] = 0
    for c in cases:
        known = c["reference"]
        if c["before"]:
            require(c["before"]["day"] < c["day"], "invalid historical cut")
        if c["after"]:
            require(c["after"]["day"] == c["day"], "invalid correction-day cut")
        if known:
            require(known["committed"][:10] < c["day"], "reference commit is too late")
            if c["before"]:
                require(known["version"] == c["before"]["version"], "reference version differs")
        for key in ("before", "after", "notice_source", "reference"):
            source = c[key]
            if source:
                require(source["license"] in PERMITTED and re.fullmatch("[0-9a-f]{64}", source["xml_sha256"]), "source receipt")
                if source.get("byline"):
                    require(source["byline"]["count"] > 0 and re.fullmatch("[0-9a-f]{64}", source["byline"]["sha256"]), "byline receipt")
        reference = known and known["byline"]
        tallies["verified_reference"] += int(bool(reference))
        for arm, key in [("mutable", "after"), ("edition", "before")]:
            answer = c[key] and c[key]["byline"]
            tallies[f"{arm}_answered"] += int(bool(answer))
            if answer and reference:
                tallies[f"{arm}_scorable"] += 1
                tallies[f"{arm}_errors"] += int(answer != reference)
    return tallies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cache", type=Path, help="verify existing local XML and version indexes offline")
    mode.add_argument("--fetch", type=Path, help="populate an external cache and verify original XML")
    parser.add_argument("--registry-cache", type=Path, help="recheck selection from cached Crossref year_OFFSET.json pages")
    args = parser.parse_args()
    sample = json.loads((ROOT / "sample.json").read_text())
    require(sha256((ROOT / "sampling.txt").read_bytes()).hexdigest() == sample["design_sha256"], "frozen design digest")
    if args.registry_cache:
        check_registry(args.registry_cache, sample)
    stats = evaluate(sample)
    require(stats == sample["expected"], "replay result changed")
    cache = args.cache or args.fetch
    if cache:
        if args.fetch:
            cache.mkdir(parents=True, exist_ok=True)
        for case in sample["cases"]:
            check_case(case, sample["revision"], cache, bool(args.fetch))
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
