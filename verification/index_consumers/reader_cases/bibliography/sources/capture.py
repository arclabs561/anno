# SPDX-License-Identifier: MIT OR Apache-2.0
"""Rebuild the evidence-only publisher feed from version indexes, version records and linked JATS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from http.client import HTTPException
import json
from pathlib import Path
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
API = "https://api.elifesciences.org/articles"
XLINK = "{http://www.w3.org/1999/xlink}href"
LICENSES = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}
FIELDS = ("prefix", "given-names", "surname", "suffix")


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def digest(data):
    return sha256(data).hexdigest()


def names(meta):
    authors = []
    for contrib in meta.findall("contrib-group/contrib[@contrib-type='author']"):
        named = [n for n in contrib if n.tag in ("name", "collab", "string-name", "name-alternatives", "anonymous")]
        if len(named) != 1:
            return None
        name = named[0]
        if name.tag == "name":
            if len(set(n.tag for n in name)) != len(name) or any(n.tag not in FIELDS for n in name):
                return None
            parts = [name.find(k) for k in FIELDS if name.find(k) is not None]
        elif name.tag in ("collab", "string-name") and len(name) == 0:
            parts = [name]
        else:
            return None
        fields = [[n.tag, unicodedata.normalize("NFC", " ".join("".join(n.itertext()).split()))] for n in parts]
        if not fields or any(not value for _, value in fields):
            return None
        authors.append(fields)
    return authors or None


def download_missing(path, url):
    if path.exists():
        return
    request = Request(url, headers={"User-Agent": "anno-publication-study/1.0 (https://github.com/arclabs561/anno)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                data = response.read()
            path.write_bytes(data)
            return
        except (HTTPError, URLError, HTTPException, TimeoutError):
            if attempt == 2:
                return
            time.sleep(2 ** attempt)


def publisher_xml_url(url, article, number):
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return (parts.scheme, parts.netloc, parts.path, parts.query, parts.fragment) == (
        "https", "cdn.elifesciences.org", f"/articles/{article}/elife-{article}-v{number}.xml", "", "",
    )


def boundary(index, article):
    releases = []
    excluded = 0
    for row in index["versions"]:
        if row.get("status") == "preprint" and "version" not in row:
            excluded += 1
            continue
        if (row.get("status") not in ("poa", "vor") or row.get("id") != article
                or row.get("doi", "").lower() != f"10.7554/elife.{article}" or type(row.get("version")) is not int):
            raise ValueError("unknown or inconsistent index entry")
        stamp = datetime.fromisoformat(row["versionDate"].replace("Z", "+00:00"))
        if stamp.utcoffset() != timedelta(0):
            raise ValueError("versionDate lacks UTC offset")
        releases.append((row["version"], stamp, row["status"]))
    releases.sort()
    if (not releases or [r[0] for r in releases] != list(range(1, len(releases) + 1))
            or [r[1] for r in releases] != sorted(r[1] for r in releases)):
        raise ValueError("index has missing, duplicate or nonchronological release keys")
    return releases, excluded


def source(cache, article, expected_target, fetch, capture_at):
    doi = f"10.7554/eLife.{article}"
    through = (capture_at.date() - timedelta(days=1)).isoformat()
    url = f"{API}/{article}/versions"
    index_file = cache / f"{article}-versions.json"
    if fetch:
        download_missing(index_file, url)
    info = {"index_url": url, "editions": []}
    if not index_file.is_file():
        return dict(info, status="index_unavailable"), None, []
    index_bytes = index_file.read_bytes()
    info["index_sha256"] = digest(index_bytes)
    try:
        releases, excluded = boundary(json.loads(index_bytes), article)
        if releases[-1][1].date() > date.fromisoformat(through):
            raise ValueError("latest release beyond complete UTC days")
    except (ValueError, KeyError, TypeError):
        return dict(info, status="incomplete_index"), None, []
    info.update(status="complete_index", through=through, external_preprints=excluded)
    editions = []
    rights = []
    for number, stamp, status in releases:
        day = stamp.date().isoformat()
        at = stamp.isoformat().replace("+00:00", "Z")
        api_url = f"{url}/{number}"
        row = {"edition": number, "published_on": day}
        proof = dict(row, version_date=at, publisher_status=status, version_url=api_url)
        info["editions"].append(proof)
        editions.append(row)
        api_path = cache / f"{article}-v{number}.json"
        if fetch:
            download_missing(api_path, api_url)
        if not api_path.is_file():
            proof["status"] = "version_record_unavailable"
            continue
        data = api_path.read_bytes()
        proof["version_sha256"] = digest(data)
        try:
            record = json.loads(data)
            source_url = record.get("xml")
            if (record.get("doi", "").lower() != doi.lower() or record.get("id") != article
                    or record.get("version") != number or record.get("status") != status
                    or datetime.fromisoformat(record["versionDate"].replace("Z", "+00:00")) != stamp
                    or not publisher_xml_url(source_url, article, number)):
                raise ValueError("unverified version")
            license_uri = LICENSES[record["copyright"]["license"]]
        except (ValueError, TypeError, KeyError):
            proof["status"] = "version_or_rights_unverified"
            continue
        proof["source_url"] = source_url
        xml_path = cache / f"{article}-v{number}.xml"
        if fetch:
            download_missing(xml_path, source_url)
        if not xml_path.is_file():
            proof["status"] = "xml_unavailable"
            continue
        xml = xml_path.read_bytes()
        xml_hash = digest(xml)
        proof["source_sha256"] = xml_hash
        try:
            meta = ET.fromstring(xml).find("./front/article-meta")
            if meta is None or meta.findtext("article-id[@pub-id-type='doi']", "").lower() != doi.lower():
                raise ValueError("different work")
            licenses = [p.get(XLINK, "").replace("http://", "https://", 1) for p in meta.findall("permissions/license")]
            if licenses != [license_uri]:
                raise ValueError("unrecognized or inconsistent rights")
            if expected_target is not None:
                targets = [p.get(XLINK, "").lower() for p in meta.findall("related-article[@related-article-type='corrected-article']")]
                if targets != [expected_target.lower()]:
                    raise ValueError("notice target differs from frame")
        except (ET.ParseError, ValueError):
            proof["status"] = "xml_identity_rights_or_relation_unverified"
            continue
        proof["license"] = license_uri
        rights.append({"doi": doi, "edition": number, "source_url": source_url,
                       "license": license_uri, "holder": record["copyright"].get("holder")})
        if expected_target is not None:
            proof.update(status="verified_notice", corrected_doi=expected_target)
            continue
        ordered = names(meta)
        if ordered is None:
            proof["status"] = "unrepresented_byline"
            continue
        proof["status"] = "supported"
        row.update(source_url=source_url, source_sha256=xml_hash, license=license_uri, byline=ordered)
    return info, {"doi": doi, "through": through, "editions": editions}, rights


def collect(cache, fetch, capture_at):
    frame = json.loads((HERE / "frame.json").read_bytes())
    original = (HERE / frame["frame_source"]).read_bytes()
    rows = frame["rows"]
    derived = (json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
    if (digest(original) != frame["frame_sha256"] or digest(derived) != frame["rows_sha256"]
            or rows != json.loads(original)["eligible"][12:] or len(rows) != 17):
        raise ValueError("frozen intake mismatch")
    identities = [(r["target"], None) for r in rows] + [(r["notice"], f"10.7554/eLife.{r['target']}") for r in rows]
    if len(set(a for a, _ in identities)) != len(identities):
        raise ValueError("repeated source key in frozen panel")
    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = list(pool.map(lambda t: source(cache, *t, fetch, capture_at), identities))
    entries = {article: proof for (article,_), (proof,_,_) in zip(identities, observed)}
    works = [w for (_, target), (_, w, _) in zip(identities, observed) if target is None and w is not None]
    rights = [a for _,_,batch in observed for a in batch]
    receipt = {"capture_at": capture_at.isoformat(), "frame_sha256": digest((HERE / "frame.json").read_bytes()), "sources": entries}
    return receipt, {"works": works}, rights


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache", type=Path, required=True)
    cli.add_argument("--fetch", action="store_true", help="download missing public source artifacts to external cache")
    cli.add_argument("--output", type=Path, required=True, help="external directory for the private reader feed and rights")
    cli.add_argument("--observe-at", help="create a fresh receipt at this UTC timestamp instead of verifying the pinned capture")
    args = cli.parse_args()
    for path in [args.cache, args.output]:
        if path.resolve().is_relative_to(HERE.parents[4]):
            cli.error("source cache and prepared bylines belong outside the repository")
        path.mkdir(parents=True, exist_ok=True)
    pinned = None if args.observe_at else json.loads((HERE / "receipt.json").read_bytes())
    stamp = datetime.fromisoformat((args.observe_at or pinned["capture_at"]).replace("Z", "+00:00"))
    if stamp.utcoffset() != timedelta(0):
        cli.error("capture timestamp must be UTC")
    receipt, wire, rights = collect(args.cache, args.fetch, stamp)
    if pinned is not None and receipt != pinned:
        (args.output / "unverified-receipt.json").write_bytes(encode(receipt))
        raise ValueError("capture differs from the pin; see unverified-receipt.json. No new feed was emitted.")
    (args.output / "receipt.json").write_bytes(encode(receipt))
    (args.output / "reader_input.json").write_bytes(encode(wire))
    (args.output / "attributions.json").write_bytes(encode(rights))
    print(f"{len(wire['works'])} complete work indexes, {len(rights)} rights-verified source editions")


if __name__ == "__main__":
    main()
