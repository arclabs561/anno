"""Project selected ONS CSV lookups and MusicBrainz core fields from public inputs."""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from http.client import RemoteDisconnected
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

ONS = {
    "2023-12": ("3acc892515aa49a8885c2deb734ebd3d", "3dee2254ad7ccd4583b3fde832afcd7d40ae092929d5cc612fc01f3e5e727694"),
    "2025-12": ("c59bd9b324644773b92d4f06d1f48dd7", "8b913fef0dfa5637807b29eadeae3c33105d8af4e59ad2eb06389313d3437b45"),
}
# Capture every outgoing Changes.csv row for these historical codes.
ORIGINS = {"E04006085", "E07000028", "E07000029", "E10000006", "E10000020"}
MUSIC = {
    "release/ad6c808f-2948-478a-aad5-3e02a9692e67": "artist-credits+artist-rels",
    "release/0e97456e-9471-4869-bf43-10ea495f666a": "artist-credits+artist-rels",
    "recording/b040d89e-190f-47a0-af7f-20534101056c": "artist-credits+artist-rels+work-rels",
    "work/dee610f0-e425-311f-be39-554747afbddf": "artist-rels",
}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def ons_date(value):
    if value:
        return datetime.strptime(value.split()[0], "%d/%m/%Y").date().isoformat()
    return None


def csv_rows(archive, name):
    with archive.open(name) as stream:
        yield from csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))


def geography(path, digest):
    with path.open("rb") as f:
        got = hashlib.file_digest(f, "sha256").hexdigest()
    if got != digest:
        raise ValueError(f"Unexpected ONS archive digest for {path}: {got}")
    with ZipFile(path) as archive:
        edges = []
        codes = set(ORIGINS)
        for row in csv_rows(archive, "Changes.csv"):
            origin = row["GEOGCD_P"]
            if origin in ORIGINS:
                target = row["GEOGCD"] or None
                edges.append({"from": origin, "to": target, "effective": ons_date(row["OPER_DATE"]), "order": row["SI_ID"]})
                if target:
                    codes.add(target)
        areas = {}
        for row in csv_rows(archive, "ChangeHistory.csv"):
            code = row["GEOGCD"]
            if code in codes:
                if code in areas:
                    raise ValueError(f"Multiple history rows for {code}")
                areas[code] = {
                    "name": row["GEOGNM"], "class": row["ENTITYCD"],
                    "operational": ons_date(row["OPER_DATE"]), "terminated": ons_date(row["TERM_DATE"]),
                }
    if set(areas) != codes:
        raise ValueError(f"Missing area history: {codes - set(areas)}")
    return {"origins": sorted(ORIGINS), "areas": areas, "changes": sorted(edges, key=lambda e: (e["from"], e["to"] or ""))}


def music_record(key, raw):
    kind, mbid = key.split("/")
    if raw["id"] != mbid:
        raise ValueError(f"Wrong entity for {key}")
    item = {"title": raw["title"], "roles": [], "works": []}
    if kind != "work":
        item["billing"] = [
            {"artist": c["artist"]["id"], "catalog_name": c["artist"]["name"], "credited": c["name"], "join": c["joinphrase"]}
            for c in raw["artist-credit"]
        ]
        field = "date" if kind == "release" else "first-release-date"
        item[field] = raw[field]
    for relation in raw["relations"]:
        target = relation["target-type"]
        if target == "artist" and relation["direction"] == "backward":
            item["roles"].append({
                "role": relation["type"], "role_id": relation["type-id"],
                "attributes": sorted(relation["attributes"]), "artist": relation["artist"]["id"],
                "catalog_name": relation["artist"]["name"],
            })
        if target == "work" and relation["type"] == "performance" and relation["direction"] == "forward":
            item["works"].append(relation["work"]["id"])
    item["roles"].sort(key=lambda r: (r["role_id"], r["artist"], r["attributes"]))
    item["works"].sort()
    return item


def cached_input(path, url, fetch):
    if not path.exists() and fetch:
        request = Request(url, headers={"User-Agent": "anno-reader-pilots/1.0 (https://github.com/arclabs561/anno)"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=60) as response:
                    data = response.read()
                break
            except HTTPError as error:
                if error.code not in (429, 502, 503, 504) or attempt == 2:
                    raise
            except (RemoteDisconnected, TimeoutError, URLError):
                if attempt == 2:
                    raise
            sleep(2 ** (attempt + 1))
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if "musicbrainz.org" in url:
            sleep(1.1)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; supply the cached file or use --fetch")
    return path


def capture(cache, fetch):
    if fetch:
        cache.mkdir(parents=True, exist_ok=True)
    geo = {}
    for edition, (item, digest) in ONS.items():
        file = cached_input(cache / f"ons-{edition}.zip", f"https://www.arcgis.com/sharing/rest/content/items/{item}/data", fetch)
        geo[edition] = geography(file, digest)
    music = {}
    for key, includes in MUSIC.items():
        url = f"https://musicbrainz.org/ws/2/{key}?" + urlencode({"inc": includes, "fmt": "json"})
        file = cached_input(cache / (key.replace("/", "-") + ".json"), url, fetch)
        music[key] = music_record(key, json.loads(file.read_bytes()))
    return {"ons": geo, "musicbrainz": music}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="separate local directory for source responses")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="download missing source files into the cache")
    args = parser.parse_args()
    args.output.write_bytes(json_bytes(capture(args.cache, args.fetch)))
