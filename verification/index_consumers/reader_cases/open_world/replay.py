# SPDX-License-Identifier: MIT OR Apache-2.0
"""Score a deliberately small catalog constructed from labeled NILK training mentions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
FILES = {"train": (239651, 9446345420), "test": (237901, 11660991447)}
WIDTH = 2 * 1024**2


def fold(text):
    value = unicodedata.normalize("NFD", text.casefold())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def window_records(cache, split, fetch, checksums):
    file_id, size = FILES[split]
    observed = {}
    for quarter in range(4):
        start = size * quarter // 4
        end = start + WIDTH - 1
        name = f"{split}-{start}.jsonl.part"
        path = cache / name
        if not path.exists() and fetch:
            req = Request(f"https://darus.uni-stuttgart.de/api/access/datafile/{file_id}?window={start}",
                          headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
            with urlopen(req, timeout=60) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise ValueError(f"{name}: server did not supply the requested byte range")
                block = response.read(WIDTH + 1)
            digest = hashlib.sha256(block).hexdigest()
            if len(block) != WIDTH or checksums[name] != digest:
                raise ValueError(f"{name}: unexpected download")
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as staging:
                staged_path = Path(staging.name)
            try:
                staged_path.write_bytes(block)
                staged_path.replace(path)
            finally:
                staged_path.unlink(missing_ok=True)
        block = path.read_bytes()
        digest = hashlib.sha256(block).hexdigest()
        if len(block) != WIDTH or checksums[name] != digest:
            raise ValueError(f"{name}: wrong cached range or checksum")
        observed[name] = digest
        for line in block.split(b"\n")[int(start > 0):-1]:
            r = json.loads(line)
            if not isinstance(r["nil"], bool) or not r["wikidata_id"].startswith("Q") or not isinstance(r["mention"], str):
                raise ValueError("invalid NILK mention or label")
            # The scorer keeps only these four fields from each source record.
            yield start, r["id"], (r["wikipedia_page_id"], r["mention"], r["wikidata_id"], r["nil"])
    if set(observed) != {k for k in checksums if k.startswith(split + "-")}:
        raise ValueError(f"{split}: unexpected source manifest")


def windows(cache, split, fetch, checksums):
    for _, key, row in window_records(cache, split, fetch, checksums):
        yield key, row


def unique_rows(rows):
    seen = {}
    duplicates = 0
    for key, row in rows:
        if key in seen:
            if seen[key] != row:
                raise ValueError(f"conflicting observation {key}")
            duplicates += 1
        else:
            seen[key] = row
    return list(seen.values()), duplicates


def cross_page_pairs(keys):
    # Count unordered pairs while excluding pairs within one source page.
    groups = Counter()
    pages = Counter()
    total = 0
    for group, page in keys:
        total += groups[group] - pages[group, page]
        groups[group] += 1
        pages[group, page] += 1
    return total


def evaluate(training, testing):
    train_pages = {r[0] for r in training}
    test_pages = {r[0] for r in testing}
    queries = [r for r in testing if r[0] not in train_pages]
    catalog_ids = {qid for _, _, qid, nil in training if not nil}
    nil_rows = [(p, name, qid) for p, name, qid, nil in queries if nil]
    result = {"train_records": len(training), "train_pages": len(train_pages), "test_records": len(testing),
              "test_pages": len(test_pages), "shared_pages": len(train_pages & test_pages),
              "excluded_records_from_shared_pages": len(testing) - len(queries),
              "queries": len(queries), "sampled_catalog_ids": len(catalog_ids), "old_catalog_nil": len(nil_rows),
              "nil_with_later_id": sum(bool(qid) for _, _, qid in nil_rows),
              "empty_test_mentions": sum(not name for _, name, _, _ in queries), "policies": {}}
    for policy, normalize in (("exact", str), ("fold", fold)):
        index = defaultdict(set)
        for _, name, qid, nil in training:
            if not nil and normalize(name):
                index[normalize(name)].add(qid)
        counts = Counter()
        for _, name, qid, nil in queries:
            candidates = index.get(normalize(name), set())
            group = "nil" if nil else ("id_sampled" if qid in catalog_ids else "id_unsampled")
            decision = "ambiguous" if candidates else "empty"
            if len(candidates) == 1:
                decision = "correct" if not nil and qid in candidates else "wrong"
            counts[group + "." + decision] += 1
        gold = cross_page_pairs((qid, p) for p, _, qid in nil_rows)
        proposed = cross_page_pairs((normalize(name), p) for p, name, _ in nil_rows if normalize(name))
        correct = cross_page_pairs(((qid, normalize(name)), p) for p, name, qid in nil_rows if normalize(name))
        result["policies"][policy] = {
            "links": dict(sorted(counts.items())),
            "nil_pairs": {"gold": gold, "correct": correct, "wrong": proposed-correct, "missed": gold-correct},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--fetch", action="store_true", help="download missing 2 MiB ranges into the local cache")
    args = parser.parse_args()
    pin = json.loads((HERE / "receipt.json").read_text())
    if hashlib.sha256((HERE / "sampling.txt").read_bytes()).hexdigest() != pin["design_sha256"]:
        raise ValueError("sampling design changed")
    if args.fetch:
        args.cache.mkdir(parents=True, exist_ok=True)
    rows = {}
    duplicate_counts = {}
    for split in FILES:
        rows[split], duplicate_counts[split] = unique_rows(windows(args.cache, split, args.fetch, pin["sha256"]))
    summary = evaluate(rows["train"], rows["test"])
    summary["duplicate_ids"] = duplicate_counts
    if summary != pin["result"]:
        raise ValueError("NILK result differs from the frozen receipt")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
