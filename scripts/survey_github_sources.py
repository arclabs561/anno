"""Inspect registry GitHub pointers and measure selected, identified data releases.

Run after plot_datasets.py or use the committed inventory:
    python scripts/survey_github_sources.py --output evidence.csv
Requires the gh CLI. Only metadata and numeric aggregates are written. Pass the
previous evidence CSV with --pins to replay its repo revisions. Repository revisions identify the source tree. SciCo uses
the author repository and checksum-verified external tar; its bytes and sha256
refer to the archive, and path identifies the member. split can also identify
a publisher partition (domain, book or category) when there is no split. A cache
directory can retain Git trees; failed lookups are retried on subsequent runs.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from survey_repo_files import measure as measure_token_file

ROOT = Path(__file__).resolve().parents[1]
FIELDS = "id registry_url repository revision path split status reason source_url sha256 bytes metric unit value".split()
RAW_EXTRAS = {"BBQ", "GAP", "MuDoCo"}
# Match only reviewed paths. The captured group denotes the split or source
# partition; labels such as 'all' and 'unsplit' are explicit where appropriate.
PATHS = {
    "BBQ": r"data/([^/]+)\.jsonl",
    "GAP": r"gap-(development|validation|test)\.tsv",
    "MuDoCo": r"mudoco_(\w+)\.json",
    "BoldBias": r"prompts/(\w+)_prompt\.json",
    "CrowSPairs": r"data/crows_pairs_anonymized\.csv",
    "StereoSet": r"data/(dev)\.json",
    "DialogRE": r"data_v2/en/data/(train|test|dev)\.json",
    "MedMentions": r"full/data/corpus_pubtator\.txt\.gz",
    "ATISFlightBooking": r"data/(?:atis-2\.(train|dev)|atis\.(test))\.w-intent\.iob",
    "SemEval2010Task8": r"corpus/SemEval2010_task8_(training/TRAIN_FILE|testing_keys/TEST_FILE_FULL)\.TXT",
    "CoNLL04RE": r"data/CoNLL04/(dev|train|test)\.txt",
    "GVC": r"gold\.conll",
    "PragmEval": r"pragmeval/([^/]+/(?:train|dev|test))\.tsv",
    "MATRES": r"(timebank|aquaint|platinum)\.txt",
    "TASTEset": r"data/TASTEset\.csv",
    "NDNER": r"ND-NER_flat/(dev|train|test)_bio\.txt",
    "Re3dDefense": r"([^/]+)/documents\.json",
    "FashionIQ": r"captions/cap\.(dress|shirt|toptee)\.(train|val|test)\.json",
    "KoCoNovel": r"data/jsonl/([^/]+)/coref\.jsonl",
    "DISAPERE": r"DISAPERE\.zip",
    "AIONER": r"data/pubtator/(\w+)\.PubTator",
}
# Do not identify a registered corpus by a substring or by its host repository.
COLLECTION = "juand-r/entity-recognition-datasets"


def repo_ref(url):
    parsed = urllib.parse.urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.hostname not in {"github.com", "raw.githubusercontent.com"}
        or len(parts) < 2
    ):
        return ""
    return "/".join(parts[:2])


def api(repository, path):
    response = subprocess.run(
        ["gh", "api", "repos/" + repository + "/" + path],
        capture_output=True,
        timeout=90,
    )
    if response.returncode:
        match = re.search(r"HTTP \d+", response.stderr.decode(errors="replace"))
        raise RuntimeError(match[0] if match else "GitHub API error")
    return json.loads(response.stdout)


def tree(repository, revision, cache):
    key = (
        repository.replace("/", "__") + ("__" + revision if revision else "") + ".json"
    )
    cached = cache / key if cache else None
    if cached and cached.exists():
        value = json.loads(cached.read_text())
        if value["status"] == "ok":
            return value
    try:
        sha = revision or api(repository, "commits/HEAD")["sha"]
        data = api(repository, f"git/trees/{sha}?recursive=1")
        result = {
            "status": "ok",
            "sha": sha,
            "truncated": data["truncated"],
            "files": {
                i["path"]: i["size"] for i in data["tree"] if i["type"] == "blob"
            },
        }
        if cached:
            cached.write_text(json.dumps(result))
        return result
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "code": str(exc)}


def metrics(dataset, raw):
    """Return tuples of (unit, count) for one identified source file."""
    if dataset == "DISAPERE":
        totals = Counter()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for path in archive.namelist():
                if re.fullmatch(
                    r"DISAPERE/final_dataset/(train|test|dev)/[^/]+\.json", path
                ):
                    item = json.loads(archive.read(path))
                    totals["review-rebuttal pairs"] += 1
                    totals["review sentences"] += len(item["review_sentences"])
                    totals["rebuttal sentences"] += len(item["rebuttal_sentences"])
        if not totals:
            raise ValueError("no final dataset members in archive")
        return list(totals.items())
    text = (gzip.decompress(raw) if dataset == "MedMentions" else raw).decode(
        "utf-8-sig"
    )
    if dataset == "KoCoNovel":
        # Choose one annotation perspective and entity convention. The others
        # annotate the same material and would inflate the mention count.
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        mentions = sum(
            len(cluster) for r in records for cluster in r["omniscent_separate"]
        )
        return [
            ("annotated segments", len(records)),
            ("omniscient separate mentions", mentions),
        ]
    if dataset == "BBQ":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not all("question" in row and "label" in row for row in records):
            raise ValueError("unexpected BBQ record")
        return [("questions", len(records))]
    if dataset in {"GAP", "CrowSPairs", "PragmEval", "TASTEset"}:
        delimiter = "," if dataset in {"CrowSPairs", "TASTEset"} else "\t"
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
        required = {
            "GAP": {"ID", "Pronoun"},
            "CrowSPairs": {"sent_more", "sent_less"},
            "PragmEval": {"label"},
            "TASTEset": {"ingredients_entities"},
        }[dataset]
        if not required <= set(reader.fieldnames or []):
            raise ValueError("unexpected tabular header")
        units = {
            "GAP": "pronoun decisions",
            "CrowSPairs": "sentence pairs",
            "PragmEval": "classification examples",
            "TASTEset": "recipe records",
        }
        records = list(reader)
        values = [(units[dataset], len(records))]
        if dataset == "TASTEset":
            values.append(
                (
                    "ingredient spans",
                    sum(len(json.loads(r["ingredients_entities"])) for r in records),
                )
            )
        return values
    if dataset == "MedMentions" or dataset == "AIONER":
        documents, mentions = set(), 0
        for line in text.splitlines():
            if match := re.match(r"^([^|\t]+)\|t\|", line):
                documents.add(match[1])
            fields = line.split("\t")
            if len(fields) >= 5 and fields[1].isdigit() and fields[2].isdigit():
                mentions += 1
        if not documents:
            raise ValueError("no PubTator document headers")
        return [("documents", len(documents)), ("mention annotations", mentions)]
    if dataset == "ATISFlightBooking":
        utterances = tokens = tagged = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            words, labels = (field.split() for field in line.split("\t"))
            if (
                len(words) < 3
                or len(words) != len(labels)
                or words[0] != "BOS"
                or words[-1] != "EOS"
                or labels[0] != "O"
                or not re.fullmatch(r"atis_\w+(?:#atis_\w+)*", labels[-1])
            ):
                raise ValueError("Unexpected ATIS token framing, alignment, or intent")
            # BOS has an O label. The EOS position holds an utterance intent.
            previous = "O"
            for slot in labels[1:-1]:
                if not re.fullmatch(r"O|[BI]-.+", slot):
                    raise ValueError("Invalid ATIS slot label")
                if slot.startswith("I-") and (
                    previous == "O" or previous[2:] != slot[2:]
                ):
                    raise ValueError("ATIS slot continues without a matching begin")
                previous = slot
                tagged += slot != "O"
            utterances += 1
            tokens += len(words) - 2
        if not utterances:
            raise ValueError("Empty ATIS release file")
        return [
            ("utterances", utterances),
            ("tokens", tokens),
            ("tagged tokens", tagged),
        ]
    if dataset == "SemEval2010Task8":
        count = sum(bool(re.match(r'^\d+\s+"', line)) for line in text.splitlines())
        if not count:
            raise ValueError("no numbered relation examples")
        return [("relation examples", count)]
    if dataset in {"GVC", "CoNLL04RE"}:
        prefix = "#begin document " if dataset == "GVC" else "#doc "
        count = sum(line.startswith(prefix) for line in text.splitlines())
        if not count:
            raise ValueError("no document markers")
        if dataset == "CoNLL04RE":
            _, tokens, tagged = measure_token_file(
                raw, {"id": dataset, "metric": "CoNLL04_entity_BIO"}
            )
            return [("documents", count), ("tokens", tokens), ("tagged tokens", tagged)]
        return [("documents", count)]
    if dataset == "MATRES":
        records = list(csv.reader(io.StringIO(text), delimiter="\t"))
        if not all(len(row) == 6 for row in records):
            raise ValueError("unexpected MATRES row")
        return [("event-pair labels", len(records))]
    if dataset == "NDNER":
        tags = [line.split()[-1] for line in text.splitlines() if line.strip()]
        if not all(re.fullmatch(r"O|[BI]-.+", tag) for tag in tags):
            raise ValueError("unexpected ND-NER label")
        return [("tokens", len(tags)), ("tagged tokens", sum(t != "O" for t in tags))]
    if dataset == "Re3dDefense":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not all("_id" in doc and "sourceName" in doc for doc in records):
            raise ValueError("missing document identifier")
        return [("documents", len(records))]
    content = json.loads(text)
    if dataset == "MuDoCo":
        dialogs = content["dialogs"].values()
        return [
            ("dialogues", len(dialogs)),
            ("turns", sum(len(d["turns"]) for d in dialogs)),
        ]
    if dataset == "DialogRE":
        if not all(len(d) == 2 and isinstance(d[1], list) for d in content):
            raise ValueError("unexpected DialogRE dialogue structure")
        return [
            ("dialogues", len(content)),
            ("annotated argument pairs", sum(len(d[1]) for d in content)),
        ]
    if dataset == "StereoSet":
        # Test terms contain no released labeled test examples.
        return [
            (
                "contexts",
                sum(
                    len(content["data"][k]) for k in ("intrasentence", "intersentence")
                ),
            )
        ]
    if dataset == "BoldBias":
        return [
            (
                "prompts",
                sum(
                    len(prompts)
                    for group in content.values()
                    for prompts in group.values()
                ),
            )
        ]
    if dataset == "FashionIQ":
        if not all("candidate" in row and "captions" in row for row in content):
            raise ValueError("unexpected FashionIQ caption structure")
        return [("image comparison records", len(content))]
    raise ValueError("unsupported measurement")


def scico(base):
    """Pinned author README links this archive; verify its independent byte digest."""
    author_commit = "e1055b74ca9e2f0a3c89424ce24e68c01d936548"
    source = "https://nlp.biu.ac.il/~ariecattan/scico/data.tar"
    expected = "324c13035caaeb4e6535cf6e998667091656f4f94c8b49f4d2986886a98237eb"
    evidence = dict(
        base, repository="ariecattan/SciCo", revision=author_commit, source_url=source
    )
    try:
        with urllib.request.urlopen(source, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("author release digest changed")
        result = []
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for split in ("train", "dev", "test"):
                member = split + ".jsonl"
                records = [
                    json.loads(line)
                    for line in archive.extractfile(member)
                    if line.strip()
                ]
                counts = {
                    "topics": len(records),
                    "concept mentions": sum(len(r["mentions"]) for r in records),
                    "relations": sum(len(r["relations"]) for r in records),
                }
                for unit, value in counts.items():
                    result.append(
                        dict(
                            evidence,
                            path=member,
                            split=split,
                            status="ok",
                            sha256=expected,
                            bytes=len(raw),
                            metric="count",
                            unit=unit,
                            value=value,
                        )
                    )
        return result
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        return [dict(evidence, status="unmeasured", reason=str(exc))]


def inspect(args):
    row, repository, inventory, local = args
    dataset = row["id"]
    revision = inventory.get("sha", "")
    base = dict(
        id=dataset, registry_url=row["url"], repository=repository, revision=revision
    )

    def skip(reason):
        return [dict(base, status="unmeasured", reason=reason)]

    if not repository:
        return skip("owner page or incomplete repository URL")
    if inventory["status"] != "ok":
        return skip(
            "repository/tree lookup failed: "
            + inventory.get("code", inventory["status"])
        )
    if dataset in local or row["access"] in {
        "Local",
        "ContactAuthors",
        "Registration",
        "DependsOnOther",
    }:
        return skip("local or access-dependent registry entry; tree metadata only")
    if dataset == "SciCo":
        return scico(base)
    if dataset not in PATHS:
        if repository == COLLECTION:
            reason = (
                "named folder has documentation only"
                if dataset == "MalwareTextDB"
                else "collection-level link; no verified matching release for this registry ID"
            )
        else:
            reason = "repository tree inspected; no reviewed release-file mapping in this survey"
        if inventory.get("truncated"):
            reason += "; GitHub returned a truncated tree"
        return skip(reason)
    records = []
    for path, size in sorted(inventory["files"].items()):
        match = re.fullmatch(PATHS[dataset], path)
        if not match:
            continue
        partition = "/".join(v for v in match.groups() if v) or "all"
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/{urllib.parse.quote(path)}"
        entry = dict(base, path=path, split=partition, source_url=url)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
            if len(raw) != size:
                raise ValueError("size differs from pinned tree")
            values = metrics(dataset, raw)
            records.extend(
                dict(
                    entry,
                    status="ok",
                    bytes=size,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    metric="count",
                    unit=unit,
                    value=value,
                )
                for unit, value in values
            )
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            records.append(
                dict(entry, status="unmeasured", reason=f"{type(exc).__name__}: {exc}")
            )
    return records or skip("expected release paths absent in pinned tree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "docs/eval/dataset-figure-inventory.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--pins", type=Path, help="use revisions from existing evidence CSV"
    )
    args = parser.parse_args()
    if args.cache_dir:
        args.cache_dir.mkdir(exist_ok=True, parents=True)
    with args.inventory.open(newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if urllib.parse.urlsplit(row["url"]).hostname == "github.com"
            or row["id"] in RAW_EXTRAS
        ]
    pins = {}
    if args.pins:
        with args.pins.open(newline="") as f:
            pins = {
                r["repository"]: r["revision"]
                for r in csv.DictReader(f)
                if r["revision"]
            }
    rules = ROOT / "crates/anno-eval/src/eval/shared_cache_exclusions.tsv"
    local = {
        l.split("\t")[0].removesuffix(".cache")
        for l in rules.read_text().splitlines()
        if l.split("\t")[1] == "local"
    }
    repos = sorted({repo_ref(row["url"]) for row in rows} - {""})
    with ThreadPoolExecutor(6) as pool:
        trees = dict(
            zip(
                repos,
                pool.map(
                    lambda repo: tree(repo, pins.get(repo), args.cache_dir), repos
                ),
            )
        )
        batches = pool.map(
            inspect,
            [
                (r, repo_ref(r["url"]), trees.get(repo_ref(r["url"]), {}), local)
                for r in rows
            ],
        )
        records = [record for batch in batches for record in batch]
    with args.output.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(
        f"{len(rows)} registry IDs, {len(repos)} repositories; file/metric rows: "
        f"{dict(Counter(r['status'] for r in records))}"
    )


if __name__ == "__main__":
    main()
