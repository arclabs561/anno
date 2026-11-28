"""Package the reviewed source files of Anno's pinned publisher survey.

  python scripts/pack_publisher_datasets.py build --cache-dir ~/.cache/anno/publisher \
      --notice-cache ~/.cache/anno/publisher-notices --fetch-missing --output publisher
  python scripts/pack_publisher_datasets.py verify publisher/publisher-general.zip --replay

A build without --fetch-missing uses local inputs only. Downloading is limited to
reviewed collections and their pinned notices. An optional existing cache also
lets the coverage report attest the excluded files' receipt hashes locally.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from survey_repo_files import measure

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
LEDGER = ROOT / "docs/eval/dataset-repo-files.csv"
TERMS = SCRIPT.parent / "data/publisher_archive_terms.json"
EXCLUSIONS = ROOT / "crates/anno-eval/src/eval/shared_cache_exclusions.tsv"
BUNDLES = {
    "general": "Publisher files with attribution, share-alike, public-domain or permissive terms as recorded individually.",
    "noncommercial": "The included release or an underlying layer has a noncommercial condition. Use the source-specific terms in manifest.csv.",
    "annotated-collection": "The original NorNE annotated collection. Third-party works may not be republished as stand-alone texts. Keep the publisher notices and source lists with the corpus.",
}
ADDED_FIELDS = [
    "bundle",
    "member",
    "bytes",
    "receipt_check",
    "terms",
    "terms_sources",
    "notices",
]
CREDIT_FIELDS = ["member", "title", "author", "sourceURL", "sourcePublication"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    if (
        not name
        or any(p in {"", ".", ".."} for p in name.split("/"))
        or "\\" in name
        or any(ord(c) < 32 for c in name)
    ):
        raise ValueError(f"Invalid relative path: {name!r}")
    return PurePosixPath(name).as_posix()


def csv_bytes(rows, fields):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def read_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode(), newline="")))


def select(row, reviewed):
    for rule in reviewed["rules"]:
        if row["id"] not in rule.get("ids", [row["id"]]):
            continue
        if not any(fnmatchcase(row["subset"], s) for s in rule.get("subsets", ["*"])):
            continue
        if any(fnmatchcase(row["path"], p) for p in rule.get("paths", ["*"])):
            return rule
    raise ValueError(f"Missing review for {row['repository']} {row['path']}")


def pinned_url(repo, rev, path):
    if len(safe_path(repo).split("/")) != 2 or not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ValueError(f"Invalid repository or revision: {repo}/{rev}")
    return f"https://raw.githubusercontent.com/{repo}/{rev}/{quote(safe_path(path), safe='/')}"


def checked_data(path, url, digest, fetch):
    if path.is_file():
        raw = path.read_bytes()
    elif fetch:
        with urlopen(url, timeout=90) as response:
            raw = response.read()
    else:
        raise FileNotFoundError(path)
    if sha(raw) != digest:
        raise ValueError(f"SHA-256 mismatch for {url}")
    if fetch and not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(raw)
        Path(tmp.name).replace(path)
    return raw


def write_member(archive, name, data):
    info = ZipInfo(safe_path(name), (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=6)


def gum_credit(raw, member):
    header = raw.split(b"# sent_id", 1)[0].decode()
    credit = {
        "member": member,
        **dict(
            re.findall(
                r"^# meta::(author|title|sourceURL|sourcePublication) = (.+)$",
                header,
                re.M,
            )
        ),
    }
    if not all(credit.get(k) for k in ("author", "title", "sourceURL")):
        raise ValueError(f"Incomplete source credits for {member}")
    return credit


def notice_location(repo, revision, path):
    return f"notices/{repo}/{revision}/{safe_path(path)}"


def documentation(bundle, rows):
    return f"""# Anno publisher data — {bundle}

{BUNDLES[bundle]}

This ZIP contains {len(rows)} original publisher files for {len({r['id'] for r in rows})} Anno registry IDs, with the repository, exact revision and original directory structure under data/. No source bytes were edited. The split, subset, source URL, original SHA-256, measurement unit and status are in manifest.csv. Some IDs contribute only selected source files. See publisher-coverage.csv in the distribution directory for every inclusion, exclusion, alias and unavailable route in the survey. These are publisher-file measurements, not model scores or S3 objects.

## Read and verify offline

Unzip this archive and run `python3 verify.py verify . --replay` in the extracted directory (Python 3.9 or later, standard library). Each expected measurement is recomputed from the original bytes with Anno's publisher-file reader. A known parser failure is checked against its recorded reason. You can also verify a ZIP directly with `python3 verify.py verify path/to/another.zip --replay`. Omit `--replay` for hash verification alone. SHA256SUMS covers data, manifests, tools and notices. The distribution's publisher-packages.json gives each entire ZIP's SHA-256 and MD5.

CSVs are UTF-8. Open a data/ member with a local text or JSON reader using its original format. rows and tokens describe that file's `metric`. A blank tagged field means that the POS or dependency unit has no NER denominator. Masked NoiseBench token positions, wherever present, contain no original words. GUM documents have author, title and source URL in their CoNLL-U headers and credits.csv. The scripts under this archive root and their Anno licenses do not govern the data.

## Attribution and terms

Read the terms and notices columns for each file. `notices` contains a JSON array of local paths. notices.csv links those notices to their pinned publishers and records their individual hashes. Preserve the original authors, copyright statements, source credits and license links if sharing permitted material. The original publisher licenses govern your use of each work. No bundle-wide dataset license is granted by Anno.
""".encode()


def build(args):
    ledger = args.manifest.read_bytes()
    config = json.loads(args.terms.read_bytes())
    if sha(ledger) != config["ledger_sha256"]:
        raise ValueError(
            "The publisher ledger changed. Review the new release and update the terms snapshot before packaging."
        )
    rows = read_rows(ledger)
    fields = list(rows[0]) + ADDED_FIELDS
    forbidden = {
        line.split("\t", 1)[0].removesuffix(".cache")
        for line in args.exclusions.read_text().splitlines()
        if line
    }
    config = config["repositories"]
    by_bundle = defaultdict(list)
    credits = defaultdict(list)
    notices = defaultdict(dict)
    hashes = defaultdict(dict)
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as tmp:
        staging = Path(tmp)
        with ExitStack() as stack:
            archives = {
                k: stack.enter_context(ZipFile(staging / f"publisher-{k}.zip", "w"))
                for k in BUNDLES
            }

            def add(bundle, name, data):
                if name in hashes[bundle]:
                    raise ValueError(f"Duplicate archive member: {name}")
                write_member(archives[bundle], name, data)
                hashes[bundle][name] = sha(data)

            for row in rows:
                repo, rev, path = row["repository"], row["revision"], row["path"]
                row.update(dict.fromkeys(ADDED_FIELDS, ""))
                if not row["url"]:
                    row.update(receipt_check="no file", terms=row["reason"])
                    continue
                expected = pinned_url(repo, rev, path)
                media = expected.replace(
                    "raw.githubusercontent.com/",
                    "media.githubusercontent.com/media/",
                    1,
                )
                if unquote(row["url"]) not in {unquote(expected), unquote(media)}:
                    raise ValueError(f"Unpinned source: {row['url']}")
                policy = config[repo]
                if rev != policy["revision"]:
                    raise ValueError(f"Unreviewed revision: {repo}/{rev}")
                rule = select(row, policy)
                bundle = rule["bundle"]
                if bundle and row["id"] in forbidden:
                    raise ValueError(f"Source acquisition constraint: {row['id']}")
                row.update(terms=rule["terms"], bundle=bundle)
                review_path = policy.get("subset_review_notices", {}).get(
                    row["subset"], policy["review_notice"]
                )
                review_paths = rule.get(
                    "notices", rule.get("review_notices", [review_path])
                )
                row["terms_sources"] = json.dumps(
                    [
                        policy.get("supplement", {}).get(n, pinned_url(repo, rev, n))
                        for n in review_paths
                    ]
                    + rule.get("references", []),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                cache = args.cache_dir / sha(row["url"].encode())
                if not bundle and not cache.is_file():
                    row["receipt_check"] = "not fetched (excluded)"
                    continue
                raw = checked_data(
                    cache,
                    row["url"],
                    row["sha256"],
                    args.fetch_missing and bool(bundle),
                )
                row.update(bytes=str(len(raw)), receipt_check="SHA-256 verified")
                if not bundle:
                    continue
                member = f"data/{repo}/{rev}/{path}"
                if "require_mask" in rule:
                    marker = rule["require_mask"].encode()
                    if any(
                        line.split()[0] != marker
                        for line in raw.splitlines()
                        if line.strip()
                    ):
                        raise ValueError(f"Unmasked source content in {member}")
                add(bundle, member, raw)
                row["member"] = member
                if row["id"] == "GUM":
                    credits[bundle].append(gum_credit(raw, member))
                local_notices = []
                for n in rule["notices"]:
                    dest = notice_location(repo, rev, n)
                    local_notices.append(dest)
                    if dest in notices[bundle]:
                        continue
                    url = policy.get("supplement", {}).get(n, pinned_url(repo, rev, n))
                    if not re.fullmatch(
                        r"https://raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/.+",
                        url,
                    ):
                        raise ValueError(f"Unpinned source notice: {url}")
                    digest = policy["notice_sha256"][n]
                    data = checked_data(
                        args.notice_cache / repo / rev / "files" / n,
                        url,
                        digest,
                        args.fetch_missing,
                    )
                    add(bundle, dest, data)
                    notices[bundle][dest] = dict(member=dest, url=url, sha256=digest)
                row["notices"] = json.dumps(
                    local_notices, ensure_ascii=False, separators=(",", ":")
                )
                by_bundle[bundle].append(row)
            for bundle, archive in archives.items():
                add(bundle, "README.md", documentation(bundle, by_bundle[bundle]))
                add(bundle, "manifest.csv", csv_bytes(by_bundle[bundle], fields))
                add(bundle, "credits.csv", csv_bytes(credits[bundle], CREDIT_FIELDS))
                add(
                    bundle,
                    "notices.csv",
                    csv_bytes(
                        sorted(notices[bundle].values(), key=lambda n: n["member"]),
                        ["member", "url", "sha256"],
                    ),
                )
                add(bundle, "verify.py", SCRIPT.read_bytes())
                add(
                    bundle,
                    "survey_repo_files.py",
                    (SCRIPT.parent / "survey_repo_files.py").read_bytes(),
                )
                for license in ["LICENSE-MIT", "LICENSE-APACHE"]:
                    add(
                        bundle,
                        f"tools-licenses/{license}",
                        (ROOT / license).read_bytes(),
                    )
                write_member(
                    archive,
                    "SHA256SUMS",
                    "".join(
                        f"{h}  {n}\n" for n, h in sorted(hashes[bundle].items())
                    ).encode(),
                )
        (staging / "publisher-coverage.csv").write_bytes(csv_bytes(rows, fields))
        packages = {}
        for bundle, included in sorted(by_bundle.items()):
            path = staging / f"publisher-{bundle}.zip"
            if not included:
                path.unlink()
                continue
            raw = path.read_bytes()
            packages[path.name] = dict(
                bytes=len(raw),
                sha256=sha(raw),
                md5=hashlib.md5(raw, usedforsecurity=False).hexdigest(),
                source_bytes=sum(int(r["bytes"]) for r in included),
                files=len(included),
                dataset_ids=sorted({r["id"] for r in included}),
                measurement_status=dict(
                    sorted(Counter(r["status"] for r in included).items())
                ),
            )
        report = dict(
            ledger_sha256=sha(ledger),
            ledger_rows=len(rows),
            named_ids=len({r["id"] for r in rows}),
            file_receipts=sum(bool(r["url"]) for r in rows),
            verified_file_receipts=sum(
                r["receipt_check"] == "SHA-256 verified" for r in rows
            ),
            included_files=sum(len(v) for v in by_bundle.values()),
            included_ids=len({r["id"] for r in rows if r["member"]}),
            packages=packages,
        )
        (staging / "publisher-packages.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        table = "\n".join(
            f"| {name} | {p['files']} | {len(p['dataset_ids'])} | {p['source_bytes']:,} |"
            for name, p in packages.items()
        )
        (staging / "README.md").write_text(
            f"""# Pinned publisher datasets for offline Anno work

The original survey covers {report['ledger_rows']} rows for {report['named_ids']} named registry IDs. There are {report['file_receipts']} pinned file checksums. This build verified {report['verified_file_receipts']} from source bytes and contains {report['included_files']} unchanged files for {report['included_ids']} distinct IDs. Some IDs contribute to more than one bundle. The full coverage CSV gives each source path, split, subset, original URL and hash, packing decision and reason. It includes input with unconfirmed redistribution terms, the aliases and the unavailable source.

| Archive | Source files | Named IDs in archive | Original bytes |
| --- | ---: | ---: | ---: |
{table}

Unzip any archive into its own directory and run `python3 verify.py verify . --replay` there. This hashes all packaged data and notices and repeats every in-scope survey measurement (or known parser error) offline with Python 3.9+. The data tree preserves publisher repository, commit, and relative file path. Read the individual notices before use. The noncommercial archive contains releases with noncommercial conditions; the annotated-collection archive carries a separate restriction on republishing individual source works. The general archive includes attribution and share-alike requirements where applicable.

Whole-package SHA-256, MD5, bytes, original-file counts and registry IDs are in publisher-packages.json. The repository generator is `scripts/pack_publisher_datasets.py`. Given the same receipts, policy and generator, archive member content, dates, permissions, ordering and compression settings are deterministic. The coverage table records additional excluded hashes checked if their files were present locally. This collection covers the publisher Git file survey only. S3 and model-evaluation evidence are separate sources.
""",
            encoding="utf-8",
        )
        for path in sorted(staging.iterdir()):
            path.replace(args.output / path.name)
    print(json.dumps(report, indent=2))


def verify(path, replay):
    with ExitStack() as stack:
        if path.is_dir():

            def read(name):
                return (path / safe_path(name)).read_bytes()

        else:
            z = stack.enter_context(ZipFile(path))
            read = z.read
        expected = dict(
            line.split("  ", 1)[::-1]
            for line in read("SHA256SUMS").decode().splitlines()
        )
        for name, digest in expected.items():
            safe_path(name)
            if sha(read(name)) != digest:
                raise ValueError(f"Archive SHA-256 mismatch: {name}")
        if path.is_file() and sorted(z.namelist()) != sorted([*expected, "SHA256SUMS"]):
            raise ValueError("Unexpected or duplicate member in archive")
        rows = read_rows(read("manifest.csv"))
        notices = {n["member"]: n for n in read_rows(read("notices.csv"))}
        counts = Counter()
        for row in rows:
            name = row["member"]
            if expected.get(name) != row["sha256"]:
                raise ValueError(f"Source ledger mismatch: {name}")
            if not all(
                n in notices and expected.get(n) == notices[n]["sha256"]
                for n in json.loads(row["notices"])
            ):
                raise ValueError(f"Notice mismatch: {name}")
            if not replay:
                continue
            raw = read(safe_path(name))
            if len(raw) != int(row["bytes"]):
                raise ValueError(f"Source size mismatch: {name}")
            try:
                observed = tuple(str(n) for n in measure(raw, row))
            except (ValueError, KeyError) as err:
                if row["status"] != "error" or row["reason"] != str(err):
                    raise ValueError(
                        f"Unexpected parser failure: {name}: {err}"
                    ) from err
                counts["recorded parser failures matched"] += 1
            else:
                if row["status"] != "ok" or observed != tuple(
                    row[k] for k in ["rows", "tokens", "tagged"]
                ):
                    raise ValueError(f"Measurement mismatch: {name}")
                counts["measurements reproduced"] += 1
    print(
        f"{len(rows)} source files and {len(notices)} notices verified. "
        + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build separately labelled, deterministic ZIPs")
    for option, default in [
        ("manifest", LEDGER),
        ("terms", TERMS),
        ("exclusions", EXCLUSIONS),
    ]:
        b.add_argument("--" + option, type=Path, default=default)
    for option in ["cache-dir", "notice-cache", "output"]:
        b.add_argument("--" + option, required=True, type=Path)
    b.add_argument(
        "--fetch-missing",
        action="store_true",
        help="download only selected, pinned source data and missing notices",
    )
    v = sub.add_parser(
        "verify",
        help="hash and optionally replay a ZIP or extracted directory, offline",
    )
    v.add_argument("path", type=Path)
    v.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        verify(args.path, args.replay)


if __name__ == "__main__":
    main()
