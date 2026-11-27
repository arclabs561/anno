# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3==1.44.12", "botocore==1.44.12"]
# ///
"""Compare live registered S3 payload bytes with measured publisher file hashes.

Run in the trusted cache-source-evidence Actions workflow. The only persisted
read results are aggregate counts and matches to already public ledger rows.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
SHA = re.compile(r"[0-9a-f]{64}\Z")
CACHE = re.compile(
    r"datasets/(?:(?:by-sha256/([0-9a-f]{64})/))?([A-Za-z][A-Za-z0-9_]*)\.cache(\.latest\.json|\.manifest\.json)?\Z"
)
# These publications include offsets, skeletons, or individually licensed text subsets.
# Their measured public files can identify bytes elsewhere, but cannot authorize reading
# an entire, separately prepared cache under the registry name.
PARTIAL_PUBLIC_SOURCES = {"DISRPT2025", "GUM", "MATRES", "TempEval3", "TweetNERD", "TwiConv"}
FIELDS = (
    "cache_id",
    "object_kind",
    "snapshot_status",
    "match",
    "publisher_id",
    "sha256",
    "payload_bytes",
    "ledger",
    "source",
    "publisher_path",
    "split_or_subset",
    "unit",
    "records",
    "tokens",
    "tagged",
)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def catalog(root):
    ledgers = root / "docs/eval"
    registry = {r["id"]: r for r in csv_rows(ledgers / "dataset-coverage.csv")}
    rules = root / "crates/anno-eval/src/eval/shared_cache_exclusions.tsv"
    definitions = (
        rules.with_name("dataset_registry.rs").read_text().split("\ndefine_datasets! {", 1)[1]
    )
    # Some public-facing registry URLs describe a gated release or an annotation sample.
    controlled = {"DELICATE"}  # no verified public clinical source
    for ident, entry in re.findall(r"^    (\w+) \{\n(.*?)^    \},?", definitions, re.M | re.S):
        if re.search(
            r'^        license: "(?:LDC(?: \+ Research)?|PhysioNet|DUA Required)"', entry, re.M
        ):
            controlled.add(ident)
    normalized = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    blocked = set()
    for line in rules.read_text().splitlines():
        name, _, aliases = line.split("\t")
        blocked.update(normalized(s) for s in [name.removesuffix(".cache"), *aliases.split(",")])
    reasons = {}
    for ident, row in registry.items():
        if normalized(ident) in blocked or normalized(row["name"]) in blocked:
            reasons[ident] = "shared_read_exclusion"
        elif (
            ident in controlled
            or row["access"] not in {"Public", "HuggingFace"}
            or not row["url"].startswith(("https://", "http://"))
        ):
            reasons[ident] = "requires_separate_source_access"
        elif ident in PARTIAL_PUBLIC_SOURCES:
            reasons[ident] = "public_release_covers_only_part_of_source"
        elif row["measured"] != "1":
            reasons[ident] = "no_independent_corpus_measurement"
        else:
            reasons[ident] = "eligible"

    refs = defaultdict(list)
    for ledger, rows in (
        ("dataset-direct-files.csv", csv_rows(ledgers / "dataset-direct-files.csv")),
        ("dataset-repo-files.csv", csv_rows(ledgers / "dataset-repo-files.csv")),
        (
            "dataset-github-sources.csv",
            csv_rows(ledgers / "dataset-github-sources.csv"),
        ),
        ("dataset-hf-splits.csv", csv_rows(ledgers / "dataset-hf-splits.csv")),
    ):
        for row in rows:
            if reasons[row["id"]] not in {"eligible", "public_release_covers_only_part_of_source"}:
                continue
            digest = row.get("sha256", row.get("payload_sha256", ""))
            if not SHA.fullmatch(digest):
                continue
            if ledger == "dataset-direct-files.csv":
                if row["status"] not in {"records measured", "tagged tokens measured"}:
                    continue
                source, subset, unit, records = (
                    row[k] for k in ("source", "split_or_subset", "unit", "records")
                )
            elif ledger == "dataset-repo-files.csv":
                if row["status"] != "ok":
                    continue
                source, subset, unit, records = (
                    row["url"],
                    "/".join(filter(None, (row["subset"], row["split"]))),
                    row["metric"],
                    row["rows"],
                )
            elif ledger == "dataset-github-sources.csv":
                if (
                    row["status"] != "ok"
                    or row["metric"] != "count"
                    or row["unit"] not in registry[row["id"]]["github_units"].split(" | ")
                ):
                    continue
                source, subset, unit, records = (
                    row["source_url"],
                    row["split"],
                    row["unit"],
                    row["value"],
                )
            else:
                if "/tree/" in urlsplit(row["payload_url"]).path or row["status"] not in {
                    "scanned",
                    "scanned_unlabelled",
                    "scanned_records",
                }:
                    continue
                source, subset, unit, records = (
                    row["payload_url"],
                    f"{row['config']}/{row['split']}",
                    "rows",
                    row["rows"],
                )
            if source.startswith(("https://", "http://")) and (
                records.isdigit() or row.get("tokens", "").isdigit()
            ):
                refs[digest].append(
                    dict(
                        publisher_id=row["id"],
                        ledger=ledger,
                        source=source,
                        publisher_path=row.get("path", ""),
                        split_or_subset=subset,
                        unit=unit,
                        records=records,
                        tokens=row.get("tokens", ""),
                        tagged=row.get("tagged", ""),
                    )
                )
    return reasons, refs


@dataclass(frozen=True)
class Object:
    key: str
    size: int
    etag: str
    dataset: str
    kind: str
    snapshot: str


def classify(row, reasons):
    match = CACHE.fullmatch(row["Key"])
    if not match:
        return None
    sha, dataset, sidecar = match.groups()
    if dataset not in reasons or (sha and sidecar):
        return None
    kind = (
        ("pointer" if sidecar == ".latest.json" else "manifest")
        if sidecar
        else ("snapshot" if sha else "legacy")
    )
    return Object(row["Key"], row["Size"], row["ETag"], dataset, kind, sha or "")


def policy_for(bucket, keys):
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "s3:GetObject",
                    "Resource": [f"arn:aws:s3:::{bucket}/{key}" for key in keys],
                }
            ],
        },
        separators=(",", ":"),
    )


def batches(bucket, entries):
    group = []
    for obj in entries:
        if len(policy_for(bucket, [item.key for item in (*group, obj)])) > 2048:
            if not group:
                raise ValueError("A listed public ID does not fit an exact-key session policy")
            yield group
            group = []
        group.append(obj)
    if group:
        yield group


def object_value(obj, chunks):
    size = 0
    digest = hashlib.sha256()
    pointer = bytearray()
    for chunk in chunks:
        size += len(chunk)
        if obj.kind == "pointer":
            if size > 65536:
                return "invalid_pointer", None
            pointer.extend(chunk)
        else:
            digest.update(chunk)
    if size != obj.size:
        return "size_changed", None
    if obj.kind == "pointer":
        try:
            parsed = json.loads(pointer)
        except (UnicodeDecodeError, ValueError):
            return "invalid_pointer", None
        if (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("sha256"), str)
            or not SHA.fullmatch(parsed["sha256"])
            or parsed.get("dataset", obj.dataset + ".cache") != obj.dataset + ".cache"
        ):
            return "invalid_pointer", None
        return "read", parsed["sha256"]
    actual = digest.hexdigest()
    if obj.kind == "snapshot" and obj.snapshot != actual:
        return "snapshot_digest_mismatch", None
    return "read", actual


def audit(reasons, refs, listing, read_batches):
    # The raw listing, unknown digests and pointers live only in memory.
    inventory = Counter()
    states = Counter()
    reported = []
    counts_by_id = defaultdict(Counter)
    known = [classify(item, reasons) for item in listing]
    keys = [obj.key for obj in known if obj]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate registered S3 listing entry")
    payload_ids = {obj.dataset for obj in known if obj and obj.kind in {"legacy", "snapshot"}}
    snapshot_ids = {obj.dataset for obj in known if obj and obj.kind == "snapshot"}
    wanted = []
    for obj in known:
        inventory[obj.kind if obj else "unrecognized"] += 1
        if not obj or obj.kind == "manifest":
            continue
        if obj.kind == "pointer":
            if reasons[obj.dataset] == "eligible" and obj.dataset in snapshot_ids:
                wanted.append(obj)
        elif reasons[obj.dataset] == "eligible":
            wanted.append(obj)
        else:
            states[reasons[obj.dataset]] += 1
    for ident, reason in reasons.items():
        counts_by_id[reason]["with_payload" if ident in payload_ids else "without_payload"] += 1

    source_ids = {ref["publisher_id"] for filenames in refs.values() for ref in filenames}
    values = dict(read_batches(wanted))
    if set(values) != set(wanted):
        raise ValueError("Incomplete scoped S3 read pass")
    pointers = {o.dataset: v for o, (s, v) in values.items() if o.kind == "pointer" and s == "read"}
    listed_snapshots = defaultdict(set)
    for obj in wanted:
        if obj.kind == "snapshot":
            listed_snapshots[obj.dataset].add(obj.snapshot)
    positions = Counter()
    pointer_states = Counter()
    by_digest = defaultdict(list)
    published_ids = set()
    own_match_ids = set()
    counted_digests = set()
    for obj, (status, digest) in values.items():
        if obj.kind == "pointer":
            pointer_states[status] += 1
            continue
        position = "legacy"
        if obj.kind == "snapshot":
            target = pointers.get(obj.dataset)
            # A dangling pointer cannot establish which listed object a loader would read.
            position = (
                "unresolved"
                if target not in listed_snapshots[obj.dataset]
                else "current" if target == obj.snapshot else "superseded"
            )
            positions[position] += 1
        if status != "read":
            states[status] += 1
            continue
        by_digest[digest].append(obj)
        found = refs.get(digest, [])
        if not found:
            states[
                (
                    "no_matching_publisher_digest"
                    if obj.dataset in source_ids
                    else "no_own_publisher_digest_and_no_cross_id_match"
                )
            ] += 1
            continue
        same_id = any(ref["publisher_id"] == obj.dataset for ref in found)
        states[
            ("matches_own_publisher_file" if same_id else "matches_other_registry_file_only")
        ] += 1
        published_ids.add(obj.dataset)
        if same_id:
            own_match_ids.add(obj.dataset)
        counted_digests.add(digest)
        for ref in found:
            reported.append(
                dict(
                    cache_id=obj.dataset,
                    object_kind=obj.kind,
                    snapshot_status=position,
                    match=(
                        "same registry ID"
                        if ref["publisher_id"] == obj.dataset
                        else "different registry ID"
                    ),
                    sha256=digest,
                    payload_bytes=obj.size,
                    **ref,
                )
            )
    if sum(states.values()) != inventory["snapshot"] + inventory["legacy"]:
        raise AssertionError("Missing payload outcome")
    return {
        "registry_entries": len(reasons),
        "objects_listed": len(listing),
        "objects_by_kind": dict(inventory),
        "registry_ids_with_payload": len(payload_ids),
        "registry_ids_by_eligibility": dict(counts_by_id),
        "payload_objects_by_outcome": dict(states),
        "read_pointer_outcomes": dict(pointer_states),
        "eligible_snapshot_positions": dict(positions),
        "publisher_digests_compared": len(refs),
        "registry_ids_with_measured_publisher_digests": len(source_ids),
        "eligible_listed_ids_without_own_publisher_digest": sum(
            reasons[i] == "eligible" and i not in source_ids for i in payload_ids
        ),
        "registry_ids_with_any_publisher_match": len(published_ids),
        "registry_ids_with_own_publisher_match": len(own_match_ids),
        "distinct_matched_publisher_digests": len(counted_digests),
        "distinct_payload_digests_read": len(by_digest),
        "digests_shared_by_multiple_objects": sum(len(items) > 1 for items in by_digest.values()),
        "digests_shared_by_multiple_cache_ids": sum(
            len({o.dataset for o in items}) > 1 for items in by_digest.values()
        ),
    }, sorted(reported, key=lambda r: tuple(str(r[f]) for f in FIELDS))


def scoped_client(session, role, bucket, group):
    from botocore import UNSIGNED
    from botocore.config import Config

    # Request a fresh, short-lived GitHub OIDC token for each narrowly scoped STS session.
    url = urlsplit(os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"])
    query = dict(parse_qsl(url.query), audience="sts.amazonaws.com")
    request = Request(
        urlunsplit(url._replace(query=urlencode(query))),
        headers={"Authorization": "Bearer " + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]},
    )
    with urlopen(request, timeout=20) as response:
        token = json.load(response)["value"]
    sts = session.client("sts", config=Config(signature_version=UNSIGNED))
    credentials = sts.assume_role_with_web_identity(
        RoleArn=role,
        RoleSessionName="anno-cache-source-evidence",
        WebIdentityToken=token,
        Policy=policy_for(bucket, [o.key for o in group]),
        DurationSeconds=3600,
    )["Credentials"]
    return session.client(
        "s3",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def run(output, bucket, role):
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise ValueError("Invalid bucket for the scoped resource policy")
    reasons, refs = catalog(ROOT)
    session = boto3.Session()
    listed_at = datetime.now(timezone.utc).isoformat()
    pages = (
        session.client("s3")
        .get_paginator("list_objects_v2")
        .paginate(Bucket=bucket, Prefix="datasets/")
    )
    listing = [obj for page in pages for obj in page.get("Contents", [])]

    def read_entries(objects):
        for group in batches(bucket, objects):
            try:
                client = scoped_client(session, role, bucket, group)
            except (BotoCoreError, ClientError, URLError, TimeoutError):
                for obj in group:
                    yield obj, ("read_session_unavailable", None)
                continue
            for obj in group:
                try:
                    response = client.get_object(Bucket=bucket, Key=obj.key, IfMatch=obj.etag)
                    with closing(response["Body"]) as stream:
                        value = object_value(obj, stream.iter_chunks(chunk_size=1 << 20))
                except ClientError as exc:
                    http = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                    value = (
                        (
                            "access_denied"
                            if http == 403
                            else (
                                "changed_after_listing" if http in {404, 412} else "s3_read_error"
                            )
                        ),
                        None,
                    )
                except BotoCoreError:
                    value = ("s3_read_error", None)
                yield obj, value

    summary, rows = audit(reasons, refs, listing, read_entries)
    summary["listing_started_utc"] = listed_at
    summary["repository_commit"] = os.environ.get("GITHUB_SHA", "")
    summary["workflow_run_id"] = os.environ.get("GITHUB_RUN_ID", "")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "publisher-file-matches.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "cache-source-coverage.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    errors = {
        "access_denied",
        "changed_after_listing",
        "s3_read_error",
        "read_session_unavailable",
        "size_changed",
        "snapshot_digest_mismatch",
        "invalid_pointer",
    }
    if errors.intersection(
        set(summary["payload_objects_by_outcome"]) | set(summary["read_pointer_outcomes"])
    ):
        raise SystemExit("Incomplete S3 verification. See the aggregate outcomes in the artifact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    run(args.output, args.bucket, args.role)
