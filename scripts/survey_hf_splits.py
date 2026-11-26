"""Survey HF URLs in an Anno inventory CSV. Usage: python scripts/survey_hf_splits.py INVENTORY OUTPUT.

Requires pyarrow. Uses anonymous HF Hub and Dataset Viewer APIs. Provider rows
come from split metadata. Rows and tokens are counted from complete pinned
Parquet partitions or named original JSONL files. Record-only scans validate
a released observation field and leave token and tag counts empty. Configurations and splits can overlap. No
example text is written to the ledger. Scan one original linked file where the
registry specifies a file. Fresh surveys follow current provider metadata.
To verify previously scanned rows at their exact revisions, run:
    python scripts/survey_hf_splits.py SNAPSHOT.csv REPLAY.csv --replay
The replay covers measured splits only and compares every count to the ledger.

"""

import argparse
import ast
import csv
import hashlib
import json
import io
import re
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HUB = "https://huggingface.co"
VIEWER = "https://datasets-server.huggingface.co"
POLICY = (
    Path(__file__).resolve().parents[1]
    / "crates/anno-eval/src/eval/shared_cache_exclusions.tsv"
)
FIELDS = (
    "id",
    "repository",
    "source_revision",
    "parquet_revision",
    "config",
    "split",
    "status",
    "provider_rows",
    "provider_tokens",
    "provider_tagged",
    "rows",
    "tokens",
    "tagged",
    "reason",
    "scope_note",
    "observation_column",
    "payload_url",
    "payload_sha256",
)
EXCLUDED = {
    "BioMNER": "Registry describes method mentions; linked source is BioNLP 2004 molecular NER",
    "BC4CHEMD": "Linked mirror declares a multi-entity tag set; chemical-only corpus identity unverified",
    "OntoNotesSample": "Registry describes a sample; licensed full-corpus partition scope unverified",
}


# One observed release field per record; never substitute a nullable target
# label for the input. Dotted names descend into a Parquet struct.
RECORD_COLUMNS = {
    "AGNews": "text",
    "AfriQA": "question",
    "AfriSenti": "tweet",
    "CADEC": "text",
    "DBPedia14": "content",
    "MasakhaNEWS": "headline",
    "OpenNER": "text",
    "REBEL": "context",
    "RealToxicityPrompts": "prompt.text",
    "TREC": "text",
    "TweetTopic": "text",
    "ViraDialogActsLive": "text",
    "YahooAnswers": "question_title",
    "DisrptDeuRstPccConlluSeg": "form",
    "DisrptDeuRstPccRels": "unit1_txt",
    "DisrptEngDepScidtbConlluSeg": "form",
    "DisrptEngDepScidtbRels": "unit1_txt",
}
SCANNED_STATUSES = {"scanned", "scanned_unlabelled", "scanned_records"}


@lru_cache(maxsize=None)
def get_json(url, expected_revision=None):
    with urlopen(url, timeout=45) as response:
        if (
            expected_revision
            and response.headers.get("x-revision") != expected_revision
        ):
            raise ValueError("Viewer source revision differs from Hub source revision")
        return json.load(response)


def viewer(repo, endpoint, revision):
    result = get_json(f"{VIEWER}/{endpoint}?" + urlencode({"dataset": repo}), revision)
    if result.get("partial") or result.get("pending") or result.get("failed"):
        raise ValueError(f"Incomplete provider {endpoint} response")
    return result


def outcome(row, status, reason):
    return dict(row, status=status, reason=reason)


def targets(inventory):
    for row in inventory:
        url = urlsplit(row["url"])
        if url.hostname == "huggingface.co":
            path = url.path.strip("/").split("/")
            if path[0] == "datasets":
                yield row, "/".join(path[1:3]), {}
            else:
                yield row, "", {}
        elif url.hostname == "datasets-server.huggingface.co":
            query = parse_qs(url.query)
            yield row, query["dataset"][0], {
                k: query[k][0] for k in ("config", "split") if k in query
            }


def card_feature(value):
    if isinstance(value, str):
        return {"dtype": value}
    if "sequence" in value:
        return {"feature": card_feature(value["sequence"])}
    if "class_label" in value:
        names = value["class_label"]["names"]
        if isinstance(names, dict):
            names = [names[str(i)] for i in range(len(names))]
        return {"names": names}
    return {"dtype": value.get("dtype")}


def uner_jobs(row, meta):
    # Read the literal file map in the pinned HF loader without running it.
    with urlopen(
        repo_file(row["repository"], meta["sha"], "universal_ner.py"), timeout=45
    ) as response:
        module = ast.parse(response.read())
    manifests = [
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_UNER_DATASETS" for t in node.targets
        )
    ]
    if len(manifests) != 1:
        raise ValueError("Expected one UniversalNER publisher file manifest")
    paths = ast.literal_eval(manifests[0])
    ready, jobs = [], []
    for config, details in card_configs(meta).items():
        for split, info in details["splits"].items():
            current = dict(
                row, config=config, split=split, provider_rows=info.get("num_examples")
            )
            try:
                path = paths[config]["dev" if split == "validation" else split]
                repo, branch, filename = path.split("/")
                if not re.fullmatch(r"UNER_[A-Za-z0-9_-]+", repo) or not re.fullmatch(
                    r"[\w.-]+\.iob2", filename
                ):
                    raise ValueError("Unexpected publisher dataset path")
                # Source branch names are mutable. Use its state at the pinned
                # HF snapshot date and save the actual immutable release URL.
                query = urlencode(
                    {"sha": branch, "until": meta["lastModified"], "per_page": 1}
                )
                commits = get_json(
                    f"https://api.github.com/repos/UniversalNER/{repo}/commits?{query}"
                )
                if not commits:
                    raise ValueError("No publisher revision at the Hub snapshot date")
                sha = commits[0]["sha"]
                url = f"https://raw.githubusercontent.com/UniversalNER/{repo}/{sha}/{filename}"
                current.update(payload_url=url)
                jobs.append((current, {"iob2": url}))
            except (OSError, ValueError, KeyError) as exc:
                ready.append(
                    outcome(current, "provider_only", f"Publisher source: {exc}")
                )
    return ready, jobs


def card_configs(meta):
    cards = (meta.get("cardData") or {}).get("dataset_info", [])
    if isinstance(cards, dict):
        cards = [cards]
    return {
        card.get("config_name", "default"): {
            "splits": {s["name"]: s for s in card.get("splits", [])},
            "features": {
                feature["name"]: card_feature(feature)
                for feature in card.get("features", [])
            },
        }
        for card in cards
    }


def repo_file(repo, rev, path):
    return f"{HUB}/datasets/{repo}/resolve/{rev}/{quote(path, safe='/')}"


def publisher_labels(repo, rev):
    if repo.startswith("tner/"):
        mapping = get_json(repo_file(repo, rev, "dataset/label.json"))
    elif repo in {"Babelscape/wikineural", "Babelscape/multinerd"}:
        with urlopen(repo_file(repo, rev, "README.md"), timeout=45) as response:
            card = response.read().decode()
        match = re.search(r"""\{\s*["']O["']\s*:\s*0\s*,[^{}]+\}""", card, re.S)
        if not match:
            return None
        mapping = ast.literal_eval(match[0])
    else:
        return None
    if not isinstance(mapping, dict) or "O" not in mapping:
        return None
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("Source label map is not a contiguous bijection")
    return [k for k, _ in sorted(mapping.items(), key=lambda item: item[1])]


publisher_labels = lru_cache(maxsize=None)(publisher_labels)


def token_columns(features):
    for token, label in (
        ("tokens", "ner_tags"),
        ("tokens", "tags"),
        ("mbert_text_tokens", "mbert_bio_labels"),
        ("words", "ner"),
    ):
        if token in features and label in features:
            return token, label
    return None


def scalar_feature(feature):
    if isinstance(feature, list):
        return feature[0]
    return feature.get("feature", {})


def pinned_shards(repo, revision, config, split):
    metadata = get_json(f"{HUB}/api/datasets/{repo}/revision/{revision}?blobs=true")
    sha = metadata["sha"]
    if re.fullmatch("[0-9a-f]{40}", revision) and sha != revision:
        raise ValueError("Hub returned a different conversion revision")
    # Exact directory only. Viewer previews use partial-<split>/, and
    # datasets may provide multiple encodings or configs of the same data.
    prefix = f"{config}/{split}/"
    files = [
        item
        for item in metadata["siblings"]
        if item["rfilename"].startswith(prefix)
        and re.fullmatch(r"[^/]+\.parquet", item["rfilename"][len(prefix) :])
    ]
    return sha, [(repo_file(repo, sha, f["rfilename"]), f["size"]) for f in files]


def record_plan(row):
    column = RECORD_COLUMNS.get(row["id"])
    if not column:
        return None, (
            "Provider rows wrap binary files; no corpus record unit verified"
            if row["id"] == "MewsliX"
            else "No verified observation column for this schema"
        )
    if row.get("provider_rows") is None:
        return None, "No provider total for split completeness check"
    try:
        revision, shards = pinned_shards(
            row["repository"], "refs%2Fconvert%2Fparquet", row["config"], row["split"]
        )
    except (OSError, ValueError, KeyError) as exc:
        return None, f"Pinned conversion unavailable: {exc}"
    if not shards:
        return None, "No complete pinned Parquet partition"
    row.update(parquet_revision=revision, observation_column=column)
    return (row, {"shards": shards, "observation_column": column}), ""


def plan(target):
    entry, repo, scope, local_only = target
    row = {"id": entry["id"], "repository": repo or entry["url"]}
    if not repo:
        return [
            outcome(row, "excluded", "Hub link identifies a model or organization")
        ], []
    if entry["id"] in local_only:
        return [
            outcome(row, "excluded", "Registry requires an authorized local source")
        ], []
    if entry["access"] in {"Local", "Registration", "ContactAuthors", "Licensed"}:
        return [outcome(row, "excluded", f"Registry access {entry['access']}")], []
    try:
        meta = get_json(f"{HUB}/api/datasets/{repo}")
        repo = meta["id"]
        row.update(repository=repo, source_revision=meta["sha"])
        if meta.get("gated") or meta.get("private"):
            return [outcome(row, "excluded", "Provider gated or private")], []
        if entry["id"] in EXCLUDED:
            exclusion = EXCLUDED[entry["id"]]
        elif entry["id"] in {"MultiCoNER", "MultiCoNERv2"}:
            exclusion = "Anno registry flags this dataset as requiring HF_TOKEN"
        elif entry["access"] == "DependsOnOther":
            exclusion = "Registry requires a separate source"
        else:
            exclusion = ""
        if entry["id"] in {"BookCoref", "BookCorefSplit"}:
            scope = {"config": "full" if entry["id"] == "BookCoref" else "split"}
        if entry["id"] == "FictionNER750M":
            path = "label_counts.csv"
            current = dict(row, config=path, split="publisher inventory")
            # The publisher provides counts per document. No document text is read.
            with urlopen(repo_file(repo, meta["sha"], path), timeout=90) as response:
                table = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))
                entities = [c for c in table.fieldnames if c.startswith(("B-", "I-"))]
                if not entities or "O" not in table.fieldnames:
                    raise ValueError("Publisher label table lacks a BIO schema")
                doc_ids = set()
                tokens = tagged = 0
                for record in table:
                    if record["doc_id"] in doc_ids:
                        raise ValueError("Duplicate document in label inventory")
                    doc_ids.add(record["doc_id"])
                    total = int(record["total"])
                    annotated = sum(int(record[c]) for c in entities)
                    if total != annotated + int(record["O"]):
                        raise ValueError("Publisher document counts fail label sum")
                    tokens += total
                    tagged += annotated
            current.update(
                provider_rows=len(doc_ids),
                provider_tokens=tokens,
                provider_tagged=tagged,
            )
            return [
                outcome(
                    current,
                    "provider_only",
                    "Summed publisher label_counts.csv across document IDs",
                )
            ], []
        if entry["id"] == "MultiCoNER":
            rows = []
            for sibling in meta["siblings"]:
                match = re.fullmatch(
                    r"([^/]+)/[^/]+_(train|dev|test)\.conll", sibling["rfilename"]
                )
                if match:
                    current = dict(row, config=match[1], split=match[2])
                    rows.append(outcome(current, "provider_only", exclusion))
            if rows:
                return rows, []
        if entry["id"] == "UniversalNER" and not exclusion:
            return uner_jobs(row, meta)
        if entry["id"] == "MewsliX":
            # The Viewer wraps unrelated candidate/retrieval archives as single
            # binary rows. The Hub also lists the actual WikiNews evaluation files.
            jobs = []
            for sibling in meta["siblings"]:
                path = sibling["rfilename"]
                match = re.fullmatch(
                    r"wikinews_mentions/([a-z]+)/((?:dev|test))\.jsonl", path
                )
                if match:
                    current = dict(
                        row,
                        config=match[1],
                        split=match[2],
                        observation_column="mention",
                    )
                    jobs.append(
                        (
                            current,
                            {"mentions_jsonl": repo_file(repo, meta["sha"], path)},
                        )
                    )
            if not jobs:
                return [
                    outcome(row, "unavailable", "No WikiNews mention release files")
                ], []
            return [], jobs
        if entry["id"] == "MultiNERD":
            url = repo_file(repo, meta["sha"], "test/test_en.jsonl")
            current = dict(row, config="test/test_en.jsonl", split="test_en")
            return [], [(current, {"jsonl": url})]
        try:
            infos = viewer(repo, "info", meta["sha"]).get("dataset_info", {})
        except (HTTPError, URLError):
            infos = {}
        if not infos:
            infos = card_configs(meta)
        if not infos:
            return [
                outcome(row, "unavailable", exclusion or "No provider split metadata")
            ], []
        files = defaultdict(list)
        if not exclusion and any(
            token_columns(x.get("features", {})) for x in infos.values()
        ):
            try:
                rev = get_json(
                    f"{HUB}/api/datasets/{repo}/revision/refs%2Fconvert%2Fparquet"
                )["sha"]
                row["parquet_revision"] = rev
                for shard in viewer(repo, "parquet", meta["sha"]).get(
                    "parquet_files", []
                ):
                    path = unquote(urlsplit(shard["url"]).path)
                    prefix = f"/datasets/{repo}/resolve/refs/convert/parquet/"
                    if "/partial-" in path or not path.startswith(prefix):
                        continue
                    files[shard["config"], shard["split"]].append(
                        (repo_file(repo, rev, path[len(prefix) :]), shard["size"])
                    )
            except (HTTPError, URLError):
                pass
        ready, jobs = [], []
        for config, info in sorted(infos.items()):
            if "config" in scope and scope["config"] != config:
                continue
            for split, split_info in sorted(info.get("splits", {}).items()):
                if "split" in scope and scope["split"] != split:
                    continue
                current = dict(
                    row,
                    config=config,
                    split=split,
                    provider_rows=split_info.get("num_examples"),
                )
                features = info.get("features", {})
                pair = token_columns(features)
                if pair and not exclusion and not files[config, split]:
                    try:
                        rev, shards = pinned_shards(
                            repo, "refs%2Fconvert%2Fparquet", config, split
                        )
                        current["parquet_revision"] = rev
                        files[config, split] = shards
                    except (HTTPError, URLError):
                        pass
                if exclusion:
                    reason = exclusion
                elif (entry["id"], config) in {
                    ("CrossNER", "conll2003"),
                    ("FabNER", "text2tech"),
                }:
                    reason = "Additional corpus in publisher repository"
                elif not pair:
                    job, reason = record_plan(current)
                    if job:
                        jobs.append(job)
                        continue
                elif not files[config, split]:
                    reason = "No complete pinned Parquet partition"
                elif current["provider_rows"] is None:
                    reason = "No provider total for split completeness check"
                else:
                    scalar = scalar_feature(features[pair[1]])
                    names = scalar.get("names")
                    string_labels = scalar.get("dtype") == "string"
                    if names is None and not string_labels:
                        names = publisher_labels(repo, meta["sha"])
                    if string_labels or (names and names.count("O") == 1):
                        jobs.append(
                            (
                                current,
                                {
                                    "pair": pair,
                                    "names": names,
                                    "shards": files[config, split],
                                },
                            )
                        )
                        continue
                    reason = "Integer tag IDs lack a verified source label map"
                ready.append(outcome(current, "provider_only", reason))
        return (
            ready
            or (
                []
                if jobs
                else [outcome(row, "unavailable", "No matching config or split")]
            ),
            jobs,
        )
    except (OSError, ValueError, KeyError) as exc:
        return [outcome(row, "unavailable", str(exc))], []


def jsonl_counts(row, url):
    labels = publisher_labels(row["repository"], row["source_revision"])
    if not labels:
        raise ValueError("No publisher label map")
    rows = tokens = tagged = 0
    with urlopen(url, timeout=90) as response:
        for line in response:
            record = json.loads(line)
            tags = record["ner_tags"]
            if len(tags) != len(record["tokens"]) or any(
                not 0 <= tag < len(labels) for tag in tags
            ):
                raise ValueError("Invalid tag alignment or ID")
            rows += 1
            tokens += len(tags)
            tagged += sum(labels[tag] != "O" for tag in tags)
    return rows, tokens, tagged


def observed_rows(parquet, column):
    """Read the projected field in every row group and reject absent inputs."""
    root, *fields = column.split(".")
    if root not in parquet.schema_arrow.names:
        raise ValueError(f"Observation column {column} is absent")
    rows = 0
    for batch in parquet.iter_batches(columns=[column], batch_size=2048):
        values = batch.column(batch.schema.get_field_index(root))
        for field in fields:
            values = pc.struct_field(values, field)
        if values.null_count:
            raise ValueError(f"Null observations in {column}")
        if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
            values = pc.list_flatten(values)
            if values.null_count:
                raise ValueError(f"Null tokens in {column}")
        if not (
            pa.types.is_string(values.type) or pa.types.is_large_string(values.type)
        ):
            raise ValueError(f"Unexpected observation type for {column}: {values.type}")
        rows += batch.num_rows
    return rows


def mention_records(url):
    rows = 0
    with urlopen(url, timeout=90) as response:
        for line in response:
            record = json.loads(line)
            if not all(
                isinstance(record.get(key), str) and record[key]
                for key in ("example_id", "entity_id", "mention")
            ):
                raise ValueError(
                    "WikiNews mention record lacks an input, linked entity or ID"
                )
            rows += 1
    return rows


def uner_counts(url):
    digest = hashlib.sha256()
    sentences = tokens = tagged = 0
    in_sentence = False
    try:
        response = urlopen(url, timeout=60)
    except HTTPError as exc:
        if exc.code < 500:
            raise
        # GitHub's raw host and its contents API serve the same pinned file.
        owner, repo, sha, path = urlsplit(url).path.strip("/").split("/", 3)
        endpoint = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={sha}"
        )
        response = urlopen(
            Request(endpoint, headers={"Accept": "application/vnd.github.raw+json"}),
            timeout=60,
        )
    with response:
        for raw in response:
            digest.update(raw)
            line = raw.decode("utf-8-sig").rstrip("\r\n")
            if not line.strip():
                in_sentence = False
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3 or not fields[1]:
                raise ValueError("Invalid UniversalNER token row")
            tag = fields[2]
            # Mirror the pinned HF loader's treatment of out-of-scope labels.
            if "OTH" in tag or tag == "B-O":
                tag = "O"
            if not re.fullmatch(r"O|[BI]-(PER|ORG|LOC)", tag):
                raise ValueError("Unexpected UniversalNER annotation")
            sentences += not in_sentence
            in_sentence = True
            tokens += 1
            tagged += tag != "O"
    return sentences, tokens, tagged, digest.hexdigest()


def measure(job):
    row, spec = job
    try:
        if "iob2" in spec:
            rows, tokens, tagged, digest = uner_counts(spec["iob2"])
            if rows != row["provider_rows"]:
                raise ValueError(
                    "UniversalNER sentence count differs from pinned provider total"
                )
            return dict(
                row,
                status="scanned",
                rows=rows,
                tokens=tokens,
                tagged=tagged,
                payload_sha256=digest,
                reason="",
            )
        if "mentions_jsonl" in spec:
            return dict(
                row,
                status="scanned_records",
                rows=mention_records(spec["mentions_jsonl"]),
                tokens="",
                tagged="",
                reason="",
            )
        if "jsonl" in spec:
            rows, tokens, tagged = jsonl_counts(row, spec["jsonl"])
        else:
            rows = tokens = tagged = 0
            for url, size in spec["shards"]:
                with tempfile.TemporaryFile() as file:
                    with urlopen(url, timeout=90) as response:
                        shutil.copyfileobj(response, file)
                    if file.tell() != size:
                        raise ValueError("Shard size disagrees with provider manifest")
                    parquet = pq.ParquetFile(file)
                    if spec.get("observation_column"):
                        rows += observed_rows(parquet, spec["observation_column"])
                        continue
                    if "pair" not in spec:
                        spec.update(replay_schema(row, parquet))
                    for batch in parquet.iter_batches(
                        columns=spec["pair"], batch_size=8192
                    ):
                        words, labels = [
                            batch.column(batch.schema.get_field_index(k))
                            for k in spec["pair"]
                        ]
                        if (
                            words.null_count
                            or labels.null_count
                            or not pc.list_value_length(words).equals(
                                pc.list_value_length(labels)
                            )
                        ):
                            raise ValueError(
                                "Token and tag lists missing or misaligned"
                            )
                        values = labels.flatten()
                        if values.null_count or words.flatten().null_count:
                            raise ValueError("Null token or label")
                        if spec["names"] is not None:
                            names = spec["names"]
                            if not pa.types.is_integer(values.type):
                                raise ValueError("Expected numeric ClassLabel array")
                            if len(values) and (
                                pc.min(values).as_py() < 0
                                or pc.max(values).as_py() >= len(names)
                            ):
                                raise ValueError("Tag ID outside source label map")
                            outside = names.index("O")
                        else:
                            if not (
                                pa.types.is_string(values.type)
                                or pa.types.is_large_string(values.type)
                            ):
                                raise ValueError("Expected literal BIO tag names")
                            pattern = (
                                r"^(O|PER|ORG|LOC)$"
                                if row["id"] == "PolyglotNER"
                                else r"^(O|[BIESUL]-.+)$"
                            )
                            valid = pc.match_substring_regex(values, pattern)
                            if len(values) and not pc.all(valid).as_py():
                                raise ValueError(
                                    "Unexpected literal tag for the source label convention"
                                )
                            outside = "O"
                        rows += batch.num_rows
                        tokens += len(values)
                        tagged += pc.sum(pc.not_equal(values, outside)).as_py() or 0
            if rows != row["provider_rows"]:
                raise ValueError("Scanned row count disagrees with provider total")
        if spec.get("observation_column"):
            return dict(
                row,
                status="scanned_records",
                rows=rows,
                tokens="",
                tagged="",
                reason="",
            )
        if row["id"] == "TweetNER7" and row["split"] in {"extra_2020", "extra_2021"}:
            return dict(
                row,
                status="scanned_unlabelled",
                rows=rows,
                tokens=tokens,
                tagged="",
                reason="Publisher identifies this extra split as unannotated",
            )
        return dict(
            row, status="scanned", rows=rows, tokens=tokens, tagged=tagged, reason=""
        )
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        return outcome(row, "scan_failed", str(exc))


def scope_note(row):
    dataset, config, split = row["id"], row.get("config"), row.get("split")
    if dataset == "UniversalNER":
        return "Publisher IOB2 from the pinned HF loader manifest; OTH and B-O normalized to O per loader"
    if dataset == "MewsliX":
        return "Publisher WikiNews mention queries; candidate entity catalog and Wikipedia training pairs are separate resources"
    if dataset == "PolyglotNER":
        return "Silver Wikipedia tags O/PER/ORG/LOC; combined config overlaps individual languages"
    if dataset == "TweetTopic":
        return "Publisher partitions include a separate COLING resplit and pooled/year-specific releases; partitions can overlap"
    if dataset == "TweetNER7":
        if split in {"extra_2020", "extra_2021"}:
            return "Unannotated tweets; omit from gold label density"
        if split == "train_all":
            return "Union of train_2020 and train_2021"
        if split in {"train_random", "validation_random"}:
            return "Random resample of the corresponding pooled two-year partition"
        return "Year-specific partition; overlaps pooled or random training/validation partitions where applicable"
    if dataset == "FabNER" and config in {"fabner", "fabner_bio", "fabner_simple"}:
        return "Alternative encodings of the same sentences; select one configuration when pooling"
    if dataset == "FewNERD":
        return "Alternative supervised, inter- and intra-domain partition protocols; count by protocol"
    if dataset == "MultiNERD":
        return "Only the registry-linked English test JSONL"
    if dataset == "CoNLL2003Sample":
        return "Full linked tner English split; registry entry is named Sample"
    if dataset in {"CrossNER", "MasakhaNER2"}:
        return "Provider conversion; corresponding upstream files may also appear in the repository survey"
    return ""


def replay_schema(row, parquet):
    metadata = parquet.schema_arrow.metadata or {}
    features = (
        json.loads(metadata.get(b"huggingface", b"{}"))
        .get("info", {})
        .get("features", {})
    )
    pair = token_columns(features or parquet.schema_arrow.names)
    if pair is None:
        raise ValueError("Pinned Parquet lacks verified token/tag columns")
    if features:
        scalar = scalar_feature(features[pair[1]])
    else:
        label_type = parquet.schema_arrow.field(pair[1]).type
        if not (pa.types.is_list(label_type) or pa.types.is_large_list(label_type)):
            raise ValueError("Pinned Parquet tag column is not a list")
        scalar = (
            {"dtype": "string"}
            if (
                pa.types.is_string(label_type.value_type)
                or pa.types.is_large_string(label_type.value_type)
            )
            else {}
        )
    names = scalar.get("names")
    if names is None and scalar.get("dtype") != "string":
        names = publisher_labels(row["repository"], row["source_revision"])
    if scalar.get("dtype") != "string" and (not names or names.count("O") != 1):
        raise ValueError(
            "No outside label in pinned Parquet schema or publisher mapping"
        )
    return {"pair": pair, "names": names}


def replay(row):
    repo, rev = row["repository"], row["parquet_revision"]
    current = dict(
        row,
        provider_rows=int(row["provider_rows"]) if row["provider_rows"] else None,
        rows="",
        tokens="",
        tagged="",
    )
    if not rev and row["id"] == "UniversalNER":
        url = row.get("payload_url", "")
        if not re.fullmatch(
            r"https://raw\.githubusercontent\.com/UniversalNER/UNER_[\w-]+/[0-9a-f]{40}/[\w.-]+\.iob2",
            url,
        ):
            raise ValueError("Missing pinned UniversalNER publisher file")
        job = {"iob2": url}
    elif not rev and row["id"] == "MewsliX":
        path = f"wikinews_mentions/{row['config']}/{row['split']}.jsonl"
        job = {"mentions_jsonl": repo_file(repo, row["source_revision"], path)}
    elif not rev and row["id"] == "MultiNERD":
        job = {"jsonl": repo_file(repo, row["source_revision"], row["config"])}
    elif rev:
        _, files = pinned_shards(repo, rev, row["config"], row["split"])
        if not files:
            raise ValueError(
                f"No pinned shards for {repo}/{row['config']}/{row['split']}"
            )
        job = {"shards": files}
        if row["status"] == "scanned_records":
            column = row["observation_column"]
            if not column or column != RECORD_COLUMNS.get(row["id"]):
                raise ValueError("Unrecognized record-scan observation column")
            job["observation_column"] = column
    else:
        raise ValueError(f"Missing replay revision for {repo}")
    result = measure((current, job))
    if (
        row.get("payload_sha256")
        and result.get("payload_sha256") != row["payload_sha256"]
    ):
        raise ValueError("UniversalNER publisher digest differs from prior scan")
    for key in ("status", "rows", "tokens", "tagged"):
        if str(result[key]) != str(row[key]):
            raise ValueError(
                f"{repo}/{row['config']}/{row['split']} replay disagrees on {key}: {result.get('reason')}"
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Verify measured rows from a previous survey CSV",
    )
    args = parser.parse_args()
    if args.replay:
        with args.inventory.open(newline="") as file:
            measured = [
                row for row in csv.DictReader(file) if row["status"] in SCANNED_STATUSES
            ]
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(replay, measured))
        write_rows(results, args.output)
        return
    with args.inventory.open(newline="") as file:
        selected = list(targets(csv.DictReader(file)))
    with ThreadPoolExecutor(args.workers) as pool:
        local_only = {
            name.removesuffix(".cache")
            for name, access, _ in csv.reader(
                POLICY.read_text().splitlines(), delimiter="\t"
            )
            if access == "local"
        }
        plans = list(pool.map(plan, ((*target, local_only) for target in selected)))
        rows = [row for ready, _ in plans for row in ready]
        jobs = [job for _, pending in plans for job in pending]
        print(
            f"Processing {len(jobs)} splits from {len(selected)} registry entries",
            flush=True,
        )
        rows.extend(pool.map(measure, jobs))
    rows.sort(key=lambda r: (r["id"], r.get("config", ""), r.get("split", "")))
    write_rows(rows, args.output)


def write_rows(rows, output):
    for row in rows:
        row["scope_note"] = scope_note(row)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
