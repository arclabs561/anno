"""An independent reader oracle for exact postings from the native anno query command."""
import json
import os
import subprocess
import time

from crossdoc_cases import _fields, _int, _items, _name, _record

RUBRIC = "anno.qa.query-postings.v1"


def validate(holdout, case, source_sha256):
    _fields(holdout, ("kind", "source_sha256", "queries"))
    if holdout["kind"] != RUBRIC or holdout["source_sha256"] != source_sha256:
        raise ValueError("query rubric or frozen source digest does not match this case")
    queries = _items(holdout["queries"], 32, 1)
    documents = case["documents"]
    for query in queries:
        _fields(query, ("after", "postings"), ("entity", "catalog_record"))
        after = _int(query["after"], 1, len(documents))
        if "entity" not in query and "catalog_record" not in query:
            raise ValueError("a reader query requires an entity or catalog_record filter")
        if "entity" in query:
            _name(query["entity"])
        if "catalog_record" in query:
            catalog, accession = _record(query["catalog_record"], qualified=True)
            if catalog.startswith("-") or accession.startswith("-"):
                raise ValueError("anno query requires catalog filter arguments without a leading hyphen")
        editions = {doc["id"]: doc for doc in documents[:after]}
        seen = set()
        for posting in _items(query["postings"], 256):
            ident, position = _items(posting, 2, 2)
            if _name(ident) not in editions:
                raise ValueError("unknown expected document at the query checkpoint")
            _int(position, 0, len(editions[ident]["signals"]) - 1)
            if (ident, position) in seen:
                raise ValueError("duplicate expected posting")
            seen.add((ident, position))
    return queries


def run(queries, reports, binary):
    env = {**os.environ, "ANNO_NO_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1"}
    deadline = time.monotonic() + 30
    checks = []
    for rule in queries:
        args = [str(binary), "query", str(reports[rule["after"]]), "--format", "json"]
        if "entity" in rule:
            args.append("--entity=" + rule["entity"])
        if "catalog_record" in rule:
            args.extend(["--catalog-record", *rule["catalog_record"]])
        try:
            result = subprocess.run(args, capture_output=True, env=env,
                                    timeout=max(0, min(10, deadline - time.monotonic())))
        except subprocess.TimeoutExpired as exc:
            raise ValueError("reader query exceeded its replay budget") from exc
        if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
            raise ValueError(f"invalid anno query execution (exit {result.returncode})")
        rows = _items(json.loads(result.stdout), 256)
        actual = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("expected a query result row")
            for posting in _items(row.get("mentions"), 256, 1):
                ident, position = _items(posting, 2, 2)
                actual.append([_name(ident), _int(position, 0, 255)])
                if len(actual) > 256:
                    raise ValueError("query returned more than 256 postings")
        actual.sort()
        checks.append({**rule, "actual": actual, "passed": actual == sorted(rule["postings"])})
    return checks
