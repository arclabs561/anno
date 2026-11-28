"""Build the offline dataset explorer and its evaluation index from recorded evidence."""

import argparse
import csv
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
from urllib.parse import urlsplit

from dataset_file_survey import registry_rows
from plot_density_ranges import is_archive

REPO = "https://github.com/arclabs561/anno/blob/main/"
FIELDS = (
    "id",
    "domain",
    "measurement",
    "declared_tasks",
    "declared_splits",
    "task",
    "backend",
    "reported_backend",
    "split",
    "split_evidence",
    "scope",
    "status",
    "examples",
    "unit",
    "detail",
    "seed",
    "metric",
    "metric_value",
    "read_source",
    "source_digest",
    "panel_digest",
    "source",
    "member",
    "record",
)
UNKNOWN = "No indexed evaluation"


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def receipt(evidence, name):
    return json.loads((evidence / name).read_text(encoding="utf-8"))


def study_outcomes(evidence):
    """A row is one explicitly identified backend/setting and scoring population."""

    def score(
        ident, task, backend, split, scope, count, unit, source, digest="", **extra
    ):
        if type(count) is not int or count <= 0:
            raise ValueError(f"Expected a positive scored sample in {source}")
        return dict(
            id=ident,
            task=task,
            backend=backend,
            split=split,
            scope=scope,
            status="Scored",
            examples=count,
            unit=unit,
            source=source,
            source_digest=digest,
            **extra,
        )

    crossre = "crossre-full-prompt-comparison.json"
    domain_ids = {}
    for domain, panel in receipt(evidence, crossre)["domains"].items():
        corpus = panel["corpus"]
        domain_ids[domain] = corpus["dataset"]
        for arm in panel["arms"]:
            sample = arm["sample"]
            if (
                arm["metrics"]["num_docs"] != sample["unit_count"]
                or sample["unit_count"] != corpus["test_docs"]
            ):
                raise ValueError(
                    f"CrossRE sample disagrees with the source in {domain}"
                )
            # The receipt records distinct registry IDs for the AI and literature releases.
            yield score(
                corpus["dataset"],
                "re",
                arm["backend"],
                f"{domain} test",
                f"{arm['prompt']} prompt",
                sample["unit_count"],
                "sentences",
                crossre,
                corpus["test_sha256"],
                reported_backend=" | ".join(
                    sorted(sample["relation_producer_documents"])
                ),
            )

    source = "crossre-full-test.json"
    data = receipt(evidence, source)
    scopes = {
        "balanced_test": "original balanced test panel",
        "additional_test": "additional held-out test texts",
        "outside_content_holdout": "official occurrences outside the unique-text holdout",
        "all_eligible_test": "all content-holdout test texts, including the original panel",
        "full_official_test": "complete official test, including all recorded subsets",
    }
    for name, panel in data["panels"].items():
        for domain, group in panel["domains"].items():
            counts = group["stock"]
            yield score(
                domain_ids[domain],
                "re",
                "GLiNER-Relex",
                f"{domain} test",
                f"{scopes[name]} · stock decoder",
                counts["sentences"],
                "sentences",
                source,
                data["source_sha256"][f"{domain}-test.json"],
                reported_backend=data["checkpoint"],
                panel_digest=group["index_text_sha256"],
                metric="exact directed relations",
                metric_value=counts["matched"],
                detail=f'{counts["gold"]} gold, {counts["predicted"]} predicted',
            )

    source = "crossre-endpoint-budget.json"
    data = receipt(evidence, source)
    selection = data["selection"]
    fraction = selection["primary_budget_percent_of_eligible"]
    cohort_roles = {
        "calibration": (
            "development",
            "calibration used to select the endpoint weight",
        ),
        "confirmation": (
            "development",
            "independent confirmation on unused development texts",
        ),
        "additional_test": (
            "test",
            "additional official test texts previously examined for other rules",
        ),
        "balanced_test": ("test", "original balanced official test panel"),
        "outside_content_holdout": (
            "test",
            "official occurrences outside the unique-text holdout",
        ),
        "full_official_test": (
            "test",
            "complete official test including these three subsets",
        ),
    }
    if set(data["panels"]) != set(cohort_roles):
        raise ValueError("Incomplete CrossRE queue cohorts")
    panels = data["panels"]
    official_subsets = ("balanced_test", "additional_test", "outside_content_holdout")
    if (
        sum(panels[k]["scope"]["sentences"] for k in official_subsets)
        != panels["full_official_test"]["scope"]["sentences"]
    ):
        raise ValueError(
            "CrossRE official release must account for its three test subsets"
        )
    for cohort_name, (split, role) in cohort_roles.items():
        panel = data["panels"][cohort_name]
        scope, budget = panel["scope"], panel["budgets"][str(fraction)]
        for key, policy in (
            ("confidence", "predicate confidence"),
            ("endpoint", "endpoint-weighted confidence"),
        ):
            weight = selection["selected_endpoint_weight"] if key == "endpoint" else 0
            yield score(
                "CrossRE",
                "relation review",
                "GLiNER-Relex",
                split,
                f"{role} · {policy} queue",
                scope["sentences"],
                "sentences",
                source,
                panel_digest=scope["input_sha256"],
                detail=f'{budget["review_slots"]} review slots at {fraction}% of eligible sentences · endpoint weight {weight}',
                metric="correct additions in review queue",
                metric_value=budget["counts"].get(key + "_correct", 0),
            )

    source = "crossre-cli-predicate-gates.json"
    data = receipt(evidence, source)
    for split, panel in data["panels"].items():
        if set(panel["index_text_sha256_by_domain"]) != set(domain_ids):
            raise ValueError(
                "The pooled predicate-gate controls require every CrossRE domain"
            )
        for gate, counts in panel["predicate_gates"].items():
            yield score(
                "CrossRE",
                "re",
                "GLiNER-Relex",
                split,
                f"six-domain pooled replay on the existing balanced panel · predicate gate {gate}",
                panel["sentences"],
                "sentences",
                source,
                metric="exact directed relations",
                metric_value=counts["matched"],
                detail=f'{panel["gold"]} gold, {counts["predicted"]} predicted',
            )

    for source in ("wnut-native-comparison.json", "wnut-stacked-comparison.json"):
        data = receipt(evidence, source)
        for setting, arm in data["arms"].items():
            yield score(
                "Wnut17",
                "ner",
                data["protocol"]["backend"],
                "test",
                f"{setting} · {arm['cli_version']}",
                data["source"]["test_sentences"],
                "sentences",
                source,
                data["source"]["sha256"]["emerging.test.annotated"],
                reported_backend=" | ".join(sorted(arm["runtime_model_calls"])),
            )

    for ident, source in (
        ("WikiGold", "wikigold-nuner-routing.json"),
        ("Wnut17", "nuner-fallback-routing.json"),
    ):
        data = receipt(evidence, source)
        splits = {"unsplit": data["arms"]} if ident == "WikiGold" else data["splits"]
        for split, arms in splits.items():
            digest = (
                data["source"]["sha256"]
                if ident == "WikiGold"
                else data["source_sha256"][
                    (
                        "emerging.dev.conll"
                        if split == "Development"
                        else "emerging.test.annotated"
                    )
                ]
            )
            for setting, arm in arms.items():
                mode = setting.replace("_", " ")
                yield score(
                    ident,
                    "ner",
                    "stacked",
                    split.lower(),
                    f"stock · {mode}",
                    arm["all_sentences"],
                    "sentences",
                    source,
                    digest,
                )
                yield score(
                    ident,
                    "ner",
                    "stacked",
                    split.lower(),
                    f"extra NuNER on stock-skipped inputs · {mode}",
                    arm["stock_skipped"],
                    "sentences",
                    source,
                    digest,
                )

    source = "nuner-misc-prompt.json"
    data = receipt(evidence, source)
    for group, ident, split in (
        ("wikigold", "WikiGold", "unsplit release"),
        ("wnut_test", "Wnut17", "test"),
    ):
        panel = data[group]
        digest = panel["source"]["sha256"]
        if group == "wnut_test":
            digest = digest["emerging.test.annotated"]
            n, panel_digest = (
                panel["source"]["test_sentences"],
                panel["source"]["test_reconstructed_texts_sha256"],
            )
        else:
            n, panel_digest = (
                panel["source"]["sentence_occurrences"],
                panel["source"]["native_gold_cases_sha256"],
            )
        if n != panel["sentences"] or panel["nuner_calls"] > n:
            raise ValueError(
                f"MISC prompt scoring population differs from {ident} source"
            )
        for arm in ("stock", "candidate"):
            prompts = data[arm + "_prompts"]
            yield score(
                ident,
                "ner",
                "stacked",
                split,
                f"{arm} · {len(prompts)} NuNER prompts, BERT {data['bert']}",
                panel["sentences"],
                "sentences",
                source,
                digest,
                panel_digest=panel_digest,
                reported_backend="nuner",
                detail=f'{panel["nuner_calls"]} sentences called NuNER via stock routing',
                metric="exact typed matches",
                metric_value=panel[arm + "_exact"],
            )

    source = "coref-panels.json"
    data = receipt(evidence, source)
    for corpus, ident in (("gap", "GAP"), ("litbank", "LitBank")):
        panel = data[corpus]
        for key, arm in panel["arms"].items():
            backend, mode = key.split("/")
            rejected = Counter(
                row["unsupported"]
                for row in arm["per_id"]
                if isinstance(row, dict) and "unsupported" in row
            )
            count = len(arm["per_id"]) - sum(rejected.values())
            if count:
                yield score(
                    ident,
                    "intra-coref",
                    backend,
                    panel["split"],
                    f"{mode} mentions",
                    count,
                    "excerpts",
                    source,
                    panel["sha256"],
                )
            for reason, count in rejected.items():
                yield dict(
                    id=ident,
                    task="intra-coref",
                    backend=backend,
                    split=panel["split"],
                    scope=f"{mode} mentions",
                    status="Unsupported input",
                    examples=count,
                    unit="excerpts",
                    detail=reason,
                    source=source,
                    source_digest=panel["sha256"],
                )
    source = "native-gap-link-retention.json"
    data = receipt(evidence, source)
    for split, digest in data["gap_sha256"].items():
        panel = data["panels"][f"GAP {split}"]
        for mode, counts in panel["operating_points"].items():
            if (
                counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
                != 2 * panel["rows"]
            ):
                raise ValueError(f"GAP decisions disagree with {split} excerpt count")
            yield score(
                "GAP",
                "intra-coref",
                "mention-ranking",
                split,
                f"{mode} · supplied mentions",
                panel["rows"],
                "excerpts",
                source,
                digest,
                metric="correct pronoun–name links",
                metric_value=counts["tp"],
                detail=(
                    f"margin {data['selected_development_margins'][mode.split()[0].lower()]}"
                    if mode in {"Rival margin", "Threshold margin"}
                    else ""
                ),
            )

    source = "ner-language-length.json"
    for corpus, cohort in receipt(evidence, source)["corpora"].items():
        if corpus.split("@")[0] != "Babelscape/multinerd":
            continue  # RED-FM is a study source outside this registry snapshot.
        for size, panel in cohort["panels"].items():
            for backend, report in panel["reports"].items():
                for cell in report["test_correct_unmatched_missed"]:
                    yield score(
                        "MultiNERD",
                        "ner",
                        backend,
                        f"test · {cell}",
                        f"{size} per length/language cell",
                        int(size),
                        "sentences",
                        source,
                        cohort["source_sha256"]["test_" + cell.split("/")[0]],
                    )

    source = "nuner-4k-labeled.json"
    data = receipt(evidence, source)
    if data["dataset"] != "Babelscape/multinerd":
        raise ValueError("Unexpected NuNER study corpus")
    for precision, arm in data["models"].items():
        for cell, sample in arm["groups"].items():
            language = cell.split("-")[0]
            yield score(
                "MultiNERD",
                "ner",
                "NuNER Zero-4k",
                f"{language} test",
                f"{precision} · {cell}",
                sample["n"],
                "packed inputs" if cell.endswith("-packed") else "sentences",
                source,
                data["test_file_sha256"][language],
            )

    source = "muxer-public-summary.json"
    data = receipt(evidence, source)
    for cell in data["cells"]:
        n = cell["evaluated_trials"]
        if (
            n != len(data["seeds"])
            or n * data["examples_per_trial"] != cell["example_evaluations"]
        ):
            raise ValueError("Muxer trial/sample counts disagree")
        yield score(
            cell["dataset"],
            cell["task"].lower(),
            cell["backend"],
            "",
            "pinned offline sampler",
            cell["example_evaluations"],
            "example evaluations",
            source,
            data["dataset_sha256"],
            seed=" | ".join(map(str, data["seeds"])),
            detail=f'{n} trials of {data["examples_per_trial"]} examples',
        )

    for filename in ("docred_0_31", "docred_32_63"):
        source = (
            f"../../scripts/qa/joint_baseline/receipts/jerex_controls/{filename}.json"
        )
        data = receipt(evidence, source)
        if "predicted_clusters" not in data["arms"]:
            raise ValueError("Missing JEREX stock prediction arm")
        yield score(
            "DocRED",
            "re",
            "JEREX",
            "authors’ joint test",
            f"documents starting at {data['document_start']} · predicted clusters",
            data["documents"],
            "documents",
            source,
            data["source_sha256"],
        )

    source = "genia-native-exports.json"
    data = receipt(evidence, source)
    for split, panel in data["splits"].items():
        yield dict(
            id=data["dataset_id"],
            task="gold-span export",
            backend="",
            split=split,
            scope="HTML and JSONL given gold mentions",
            status="Exporter replay",
            examples=panel["sentences_verified"],
            unit="sentences",
            source=source,
            source_digest=panel["source"]["sha256"],
        )


def matched_splits(matches):
    result = defaultdict(set)
    aliases = {
        "test.tsv": "test",
        "test.txt": "test",
        "gap-test.tsv": "test",
        "unsplit": "unsplit release",
    }
    for row in matches:
        if row["cache_id"] == row["publisher_id"] and not is_archive(row["source"]):
            label = row["split_or_subset"]
            if row["cache_id"] == "WikiGold" and label == "complete linked release":
                label = "unsplit release"  # This particular publisher release has no official split.
            if label:
                result[(row["cache_id"], row["sha256"])].add(aliases.get(label, label))
    return result


def assemble(coverage, declarations, outcomes, matches):
    inventory_ids = {row["id"] for row in coverage}
    registered = set(declarations)
    if len(inventory_ids) != len(coverage) or inventory_ids != registered:
        raise ValueError(
            "The coverage inventory must include each registry ID once. "
            f"Missing from the survey: {sorted(registered - inventory_ids)}. "
            f"Missing from the registry: {sorted(inventory_ids - registered)}."
        )
    by_id = defaultdict(list)
    for row in outcomes:
        if row["id"] not in declarations:
            raise ValueError(f"Unmatched evaluation dataset {row['id']}")
        split = row.get("split")
        evidence_type = "study or execution receipt"
        if not split:
            scopes = matches.get((row["id"], row.get("source_digest", "")), set())
            if row.get("scope") == "checkout fixture":
                split, evidence_type = "fixture", "explicit checkout fixture"
            elif len(scopes) == 1:
                split, evidence_type = (
                    next(iter(scopes)),
                    "same-ID publisher file SHA-256",
                )
            else:
                split = "unreported"
                evidence_type = (
                    "ambiguous same-ID file scopes" if scopes else "unreported"
                )
        by_id[row["id"]].append(dict(row, split=split, split_evidence=evidence_type))

    results = []
    for item in sorted(coverage, key=lambda r: r["id"]):
        ident = item["id"]
        measurement = (
            "Tagged-token counts"
            if int(item["tagged_tokens_measured"])
            else (
                "Other corpus counts"
                if int(item["measured"])
                else "No independent measurement"
            )
        )
        base = dict(
            id=ident,
            domain=item["domain"],
            measurement=measurement,
            declared_tasks=declarations[ident].get("declared_tasks", ""),
            declared_splits=declarations[ident].get("declared_splits", ""),
        )
        evidence = by_id.get(ident) or [dict(status=UNKNOWN)]
        for record in sorted(
            evidence,
            key=lambda r: tuple(
                str(r.get(k, ""))
                for k in ("task", "backend", "split", "scope", "source", "record")
            ),
        ):
            results.append({**base, **record})
    return results


def link(url, label):
    if urlsplit(url).scheme not in {"http", "https"}:
        return escape(label)
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'


def receipt_link(row):
    url = row["source"]
    if url.startswith("https://"):
        return link(url, "CI artifact")
    return link(
        REPO + (url[6:] if url.startswith("../../") else "docs/eval/" + url),
        "Study receipt",
    )


def cells(values, header=False):
    tag = "th" if header else "td"
    attr = ' scope="col"' if header else ""
    return "<tr>" + "".join(f"<{tag}{attr}>{v}</{tag}>" for v in values) + "</tr>"


def table(heads, content):
    return (
        '<div class="scroll"><table><thead>'
        + cells(heads, True)
        + "</thead><tbody>"
        + "".join(content)
        + "</tbody></table></div>"
    )


def explorer(rendered, indexed, coverage):
    entries = defaultdict(list)
    for row in indexed:
        entries[row["id"]].append(row)
    units = defaultdict(list)
    for row in csv_rows(rendered / "dataset-density-units.csv"):
        units[row["id"]].append(row)
    summaries = {r["id"]: r for r in csv_rows(rendered / "dataset-density-ranges.csv")}
    if set(units) != set(summaries) or any(
        len(group) != int(summaries[k]["n_observation_units"])
        for k, group in units.items()
    ):
        raise ValueError("Every density observation must have one drill-down row")
    datasets = []
    for item in sorted(coverage, key=lambda r: (r["name"].casefold(), r["id"])):
        ident = item["id"]
        records = entries[ident]
        first = records[0]
        count = Counter(r["status"] for r in records)
        state = " · ".join(
            f"{n} {s.lower()}" if s != UNKNOWN else UNKNOWN
            for s, n in sorted(count.items())
        )
        den = summaries.get(ident, {})
        annotation = den.get("annotation", "")
        attributes = {
            "id": ident,
            "domain": item["domain"],
            "annotation": annotation,
            "measurement": first["measurement"],
            "keywords": " ".join(
                (ident, item["name"], item["domain"], first["declared_tasks"])
            ).lower(),
        }
        attrs = " ".join(
            f'data-{key}="{escape(str(value), quote=True)}"'
            for key, value in attributes.items()
        )
        part = [
            f'<details class="dataset" {attrs}><summary>{escape(item["name"])} <span>{escape(item["domain"] or "unspecified domain")} · {escape(state)}</span></summary>',
            f'<p><strong>{escape(ident)}</strong> · {escape(first["measurement"])} · {escape(item["source_disposition"])}</p>',
            f'<p>Declared tasks · {escape(first["declared_tasks"].replace("|", ", ") or "unspecified")}<br>'
            f'Declared splits · {escape(first["declared_splits"].replace("|", ", ") or "unspecified")}</p>',
        ]
        reasons = item["outstanding_reason"]
        if reasons:
            part.append(
                f'<p><strong>{escape(item["coverage_action"])}</strong> · {escape(reasons)}</p>'
            )
        if item["source_review_url"]:
            part.append(
                f'<p>{link(item["source_review_url"], "Reviewed source")} · {escape(item["source_review_scope"])}</p>'
            )
        elif item["url"]:
            part.append(f'<p>{link(item["url"], "Registry source link")}</p>')
        if count[UNKNOWN]:
            part.append(
                '<p class="outcome" data-status="No indexed evaluation">No evaluation receipt for this ID in the indexed sources.</p>'
            )
        else:
            scored = []
            for r in records:
                backend = r.get("backend") or "—"
                actual = r.get("reported_backend", "")
                display = (
                    backend
                    if not actual or actual == backend
                    else f"{backend} · reported {actual}"
                )
                notes = " · ".join(
                    str(x)
                    for x in (r.get("scope"), r.get("read_source"), r.get("detail"))
                    if x
                )
                n = r.get("examples", "")
                sample = f'{int(n):,} {r.get("unit", "")}' if n != "" else "—"
                citation = receipt_link(r)
                if r.get("member"):
                    citation += f'<small>{escape(r["member"].split("/")[-1])} · record {escape(str(r["record"]))}</small>'
                status_label = escape(r["status"])
                if r.get("metric_value", "") != "":
                    number = float(r["metric_value"])
                    value = (
                        f"{number:.3f}"
                        if r["metric"] in {"primary_f1", "f1"}
                        else f"{number:g}"
                    )
                    status_label += f'<small>{escape(r["metric"])} = {value}</small>'
                split = escape(r["split"])
                provenance = r["split_evidence"]
                if provenance.startswith(("same-ID", "ambiguous")):
                    split += f"<small>{escape(provenance)}</small>"
                body = cells(
                    [
                        escape(r.get("task", "")),
                        escape(display),
                        split,
                        escape(notes),
                        status_label,
                        escape(sample),
                        citation,
                    ]
                )
                record_fields = {
                    "status": r["status"],
                    "backend": r.get("backend", ""),
                    "reported": actual,
                    "task": r.get("task", ""),
                    "split": r["split"],
                }
                attrs = " ".join(
                    f'data-{key}="{escape(value, quote=True)}"'
                    for key, value in record_fields.items()
                )
                scored.append(body.replace("<tr>", f'<tr class="outcome" {attrs}>', 1))
            part.append(
                table(
                    [
                        "Task",
                        "Backend",
                        "Split",
                        "Setting",
                        "Outcome",
                        "Inputs",
                        "Evidence",
                    ],
                    scored,
                )
            )
        if den:
            part.append(
                f"<h3>{escape(annotation)} · {len(units[ident]):,} measured units</h3>"
                f'<p>Median partition density {float(den["median_percent"]):.1f}%. Open the units to inspect the numerator and denominator.</p>'
            )
            source_rows = []
            for u in units[ident]:
                url = u["source"]
                if not urlsplit(url).scheme and url:
                    url = REPO + url
                source_rows.append(
                    cells(
                        [
                            escape(u["partition"] or "unspecified"),
                            f'{int(u["tagged"]):,}',
                            f'{int(u["tokens"]):,}',
                            f'{float(u["percent"]):.2f}%',
                            link(url, u["route"]),
                        ]
                    )
                )
            part.append(
                "<details><summary>Measured files and partitions</summary>"
                + table(
                    ["File or partition", "Tagged", "Tokens", "Share", "Source"],
                    source_rows,
                )
                + "</details>"
            )
        datasets.append("".join(part) + "</details>")
    html = (
        Path(__file__).with_name("dataset_explorer.html.in").read_text(encoding="utf-8")
    )
    nominal = {r.get("backend", "") for r in indexed}
    actual = {
        b
        for r in indexed
        for b in r.get("reported_backend", "").split(" | ")
        if b != "unreported"
    }
    options = {
        "domains": {r["domain"] for r in indexed},
        "measurements": {r["measurement"] for r in indexed},
        "states": {r["status"] for r in indexed},
        "schemes": {r["annotation"] for r in summaries.values()},
        "tasks": {r.get("task", "") for r in indexed},
        "splits": {r.get("split", "") for r in indexed},
        "backends": nominal | actual,
    }
    for key, items in options.items():
        html = html.replace(
            "{{" + key + "}}",
            "".join(f"<option>{escape(val)}</option>" for val in sorted(items) if val),
        )
    html = html.replace("{{count}}", str(len(entries))).replace(
        "{{units}}", f"{sum(map(len, units.values())):,}"
    )
    html = html.replace("{{tagged_ids}}", str(len(units))).replace(
        "{{datasets}}", "".join(datasets)
    )
    (rendered / "dataset-explorer.html").write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path)
    p.add_argument("rendered", type=Path)
    args = p.parse_args()
    coverage = csv_rows(args.rendered / "dataset-coverage.csv")
    declarations = {
        row["id"]: row for row in registry_rows(include_declared_tasks=True)[0]
    }
    indexed = assemble(
        coverage,
        declarations,
        [
            *csv_rows(args.evidence / "dataset-ci-outcomes.csv"),
            *study_outcomes(args.evidence),
        ],
        matched_splits(csv_rows(args.evidence / "s3-publisher-file-matches.csv")),
    )
    with (args.rendered / "dataset-evaluation-coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.DictWriter(out, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(indexed)
    explorer(args.rendered, indexed, coverage)


if __name__ == "__main__":
    main()
