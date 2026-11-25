"""Summarize live web QA coverage without treating retrieval as labeled accuracy."""
from collections import Counter, defaultdict
import html
import re
from urllib.parse import urlsplit


def _label(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value) is not None


# ModelBackend CLI aliases and runtime names returned by Model::name().
# A shared architecture alone (e.g. bert-onnx for multiple checkpoints)
# cannot establish the identity of a specialized requested model.
def canonical_backend(name):
    normalized = name.lower().replace("_", "-")
    aliases = {
        "regex": "pattern", "statistical": "heuristic", "tplink": "tplinker",
        "bert": "bert-onnx", "deberta": "deberta-v3", "biomedical-ner": "biomedical",
        "pii": "gliner-pii", "gliner-onnx": "gliner",
        "glinermultitask-onnx": "gliner-multitask", "candlener": "candle-ner",
    }
    return aliases.get(normalized, normalized)


def backend_matches(requested, runtime):
    if not isinstance(runtime, str) or not runtime:
        return False
    backend = canonical_backend(requested)
    if backend in {"stacked", "ensemble"}:
        separator = r"\+" if backend == "stacked" else r"\|"
        if re.fullmatch(rf"{backend}\([A-Za-z0-9_-]+(?:{separator}[A-Za-z0-9_-]+)*\)", runtime):
            return True
    return canonical_backend(runtime) == backend


def valid_domain_name(name):
    return isinstance(name, str) and bool(name) and name == name.strip() and name.isprintable()


def _host(url):
    if not isinstance(url, str) or any(ch.isspace() or ord(ch) < 32 for ch in url):
        raise ValueError("expected an HTTPS URL")
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None:
        raise ValueError("expected an HTTPS URL without credentials")
    return parts.hostname


def coverage_summary(receipt):
    """Count sampled catalog entries, paired inputs and independent source content."""
    if (not isinstance(receipt, dict) or receipt.get("kind") != "anno.qa.cross-domain"
            or type(receipt.get("schema_version")) is not int or receipt["schema_version"] not in (1, 2)
            or type(receipt.get("seed")) is not int):
        raise ValueError("expected a version 1 or 2 cross-domain receipt with a seed")
    backends = receipt.get("backends")
    if (not isinstance(backends, list) or not backends or not all(_label(x) for x in backends)
            or len(set(backends)) != len(backends)):
        raise ValueError("expected distinct requested backends")
    if receipt["schema_version"] == 2 and len({canonical_backend(b) for b in backends}) != len(backends):
        raise ValueError("expected distinct requested backends")
    eligible = receipt.get("eligible_sources")
    if (not isinstance(eligible, dict) or not eligible
            or not all(_label(k) and type(v) is int and v > 0 for k, v in eligible.items())):
        raise ValueError("expected positive eligible source counts by domain")
    sources = receipt.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("empty selection")
    sampled = Counter()
    matched = Counter()
    failures = Counter()
    seen_urls = set()
    input_matched_urls = set()
    final_hosts = set()
    domains_per_hash = defaultdict(set)
    texts_per_domain = defaultdict(set)
    for source in sources:
        if not isinstance(source, dict) or not _label(source.get("domain")):
            raise ValueError("invalid source domain")
        topic = source["domain"]
        url = source.get("source_url")
        _host(url)
        if url in seen_urls:
            raise ValueError("duplicate source URL")
        seen_urls.add(url)
        sampled[topic] += 1
        if sampled[topic] > eligible.get(topic, 0):
            raise ValueError("selection exceeds the eligible source count")
        checks = source.get("checks")
        if not isinstance(checks, list) or "selected_url" not in source:
            raise ValueError("source needs a selected URL or a recorded discovery failure, and checks")
        if source["selected_url"] is None:
            if checks:
                raise ValueError("undiscovered source has backend checks")
            failures["discovery"] += 1
            continue
        _host(source["selected_url"])
        observed = {}
        backend_failures = set()
        for check in checks:
            if not isinstance(check, dict) or not _label(check.get("backend")):
                raise ValueError("invalid backend check")
            backend = check["backend"]
            if backend in observed or backend not in backends:
                raise ValueError("duplicate or unexpected backend")
            status = check.get("status")
            if status not in ("ok", "error", "timeout", "backend_mismatch", "backend_unverified"):
                raise ValueError("invalid backend status")
            observed[backend] = None
            if status in {"ok", "backend_mismatch", "backend_unverified"}:
                runtime = check.get("reported_model")
                if runtime is not None and (not isinstance(runtime, str) or not runtime.strip()):
                    raise ValueError("invalid runtime backend identity")
                if receipt["schema_version"] == 1 or runtime is None or status == "backend_unverified":
                    backend_failures.add("backend_unverified")
                elif status == "backend_mismatch" or not backend_matches(backend, runtime):
                    backend_failures.add("backend_mismatch")
                final_url, digest = check.get("final_url"), check.get("text_hash")
                _host(final_url)
                if not isinstance(digest, str) or not re.fullmatch(r"xxh3:[0-9a-f]{16}", digest):
                    raise ValueError("successful backend lacks a valid rendered text hash")
                observed[backend] = (final_url, digest)
        if observed.keys() != set(backends):
            raise ValueError("missing requested backend check")
        if None in observed.values():
            failures["backend"] += 1
            continue
        identities = set(observed.values())
        if len(identities) != 1:
            failures["input_mismatch"] += 1
            continue
        input_matched_urls.add(url)
        if backend_failures:
            failures["backend_mismatch" if "backend_mismatch" in backend_failures else "backend_unverified"] += 1
            continue
        final_url, digest = identities.pop()
        final_hosts.add(_host(final_url))
        matched[topic] += 1
        texts_per_domain[topic].add(digest)
        domains_per_hash[digest].add(topic)
    replay_counts = None
    if "replay" in receipt:
        replay = receipt["replay"]
        if (not isinstance(replay, dict) or not isinstance(replay.get("reference_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", replay["reference_sha256"])):
            raise ValueError("replay needs the digest of its reference receipt")
        outcomes = replay.get("outcomes")
        if not isinstance(outcomes, dict) or outcomes.keys() != seen_urls:
            raise ValueError("replay outcomes must cover exactly the selected sources")
        for url, outcome in outcomes.items():
            if outcome not in ("matched", "changed", "unverified"):
                raise ValueError("invalid replay outcome")
            if url not in input_matched_urls and outcome != "unverified":
                raise ValueError("replay agreement requires a verified current input")
        replay_counts = dict(Counter(outcomes.values()))
    return {
        "replay": replay_counts, "input_matched": len(input_matched_urls), "selected": len(sources), "matched": sum(matched.values()), "failures": dict(failures),
        "final_hosts": len(final_hosts), "distinct_texts": len(domains_per_hash),
        "shared_across_domains": sum(len(domains) > 1 for domains in domains_per_hash.values()),
        "domains": {topic: {
            "eligible": eligible[topic], "selected": sampled[topic], "matched": matched[topic],
            "distinct_texts": len(texts_per_domain[topic]),
            "exclusive_texts": sum(len(domains_per_hash[h]) == 1 for h in texts_per_domain[topic]),
        } for topic in sorted(eligible)},
    }


def render_coverage(receipt):
    summary = coverage_summary(receipt)
    domains = summary["domains"]
    counts = summary["failures"]
    lines = [
        f"Catalog topics sampled: {sum(x['selected'] > 0 for x in domains.values())}/{len(domains)}. "
        f"Inputs verified on every requested backend: {summary['matched']}/{summary['selected']}.",
        f"Matching input URL and text hash across requests: {summary['input_matched']}/{summary['selected']}.",
        f"Verified inputs reached {summary['final_hosts']} distinct final hosts and "
        f"{summary['distinct_texts']} distinct rendered texts. "
        f"{summary['shared_across_domains']} text hashes appeared under multiple catalog topics.",
        f"Sources omitted from verified coverage: {counts.get('discovery', 0)} discovery failures, "
        f"{counts.get('backend', 0)} backend failures, {counts.get('input_mismatch', 0)} input mismatches, "
        f"{counts.get('backend_mismatch', 0)} runtime backend mismatches, "
        f"{counts.get('backend_unverified', 0)} unverified runtime identities.",
        "",
        "| Catalog topic | Eligible | Selected | Verified | Distinct texts | Texts unique to this topic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for topic, values in domains.items():
        lines.append(f"| {topic} | {values['eligible']} | {values['selected']} | {values['matched']} | "
                     f"{values['distinct_texts']} | {values['exclusive_texts']} |")
    if summary["replay"] is not None:
        replay = summary["replay"]
        lines.extend(["", f"Compared with the reference receipt: {replay.get('matched', 0)} matched, "
                      f"{replay.get('changed', 0)} changed, {replay.get('unverified', 0)} unverified."])
    lines.extend(["", "Topics describe the catalog. Check the rendered content and independent labels "
                  "before drawing conclusions about extraction accuracy."])
    return "\n".join(lines)


def _count(row, field):
    value = row.get(field) if isinstance(row, dict) else None
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid {field} count in labeled QA report")
    return value


def render_domain_scores(report):
    """Use aggregate scores for the handoff, keeping the fixture text and model output local."""
    digest = report.get("fixture_sha256")
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("expected a version 1 labeled QA report with its fixture SHA-256")
    groups = report.get("domains")
    if not isinstance(groups, list) or not groups:
        raise ValueError("labeled QA report is missing its domain counts")
    paired = "comparison" in report
    lines = [f"Labeled span assertions for fixture `{digest}`.",
             "", "| Domain | Documents | Exact spans | Protected ranges | Gains | Losses |",
             "|---|---:|---:|---:|---:|---:|"]
    totals = Counter()
    seen = set()
    for row in groups:
        if not isinstance(row, dict):
            raise ValueError("invalid domain summary")
        topic = row.get("domain")
        if (topic is not None and not valid_domain_name(topic)) or topic in seen:
            raise ValueError("expected distinct domain identifiers in labeled QA report")
        seen.add(topic)
        docs = _count(row, "documents")
        columns = []
        passes = count = 0
        for field in ("exact_span", "no_overlap"):
            hit, n = _count(row.get(field), "passed"), _count(row.get(field), "total")
            if hit > n:
                raise ValueError("passed count exceeds its denominator")
            passes += hit
            count += n
            columns.append(f"{hit}/{n}")
        if not 0 < docs <= count:
            raise ValueError("invalid labeled document count")
        totals.update(passed=passes, total=count)
        if paired != ("comparison" in row):
            raise ValueError("incomplete per-domain comparison")
        if paired:
            comparison = row["comparison"]
            base, gained, lost = (_count(comparison, key) for key in ("baseline_passed", "gained", "lost"))
            if base > count or lost > base or gained > count - base or base + gained - lost != passes:
                raise ValueError("inconsistent paired outcomes")
            totals.update(baseline_passed=base, gained=gained, lost=lost)
            columns.extend([str(gained), str(lost)])
        else:
            columns.extend(["—", "—"])
        # A caller can supply a display name such as "Finance | Markets".
        name = "(unlabeled domain)" if topic is None else html.escape(topic, quote=False)
        name = re.sub(r"([\\|`*_\[\]])", r"\\\1", name)
        lines.append("| " + " | ".join([name, str(docs), *columns]) + " |")
    if any(totals[key] != _count(report, key) for key in ("passed", "total")):
        raise ValueError("per-domain totals disagree with report")
    if paired and any(totals[key] != _count(report["comparison"], key)
                      for key in ("baseline_passed", "gained", "lost")):
        raise ValueError("per-domain comparison disagrees with report")
    lines.extend(["", "Each denominator counts the supplied assertions. Review source families "
                  "and annotation coverage before generalizing beyond this fixture."])
    if "prediction_coverage" in report:
        coverage = report["prediction_coverage"]
        total = _count(coverage, "total")
        unassessed = _count(coverage, "unassessed")
        if coverage.get("scope") != "supplied_assertions" or unassessed > total:
            raise ValueError("invalid prediction coverage for supplied assertions")
        lines.append(f"Of {total} emitted predictions, {total - unassessed} matched a typed exact "
                     f"assertion or violated a typed protected range. {unassessed} remain unassessed.")
    else:
        lines.append("This receipt did not record how many predictions the assertions assessed.")
    lines.append("Whole-document precision requires judgments for additional predictions.")
    return "\n".join(lines)
