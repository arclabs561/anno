"""Summarize live web QA coverage without treating retrieval as labeled accuracy."""
from collections import Counter, defaultdict
import re
from urllib.parse import urlsplit


def _label(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value) is not None


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
            or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or type(receipt.get("seed")) is not int):
        raise ValueError("expected a version 1 cross-domain receipt with a seed")
    backends = receipt.get("backends")
    if (not isinstance(backends, list) or not backends or not all(_label(x) for x in backends)
            or len(set(backends)) != len(backends)):
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
        for check in checks:
            if not isinstance(check, dict) or not _label(check.get("backend")):
                raise ValueError("invalid backend check")
            backend = check["backend"]
            if backend in observed or backend not in backends:
                raise ValueError("duplicate or unexpected backend")
            status = check.get("status")
            if status not in ("ok", "error", "timeout"):
                raise ValueError("invalid backend status")
            observed[backend] = None
            if status == "ok":
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
        final_url, digest = identities.pop()
        final_hosts.add(_host(final_url))
        matched[topic] += 1
        texts_per_domain[topic].add(digest)
        domains_per_hash[digest].add(topic)
    return {
        "selected": len(sources), "matched": sum(matched.values()), "failures": dict(failures),
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
        f"Verified inputs reached {summary['final_hosts']} distinct final hosts and "
        f"{summary['distinct_texts']} distinct rendered texts. "
        f"{summary['shared_across_domains']} text hashes appeared under multiple catalog topics.",
        f"Sources omitted from verified coverage: {counts.get('discovery', 0)} discovery failures, "
        f"{counts.get('backend', 0)} backend failures, {counts.get('input_mismatch', 0)} input mismatches.",
        "",
        "| Catalog topic | Eligible | Selected | Verified | Distinct texts | Texts unique to this topic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for topic, values in domains.items():
        lines.append(f"| {topic} | {values['eligible']} | {values['selected']} | {values['matched']} | "
                     f"{values['distinct_texts']} | {values['exclusive_texts']} |")
    lines.extend(["", "Topics describe the catalog. Check the rendered content and independent labels "
                  "before drawing conclusions about extraction accuracy."])
    return "\n".join(lines)
