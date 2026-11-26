# SPDX-License-Identifier: MIT OR Apache-2.0
"""Attribute exact gold links to the first missed FCoref stage on the emitted-boundary panel."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from coref_gap import decode_clusters, index_partition
from gap_receipts import digest


REFERENCE = Path(__file__).resolve().parents[2] / "docs/eval/fcoref-emitted-boundaries.json"
STAGES = ("no_exact_candidate", "below_stock_budget", "retained_but_not_emitted",
          "emitted_separately", "linked")
ROLES = ("pronoun", "name", "both")
# Rust str::trim uses Unicode White_Space. Python's default strip also removes U+001C..U+001F.
WHITE_SPACE = "\t\n\v\f\r \u0085\u00a0\u1680" + "".join(chr(i) for i in range(0x2000, 0x200B)) + "\u2028\u2029\u202f\u205f\u3000"


def span_ranks(text, encoding, trace):
    """Find the best native rank mapping to each nonempty source-character span."""
    k, candidates = trace
    offsets, mask = encoding.offsets, encoding.attention_mask
    if len(offsets) != len(mask) or any(bit not in (0, 1) for bit in mask):
        raise ValueError("Invalid tokenizer attention mask")
    if any(not 0 <= a <= b <= len(text) for a, b in offsets):
        raise ValueError("Tokenizer returned invalid source offsets")
    if type(k) is not int or k != max(1, (sum(mask) + 3) // 4):
        raise ValueError("Unexpected stock cutoff")
    legal = set()
    for i in range(len(mask)):
        for j in range(i, min(i + 30, len(mask))):
            if not mask[j]:
                break
            legal.add((i, j))
    if len(candidates) != len(legal):
        raise ValueError("Incomplete native candidate table")

    best = {}
    previous_score = None
    previous_pair = None
    for rank, (i, j, logit) in enumerate(candidates, 1):
        pair = (i, j)
        if type(i) is not int or type(j) is not int or pair not in legal:
            raise ValueError("Illegal or repeated token pair in trace")
        legal.remove(pair)
        if type(logit) not in (float, int) or not math.isfinite(logit):
            raise ValueError("Invalid native logit")
        # Rust f32::total_cmp distinguishes signed zero. Stable ties retain (i,j) order.
        score_key = (logit, math.copysign(1, logit))
        if previous_score is not None and (score_key > previous_score or
                (score_key == previous_score and pair < previous_pair)):
            raise ValueError("Candidate trace differs from the native rank order")
        previous_score, previous_pair = score_key, pair
        start, end = offsets[i][0], offsets[j][1]
        raw = text[start:end]
        leading_trimmed = raw.lstrip(WHITE_SPACE)
        surface = leading_trimmed.rstrip(WHITE_SPACE)
        if surface:
            start += len(raw) - len(leading_trimmed)
            best.setdefault((start, start + len(surface)), rank)
    return best


def endpoint_role(pronoun, name):
    return "both" if pronoun and name else "pronoun" if pronoun else "name"


def classify(pronoun, name, ranks, k, clusters):
    """The role identifies the endpoint(s) first lost at the reported stage."""
    endpoints = (pronoun, name)
    for stage, missing in (
        (STAGES[0], [s not in ranks for s in endpoints]),
        (STAGES[1], [ranks.get(s, 0) > k for s in endpoints]),
        (STAGES[2], [s not in clusters for s in endpoints]),
    ):
        if any(missing):
            return stage, endpoint_role(*missing)
    return ("linked" if clusters[pronoun] == clusters[name] else "emitted_separately"), None


def strictly_contains(outer, inner):
    return outer != inner and outer[0] <= inner[0] and inner[1] <= outer[1]


def rival_directions(spans, name):
    longer = any(strictly_contains(s, name) for s in spans)
    shorter = any(strictly_contains(name, s) for s in spans)
    if longer and shorter:
        return "both"
    if longer:
        return "longer"
    return "shorter" if shorter else "none"


def marked(row, field):
    start = int(row[field + "-offset"])
    value = row[field]
    end = start + len(value)
    if not value or start < 0 or row["Text"][start:end] != value:
        raise ValueError("Invalid marked GAP endpoint")
    return start, end


def score(rows, records, tokenizer, *, reference_path=REFERENCE):
    """Score a sequence of {id, stock, trace} records against the pinned native run."""
    reference = json.loads(reference_path.read_text())
    partitions = []
    stages = dict.fromkeys(STAGES, 0)
    roles = {stage: dict.fromkeys(ROLES, 0) for stage in STAGES[:3]}
    ranks_below_k = {role: [] for role in ("pronoun", "name")}
    # Each positive pair contributes once, even when several rivals overlap its name.
    rivals = {key: dict.fromkeys(("longer", "shorter", "both", "none"), 0)
              for key in ("among_retained", "among_pronoun_cluster")}
    would_reach = 0
    confusion = Counter()
    for row, record in zip(rows, records, strict=True):
        doc_id, text = row["ID"], row["Text"]
        if record["id"] != doc_id:
            raise ValueError("Trace document mismatch")
        groups = decode_clusters(record["stock"], doc_id, text, "fcoref")
        clusters = index_partition(groups)
        partitions.append((doc_id, sorted(sorted(g) for g in groups)))
        k = record["trace"][0]
        ranks = span_ranks(text, tokenizer.encode(text), record["trace"])
        retained = {s for s, rank in ranks.items() if rank <= k}
        if not retained.issuperset(clusters):
            raise ValueError("Emitted span absent from retained candidates")

        pronoun = marked(row, "Pronoun")
        for field in ("A", "B"):
            label = row[field + "-coref"]
            if label not in ("TRUE", "FALSE"):
                raise ValueError("Invalid marked GAP link label")
            name = marked(row, field)
            stage, role = classify(pronoun, name, ranks, k, clusters)
            if label == "FALSE":
                confusion["fp" if stage == "linked" else "tn"] += 1
                continue
            confusion["tp" if stage == "linked" else "fn"] += 1
            stages[stage] += 1
            if role:
                roles[stage][role] += 1
            if stage == "below_stock_budget":
                would_reach += max(ranks[pronoun], ranks[name]) <= 2 * k
                if ranks[name] > k:
                    group = groups[clusters[pronoun]] if pronoun in clusters else ()
                    rivals["among_retained"][rival_directions(retained, name)] += 1
                    rivals["among_pronoun_cluster"][rival_directions(group, name)] += 1
                for key, span in (("pronoun", pronoun), ("name", name)):
                    if ranks[span] > k:
                        ranks_below_k[key].append(ranks[span] - k)

    cohort_digest = hashlib.sha256("\n".join(doc_id for doc_id, _ in partitions).encode()).hexdigest()
    partition_digest = digest(partitions)
    if cohort_digest != reference["cohort_ids_sha256"] or partition_digest != reference["stock_partitions_sha256"]:
        raise ValueError("Cohort or returned clusters differ from the frozen stock run")
    stock = reference["stock"]
    if any(confusion[key] != stock[key] for key in ("tp", "fp", "fn", "tn")) or (
            stages["linked"] + stages["emitted_separately"] != stock["gold_both_covered"]):
        raise ValueError("Marked decisions differ from the frozen stock run")

    return dict(documents=len(partitions), cohort_ids_sha256=cohort_digest,
                stock_partitions_sha256=partition_digest, gold_links=sum(stages.values()),
                stock_decisions={key: confusion[key] for key in ("tp", "fp", "fn", "tn")},
                first_stage=stages, endpoint_at_first_loss=roles,
                below_budget_pairs_with_both_in_first_2k=would_reach,
                below_budget_name_rivals={"gold_pairs": sum(rivals["among_retained"].values()), **rivals},
                ranks_below_stock_cutoff={k: dict(count=len(v), median=median(v), min=min(v), max=max(v))
                                          for k, v in ranks_below_k.items() if v})
