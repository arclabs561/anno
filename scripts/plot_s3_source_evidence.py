# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.8", "numpy==2.3.5", "pillow==13.0.0"]
# ///
"""Plot the live cache-to-publisher comparison using its aggregate CI receipt."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The comparator gives each recognized legacy/snapshot object exactly one outcome.
# Pointers, manifests and unrecognized names never enter these denominators.
OUTCOMES = (
    ("Same dataset and publisher file", "read", {"matches_own_publisher_file"}),
    (
        "File matched a different dataset only",
        "read",
        {"matches_other_registry_file_only"},
    ),
    ("Read, no match to measured files", "read", {"no_matching_publisher_digest"}),
    (
        "Read, no reference hash for this ID",
        "read",
        {"no_own_publisher_digest_and_no_cross_id_match"},
    ),
    (
        "Skipped, no corpus measurement",
        "skipped",
        {"no_independent_corpus_measurement"},
    ),
    (
        "Skipped, source access or scope",
        "skipped",
        {
            "shared_read_exclusion",
            "requires_separate_source_access",
            "public_release_covers_only_part_of_source",
        },
    ),
    (
        "Read failed or bytes changed",
        "failed",
        {
            "access_denied",
            "changed_after_listing",
            "s3_read_error",
            "read_session_unavailable",
            "size_changed",
            "snapshot_digest_mismatch",
        },
    ),
)


def summary_counts(summary):
    observed = summary["payload_objects_by_outcome"]
    kinds = summary["objects_by_kind"]
    ids = summary["registry_ids_with_payload"]
    matched = summary["registry_ids_with_own_publisher_match"]
    own = observed.get("matches_own_publisher_file", 0)
    values = [
        *kinds.values(),
        *observed.values(),
        ids,
        matched,
        summary["objects_listed"],
    ]
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("Inventory and payload outcomes require nonnegative integers")
    n = kinds.get("legacy", 0) + kinds.get("snapshot", 0)
    if (
        set(observed) - set.union(*(keys for _, _, keys in OUTCOMES))
        or sum(observed.values()) != n
        or sum(kinds.values()) != summary["objects_listed"]
    ):
        raise ValueError("Every registered payload needs one known outcome")
    if not (matched <= ids <= n and matched <= own and bool(own) == bool(matched) and bool(ids) == bool(n)):
        raise ValueError("Inconsistent distinct dataset and payload match counts")
    totals = {"payload": n, "own": own, "read": 0, "skipped": 0, "failed": 0}
    grouped = {}
    for label, status, keys in OUTCOMES:
        count = sum(observed.get(key, 0) for key in keys)
        grouped[label] = count
        totals[status] += count
    return totals, grouped


def draw(summary):
    totals, grouped = summary_counts(summary)
    kinds = summary["objects_by_kind"]
    counts = list(grouped.values())
    n, read, own = totals["payload"], totals["read"], totals["own"]
    ids = summary["registry_ids_with_payload"]
    matched = summary["registry_ids_with_own_publisher_match"]
    fig, ax = plt.subplots(figsize=(8.2, 5.25))
    fig.subplots_adjust(left=0.405, right=0.92, bottom=0.21, top=0.65)
    ink = "#203945"
    fig.text(
        0.035,
        0.92,
        "Which cached inputs match a measured publisher file?",
        size=16,
        weight="bold",
        color=ink,
    )
    fig.text(0.035, 0.825, "Payload objects", size=12, weight="bold", color=ink)
    fig.text(
        0.306,
        0.825,
        f"{n:,} recognized   →   {read:,} read   →   {own:,} own-publisher matches",
        size=11.3,
        color=ink,
    )
    fig.text(0.035, 0.745, "Distinct registry IDs", size=12, weight="bold", color=ink)
    fig.text(
        0.306,
        0.745,
        f"{ids:,} with a payload   →   {matched:,} with an own-publisher match",
        size=11.3,
        color=ink,
    )
    y = list(range(len(counts)))
    bars = ax.barh(
        y,
        counts,
        height=0.56,
        color=[
            "#16756c",
            "#438b82",
            "#b57830",
            "#c3965e",
            "#9fadb2",
            "#6c8491",
            "#964b45",
        ],
    )
    ax.bar_label(
        bars, labels=[str(c) for c in counts], padding=4, fontsize=10, color=ink
    )
    ax.set_yticks(y, list(grouped), fontsize=9.5)
    ax.set_xlim(0, 1.15 * max(1, max(counts)))
    ax.invert_yaxis()
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, colors=ink)
    ax.set_xlabel("Payload objects in each outcome", size=10)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e3e9eb")
    fig.text(
        0.035,
        0.047,
        f"{summary['listing_started_utc'][:10]} UTC   ·   {summary['objects_listed']:,} objects listed   ·   "
        f"{kinds.get('unrecognized', 0):,} unrecognized keys counted only in the listing",
        size=9.7,
        color="#49616b",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    figure = draw(json.loads(args.receipt.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        figure.savefig(args.output / f"s3-publisher-matches.{ext}", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
