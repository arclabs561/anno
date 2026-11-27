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

OUTCOMES = {
    "Same dataset and publisher file": {"matches_own_publisher_file"},
    "File matched a different dataset only": {"matches_other_registry_file_only"},
    "Read, no match to measured files": {"no_matching_publisher_digest"},
    "Read, no reference hash for this ID": {"no_own_publisher_digest_and_no_cross_id_match"},
    "Skipped, no corpus measurement": {"no_independent_corpus_measurement"},
    "Skipped, source access or scope": {
        "shared_read_exclusion", "requires_separate_source_access", "public_release_covers_only_part_of_source"
    },
    "Read failed or bytes changed": {
        "access_denied", "changed_after_listing", "s3_read_error", "read_session_unavailable",
        "size_changed", "snapshot_digest_mismatch"
    },
}


def draw(summary):
    observed = summary["payload_objects_by_outcome"]
    kinds = summary["objects_by_kind"]
    n = kinds.get("legacy", 0) + kinds.get("snapshot", 0)
    if set(observed) - set.union(*OUTCOMES.values()) or sum(observed.values()) != n:
        raise ValueError("Every registered payload needs one known outcome")
    counts = [sum(observed.get(key, 0) for key in keys) for keys in OUTCOMES.values()]
    ids = summary["registry_ids_with_payload"]
    matched = summary["registry_ids_with_own_publisher_match"]
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    fig.subplots_adjust(left=0.38, right=0.93, bottom=0.20, top=0.74)
    ink = "#203945"
    fig.text(.035, .92, "Which cached inputs match a measured publisher file?", size=16, weight="bold", color=ink)
    fig.text(.035, .85, f"{matched} of {ids} cached registry IDs have a file matching their own publisher reference.", size=11, color=ink)
    y = list(range(len(counts)))
    bars = ax.barh(y, counts, height=.56, color=["#16756c", "#438b82", "#b57830", "#c3965e", "#9fadb2", "#6c8491", "#964b45"])
    ax.bar_label(bars, labels=[str(c) for c in counts], padding=4, fontsize=10, color=ink)
    ax.set_yticks(y, list(OUTCOMES), fontsize=9.5)
    ax.set_xlim(0, 1.15 * max(1, max(counts)))
    ax.invert_yaxis()
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, colors=ink)
    ax.set_xlabel(f"Registered payload objects ({n} legacy files and snapshots combined)", size=9)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e3e9eb")
    fig.text(.035, .045,
             f"{summary['listing_started_utc'][:10]} UTC   ·   {summary['objects_listed']} objects listed   ·   "
             f"{summary['publisher_digests_compared']:,} independent publisher digests", size=9, color="#49616b")
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
