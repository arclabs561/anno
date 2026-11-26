"""Plot the aggregate JSON emitted by the cache_inventory verifier.

Usage: python3 scripts/plot_s3_cache_inventory.py INVENTORY.json OUTPUT_DIRECTORY
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#243748"
MUTED = "#506373"
TEAL = "#216e69"
OBJECTS = {
    "unrecognized_keys": ("Unrecognized names", "#8b989f"),
    "registered_legacy_payloads": ("Legacy payloads", TEAL),
    "registered_snapshots": ("Payload snapshots", TEAL),
    "registered_pointers": ("Pointers", "#52749a"),
    "registered_manifests": ("Manifests", "#52749a"),
}
POLICIES = {
    "payloads_excluded_from_shared_reads": "Excluded from shared reads",
    "payloads_requiring_local_source": "Require a local source",
    "payloads_approved_for_shared_publication": "Approved for shared publication",
}


def check_counts(report):
    verified = "payloads_with_digest_matched_rights_evidence"
    if not isinstance(report, dict):
        raise ValueError(
            "Expected the aggregate JSON object emitted by cache_inventory"
        )
    for key in (*OBJECTS, *POLICIES, "objects", "registry_ids_with_payload", verified):
        if type(report.get(key)) is not int or report[key] < 0:
            raise ValueError(f"Expected a nonnegative inventory count for {key}")
    if sum(report[key] for key in OBJECTS) != report["objects"]:
        raise ValueError("The five object classes must partition the listing")
    payloads = report["registered_legacy_payloads"] + report["registered_snapshots"]
    if any(report[key] > payloads for key in (*POLICIES, "registry_ids_with_payload")):
        raise ValueError(
            "A registered payload count exceeds the number of payload objects"
        )
    if report[verified] > min(1, payloads):
        raise ValueError(
            "This verifier checks the bytes of one WikiGold payload at most"
        )
    return payloads


def draw(report):
    payloads = check_counts(report)
    fig = plt.figure(figsize=(9, 4.35))
    fig.text(
        0.025,
        0.90,
        "What did the S3 cache verifier find?",
        size=20,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.025,
        0.82,
        f"{report['objects']:,} objects listed. Registered payload names resolve to "
        f"{report['registry_ids_with_payload']:,} distinct dataset IDs.",
        size=11.5,
        color=INK,
    )
    fig.text(
        0.025,
        0.69,
        "Objects by registry classification",
        size=13,
        weight="bold",
        color=INK,
    )
    ax = fig.add_axes([0.207, 0.250, 0.275, 0.365])
    counts = [report[key] for key in OBJECTS]
    bars = ax.barh(
        range(len(counts)), counts, height=0.58, color=[c for _, c in OBJECTS.values()]
    )
    ax.bar_label(
        bars,
        labels=[f"{count:,}" for count in counts],
        padding=4,
        fontsize=10.5,
        color=INK,
    )
    ax.set_xlim(0, max(max(counts), 1) * 1.22)
    ax.set_ylim(len(counts) - 0.5, -0.5)
    ax.set_yticks(range(len(counts)), [label for label, _ in OBJECTS.values()])
    ax.tick_params(axis="y", labelcolor=INK, length=0, labelsize=10.5)
    ax.set_xticks([])
    ax.spines[:].set_visible(False)

    fig.text(
        0.54, 0.69, "Registered payload policies", size=13, weight="bold", color=INK
    )
    fig.text(
        0.54,
        0.605,
        f"Counts can overlap across {payloads:,} payload objects.",
        size=10,
        color=MUTED,
    )
    for index, (key, label) in enumerate(POLICIES.items()):
        y = 0.505 - 0.093 * index
        fig.text(0.54, y, label, size=10.5, color=INK)
        fig.text(
            0.955, y, f"{report[key]:,}", ha="right", size=12, weight="bold", color=INK
        )

    verified = report["payloads_with_digest_matched_rights_evidence"]
    evidence = (
        "1 WikiGold payload matched the SHA-256 of the reviewed release."
        if verified
        else "0 WikiGold payloads were verified against the reviewed release."
    )
    fig.text(0.025, 0.125, evidence, color=TEAL, fontsize=12, weight="bold")
    fig.text(
        0.025,
        0.051,
        "The listing counts come from object names. Byte verification requires reading a payload.",
        size=10,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fig = draw(json.loads(args.inventory.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "s3-cache-inventory.png", dpi=160)
    with plt.rc_context(
        {"svg.hashsalt": "anno-s3-cache-inventory", "svg.fonttype": "none"}
    ):
        fig.savefig(args.output / "s3-cache-inventory.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
