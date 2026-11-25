"""Merge the five arm receipts per dataset and compute common-document LitBank scores."""

import argparse
import json
from pathlib import Path
from coref_litbank import summarize


def collect(directory):
    arms = [
        ("native", "mention-ranking", "predicted"),
        ("native", "mention-ranking", "supplied"),
        ("fcoref", "fcoref", "predicted"),
        ("maverick", "maverick", "predicted"),
        ("maverick", "maverick", "supplied"),
    ]
    result = {"models": {}, "gap": {"arms": {}}, "litbank": {"arms": {}}}
    for dataset in ["gap", "litbank"]:
        block = result[dataset]
        for stem, backend, mode in arms:
            filename = ("" if dataset == "gap" else "litbank-") + f"{stem}-{mode}.json"
            arm = json.loads((directory / filename).read_text())
            if arm["model"]["backend"] != backend or arm["mode"] != mode:
                raise ValueError(f"wrong arm in {filename}")
            model = arm.pop("model")
            if result["models"].setdefault(backend, model) != model:
                raise ValueError(f"inconsistent model identity for {backend}")
            arm.pop("mode")
            for key in ["revision", "sha256", "split"] + (
                ["ids", "gold_pairs"] if dataset == "gap" else []
            ):
                value = arm.pop(key)
                if block.setdefault(key, value) != value:
                    raise ValueError("different evaluation inputs")
            if dataset == "litbank":
                metadata = [
                    {
                        k: d[k]
                        for k in [
                            "id",
                            "text_sha256",
                            "tokens",
                            "sentences",
                            "source_chars",
                        ]
                    }
                    for d in arm["documents"]
                ]
                if block.setdefault("documents", metadata) != metadata:
                    raise ValueError("different document editions")
                arm["per_id"] = [
                    d["counts"] if "counts" in d else {"unsupported": d["unsupported"]}
                    for d in arm.pop("documents")
                ]
            block["arms"][backend + "/" + mode] = arm

    lit = result["litbank"]
    common = [
        i
        for i in range(len(lit["documents"]))
        if all("unsupported" not in arm["per_id"][i] for arm in lit["arms"].values())
    ]
    lit["common_ids"] = [lit["documents"][i]["id"] for i in common]
    for arm in lit["arms"].values():
        counts = {}
        for i in common:
            for k, values in arm["per_id"][i].items():
                before = counts.get(k, [0] * len(values))
                counts[k] = [a + b for a, b in zip(before, values, strict=True)]
        arm["common_scores"] = summarize(counts)
    labels = result["gap"]["gold_pairs"]
    arms = result["gap"]["arms"]
    fc = arms["fcoref/predicted"]["per_id"]
    mv = arms["maverick/predicted"]["per_id"]
    transitions = {
        "both_correct": 0,
        "fcoref_only": 0,
        "maverick_only": 0,
        "neither_correct": 0,
    }
    for a, b, g in zip(fc, mv, labels, strict=True):
        ok_a, ok_b = a[-2:] == g, b[-2:] == g
        key = (
            "both_correct"
            if ok_a and ok_b
            else "fcoref_only"
            if ok_a
            else "maverick_only"
            if ok_b
            else "neither_correct"
        )
        transitions[key] += 1
    result["gap"]["paired_raw_exact_decisions"] = transitions
    deltas = []
    for held_out in range(len(lit["documents"])):
        scores = []
        for name in ["fcoref/predicted", "maverick/predicted"]:
            rows = [
                c for i, c in enumerate(lit["arms"][name]["per_id"]) if i != held_out
            ]
            totals = {
                k: [sum(x) for x in zip(*(c[k] for c in rows), strict=True)]
                for k in rows[0]
            }
            scores.append(summarize(totals)["conll_f1"])
        deltas.append(round(scores[1] - scores[0], 6))
    lit["leave_one_work_out_raw_maverick_minus_fcoref_conll"] = deltas
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect(args.directory), indent=2) + "\n")
