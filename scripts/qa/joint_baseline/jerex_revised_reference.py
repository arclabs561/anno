# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check previously unannotated aligned JEREX predictions against independent Re-DocRED labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from aggregate_receipt import equal, project
from jerex_loss import gold_clusters
from jerex_stages import SPLIT, digest

REVISION = "ccfb54f5ddf5836027c87badda10f6dfc56efaac"
FILES = {
    "dev_revised.json": "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f",
    "test_revised.json": "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c",
    "train_revised.json": "c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2",
}


def classify(source, predictions, revised):
    if source["title"] != revised["title"] or source["sents"] != revised["sents"]:
        raise ValueError("revised document has a different title or source token sequence")
    old, new = gold_clusters(source), gold_clusters(revised)
    original_labels = {(r["h"], r["r"], r["t"]) for r in source["labels"]}
    revised_labels = {(r["h"], r["r"], r["t"]) for r in revised["labels"]}
    summary = Counter()
    for edge in predictions:
        if edge["outcome"] != "directed_pair_unannotated":
            continue
        head, = edge["head_alignment"]["exact"]
        tail, = edge["tail_alignment"]["exact"]
        if any(h == head and t == tail for h, _, t in original_labels):
            raise ValueError("baseline directed pair was actually annotated")
        heads = [i for i, members in enumerate(new) if members == old[head]]
        tails = [i for i, members in enumerate(new) if members == old[tail]]
        if len(heads) != 1 or len(tails) != 1:
            outcome = "endpoint_membership_changed_or_ambiguous"
        elif (heads[0], edge["predicate"], tails[0]) in revised_labels:
            outcome = "exact_directed_predicate_added"
        elif any(h == heads[0] and t == tails[0] for h, _, t in revised_labels):
            outcome = "another_predicate_on_revised_pair"
        else:
            outcome = "directed_pair_still_unannotated"
        summary[outcome] += 1
    return summary


def load_revised(directory, documents):
    wanted = {doc["title"] for doc in documents}
    lookup = {}
    for name, sha in FILES.items():
        path = directory / name
        equal(digest(path), sha, "unexpected revised edition")
        for row in json.loads(path.read_bytes()):
            if row["title"] in wanted:
                if row["title"] in lookup:
                    raise ValueError("ambiguous revised title")
                lookup[row["title"]] = (name, row)
    return lookup


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("test_joint", type=Path)
    cli.add_argument("external_predictions_receipt", type=Path)
    cli.add_argument("revised_directory", type=Path)
    args = cli.parse_args()
    equal(digest(args.test_joint), SPLIT, "unexpected original split")
    receipt = json.loads(args.external_predictions_receipt.read_bytes())
    published = Path(__file__).parent / "receipts/jerex_predictions32.json"
    equal(project(receipt, "jerex-predictions"), json.loads(published.read_bytes()), "unexpected prediction receipt")
    source = json.loads(args.test_joint.read_bytes())[:len(receipt["documents"])]
    lookup = load_revised(args.revised_directory, source)
    totals, documents = Counter(), Counter()
    for doc, row in zip(source, receipt["documents"], strict=True):
        equal(doc["title"], row["document"], "unexpected source order")
        name, revised = lookup[doc["title"]]
        totals.update(classify(doc, row["predictions"], revised))
        documents[name] += 1
    equal(sum(totals.values()), receipt["outcomes"]["directed_pair_unannotated"], "incomplete projection")
    print(json.dumps(dict(source_sha256=SPLIT, revised_revision=REVISION, documents_by_revised_split=documents,
                          aligned_predictions=sum(totals.values()), outcomes=totals), indent=2))


if __name__ == "__main__":
    main()
