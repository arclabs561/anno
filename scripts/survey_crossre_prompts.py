"""Count what training and per-sentence gold prompts expose in all CrossRE splits.

Run `python3 scripts/survey_crossre_prompts.py OUTPUT.csv`. Reuse downloaded
publisher files with `--cache-dir`. Each file is checked against the pinned
repository ledger. Only aggregate counts and source digests are written.
"""

import argparse
import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from survey_repo_files import MANIFEST, crossre_counts, file_contents


def read_split(entry, cache):
    data = file_contents(entry, cache)
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Digest mismatch for {entry['path']}")
    lines = data.splitlines()
    if crossre_counts(lines) != tuple(
        int(entry[k]) for k in ("rows", "tokens", "tagged")
    ):
        raise ValueError(f"Corpus counts disagree with the ledger for {entry['path']}")
    return [json.loads(line) for line in lines if line.strip()]


def typed_examples(records):
    """One list of directed (head type, predicate, tail type) rows per sentence."""
    all_entity_types = set()
    examples = []
    for row in records:
        entities = {}
        for start, end, label in row["ner"]:
            if entities.setdefault((start, end), label) != label:
                raise ValueError("Conflicting entity types at the same CrossRE span")
            all_entity_types.add(label)
        relations = []
        for head, head_end, tail, tail_end, label, *_ in row["relations"]:
            if not isinstance(label, str) or not label:
                raise ValueError("Missing CrossRE predicate label")
            if (head, head_end) not in entities or (tail, tail_end) not in entities:
                raise ValueError("CrossRE relation endpoint has no annotated entity")
            relations.append(
                (entities[head, head_end], label, entities[tail, tail_end])
            )
        examples.append(relations)
    return all_entity_types, examples


def split_counts(
    entity_types, examples, training_types, training_heads, training_tails
):
    empty = single = relations = unseen_predicate = unseen_endpoint = outside_roles = 0
    predicates = set()
    for sentence in examples:
        labels = {label for _, label, _ in sentence}
        empty += not labels
        single += len(labels) == 1
        predicates.update(labels)
        for head, label, tail in sentence:
            relations += 1
            unseen_predicate += label not in training_heads
            unseen_endpoint += head not in training_types or tail not in training_types
            outside_roles += head not in training_heads.get(
                label, ()
            ) or tail not in training_tails.get(label, ())
    return {
        "sentences": len(examples),
        "no_predicate_sentences": empty,
        "one_predicate_type_sentences": single,
        "multiple_predicate_type_sentences": len(examples) - empty - single,
        "entity_types": len(entity_types),
        "predicate_types": len(predicates),
        "gold_relations": relations,
        "relations_with_unseen_predicate": unseen_predicate,
        "relations_with_unseen_endpoint_type": unseen_endpoint,
        "relations_outside_train_predicate_roles": outside_roles,
    }


def survey(cache):
    with MANIFEST.open(newline="", encoding="utf-8") as source:
        domains = defaultdict(dict)
        for entry in csv.DictReader(source):
            if entry["id"] != "CrossRE":
                continue
            splits = domains[entry["subset"]]
            if entry["status"] != "ok" or entry["split"] in splits:
                raise ValueError(
                    f"Missing or duplicate CrossRE measurement for {entry['path']}"
                )
            splits[entry["split"]] = entry
    if not domains:
        raise ValueError("No CrossRE release files in the publisher ledger")
    rows = []
    for domain, files in sorted(domains.items()):
        if set(files) != {"train", "dev", "test"}:
            raise ValueError(f"Incomplete CrossRE train/dev/test files for {domain}")
        # Freeze the vocabulary using this domain's training data alone.
        training_types, train = typed_examples(read_split(files["train"], cache))
        heads, tails = defaultdict(set), defaultdict(set)
        for sentence in train:
            for head, label, tail in sentence:
                heads[label].add(head)
                tails[label].add(tail)
        for split in ("train", "dev", "test"):
            entry = files[split]
            types, examples = (
                (training_types, train)
                if split == "train"
                else typed_examples(read_split(entry, cache))
            )
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "sha256": entry["sha256"],
                    **split_counts(types, examples, training_types, heads, tails),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, help="Reuse the publisher file survey's local cache"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        cache = args.cache_dir or Path(temp)
        cache.mkdir(exist_ok=True, parents=True)
        rows = survey(cache)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
