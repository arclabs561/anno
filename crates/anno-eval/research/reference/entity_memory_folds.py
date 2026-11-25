"""Run the frozen entity-memory pilot across the original LitBank folds."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from entity_memory_replay import (
    POLICIES, compare_wrong_attachments, load_conll, load_features, run_experiment, summarize,
)


def evaluate_fold(task):
    fold, splits, corpus, features, dimension, budget = task

    def load(ids):
        documents = [load_conll(corpus / f"{name}.conll") for name in sorted(ids)]
        return [(doc, load_features(doc, dimension, features)) for doc in documents]

    dev, test = load(splits["dev"]), load(splits["test"])
    return fold, run_experiment(dev, test, budget, 8, 128, 0.35)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--encoder-id")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if bool(args.features) != bool(args.encoder_id):
        parser.error("--features and --encoder-id must be provided together")
    if args.jobs < 1 or args.dimension < 1:
        parser.error("jobs and dimension must be positive")
    budget = 24 * (args.dimension * 8 + 24)
    folds = {}
    for fold in range(10):
        folds[fold] = {}
        for split in ("dev", "test"):
            ids = (args.splits / str(fold) / f"{split}.ids").read_text(encoding="utf-8").splitlines()
            # Reserve this BookCoref gold work for a potential scale follow-up.
            folds[fold][split] = [
                Path(name).stem for name in ids if name.strip() and not name.startswith("1342_")
            ]
        if set(folds[fold]["dev"]) & set(folds[fold]["test"]):
            raise ValueError(f"overlapping development and test books in fold {fold}")
    test_ids = [name for splits in folds.values() for name in splits["test"]]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("a work appears in more than one test fold")

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        runs = dict(pool.map(evaluate_fold, (
            (fold, splits, args.corpus, args.features, args.dimension, budget)
            for fold, splits in folds.items()
        )))
    closed_loop = {}
    oracle_history = {}
    for policy in POLICIES:
        for stage, destination in (
            ("closed_loop_documents", closed_loop), ("oracle_documents", oracle_history)
        ):
            destination[policy] = [
                row for run in runs.values() for row in run[policy][stage]
            ]
    print(json.dumps({
        "test_works": len(test_ids),
        "encoder": args.encoder_id or "blake2b-lexical-prefix-v1",
        "dimension": args.dimension,
        "budget": budget,
        "folds": runs,
        "closed_loop": {policy: summarize(rows) for policy, rows in closed_loop.items()},
        "oracle_history": {policy: summarize(rows) for policy, rows in oracle_history.items()},
        "paired_closed_loop": {
            policy: compare_wrong_attachments(closed_loop["box"], closed_loop[policy])
            for policy in POLICIES if policy != "box"
        },
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
