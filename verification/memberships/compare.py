"""Compare native components and exact unweighted cluster editing at equal coverage.

Connected components factor the objective: joining disconnected components can
only add unsupported pairs. Enumerate all partitions inside each component.
Break ties by the lexicographically first membership assignment in source order.
Gold is consulted only after choosing the partitions.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from replay import capture, document, edition


def partitions(n):
    def extend(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for label in range(max(prefix) + 2):
            yield from extend(prefix + [label])
    yield from extend([0])


def predictions(cli, names):
    docs = [document(str(i), name) for i, name in enumerate(names)]
    snapshot = capture(cli, edition(1, docs))
    groups = defaultdict(list)
    observed = set()
    for key, row in snapshot["rows"]:
        i = int(key[0])
        groups[row["component"]].append(i)
        observed.update((i, j) for edge in row["direct"] if i < (j := int(edge["track"][0])))
    native, edited = set(), set()
    ties = False
    for members in groups.values():
        members.sort()
        if len(members)>9:
            raise ValueError("exact enumeration limited to nine tracks per connected component")
        pairs = list(itertools.combinations(members, 2))
        native.update(pairs)
        rank = {v:i for i,v in enumerate(members)}
        cost, optimum, overlap = float("inf"), None, 0
        for assignment in partitions(len(members)):
            joined = {(i,j) for i,j in pairs if assignment[rank[i]] == assignment[rank[j]]}
            loss = sum((p in joined) != (p in observed) for p in pairs)
            if loss < cost:
                cost, optimum, overlap = loss, joined, 1
            elif loss == cost:
                overlap += 1
        edited.update(optimum)
        ties |= overlap>1
    return {"native": native, "cluster_edit": edited}, ties


def evaluate(cli, split, topics):
    counts = {name:dict(false=0, missed=0, deferred=0) for name in ["native","cluster_edit"]}
    total = positives = ties = 0
    for names, ids, pairs in topics:
        decisions, tied = predictions(cli,names)
        gold = {p for p in pairs if ids[p[0]] == ids[p[1]]}
        total += len(pairs)
        positives += len(gold)
        ties += tied
        for method, joined in decisions.items():
            predictions_here = joined & pairs
            counts[method]["false"] += len(predictions_here - gold)
            counts[method]["missed"] += len(gold - predictions_here)
    for result in counts.values():
        result["discovery_loss_FP_plus_2FN"] = result["false"] + 2 * result["missed"]
        result["attribution_loss_8FP_plus_FN"] = 8 * result["false"] + result["missed"]
    print(json.dumps(dict(split=split, pairs=total, gold=positives, coverage=1,
                          topics_with_tied_optima=ties, methods=counts)))


def real_roster(cli, path):
    # The existing public CC0 extraction supplies dated official names and Bioguide IDs.
    if hashlib.sha256(path.read_bytes()).hexdigest() != "f6c4246cdaff7b3094ae00cf820b9ed27462970c75bda0319460e0350290a337":
        raise ValueError("use the roster.jsonl pinned by the Congress pilot")
    releases = [json.loads(line) for line in path.read_text().splitlines()]
    observations = []
    for year in [2021,2023,2025]:
        roster = {}
        cut = f"{year}-07-01"
        for change in releases:
            if change["published_at"][:10]>=cut:
                break
            for id in change["remove"]:
                del roster[id]
            roster.update((r["id"],r) for r in change["upsert"])
        # Select solely on printed first token and use all matching rows. Sort by
        # observed chamber/name; ties keep source order. Never order by person ID.
        sample = [r for r in roster.values() if r["official_full"] and
                  r["official_full"].split()[0] in {"Mark","Robert"}]
        for record in sorted(sample,key=lambda r:(r["chamber"],r["official_full"])):
            observations.append((year,record["official_full"],record["id"]))
        if year==2021:
            continue
        pairs = {(i,j) for i,j in itertools.combinations(range(len(observations)),2)
                 if observations[i][0]<year and observations[j][0]==year}
        evaluate(cli,"roster-"+cut,[([r[1] for r in observations],[r[2] for r in observations],pairs)])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("cli")
    parser.add_argument("--roster",type=Path)
    args=parser.parse_args()
    panel=json.loads(Path(__file__).with_name("panel.json").read_text())
    for split in ["development","later"]:
        topics=[]
        for topic in panel[split]["topics"]:
            names,ids=zip(*topic["rows"])
            topics.append((names,ids,set(itertools.combinations(range(len(names)),2))))
        evaluate(args.cli,split,topics)
    if args.roster:
        real_roster(args.cli,args.roster)
