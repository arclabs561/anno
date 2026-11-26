"""Render Anno dataset figures (requires Matplotlib >= 3.6 and upstream HTTP access).

Usage: python scripts/plot_datasets.py OUTPUT_DIRECTORY
The CSV records per-file read failures. Remote URLs may change between runs.
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import re
import csv
import urllib.request
import collections
import concurrent.futures
import pathlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=pathlib.Path)
root = parser.parse_args().output
root.mkdir(exist_ok=True, parents=True)
repo = pathlib.Path(__file__).resolve().parents[1]
task_counts = collections.Counter()
comparison = {}

registry = (
    (repo / "crates/anno-eval/src/eval/dataset_registry.rs")
    .read_text()
    .split("\ndefine_datasets! {")[1]
)
rows = []
for key, entry in re.findall(r"^    (\w+) \{\n(.*?)^    \},?", registry, re.M | re.S):
    row = dict(
        re.findall(
            r'^        (name|url|format|domain|language): "([^"\n]*)"', entry, re.M
        )
    )
    if "name" not in row:
        continue
    row["id"] = key
    access_match = re.search(r"access_status: (\w+)", entry)
    row["access"] = access_match[1] if access_match else "Public"
    row.update(result="Registry metadata only", tokens="", tagged="", bytes="")
    tasks = re.search(r"^        tasks: \[(.*?)\]", entry, re.M | re.S)
    if tasks:
        task_counts.update(set(re.findall(r'"([^"]+)"', tasks[1])))
    rows.append(row)
if not rows:
    raise ValueError("No registry entries found")


def measure(row):
    url = row.get("url", "")
    if not (
        "raw.githubusercontent.com/" in url
        or "/resolve/" in url
        or re.search(r"\.(?:conllu|tsv|jsonl|csv|txt|zip|gz|json)$", url)
    ) or not url.startswith("https://"):
        return
    try:
        with urllib.request.urlopen(url, timeout=20) as f:
            data = f.read()
        row["bytes"] = len(data)
        row["result"] = "File read"
        if data.startswith(b"PK") or b"\0" in data[:1000]:
            return
        records = [
            line.split()
            for line in data.decode().splitlines()
            if line.strip() and not line.startswith(("-DOCSTART-",))
        ]
        if row["id"] == "ZELDA":
            records = [record for record in records if record[0] != "#"]
        if row["id"] == "GENIANested":
            records = [
                [record[0], next((t for t in record[1:] if t != "O"), "O")]
                for record in records
            ]
        if not any(
            record[-1].startswith(("B-", "I-")) for record in records
        ) or not all(
            len(record) >= 2 and re.fullmatch(r"O|[BI]-.+", record[-1])
            for record in records
        ):
            return
        row.update(
            tokens=len(records),
            tagged=sum(record[-1] != "O" for record in records),
            result="BIO token counts",
        )
        if row["name"] in ("WikiGold", "WNUT-16"):
            mentions = collections.Counter()
            words = []
            last = None
            for line in data.decode().splitlines() + [""]:
                parts = line.split()
                tag = parts[-1] if parts and parts[0] != "-DOCSTART-" else "O"
                kind = tag[2:] if tag != "O" else None
                if words and (kind != last or tag.startswith("B-")):
                    mentions[" ".join(words).casefold()] += 1
                    words = []
                if kind:
                    words.append(parts[0])
                last = kind
            comparison[row["name"]] = (
                100 * row["tagged"] / row["tokens"],
                100 * sum(n == 1 for n in mentions.values()) / len(mentions),
            )
    except UnicodeDecodeError:
        pass
    except Exception as e:
        row["result"] = str(e)[:80]


with concurrent.futures.ThreadPoolExecutor(10) as ex:
    list(ex.map(measure, rows))
with (root / "anno-all.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
measured_rows = sorted(
    (row for row in rows if row["tokens"]),
    key=lambda row: row["tagged"] / row["tokens"],
)
print(
    collections.Counter(row["result"] for row in rows),
    [
        (row["name"], round(100 * row["tagged"] / row["tokens"], 1))
        for row in measured_rows
    ],
)
fig, (access_axis, density_axis) = plt.subplots(
    1, 2, figsize=(13, 6), layout="constrained"
)
groups = collections.Counter(row["access"] for row in rows).most_common()[::-1]
for axis, labels, values, title, color in [
    (
        access_axis,
        [x for x, n in groups],
        [n for x, n in groups],
        f"Registry access labels · {len(rows)} entries",
        "#315b75",
    ),
    (
        density_axis,
        [row["name"] for row in measured_rows],
        [100 * row["tagged"] / row["tokens"] for row in measured_rows],
        "Entity-tagged tokens · % of file",
        "#008675",
    ),
]:
    bars = axis.barh(labels, values, color=color, height=0.63)
    axis.bar_label(bars, fmt="%.1f" if axis == density_axis else "%d", padding=4)
    axis.margins(x=0.15)
    axis.set_title(title, loc="left")
    axis.spines[["top", "right", "left"]].set_visible(False)
fig.suptitle("Anno dataset coverage and annotation density", fontsize=18)
for ext in ("png", "svg"):
    fig.savefig(root / ("anno-all." + ext), dpi=150)


def bar_chart(name, title, counts):
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    labels, values = zip(*counts)
    bars = axis.barh(labels, values, color="#187673")
    axis.invert_yaxis()
    axis.bar_label(bars, padding=4)
    axis.margins(x=0.13)
    axis.set_title(title, loc="left")
    for ext in ("png", "svg"):
        fig.savefig(root / f"{name}.{ext}", dpi=150)
    plt.close(fig)


tags = ["ner", "coref", "el", "re", "temporal"]
bar_chart(
    "anno-tasks",
    f"Selected explicit task tags across {len(rows)} registry entries",
    [(tag, task_counts[tag]) for tag in tags],
)
fixture = repo / "crates/anno-eval/fixtures/pii_masking_fixture.jsonl"
counts = collections.Counter(
    entity["type"]
    for line in fixture.read_text().splitlines()
    for entity in json.loads(line)["entities"]
)
bar_chart(
    "anno-pii", "Annotated spans in the PII masking fixture", counts.most_common()
)
names = ["WikiGold", "WNUT-16"]
if any(name not in comparison for name in names):
    raise SystemExit(
        "Corpus comparison incomplete. See anno-all.csv for source status."
    )
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for i, title in enumerate(
    ["Tokens inside entity mentions", "Distinct mention strings seen once"]
):
    bars = axes[i].bar(
        names, [comparison[name][i] for name in names], color=["#176e70", "#d58642"]
    )
    axes[i].set_ylim(0, 100)
    axes[i].bar_label(bars, fmt="%.1f%%")
    axes[i].set_title(title)
for ext in ("png", "svg"):
    fig.savefig(root / f"anno-corpora.{ext}", dpi=150)
