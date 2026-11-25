"""Run CPU label-prototype and lexical typing controls on the UFET crowd splits."""
import argparse
import hashlib
import json
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SOURCE = "https://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz"
WORDNET = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
SPLITS = ("train", "dev", "test")
HASHES = {'train.json': '97827cfc639cad63005dcbbc656140ea7281c29880d93cd8a3f0f3601980d3a5', 'dev.json': '95d0d817ac80615747f8b57dc70005e42c5fd0e35a8268cbab1a8c56696bcc0a', 'test.json': '76cc77039bef0c635ffe7f956b6bcc76d1d299c35a7ca9bce5c5a3b1ed0fc11f', 'wordnet.zip': 'cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59'}


def prepare(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(not (path / (s + ".json")).exists() for s in SPLITS):
        wanted = {f"./release/crowd/{s}.json" for s in SPLITS}
        with urllib.request.urlopen(SOURCE) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if member.name in wanted:
                    (path / Path(member.name).name).write_bytes(archive.extractfile(member).read())
                    wanted.remove(member.name)
                if not wanted:
                    break
    if not (path / "wordnet.zip").exists():
        urllib.request.urlretrieve(WORDNET, path / "wordnet.zip")
    for name, digest in HASHES.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"unexpected digest for {name}")


def document(row):
    return row["annot_id"].rsplit(":", 2)[0]


def design(parts):
    all_rows = sum(parts, [])
    fields = [
        ([r["mention_span"] for r in all_rows], {"analyzer": "char_wb", "ngram_range": (2, 4)}, 2),
        ([" ".join(r["left_context_token"][-24:] + r["right_context_token"][:24]) for r in all_rows],
         {"ngram_range": (1, 2)}, 1),
    ]
    vectors = []
    for field, settings, weight in fields:
        encoder = TfidfVectorizer(sublinear_tf=True, **settings).fit(field[:len(parts[0])])
        vectors.append(encoder.transform(field) * weight)
    return normalize(hstack(vectors, format="csr"))


def labels(rows, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    y = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for n, r in enumerate(rows):
        for typ in r["y_str"]:
            if typ in idx:
                y[n, idx[typ]] = 1
    return y


def metrics(order, gold, gold_sizes, k):
    hits = np.take_along_axis(gold, order[:, :k], axis=1).sum(axis=1)
    p, r = hits.mean() / k, (hits / gold_sizes).mean()
    return dict(k=k, correct=int(hits.sum()), macro_f1=float(2*p*r/(p+r)) if p+r else 0.,
                micro_f1=float(2*hits.sum()/(gold_sizes.sum()+len(gold)*k)))


def prototype_scores(train, held, y):
    """Quantile envelopes and centroids, with the same latent coordinates."""
    scores = {name: [] for name in ("centroid", "midpoint", "box")}
    for col in y.T:
        samples = train[col.astype(bool)]
        lower, upper = np.quantile(samples, [0.1, 0.9], axis=0)
        center, mid = samples.mean(axis=0), (lower + upper) / 2
        middle = ((held-mid)**2).sum(axis=1)
        outside = np.maximum(np.maximum(lower-held, held-upper), 0)
        scores["box"].append(-((outside**2).sum(axis=1) + 0.25 * middle))
        scores["centroid"].append(-((held-center)**2).sum(axis=1))
        scores["midpoint"].append(-middle)
    return {k: np.array(v).T for k, v in scores.items()}


def taxonomy(path, vocab, y):
    """Check direct WordNet noun hypernyms against training-label evidence."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.read("wordnet/data.noun").decode().splitlines()
    lexemes, parents = {}, {}
    idx = {t: i for i,t in enumerate(vocab)}
    for line in entries:
        if not line or not line[0].isdigit():
            continue
        fields = line.split("|")[0].split()
        n = int(fields[3], 16)
        lexemes[fields[0]] = {w for w in fields[4:4+2*n:2] if w in idx}
        ptr = 5+2*n
        parents[fields[0]] = [fields[i+1] for i in range(ptr, ptr+4*int(fields[ptr-1]), 4)
                              if fields[i] in {"@", "@i"}]
    edges = {(idx[child], idx[parent]) for syn, src in lexemes.items() for dst in parents[syn]
             for child in src for parent in lexemes[dst] if child != parent}
    edges = sorted((a,b) for a,b in edges if (b,a) not in edges)
    counts = y.sum(axis=0)
    co = y.T @ y
    observed = [(a,b) for a,b in edges if co[a,b] > 0]
    freq = sum(counts[b] > counts[a] for a,b in observed)
    ties = sum(counts[b] == counts[a] for a,b in observed)
    cond = sum(co[a,b]/counts[a] > co[a,b]/counts[b] for a,b in observed)
    assert cond == freq
    return edges, dict(direct_edges=len(edges),coannotated=len(observed),
                other_edges_without_positive_coannotation=len(edges)-len(observed),
                strict_correct_conditional=int(cond),strict_correct_frequency=int(freq),ties=int(ties))


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("cache", type=Path)
    args = arg.parse_args()
    prepare(args.cache)
    raw = [[json.loads(x) for x in (args.cache/(s+".json")).read_text().splitlines()] for s in SPLITS]
    test_docs = set(map(document, raw[2]))
    dev = [r for r in raw[1] if document(r) not in test_docs]
    held_docs = test_docs | set(map(document, dev))
    train = [r for r in raw[0] if document(r) not in held_docs]
    test = raw[2]
    x = design([train,dev,test])
    vocab = sorted({t for r in train for t in r["y_str"]})
    y = labels(train,vocab)
    size, split = len(train), len(dev)
    projection = TruncatedSVD(32, random_state=7).fit(x[:size])
    latent = normalize(projection.transform(x))
    scores = prototype_scores(latent[:size], latent[size:], y)
    kernel = (x[:size] @ x[:size].T).toarray()
    kernel.flat[::size+1] += 1
    scores["ridge"] = (x[size:] @ x[:size].T) @ solve(kernel,y,assume_a="pos")
    freq = np.log(y.sum(axis=0)/size)
    scores["prior"] = np.broadcast_to(freq, scores["ridge"].shape)
    yd, yt = labels(dev,vocab), labels(test,vocab)
    nd, nt = [np.array([len(set(r["y_str"])) for r in rows]) for rows in (dev,test)]
    edges, taxonomy_stats = taxonomy(args.cache/"wordnet.zip",vocab,y)
    # Decode the full type set with a fixed, external lexical relation. Remove
    # identity edges and use only one propagation step to prevent feedback.
    ridge = scores["ridge"]
    gains = {}
    for key, pairs in (("child_to_parent", edges), ("parent_to_child", [(b,a) for a,b in edges])):
        gain = np.zeros_like(ridge)
        for source, target in pairs:
            np.maximum(gain[:,target], ridge[:,source], out=gain[:,target])
        gains[key] = gain
    results = {}
    for method,score in scores.items():
        best = None
        for strength in ([0,.01,.03,.1,.3,1] if method in {"centroid","box","midpoint"} else [0]):
            ranks = np.argsort(-(score[:split] + strength*freq), axis=1, kind="stable")
            for k in (1,2,3,5,8):
                row = metrics(ranks,yd,nd,k)
                if best is None or row["macro_f1"] > best[0]["macro_f1"]:
                    best = row,strength
        row,strength = best
        ranks = np.argsort(-(score[split:]+strength*freq), axis=1, kind="stable")
        results[method] = dict(prior_weight=strength,dev=row,test=metrics(ranks,yt,nt,row["k"]))
    joint = {}
    for key, gain in gains.items():
        selected = None
        for weight in (0.,.05,.1,.25,.5,1.):
            ranks = np.argsort(-(ridge[:split] + weight * gain[:split]), axis=1, kind="stable")
            for k in (1,2,3,5,8):
                row = metrics(ranks,yd,nd,k)
                if selected is None or row["macro_f1"] > selected[0]["macro_f1"]:
                    selected = row,weight
        row,weight = selected
        ranks = np.argsort(-(ridge[split:]+weight*gain[split:]),axis=1,kind="stable")
        joint[key] = dict(weight=weight,dev=row,test=metrics(ranks,yt,nt,row["k"]))
    print(json.dumps(dict(split_counts=dict(zip(SPLITS,map(len,(train,dev,test)))),training_types=len(vocab),
                         test_gold_labels=int(nt.sum()),test_seen_gold_labels=int(yt.sum()),
                         methods=results,wordnet=taxonomy_stats,joint_typing_taxonomy=joint),indent=2))


if __name__ == "__main__":
    main()
