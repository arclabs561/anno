"""Fit small, parameter-matched type geometries to annotated GUMBridge pairs."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from bridge_joint import DEPTH, REVISION, read_doc

DIM = 4
SMOOTH = 0.1


def smooth_length(x):
    return SMOOTH * np.logaddexp(0., x / SMOOTH)


def loglength(x):
    length = smooth_length(x)
    return np.log(length), expit(x / SMOOTH) / length


def geometry(weights, mode):
    """Return pair scores and derivatives for each row and column role."""
    a, b = weights[:, 0], weights[:, 1]
    if mode == "point":
        delta = a[:, None] - b[None, :]
        score = -(delta**2).sum(-1)
        zeros = np.zeros_like(delta)
        return score, np.stack((-2*delta, zeros), 2), np.stack((zeros, 2*delta), 2)

    width = np.logaddexp(0., b) + 0.01
    lower, upper = a - width, a + width
    # Anaphor is the row, proposed antecedent the column.
    span = np.minimum(upper[:, None], upper[None, :]) - np.maximum(lower[:, None], lower[None, :])
    log_inter, grad = loglength(span)
    log_vol, grad_vol = loglength(2*width)
    upper_owns = (upper[:, None] < upper[None, :]) + .5*(upper[:, None] == upper[None, :])
    lower_owns = (lower[:, None] > lower[None, :]) + .5*(lower[:, None] == lower[None, :])
    row = np.stack((grad*(upper_owns-lower_owns),
                    grad*(upper_owns+lower_owns)*expit(b)[:, None]), 2)
    col = np.stack((grad*(lower_owns-upper_owns),
                    grad*(2-upper_owns-lower_owns)*expit(b)[None, :]), 2)
    normalization = 2*grad_vol*expit(b)
    if mode == "directional_box":
        # Fraction of anaphor volume covered by antecedent: log P(ante | ana).
        score = (log_inter - log_vol[:, None]).sum(-1)
        row[:, :, 1] -= normalization[:, None]
    else:
        score = (log_inter - .5*(log_vol[:, None]+log_vol[None, :])).sum(-1)
        row[:, :, 1] -= .5*normalization[:, None]
        col[:, :, 1] -= .5*normalization[None, :]
    return score, row, col


def fit(train, mode, seed):
    types = sorted({t for q, cs, _ in train for t in [q, *(t for _, t in cs)]})
    ids = {typ: i for i, typ in enumerate(types)}
    buckets = DEPTH.bit_length()
    shape = (len(types), len(types), buckets)
    total, positive = np.zeros(shape), np.zeros(shape)
    for q, cs, gold in train:
        for i, (mid, t) in enumerate(cs):
            coord = ids[q], ids[t], (i+1).bit_length()-1
            total[coord] += 1
            positive[coord] += mid in gold

    def unpack(params):
        return params[:-buckets].reshape(len(types), 2, DIM), params[-buckets:]

    def loss(params):
        weights, distance = unpack(params)
        geo, ga, gb = geometry(weights, mode)
        logits = geo[:, :, None] + distance
        delta = (total*expit(logits) - positive) / total.sum()
        cost = (total*np.logaddexp(0., logits) - positive*logits).sum()/total.sum()
        penalty = .0001
        pair_grad = delta.sum(-1)[:, :, None, None]
        grad = (pair_grad*ga).sum(1) + (pair_grad*gb).sum(0)
        cost += penalty * np.mean(weights**2)
        grad += 2 * penalty * weights / weights.size
        return cost, np.r_[grad.ravel(), delta.sum((0, 1))]

    rng = np.random.default_rng(seed)
    initial = np.r_[rng.normal(0, .15, (len(types), 2, DIM)).ravel(), np.full(buckets, -3.)]
    # Catch sign/axis errors in hand-coded geometry gradients.
    direction = rng.normal(size=initial.size)
    eps = 1e-6
    numeric = (loss(initial+eps*direction)[0] - loss(initial-eps*direction)[0])/(2*eps)
    assert np.isclose(numeric, np.dot(loss(initial)[1], direction), atol=1e-7)
    result = minimize(loss, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-6})
    weights, distance = unpack(result.x)
    scores = geometry(weights, mode)[0]
    return (ids, scores, distance), dict(parameters=len(initial), iterations=int(result.nit),
            training_loss=float(result.fun), termination=str(result.message))


def evaluate(fit, examples, reverse=False):
    ids, scores, distance = fit
    counts = Counter()
    for q, candidates, gold in examples:
        ranked = []
        for i, (mid, t) in enumerate(candidates):
            row, col = (t, q) if reverse else (q, t)
            # Unseen entity types fall back to the global mean type-pair score.
            geom = scores[ids[row], ids[col]] if row in ids and col in ids else scores.mean()
            ranked.append((geom + distance[(i+1).bit_length()-1], mid))
        ranked.sort(key=lambda x: x[0], reverse=True)
        counts["queries"] += 1
        counts["top1"] += bool(ranked and ranked[0][1] in gold)
        counts["top5"] += any(mid in gold for _, mid in ranked[:5])
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    splits = {s: [row for p in sorted((args.data/s).glob("*.tsv")) for row in read_doc(p)]
              for s in ("train", "dev", "test", "test2")}
    if not all(splits.values()):
        raise ValueError("Missing GUMBridge split")
    results = {"revision": REVISION, "numpy": np.__version__, "scipy": scipy.__version__,
               "dimension": DIM, "seeds": {}}
    for seed in (7, 11, 23):
        models = results["seeds"][seed] = {}
        for mode in ("point", "symmetric_box", "directional_box"):
            model, receipt = fit(splits["train"], mode, seed)
            models[mode] = {"optimizer": receipt, "normal": {
                s: evaluate(model, rows) for s, rows in splits.items() if s != "train"}}
            if mode == "directional_box":
                models[mode]["swapped"] = {s: evaluate(model, rows, True)
                                            for s, rows in splits.items() if s != "train"}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
