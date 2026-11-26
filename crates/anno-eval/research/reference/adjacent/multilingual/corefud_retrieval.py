"""Compare three source-span retrievers across CorefUD 1.4 development corpora."""
import argparse
import hashlib
import heapq
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


REV = "3ded5a1638e43d89b81ee24bf5b98f66943e1c60"
URL = f"https://huggingface.co/datasets/lattice-nlp/corefud-1-4/resolve/{REV}/data/dev-00000-of-00001.parquet"
CHECKSUM = "d45d2a76003a1a692aed994b3ce35b4b617a114c3c5882afb2e272412d387aff"


def norm(s):
    return " ".join(s.casefold().split())


def score(doc, predicted_heads=None):
    text = doc['text']
    counts = Counter(documents=1)
    occurrences = []
    for m in doc['mentions']:
        counts['source_mentions'] += 1
        l, r = m['onset'], m['offset']
        if l is None or r is None or l < 0 or r > len(text) or r <= l or not any(c.isalnum() for c in text[l:r]):
            counts['excluded_spans'] += 1
            continue
        if predicted_heads is None:
            h, t = m['head_onset'], m['head_offset']
        else:
            h, t = predicted_heads.get((l, r), (None, None))
        head = norm(text[h:t]) if h is not None and t is not None and l <= h < t <= r else ''
        occurrences.append((l, r, m['COREF'], norm(text[l:r]), head, m['start_token'], m['end_token']))
    # Only unambiguous observable spans are scored. Sort ties by offsets;
    # no gold ID participates in tie-breaking or candidate selection.
    multiplicity = Counter((m[0], m[1]) for m in occurrences)
    counts['duplicate_span_mentions'] += sum(n for n in multiplicity.values() if n > 1)
    mentions = sorted((m for m in occurrences if multiplicity[m[:2]] == 1), key=lambda x: x[:2])
    latest = None
    by_shape, by_head, gold_last = {}, {}, {}
    gold_shapes = set()
    pending = []
    clusters = {'surface': [], 'head_backoff': []}
    gold_counts = Counter()
    predicted_counts = defaultdict(Counter)
    intersections = defaultdict(Counter)
    for i, (start, end, target, shape, head, begin_token, _) in enumerate(mentions):
        while pending and pending[0][0] <= start:
            _, j = heapq.heappop(pending)
            _, _, entity, surface, hform, _, _ = mentions[j]
            latest = j
            by_shape[surface] = j
            if hform:
                by_head[hform] = j
            gold_last[entity] = j
            gold_shapes.add((entity, surface))
        heapq.heappush(pending, (end, i))
        if predicted_heads is not None:
            counts['usable_mentions'] += 1
            counts['mentions_with_head'] += bool(head)
            gold_counts[target] += 1
            surface_pick = by_shape.get(shape)
            head_pick = surface_pick if surface_pick is not None else by_head.get(head)
            for policy, pick in [('surface', surface_pick), ('head_backoff', head_pick)]:
                # Construct one cluster per chain of affirmative matches. Unmatched
                # mentions start a cluster; the ranker's recency guess is not a merge.
                cluster = i if pick is None else clusters[policy][pick]
                clusters[policy].append(cluster)
                predicted_counts[policy][cluster] += 1
                intersections[policy][cluster, target] += 1
        if target not in gold_last:
            counts['no_eligible_gold'] += 1
            continue
        labels = ['all', 'repeated_surface' if (target, shape) in gold_shapes else 'new_surface']
        gap = begin_token - mentions[gold_last[target]][6] - 1
        if gap >= 128:
            labels.append('long_gap')
        exact = by_shape.get(shape, latest)
        head_backoff = by_shape.get(shape, by_head.get(head, latest))
        if predicted_heads is not None:
            baseline_hit = mentions[exact][2] == target
            candidate_hit = mentions[head_backoff][2] == target
            counts['gained'] += candidate_hit and not baseline_hit
            counts['lost'] += baseline_hit and not candidate_hit
        for label in labels:
            counts[label + '.queries'] += 1
            for policy, picked in [('recent', latest), ('surface', exact), ('head_backoff', head_backoff)]:
                counts[label + '.' + policy] += mentions[picked][2] == target
    if predicted_heads is not None:
        def pairs(histogram):
            return sum(n * (n - 1) // 2 for n in histogram.values())
        counts['gold_pairs'] = pairs(gold_counts)
        for policy in clusters:
            counts[f'{policy}.predicted_pairs'] = pairs(predicted_counts[policy])
            counts[f'{policy}.correct_pairs'] = pairs(intersections[policy])
    return counts


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    if not args.cache.exists():
        urllib.request.urlretrieve(URL, args.cache)
    if hashlib.sha256(args.cache.read_bytes()).hexdigest() != CHECKSUM:
        raise ValueError('unexpected development shard checksum')
    languages, corpora = defaultdict(Counter), defaultdict(Counter)
    total = Counter()
    for batch in pq.ParquetFile(args.cache).iter_batches(columns=['language','file_name','text','mentions'], batch_size=16):
        for doc in batch.to_pylist():
            corpus = doc['file_name'].split('-corefud-dev')[0]
            counts = score(doc)
            languages[doc['language']].update(counts)
            corpora[corpus].update(counts)
            total.update(counts)
    args.result.write_text(json.dumps(dict(revision=REV, sha256=CHECKSUM, total=total,
                                           languages=languages, corpora=corpora), indent=2)+'\n')


if __name__ == '__main__':
    main()
