"""Causal, text-free candidate retrieval on complete BookCoref gold novels."""
import argparse
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
import hashlib
import json

SHA = '2c2587aca956a4225712abdcf1af72b93f18a28f'
ANNOTATION_SHA256 = 'c4d426a722d3901489bf7bfcb3554d734db29a689876fcff269719ec7288dc9a'
URL = f'https://huggingface.co/datasets/sapienzanlp/bookcoref/resolve/{SHA}/bookcoref_annotations/full/test.jsonl'


def replay(doc, capacity):
    # Canonical order is by observed coordinates, never by source character ID.
    events = sorted((end,start,who) for who,cluster in enumerate(doc['clusters']) for start,end in cluster)
    history = {}
    retained = {}
    out = defaultdict(Counter)
    serial = 0
    for end,at_end in groupby(events,key=lambda row: row[0]):
        batch = list(at_end)
        # All mentions ending at this token see the same strict prefix.
        recency = sorted(retained,key=lambda x:-retained[x][0])
        frequency = sorted(retained,key=lambda x:(-retained[x][1],-retained[x][0]))
        for _,start,who in batch:
            if who not in history:
                out['all']['new'] += 1
                continue
            gap = end - history[who]
            groups = ['all'] + [f'gap_ge_{g}' for g in (128,1024,8192) if gap >= g]
            for group in groups:
                counts = out[group]
                counts['returning'] += 1
                counts['retained'] += who in retained
                counts['recency_at_1'] += who in recency[:1]
                counts['recency_at_4'] += who in recency[:4]
                counts['frequency_at_1'] += who in frequency[:1]
                counts['frequency_at_4'] += who in frequency[:4]
        # Overlapping mentions ending together update by observed start.
        # For equally placed distinct people, group mentions are ambiguous and
        # would require a tied admission policy. Check rather than leak label order.
        assert len({start for _,start,_ in batch}) == len(batch)
        for _,start,who in batch:
            seen = retained.get(who,(0,0))[1]
            retained[who] = (serial,seen+1)
            serial += 1
            if len(retained) > capacity:
                del retained[min(retained,key=lambda x:retained[x][0])]
            history[who] = end
    return dict(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    args=parser.parse_args()
    payload = args.data.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ANNOTATION_SHA256:
        parser.error(f'expected the pinned BookCoref test annotations ({ANNOTATION_SHA256}), got {digest}')
    docs = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
    result={'source':URL,'sha256':digest,'capacities':{}}
    for capacity in (8,24,64):
        books={doc['doc_key']:replay(doc,capacity) for doc in docs}
        total=defaultdict(Counter)
        for rows in books.values():
            for k,v in rows.items():
                total[k].update(v)
        result['capacities'][capacity]={'books':books,'total':total}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
