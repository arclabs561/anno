"""Local vs pipeline vs joint decoding on SciERC gold mention boundaries."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations, permutations
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

NONE = 'NONE'


def read_split(path):
    out = []
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        words = [w.lower() for s in raw['sentences'] for w in s]
        spans, types, sentence = [], [], []
        for sid, annotations in enumerate(raw['ner']):
            for a, b, t in annotations:
                spans.append((a, b))
                types.append(t)
                sentence.append(sid)
        assert len(set(spans)) == len(spans)
        rel = {(tuple(r[:2]), tuple(r[2:4])): r[4] for s in raw['relations'] for r in s}
        assert len(rel) == len({tuple(r) for group in raw['relations'] for r in group})
        clusters = {tuple(m): k for k, c in enumerate(raw['clusters']) for m in c}
        fx = [[], [], []]
        labels = [types, [], []]
        edges = [[], []]
        for a, b in spans:
            fx[0].append({'surface': ' '.join(words[a:b+1]), 'head': words[b],
                          'first': words[a], 'length': str(min(b-a, 5)),
                          'prev': words[a-1] if a else '', 'next': ' '.join(words[b+1:b+2])})
        for task, pairs in enumerate([permutations(range(len(spans)), 2), combinations(range(len(spans)), 2)]):
            for i, j in pairs:
                a, b = spans[i]
                c, d = spans[j]
                if task == 0 and sentence[i] != sentence[j]:
                    continue
                same = spans[i] in clusters and clusters[spans[i]] == clusters.get(spans[j])
                left, right = sorted([spans[i], spans[j]])
                between = words[left[1]+1:right[0]]
                fi, fj = fx[0][i], fx[0][j]
                f = {'first_head': words[b], 'second_head': words[d],
                     'head_pair': words[b]+' '+words[d], 'direction': str(a < c),
                     'same_head': str(words[b] == words[d]), 'same_surface': str(fi['surface'] == fj['surface']),
                     'gap': str(min(len(between), 8)), 'sentence_gap': str(min(abs(sentence[i]-sentence[j]), 3))}
                if task == 0:
                    f['between'] = ' '.join(between[:6])
                    f['nested'] = str(a <= c <= d <= b or c <= a <= b <= d)
                edges[task].append((i,j))
                fx[task+1].append(f)
                labels[task+1].append(rel.get((spans[i],spans[j]), NONE) if task == 0 else ('SAME' if same else NONE))
        assert len(rel) == sum(label != NONE for label in labels[1])
        out.append({'id': raw['doc_key'], 'fx':fx, 'y':labels, 'edges':edges})
    return out


def fit(train):
    models, classes = [], []
    for task in range(3):
        v = DictVectorizer()
        xx = v.fit_transform([f for doc in train for f in doc['fx'][task]])
        yy = [y for doc in train for y in doc['y'][task]]
        model = LogisticRegression(solver='saga', max_iter=500, random_state=7, tol=0.001).fit(xx, yy)
        models.append((v,model))
        classes.append(model.classes_.tolist())
    factors = []
    for k in (1,2):
        m = len(classes[0])
        freq = np.ones((len(classes[k]),m,m))
        for doc in train:
            t = [classes[0].index(y) for y in doc['y'][0]]
            for (i,j), y in zip(doc['edges'][k-1],doc['y'][k],strict=True):
                freq[classes[k].index(y),t[i],t[j]] += 1
        logp = np.log(freq / freq.sum(axis=(1,2),keepdims=True))
        factors.append(logp-logp[classes[k].index(NONE)])
    return models, classes, factors


def prepare(docs, models, classes):
    for doc in docs:
        doc['local'] = [m.predict_log_proba(v.transform(f)) if f else np.empty((0,len(m.classes_)))
                        for (v,m),f in zip(models,doc['fx'],strict=True)]
        doc['truth'] = [np.array([c.index(y) for y in ys], dtype=int) for c,ys in zip(classes,doc['y'],strict=True)]
        doc['ij'] = [np.array(es,dtype=int).reshape((-1,2)).T for es in doc['edges']]


def decode(doc, factors, weight, joint):
    unary = doc['local']
    types = unary[0].argmax(axis=1)
    for iteration in range(6):
        preds = [np.argmax(local + weight*factor[:,types[ii],types[jj]].T,axis=1)
                 for local, factor, (ii,jj) in zip(unary[1:],factors,doc['ij'],strict=True)]
        if not joint or iteration == 5:
            break
        old = types.copy()
        for i in range(len(types)):
            scores = unary[0][i].copy()
            for pred,fac,(ii,jj) in zip(preds,factors,doc['ij'],strict=True):
                src, tgt = ii==i, jj==i
                scores += weight*(fac[pred[src],:,types[jj[src]]].sum(axis=0)
                                  +fac[pred[tgt],types[ii[tgt]],:].sum(axis=0))
            types[i] = scores.argmax()
        if np.array_equal(old,types): break
    return types, preds


def score(docs, factors, classes, weight, joint):
    c = Counter()
    for doc in docs:
        tt, ps = decode(doc,factors,weight,joint)
        good = tt == doc['truth'][0]
        c['mentions'] += len(tt)
        c['correct_types'] += int(good.sum())
        for name,k,p,(ii,jj) in zip(('relations','coref'),(1,2),ps,doc['ij'],strict=True):
            y = doc['truth'][k]
            none = classes[k].index(NONE)
            correct = (p!=none) & (p==y)
            c[name+'_p'] += int((p!=none).sum())
            c[name+'_g'] += int((y!=none).sum())
            c[name+'_tp'] += int(correct.sum())
            c[name+'_typed_tp'] += int((correct & good[ii] & good[jj]).sum())
    for name in ('relations','coref'):
        den = c[name+'_p'] + c[name+'_g']
        for key in ('','typed_'):
            c[name+'_'+key+'f1'] = 2*c[name+'_'+key+'tp']/den if den else 0
    c['type_accuracy'] = c['correct_types']/c['mentions']
    return dict(c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    paths = [args.directory/(p+'.json') for p in ('train','dev','test')]
    train,dev,test = docs = [read_split(p) for p in paths]
    ids = [d['id'] for split in docs for d in split]
    assert len(set(ids)) == len(ids)
    models, classes, factors = fit(train)
    for split in [dev,test]:
        prepare(split,models,classes)
    result = {'split_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'documents':list(map(len,docs)), 'arms':{}}
    for name, joint, weights in [('local',False,[0]),('pipeline',False,[0,.25,.5,1]),
                                ('joint',True,[0,.25,.5,1])]:
        dev_scores = [score(dev,factors,classes,w,joint) for w in weights]
        # Both pair tasks require correctly typed endpoints. We give each equal weight.
        best = max(range(len(weights)),key=lambda i:sum(dev_scores[i][m+'_typed_f1'] for m in ('relations','coref')))
        result['arms'][name] = {'development':dict(zip(weights,dev_scores)), 'weight':weights[best],
                                'test':score(test,factors,classes,weights[best],joint)}
    # Compare the two coupling directions on paired held-out documents.
    boot = np.random.default_rng(7).integers(len(test), size=(2000, len(test)))
    paired = {}
    for arm in ('pipeline', 'joint'):
        rows = [score([d],factors,classes,result['arms'][arm]['weight'],arm == 'joint') for d in test]
        def pooled(key):
            return np.array([r[key] for r in rows])[boot].sum(axis=1)
        paired[arm] = {m: 2*pooled(m+'_typed_tp')/(pooled(m+'_p')+pooled(m+'_g'))
                       for m in ('relations','coref')}
    result['joint_minus_pipeline_typed_f1_bootstrap95'] = {
        m: np.quantile(paired['joint'][m]-paired['pipeline'][m], [.025,.975]).tolist()
        for m in ('relations','coref')}
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
