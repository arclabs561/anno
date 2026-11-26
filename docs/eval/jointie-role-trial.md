# How JointIE ranks relation arguments and endpoint types

On the original 64 CrossRE AI test sentences and broad training-derived schema, both annotated argument spans reached their predicate's raw role proposals for all 137 previously unlocalized losses. At the checkpoint's 0.20 argument-probability gate, only the head fell below threshold on 34 edges, only the tail on 36, and both on 67. The observer reconstructs per-role top candidates from the real generator input, checks each returned ordered pair against those shortlists, then reconciles the historical per-document receipt with the published digest. It uses the upstream FP32 sigmoid and the model's stable ranking rules. [Role observer](../../scripts/qa/joint_baseline/jointie_roles.py)

We froze a lower gate of 0.10 and compared it with stock on the next 32 CrossRE AI sentences in publisher order (indices 64–95). They share no text hashes with the first 64. Both arms used the same trained weights, 384-word windows, broad training vocabulary and downstream decoder. There were 90 distinct gold typed edges.

| Argument gate | Typed edges scored | Gold among them | Gold reaching optimizer | Gold decoded | Unmatched output |
|---|---:|---:|---:|---:|---:|
| Stock 0.20 | 191,100 | 15 | 1 | 0 | 1 |
| Experimental 0.10 | 1,035,860 | 34 | 1 | 0 | 3 |

The lower gate added three unmatched output edges and removed the stock arm's one unmatched edge. With 14 endpoint types on each side, the provider expanded each distinct ordered span pair and predicate to 196 typed edges with the same relation logit. The stock run had 975 such span-pair/predicate triples and the lower-gate run had 5,285. The subsequent per-predicate cap of 128 dropped 33 of the lower-gate arm's 34 gold candidates. On this shared CPU host, end-to-end process CPU time including load and diagnostics rose from 56.2 to 114.8 seconds, wall time from 56.9 to 175.8 seconds, and peak RSS was about 1.02 GiB in both arms. The trial supports keeping the stock gate. [Paired comparison](../../scripts/qa/joint_baseline/compare_jointie_roles.py)

We then kept the stock 0.20 argument gate and the same 128 slots per predicate. On those same 32 sentences, the alternative admitted one typed expansion per directed span pair before admitting second expansions. It prioritized pairs by the existing relation logit and ranked types for each pair by the sum of the two independent mention logits. Neither the model scores nor the downstream solver changed. The comparison reproduced the full stock arm before scoring the alternative.

| Stage | Stock | Alternative | Paired gold gains / losses |
|---|---:|---:|---:|
| Exact gold among 35,968 admitted typed edges | 1 | 12 | +11 / 0 |
| Gold in the final graph (90 in the reference) | 0 | 1 | +1 / 0 |
| Unmatched edges in the final graph | 1 | 12 | 11 added / 0 removed |

Preferentially representing different pairs recovered candidates at the same cap. The deployed decision would trade one recovered labeled relation for eleven extra unmatched outputs on six documents. This small, reused diagnostic slice supports a ranking mechanism and does not justify changing the runtime default. [Experimental ranker](../../scripts/qa/joint_baseline/jointie_ranking.py)

The original replay documents the model and corpus revisions, environment and authorized external source paths. [Prerequisites and local paths](joint-edge-loss.md#source-terms-and-replay)

```sh
python "$qa/jointie_coverage.py" "$test" "$jointie" "$out/roles.json" --documents 64 --window-words 384 --overlap-words 64 --sparse-roles
python "$qa/jointie_coverage.py" "$test" "$jointie" "$out/stock.json" --first-index 64 --documents 32 --window-words 384 --overlap-words 64 --sparse-roles
python "$qa/jointie_coverage.py" "$test" "$jointie" "$out/lower.json" --first-index 64 --documents 32 --window-words 384 --overlap-words 64 --sparse-roles --argument-threshold 0.10
python "$qa/jointie_coverage.py" "$test" "$jointie" "$out/ranked.json" --first-index 64 --documents 32 --window-words 384 --overlap-words 64 --sparse-roles --rank-typed-alternatives
python "$qa/compare_jointie_roles.py" "$test" "$out/roles.json" "$out/stock.json" "$out/lower.json" --ranked "$out/ranked.json"
```
