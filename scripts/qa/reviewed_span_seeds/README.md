# Replaying a reviewed NER regression seed

A judge can propose an organization-span case. A maintainer can admit the case as an
exposed regression once the fixture has an independent oracle and a native replay.
The registry pins the reviewed bytes, trace and oracle dispatch. Only original synthetic
fixtures are eligible here. Keep fresh evaluation holdouts out of this registry.

The first registered case came from the offline organization-alias round for PR #243. A scout
and skeptic wrote two synthetic cases. A fresh-context oracle received only the text, units and
annotation convention. It froze three exact organization spans before the native comparison.
The affected implementation missed two spans on one case and preserved the neighboring legal
name. Count that as one reproduced case. [Original repair](https://github.com/arclabs561/anno/pull/243)

Use the fixture and trace directory retained by the round coordinator and a locally built Anno CLI.
The trace directory must contain the exported trace, dispatch receipt, oracle prompt and response,
and the source-bound native baseline and comparison prediction JSONLs. These inputs stay local.

```sh
python3 scripts/qa/reviewed_span_seeds/replay.py fixture.json final /path/to/anno > seed-replay.json
```

The adapter verifies the reviewed digests and recorded fresh-context dispatch. It checks
that the oracle finished before the native comparison, checks its response against the fixture,
rescores both historical runs, and runs `anno batch` on the
supplied binary. The output separates fixed historical failures from failures reproduced on the
tested binary. Its shareable fields contain hashes, counts and fixed category names. A local
trace may require additional rights review even for a redistributable fixture.

For a Muxer campaign, use `seed_sha256` as the arm within this backend and assertion domain.
Give each selected execution one exposure. Credit one success for the first independent,
runner-confirmed reproduction of that case on the tested target. Repeat confirmations and
multiple failed assertions on one case add no novelty. Keep binary digests with the campaign
so that repetitions on a repaired implementation remain distinguishable from new regressions.

For another original synthetic case, review the inputs and rights, freeze the independent
exact-ORG oracle, and pin the fixture, dispatch and completed trace through repository review.
The registry reviewed here handles this specific oracle response and trace layout.
