# SPDX-License-Identifier: MIT OR Apache-2.0
"""Observe routed boundary proposals for the pinned GLiNER2 2.2.0 CPU scorer."""

from unittest.mock import patch


def ranked_roles(routed, probabilities, valid, indices, threshold, cap):
    # Upstream's stable tie order is probability, then start, end and flat index.
    rows = sorted(
        (-score, *span, query, index)
        for query in routed
        for index, (score, keep, span) in enumerate(
            zip(probabilities[query], valid[query], indices[query], strict=True)
        )
        if keep
    )
    eligible = [row for row in rows if -row[0] >= threshold]
    return tuple({(a, b) for _, a, b, _, _ in stage} for stage in (rows, eligible, eligible[:cap]))


class RoleTrace:
    def __init__(self):
        self.windows = []

    def watch(self, stack, engine):
        from gliner2.models.boundary.relations import TypedRelationPairGenerator

        scorer = engine.scorer
        build = scorer._build_boundary_score_set
        generate = TypedRelationPairGenerator.generate

        def build_window(text, schema, batch, index, core, *args):
            if index != 0:
                raise ValueError("Expected a single window per score call")
            offset = int(core["word_offsets"][index])
            starts, ends = batch.start_mappings[index], batch.end_mappings[index]
            routes = {}

            def chars(span):
                a, b = span
                if 0 <= a - offset < b - offset <= len(ends):
                    return starts[a - offset], ends[b - offset - 1]
                return None

            def observe(generator, candidates, layouts, specs, **kwargs):
                result = generate(generator, candidates, layouts, specs, **kwargs)
                settings = generator.settings
                probs = candidates.pair_logits[0].sigmoid().tolist()
                valid = (candidates.valid_mask[0] & candidates.query_mask[0].unsqueeze(-1)).tolist()
                indices = candidates.indices[0].tolist()
                role_cache = {}

                def roles(ids, cap):
                    key = tuple(ids), cap
                    if key not in role_cache:
                        stages = ranked_roles(ids, probs, valid, indices, settings.argument_threshold, cap)
                        role_cache[key] = tuple({chars(s) for s in stage} for stage in stages)
                    return role_cache[key]

                for spec in specs:
                    heads = roles(spec.head_query_ids, settings.heads_per_relation)
                    tails = roles(spec.tail_query_ids, settings.tails_per_relation)
                    routes[spec.relation_type] = (heads, tails, spec.allow_self, set())
                for name, a, b, c, d in zip(result.relation_types, result.head_start.tolist(),
                        result.head_end.tolist(), result.tail_start.tolist(), result.tail_end.tolist(), strict=True):
                    head, tail = chars((a, b)), chars((c, d))
                    heads, tails, self_ok, actual = routes[name]
                    if head not in heads[2] or tail not in tails[2] or (head == tail and not self_ok):
                        raise ValueError("Observed pair contradicts reconstructed role shortlist")
                    actual.add((head, tail))
                return result

            with patch.object(TypedRelationPairGenerator, "generate", observe):
                result = build(text, schema, batch, index, core, *args)
            self.windows.append(routes)
            return result

        stack.enter_context(patch.object(scorer, "_build_boundary_score_set", build_window))

    def classify(self, edge, windows):
        if len(windows) != len(self.windows):
            raise ValueError("Upstream role and scoring windows differ")
        predicate = edge["predicate"]
        observations = []
        for i in edge["joint_mention_windows"]:
            delta = windows[i]["start"]
            head, tail = ((s[1] - delta, s[2] - delta) for s in (edge["head"], edge["tail"]))
            if predicate not in self.windows[i]:
                continue
            (raw_heads, head_pool, top_heads), (raw_tails, tail_pool, top_tails), self_ok, pairs = self.windows[i][predicate]
            if head not in raw_heads or tail not in raw_tails:
                stage, suffix = -1, self._missing(head in raw_heads, tail in raw_tails)
                reason = "raw_role_missing_" + suffix
            elif head not in head_pool or tail not in tail_pool:
                stage, suffix = 0, self._missing(head in head_pool, tail in tail_pool)
                reason = "argument_threshold_missing_" + suffix
            elif head not in top_heads or tail not in top_tails:
                stage, suffix = 1, self._missing(head in top_heads, tail in top_tails)
                reason = "role_shortlist_missing_" + suffix
            elif head == tail and not self_ok:
                stage, reason = 2, "self_pair_disallowed"
            elif (head, tail) not in pairs:
                stage, reason = 3, "pair_cap_before_relation_scoring"
            else:
                stage, reason = 4, "generated_ordered_pair"
            observations.append((stage, reason))
        if not observations:
            return []
        furthest = max(stage for stage, _ in observations)
        return sorted({reason for stage, reason in observations if stage == furthest})

    @staticmethod
    def _missing(head, tail):
        return "tail" if head else "head" if tail else "both"
