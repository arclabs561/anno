"""Instrument the upstream 2.2.0 sparse scorer and solver without changing their decisions."""

from collections import defaultdict
from contextlib import ExitStack
from unittest.mock import patch


class DecoderTrace:
    def __init__(self):
        self.windows = []
        self.problem = None
        self.solution = None

    def watch(self, stack: ExitStack, model):
        score = model.score
        make_optimizer = model._make_optimizer

        def score_window(*args, **kwargs):
            result = score(*args, **kwargs)
            self.windows.append(result)
            return result

        def optimizer(config):
            real = make_optimizer(config)
            optimize = real.optimize

            def solve(problem):
                self.problem = problem
                self.solution = optimize(problem)
                return self.solution

            real.optimize = solve
            return real

        stack.enter_context(patch.object(model, "score", score_window))
        stack.enter_context(patch.object(model, "_make_optimizer", optimizer))

    def losses(self, gold, merged, config, windows, decoded):
        def occurrence(scores, key, offset=0):
            label, lo, hi = key
            return (
                label,
                offset + scores.start_mappings[lo],
                offset + scores.end_mappings[hi - 1],
            )

        def key(edge):
            return (
                occurrence(merged, edge.head),
                edge.relation_type,
                occurrence(merged, edge.tail),
            )

        scored = {key(e): e for e in merged.edges}
        # The pinned upstream converter deduplicates (predicate, typed head, typed tail),
        # rejects sub-threshold scores, then sorts per predicate before applying its cap.
        by_predicate = defaultdict(list)
        for edge in scored.values():
            gate = (
                config.relation_role_threshold
                if edge.candidate_threshold is None
                else edge.candidate_threshold
            )
            if edge.probability >= gate:
                by_predicate[edge.relation_type].append(edge)
        rank = {}
        for edges in by_predicate.values():
            ordered = sorted(edges, key=lambda e: (-e.logit, str(e.head), str(e.tail)))
            rank.update({key(e): i + 1 for i, e in enumerate(ordered)})
        cap = min(config.relation_pair_cap, config.max_edges_per_type)
        nodes = {occurrence(merged, n.candidate_id) for n in self.problem.nodes}
        problem = {key(e): e for e in self.problem.edges}
        solution = {key(e) for e in self.solution.edges}
        seen = []
        for segment, scores in zip(windows, self.windows, strict=True):
            seen.append(
                {occurrence(scores, m.key, segment["start"]) for m in scores.mentions}
            )
        results = []
        for edge in sorted(gold):
            h, r, t = edge
            together = [
                i
                for i, w in enumerate(windows)
                if w["start"] <= min(h[1], t[1]) and max(h[2], t[2]) <= w["end"]
            ]
            local = [i for i in together if h in seen[i] and t in seen[i]]
            attempt = scored.get(edge)
            record = {
                "head": h,
                "predicate": r,
                "tail": t,
                "co_windows": together,
                "joint_mention_windows": local,
            }
            if not together:
                reason = "outside_shared_window"
            elif not local:
                reason = "missing_typed_mention_in_shared_window"
            elif attempt is None:
                reason = "not_in_sparse_edge_output"
            else:
                gate = (
                    config.relation_role_threshold
                    if attempt.candidate_threshold is None
                    else attempt.candidate_threshold
                )
                record.update(
                    probability=attempt.probability,
                    candidate_threshold=gate,
                    optimizer_endpoint_count=int(h in nodes) + int(t in nodes),
                    per_predicate_cap=cap,
                    rank_after_gate=rank.get(edge),
                )
                expected_in_problem = (
                    edge in rank and rank[edge] <= cap and h in nodes and t in nodes
                )
                if expected_in_problem != (edge in problem):
                    raise ValueError(
                        "threshold, rank and endpoint survival disagree with the real optimizer input"
                    )
                if attempt.probability < gate:
                    reason = "below_edge_candidate_threshold"
                elif rank[edge] > cap:
                    reason = "edge_cap"
                elif edge not in problem:
                    reason = "optimizer_endpoint_pruning"
                else:
                    record["optimizer_edge_utility"] = problem[edge].score
                    if edge not in solution:
                        reason = "optimizer_did_not_select"
                    elif edge not in decoded:
                        reason = "result_builder_omitted"
                    else:
                        reason = "matched"
            record["first_loss"] = reason
            results.append(record)
        return results
