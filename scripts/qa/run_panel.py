"""Run and validate anno's fixed, cached-only evaluation panels.

The manifest deliberately names only evaluation cells and expected incompatibilities.
Dataset/model registry and scoring remain owned by `anno benchmark`; this helper only
executes that CLI and rejects incomplete or malformed result artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """The manifest or a benchmark result does not meet the panel contract."""


@dataclass(frozen=True, order=True)
class Cell:
    task: str
    dataset: str
    backend: str
    seed: int

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> Cell:
        try:
            return cls(
                task=str(result["task"]),
                dataset=str(result["dataset"]),
                backend=str(result["backend"]),
                seed=int(result["seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("result lacks a valid task, dataset, backend, or seed") from exc


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ContractError("manifest must be an object with schema_version 1")
    suites = manifest.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ContractError("manifest must contain at least one suite")
    ids: set[str] = set()
    for suite in suites:
        if not isinstance(suite, dict):
            raise ContractError("suite must be an object")
        suite_id = suite.get("id")
        if not isinstance(suite_id, str) or not suite_id or suite_id in ids:
            raise ContractError("suite ids must be unique nonempty strings")
        ids.add(suite_id)
        for field in ("tasks", "datasets", "backends", "seeds"):
            value = suite.get(field)
            if not isinstance(value, list) or not value:
                raise ContractError(f"suite {suite_id!r} needs a nonempty {field}")
        if not isinstance(suite.get("max_examples"), int) or suite["max_examples"] <= 0:
            raise ContractError(f"suite {suite_id!r} needs a positive max_examples")
        allowed = suite.get("allowed_incompatibilities", [])
        if not isinstance(allowed, list):
            raise ContractError(f"suite {suite_id!r} allowed_incompatibilities must be a list")
    return manifest


def normalized_task(task: str) -> str:
    names = {"ner": "NER", "coref": "IntraDocCoref", "relation": "RelationExtraction"}
    try:
        return names[task.lower()]
    except KeyError as exc:
        raise ContractError(f"unsupported manifest task {task!r}") from exc


def suite_cells(suite: dict[str, Any]) -> set[Cell]:
    return {
        Cell(normalized_task(task), dataset, backend, int(seed))
        for task in suite["tasks"]
        for dataset in suite["datasets"]
        for backend in suite["backends"]
        for seed in suite["seeds"]
    }


def allowed_cells(suite: dict[str, Any]) -> set[Cell]:
    cells: set[Cell] = set()
    for skip in suite.get("allowed_incompatibilities", []):
        if not isinstance(skip, dict):
            raise ContractError(f"suite {suite['id']!r} has a non-object incompatibility")
        try:
            cell_without_seed = (
                str(skip["task"]),
                str(skip["dataset"]),
                str(skip["backend"]),
            )
        except KeyError as exc:
            raise ContractError(f"suite {suite['id']!r} incompatibility lacks {exc.args[0]}") from exc
        matching = {
            cell for cell in suite_cells(suite)
            if (cell.task, cell.dataset, cell.backend) == cell_without_seed
        }
        if not matching:
            raise ContractError(f"suite {suite['id']!r} declares an incompatibility outside its cells")
        cells.update(matching)
    return cells


def require_finite_metrics(result: dict[str, Any], cell: Cell) -> None:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ContractError(f"{cell}: successful result lacks metrics")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ContractError(f"{cell}: metric {name!r} is not finite")
    primary = {
        "NER": "f1",
        "IntraDocCoref": "conll_f1",
        "RelationExtraction": "f1",
    }.get(cell.task)
    if primary is None or primary not in metrics:
        raise ContractError(f"{cell}: successful result lacks primary metric {primary!r}")


def require_current_provenance(result: dict[str, Any], cell: Cell, max_examples: int) -> None:
    provenance = result.get("provenance")
    schema_version = provenance.get("schema_version") if isinstance(provenance, dict) else None
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise ContractError(f"{cell}: missing or legacy provenance")
    for field in ("build", "dataset", "backend", "runtime"):
        if not isinstance(provenance.get(field), dict):
            raise ContractError(f"{cell}: provenance.{field} must be an object")
    runtime = provenance["runtime"]
    if runtime.get("cached_only") is not True:
        raise ContractError(f"{cell}: result was not recorded as cached-only")
    if runtime.get("seed") != cell.seed or runtime.get("max_examples") != max_examples:
        raise ContractError(f"{cell}: provenance runtime does not match requested seed or cap")

    build = provenance["build"]
    if not isinstance(build.get("package_version"), str) or not build["package_version"]:
        raise ContractError(f"{cell}: provenance build lacks package_version")
    if not isinstance(build.get("enabled_features"), list):
        raise ContractError(f"{cell}: provenance build lacks enabled_features")
    if "eval" not in build["enabled_features"]:
        raise ContractError(f"{cell}: provenance build does not record the eval feature")

    dataset = provenance["dataset"]
    if not isinstance(dataset.get("source_url"), str) or not dataset["source_url"]:
        raise ContractError(f"{cell}: provenance dataset lacks source_url")
    if not isinstance(dataset.get("data_source"), str) or not dataset["data_source"]:
        raise ContractError(f"{cell}: provenance dataset lacks data_source")
    if dataset.get("sentence_count") != result.get("num_examples"):
        raise ContractError(f"{cell}: provenance dataset sentence_count disagrees with result")

    backend = provenance["backend"]
    if backend.get("requested") != cell.backend or backend.get("effective") != cell.backend:
        raise ContractError(f"{cell}: provenance backend does not match result backend")


def validate_suite(suite: dict[str, Any], artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(artifact, dict):
        raise ContractError(f"suite {suite['id']!r}: artifact must be an object")
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ContractError(f"suite {suite['id']!r}: artifact lacks a results array")
    expected = suite_cells(suite)
    allowed = allowed_cells(suite)
    observed: dict[Cell, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ContractError(f"suite {suite['id']!r}: result is not an object")
        cell = Cell.from_result(result)
        if cell not in expected:
            raise ContractError(f"suite {suite['id']!r}: unexpected result {cell}")
        if cell in observed:
            raise ContractError(f"suite {suite['id']!r}: duplicate result {cell}")
        observed[cell] = result

    missing = expected - observed.keys()
    if missing:
        raise ContractError(f"suite {suite['id']!r}: missing expected cells: {sorted(missing)}")

    rendered: list[dict[str, Any]] = []
    for cell in sorted(expected):
        result = observed[cell]
        if cell in allowed:
            if result.get("success") is not False or not str(result.get("error", "")).startswith("incompatible:"):
                raise ContractError(f"{cell}: declared incompatibility must be an incompatible skipped result")
            outcome = "declared-incompatible"
        else:
            if result.get("success") is not True:
                raise ContractError(f"{cell}: expected success, got {result.get('error')!r}")
            if not isinstance(result.get("num_examples"), int) or result["num_examples"] <= 0:
                raise ContractError(f"{cell}: successful result is empty")
            require_finite_metrics(result, cell)
            require_current_provenance(result, cell, suite["max_examples"])
            outcome = "success"
        rendered.append({
            "task": cell.task,
            "dataset": cell.dataset,
            "backend": cell.backend,
            "seed": cell.seed,
            "outcome": outcome,
            "num_examples": result.get("num_examples", 0),
            "metrics": result.get("metrics", {}),
            "provenance": result.get("provenance"),
        })
    return rendered


def render_markdown(rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    success = sum(row["outcome"] == "success" for row in rows)
    skipped = len(rows) - success
    lines = [
        "# anno fixed QA panel",
        "",
        f"Validated {len(rows)} expected cells: {success} successes and {skipped} declared incompatibilities.",
        "",
        "| Task | Dataset | Backend | Seed | Outcome | Examples | Primary F1 |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["metrics"]
        primary = "conll_f1" if row["task"] == "IntraDocCoref" else "f1"
        f1 = metrics.get(primary) if isinstance(metrics, dict) else None
        f1_text = f"{f1:.4f}" if isinstance(f1, (int, float)) else ""
        lines.append(
            f"| {row['task']} | {row['dataset']} | {row['backend']} | {row['seed']} | "
            f"{row['outcome']} | {row['num_examples']} | {f1_text} |"
        )
    if any(row["task"] == "IntraDocCoref" for row in rows):
        lines.extend([
            "",
            "Coreference uses conll_f1. GAP annotations are partial and CEAF alignment is approximate; "
            "these diagnostic scores are not official benchmark results.",
        ])
    return "\n".join(lines) + "\n"


def selected_suites(manifest: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    suites = manifest["suites"]
    if not names:
        return suites
    wanted = set(names)
    selected = [suite for suite in suites if suite["id"] in wanted]
    missing = wanted - {suite["id"] for suite in selected}
    if missing:
        raise ContractError(f"unknown suite(s): {', '.join(sorted(missing))}")
    return selected


def write_log(
    path: Path,
    command: list[str],
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    *,
    exit_status: int | None = None,
    note: str | None = None,
) -> None:
    def text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        return value.decode(errors="replace") if isinstance(value, bytes) else value

    status = "unknown" if exit_status is None else str(exit_status)
    header = "$ " + " ".join(command) + f"\n# exit-status: {status}\n"
    if note:
        header += f"# note: {note}\n"
    path.write_text(header + "\n" + text(stdout) + text(stderr))


def run_suite(anno_bin: Path, suite: dict[str, Any], output_dir: Path, timeout_seconds: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    for seed in suite["seeds"]:
        artifact = output_dir / f"{suite['id']}-seed{seed}.json"
        markdown = output_dir / f"{suite['id']}-seed{seed}.md"
        log = output_dir / f"{suite['id']}-seed{seed}.log"
        existing = [path for path in (artifact, markdown, log) if path.exists()]
        if existing:
            raise ContractError(
                f"suite {suite['id']!r}, seed {seed}: refusing to reuse existing artifact(s): "
                + ", ".join(str(path) for path in existing)
            )
        command = [
            str(anno_bin), "benchmark",
            "--tasks", ",".join(suite["tasks"]),
            "--datasets", ",".join(suite["datasets"]),
            "--backends", ",".join(suite["backends"]),
            "--max-examples", str(suite["max_examples"]),
            "--seed", str(seed),
            "--cached-only",
            "--split-heavy-backends=false",
            "--output", str(markdown),
            "--output-json", str(artifact),
        ]
        try:
            child_env = os.environ.copy()
            # `--cached-only` constrains dataset loading. HF-backed model loading
            # uses this separate process setting, so enforce both layers without
            # changing the caller's environment.
            child_env["ANNO_NO_DOWNLOADS"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                env=child_env,
            )
        except subprocess.TimeoutExpired as exc:
            write_log(log, command, exc.stdout, exc.stderr, note=f"timed out after {timeout_seconds}s")
            raise ContractError(
                f"suite {suite['id']!r}, seed {seed}: anno benchmark timed out after {timeout_seconds}s"
            ) from exc
        except OSError as exc:
            write_log(log, command, None, str(exc), note="process could not start")
            raise ContractError(f"suite {suite['id']!r}, seed {seed}: could not execute anno: {exc}") from exc
        write_log(log, command, completed.stdout, completed.stderr, exit_status=completed.returncode)
        if completed.returncode:
            raise ContractError(f"suite {suite['id']!r}, seed {seed}: anno benchmark exited {completed.returncode}")
        artifacts.append(artifact)
    return artifacts[0] if len(artifacts) == 1 else output_dir


def validate_paths(manifest: dict[str, Any], suites: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in suites:
        for seed in suite["seeds"]:
            path = output_dir / f"{suite['id']}-seed{seed}.json"
            rows.extend(validate_suite(suite | {"seeds": [seed]}, load_json(path)))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("scripts/qa/core-panel.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", action="append", dest="suites", help="run only this suite (repeatable)")
    parser.add_argument("--anno-bin", type=Path, help="already-built anno binary; required unless --validate-only")
    parser.add_argument("--validate-only", action="store_true", help="validate existing JSON artifacts without executing anno")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="per seed execution limit (default: 900)")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        suites = selected_suites(manifest, args.suites)
        args.output_dir = args.output_dir.resolve()
        if args.timeout_seconds <= 0:
            raise ContractError("--timeout-seconds must be positive")
        if not args.validate_only:
            if args.anno_bin is None:
                raise ContractError("--anno-bin is required unless --validate-only is set")
            args.anno_bin = args.anno_bin.resolve()
            if not args.anno_bin.is_file():
                raise ContractError(f"anno binary does not exist: {args.anno_bin}")
            old_summaries = [
                path
                for path in (args.output_dir / "panel-summary.json", args.output_dir / "panel-summary.md")
                if path.exists()
            ]
            if old_summaries:
                raise ContractError(
                    "refusing to replace existing panel summary artifact(s): "
                    + ", ".join(str(path) for path in old_summaries)
                )
            for suite in suites:
                run_suite(args.anno_bin, suite, args.output_dir, args.timeout_seconds)
        rows = validate_paths(manifest, suites, args.output_dir)
        (args.output_dir / "panel-summary.json").write_text(json.dumps({"results": rows}, indent=2) + "\n")
        (args.output_dir / "panel-summary.md").write_text(render_markdown(rows))
        print(f"Validated {len(rows)} expected cells in {args.output_dir}")
        return 0
    except ContractError as exc:
        print(f"qa panel contract failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
