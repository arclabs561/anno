#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Comprehensive, resumable evaluation across all backends × datasets.

Usage:
    uv run scripts/eval_comprehensive.py [--resume] [--max-examples 50]
    
Features:
- Incremental JSONL output (resumable)
- Skip completed combinations on resume
- Graceful error handling per combination
- Progress tracking
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# All NER-capable backends
BACKENDS = [
    # Zero-dep (always available)
    "heuristic",
    "stacked",
    "crf",
    # ONNX backends (require models)
    "nuner",
    "gliner_onnx",
    "gliner_multitask",
    "bert_onnx",
]

# Standard NER datasets (PER/ORG/LOC/MISC)
NER_DATASETS = [
    "WikiGold",
    "CoNLL2003Sample",
    "Wnut17",
    "MultiNERD",
]

# Biomedical NER datasets (Disease/Chemical)
BIO_DATASETS = [
    "BC5CDR",       # Disease + Chemical
    "NCBIDisease",  # Disease only
]

# Coreference datasets
COREF_DATASETS = [
    "GAP",          # Gender Ambiguous Pronouns
    # "PreCo",      # Large-scale coref (may be slow)
    # "LitBank",    # Literary coref (requires HF token)
]

# Coref backends
COREF_BACKENDS = [
    "mention_ranking",  # Default coref resolver
]

# Slot-filling datasets (domain-specific entity types)
# Only zero-shot models (nuner, gliner*) can attempt these
SLOT_DATASETS = [
    "MitMovie",      # ACTOR, DIRECTOR, GENRE, etc.
    "MitRestaurant", # CUISINE, DISH, PRICE, etc.
]

# Default: NER datasets (now includes FewNERD after fixing API pagination)
DEFAULT_DATASETS = NER_DATASETS + ["FewNERD"]

# All datasets (including biomedical)
ALL_DATASETS = DEFAULT_DATASETS + BIO_DATASETS

# Backends that can handle arbitrary labels (zero-shot)
ZERO_SHOT_BACKENDS = ["nuner", "gliner_onnx", "gliner_multitask"]

RESULTS_FILE = Path("reports/eval-comprehensive.jsonl")
SUMMARY_FILE = Path("reports/eval-summary.json")
MARKDOWN_FILE = Path("reports/RESULTS.md")


def get_completed_combinations(results_file: Path) -> set[tuple[str, str, str]]:
    """Load already-completed (backend, dataset, task) tuples."""
    completed = set()
    if results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("status") in ("success", "skipped"):
                        task = r.get("task", "ner")  # Default to "ner" for old entries
                        completed.add((r["backend"], r["dataset"], task))
                except json.JSONDecodeError:
                    continue
    return completed


def run_single_eval(backend: str, dataset: str, max_examples: int, seed: int, task: str = "ner") -> dict:
    """Run evaluation for a single backend/dataset combination."""
    start = time.time()
    result = {
        "backend": backend,
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "max_examples": max_examples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    
    # Build command
    cmd = [
        "./target/release/anno",
        "benchmark",
        "--tasks", task,
        "--backends", backend,
        "--datasets", dataset,
        "--max-examples", str(max_examples),
        "--seed", str(seed),
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per combination
            cwd=Path(__file__).parent.parent,
        )
        
        duration = time.time() - start
        result["duration_s"] = round(duration, 2)
        result["exit_code"] = proc.returncode
        
        output = proc.stdout + proc.stderr
        
        # Check for dataset loading errors (e.g., HuggingFace 422)
        if "Failed to load dataset" in output or "status code 422" in output:
            result["status"] = "dataset_error"
            result["reason"] = "Dataset loading failed (API error or unavailable)"
        elif proc.returncode == 0:
            result["status"] = "success"
            
            # Extract F1 from markdown table: | Dataset | Backend | F1 | P | R | N | ms |
            for line in output.split("\n"):
                # Match lines like: | WikiGold | nuner | 76.9 | 76.9 | 76.9 | 5 | 5591 |
                if "|" in line and backend in line.lower():
                    parts = [p.strip() for p in line.split("|")]
                    # parts[0] is empty, parts[1] is dataset, parts[2] is backend, parts[3] is F1...
                    if len(parts) >= 7:
                        try:
                            f1_val = parts[3]
                            p_val = parts[4]
                            r_val = parts[5]
                            if f1_val and f1_val != "F1" and f1_val != "-":
                                result["f1"] = float(f1_val)
                                result["precision"] = float(p_val)
                                result["recall"] = float(r_val)
                                break
                        except (ValueError, IndexError):
                            pass
            
            # Check for 0% F1 with incompatible entity types
            if result.get("f1") == 0.0 and "doesn't support dataset entity types" in output:
                result["status"] = "incompatible"
                result["reason"] = "Backend entity types don't match dataset"
            
            # Check if backend was skipped (0 runs)
            if "Total: 0" in output:
                result["status"] = "skipped"
                result["reason"] = "No compatible task/dataset combination"
        else:
            result["status"] = "error"
            result["error"] = proc.stderr[:500] if proc.stderr else "Unknown error"
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["duration_s"] = 300
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
    
    return result


def append_result(result: dict, results_file: Path):
    """Append a single result to the JSONL file."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def generate_summary(results_file: Path) -> dict:
    """Generate summary statistics from results."""
    results = []
    if results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    
    successful = [r for r in results if r.get("status") == "success" and r.get("f1") is not None]
    
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(results),
        "successful": len(successful),
        "skipped": len([r for r in results if r.get("status") == "skipped"]),
        "incompatible": len([r for r in results if r.get("status") == "incompatible"]),
        "dataset_errors": len([r for r in results if r.get("status") == "dataset_error"]),
        "errors": len([r for r in results if r.get("status") == "error"]),
        "timeouts": len([r for r in results if r.get("status") == "timeout"]),
    }
    
    if successful:
        f1_scores = [r["f1"] for r in successful]
        summary["avg_f1"] = round(sum(f1_scores) / len(f1_scores), 1)
        summary["best_f1"] = max(f1_scores)
        best = max(successful, key=lambda r: r["f1"])
        summary["best"] = f"{best['backend']}/{best['dataset']}"
        
        # By backend
        summary["by_backend"] = {}
        for backend in set(r["backend"] for r in successful):
            backend_results = [r for r in successful if r["backend"] == backend]
            f1s = [r["f1"] for r in backend_results]
            summary["by_backend"][backend] = {
                "avg_f1": round(sum(f1s) / len(f1s), 1),
                "best_f1": max(f1s),
                "runs": len(backend_results),
            }
        
        # By dataset
        summary["by_dataset"] = {}
        for dataset in set(r["dataset"] for r in successful):
            dataset_results = [r for r in successful if r["dataset"] == dataset]
            f1s = [r["f1"] for r in dataset_results]
            summary["by_dataset"][dataset] = {
                "avg_f1": round(sum(f1s) / len(f1s), 1),
                "best_f1": max(f1s),
                "runs": len(dataset_results),
            }
    
    return summary


def generate_markdown(results_file: Path, summary: dict) -> str:
    """Generate markdown report from results."""
    results = []
    if results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    
    successful = [r for r in results if r.get("status") == "success" and r.get("f1") is not None]
    
    md = ["# Comprehensive Evaluation Results\n"]
    md.append(f"Generated: {summary['generated']}\n")
    md.append(f"\n## Summary\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Total runs | {summary['total_runs']} |")
    md.append(f"| Successful | {summary['successful']} |")
    md.append(f"| Skipped | {summary.get('skipped', 0)} |")
    md.append(f"| Incompatible | {summary.get('incompatible', 0)} |")
    md.append(f"| Dataset errors | {summary.get('dataset_errors', 0)} |")
    md.append(f"| Errors | {summary.get('errors', 0)} |")
    if summary.get('avg_f1'):
        md.append(f"| Avg F1 | {summary['avg_f1']}% |")
        md.append(f"| Best F1 | {summary['best_f1']}% |")
        md.append(f"| Best | {summary['best']} |")
    md.append("")
    
    # By backend
    if summary.get("by_backend"):
        md.append("\n## By Backend\n")
        md.append("| Backend | Avg F1 | Best F1 | Runs |")
        md.append("|---------|--------|---------|------|")
        for backend, stats in sorted(summary["by_backend"].items(), key=lambda x: -x[1]["avg_f1"]):
            md.append(f"| {backend} | {stats['avg_f1']}% | {stats['best_f1']}% | {stats['runs']} |")
        md.append("")
    
    # By dataset
    if summary.get("by_dataset"):
        md.append("\n## By Dataset\n")
        md.append("| Dataset | Avg F1 | Best F1 | Runs |")
        md.append("|---------|--------|---------|------|")
        for dataset, stats in sorted(summary["by_dataset"].items(), key=lambda x: -x[1]["avg_f1"]):
            md.append(f"| {dataset} | {stats['avg_f1']}% | {stats['best_f1']}% | {stats['runs']} |")
        md.append("")
    
    # Full results matrix
    if successful:
        md.append("\n## Results Matrix\n")
        md.append("| Dataset | Backend | F1 | P | R | Time |")
        md.append("|---------|---------|-----|-----|-----|------|")
        for r in sorted(successful, key=lambda x: (-x.get("f1", 0), x["dataset"])):
            md.append(f"| {r['dataset']} | {r['backend']} | {r.get('f1', '-')} | {r.get('precision', '-')} | {r.get('recall', '-')} | {r.get('duration_s', '-')}s |")
        md.append("")
    
    # Errors
    errors = [r for r in results if r.get("status") == "error"]
    if errors:
        md.append("\n## Errors\n")
        for r in errors:
            md.append(f"- **{r['backend']}/{r['dataset']}**: {r.get('error', 'Unknown')[:100]}")
        md.append("")
    
    md.append(f"\n---\n*Raw data: [eval-comprehensive.jsonl](eval-comprehensive.jsonl)*\n")
    
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive resumable evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--max-examples", type=int, default=50, help="Max examples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--backends", type=str, help="Comma-separated backends (default: all)")
    parser.add_argument("--datasets", type=str, help="Comma-separated datasets (default: NER only)")
    parser.add_argument("--include-slots", action="store_true", 
                        help="Include slot-filling datasets (MitMovie, MitRestaurant) with zero-shot backends")
    parser.add_argument("--include-bio", action="store_true",
                        help="Include biomedical datasets (BC5CDR, NCBIDisease)")
    parser.add_argument("--include-coref", action="store_true",
                        help="Include coreference datasets (GAP) with coref backends")
    parser.add_argument("--all", action="store_true",
                        help="Run all datasets (NER + FewNERD + biomedical + coref)")
    args = parser.parse_args()
    
    backends = args.backends.split(",") if args.backends else BACKENDS
    
    # Determine dataset list
    if args.datasets:
        datasets = args.datasets.split(",")
    elif args.all:
        datasets = list(ALL_DATASETS)
    else:
        datasets = list(DEFAULT_DATASETS)
    
    # Add biomedical if requested
    if args.include_bio and not args.all:
        for ds in BIO_DATASETS:
            if ds not in datasets:
                datasets.append(ds)
    
    # Add coref datasets and backends if requested
    coref_combos = []
    if args.include_coref or args.all:
        for backend in COREF_BACKENDS:
            for ds in COREF_DATASETS:
                coref_combos.append((backend, ds, "coref"))
    
    # If including slots, add slot datasets for zero-shot backends only
    slot_combos = set()
    if args.include_slots:
        for ds in SLOT_DATASETS:
            if ds not in datasets:
                datasets.append(ds)
        # Track which combos are slot-filling for skip logic
        for b in backends:
            for ds in SLOT_DATASETS:
                if b not in ZERO_SHOT_BACKENDS:
                    slot_combos.add((b, ds))
    
    # Get completed combinations if resuming
    completed = get_completed_combinations(RESULTS_FILE) if args.resume else set()
    if not args.resume and RESULTS_FILE.exists():
        # Start fresh
        RESULTS_FILE.unlink()
    
    # Build all combinations: NER + coref
    ner_combos = [(b, d, "ner") for b in backends for d in datasets]
    all_combos = ner_combos + coref_combos
    
    total = len(all_combos)
    done = len(completed)
    
    print(f"=== Comprehensive Evaluation ===")
    print(f"NER: {len(backends)} backends × {len(datasets)} datasets = {len(ner_combos)}")
    if coref_combos:
        print(f"Coref: {len(COREF_BACKENDS)} backends × {len(COREF_DATASETS)} datasets = {len(coref_combos)}")
    print(f"Total combinations: {total}")
    print(f"Already completed: {done}")
    print(f"Remaining: {total - done}")
    print()
    
    for idx, (backend, dataset, task) in enumerate(all_combos, 1):
        combo = (backend, dataset, task)
        
        if combo in completed:
            print(f"[{idx}/{total}] SKIP {backend}/{dataset}/{task} (already done)")
            continue
        
        # Skip non-zero-shot backends on slot-filling datasets
        if (backend, dataset) in slot_combos:
            print(f"[{idx}/{total}] SKIP {backend}/{dataset} (slot dataset needs zero-shot)")
            result = {
                "backend": backend,
                "dataset": dataset,
                "task": task,
                "status": "skipped",
                "reason": "Slot-filling dataset requires zero-shot backend",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            append_result(result, RESULTS_FILE)
            continue
        
        print(f"[{idx}/{total}] Running {backend}/{dataset}/{task}...", end=" ", flush=True)
        
        result = run_single_eval(backend, dataset, args.max_examples, args.seed, task)
        append_result(result, RESULTS_FILE)
        
        status = result["status"]
        if status == "success":
            f1_str = f"{result.get('f1', 0):.1f}" if result.get('f1') is not None else "?"
            print(f"OK F1={f1_str}% ({result.get('duration_s', '?')}s)")
        elif status == "skipped":
            print(f"-- skipped")
        elif status == "incompatible":
            print(f"-- incompatible types")
        elif status == "dataset_error":
            print(f"!! dataset load failed")
        elif status == "timeout":
            print(f"!! timeout")
        else:
            print(f"!! {result.get('error', 'error')[:50]}")
    
    # Generate summary and markdown
    print("\n=== Generating reports ===")
    summary = generate_summary(RESULTS_FILE)
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {SUMMARY_FILE}")
    
    markdown = generate_markdown(RESULTS_FILE, summary)
    with open(MARKDOWN_FILE, "w") as f:
        f.write(markdown)
    print(f"Markdown: {MARKDOWN_FILE}")
    
    print("\n=== Final Summary ===")
    print(f"Total: {summary['total_runs']}")
    print(f"Successful: {summary['successful']}")
    if summary.get('avg_f1'):
        print(f"Avg F1: {summary['avg_f1']}%")
        print(f"Best: {summary['best']} ({summary['best_f1']}%)")


if __name__ == "__main__":
    main()

