#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.34", "rich>=13.0"]
# ///
"""
Orchestrator for distributed spot evaluation.

Generates tasks, launches fleet, monitors progress, aggregates results.

Usage:
    uv run scripts/spot/orchestrate.py generate   # Generate tasks for queue
    uv run scripts/spot/orchestrate.py launch     # Launch spot fleet
    uv run scripts/spot/orchestrate.py status     # Check progress
    uv run scripts/spot/orchestrate.py results    # Aggregate results
    uv run scripts/spot/orchestrate.py teardown   # Clean up
    uv run scripts/spot/orchestrate.py full       # Generate + launch + wait + results
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import boto3
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    region: str = "us-east-1"
    bucket: str = "arc-anno-data"
    queue_name: str = "anno-eval-tasks"
    queue_url: str = ""
    launch_template: str = "anno-eval-worker"
    fleet_size: int = 4
    instance_types: list[str] = None
    
    def __post_init__(self):
        if self.instance_types is None:
            self.instance_types = ["c7i.xlarge", "c7a.xlarge", "m7i.xlarge", "c6i.xlarge"]
        
        # Load from environment
        self.region = os.environ.get("ANNO_SPOT_REGION", self.region)
        self.bucket = os.environ.get("ANNO_SPOT_BUCKET", self.bucket)
        self.queue_name = os.environ.get("ANNO_SPOT_QUEUE", self.queue_name)
        self.fleet_size = int(os.environ.get("ANNO_SPOT_FLEET_SIZE", self.fleet_size))
        
        # Try loading from config file
        config_path = Path(__file__).parent / "config.env"
        if config_path.exists():
            for line in config_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    if key == "ANNO_SPOT_QUEUE_URL":
                        self.queue_url = value


# ============================================================================
# Task Generation
# ============================================================================

# Backend groups for different evaluation profiles
BACKENDS_ZERO_DEP = [
    # Always available (pure Rust, no ML models)
    "heuristic",   # PER/ORG/LOC via heuristics
    "stacked",     # pattern + heuristic combined
]

BACKENDS_ONNX = [
    # Require ONNX runtime + models (feature: onnx)
    "gliner",      # GLiNER multi-v2.1
    "gliner2",     # GLiNER v2 variant
    "nuner",       # NuNER model
    # "w2ner",     # W2NER (if model available)
    # "bert_onnx", # BERT-based NER (if model available)
]

BACKENDS_CANDLE = [
    # Require Candle runtime (feature: candle)
    # "gliner_candle",  # GLiNER via Candle (Metal on Mac)
]

# Default: all available backends
BACKENDS = BACKENDS_ZERO_DEP + BACKENDS_ONNX + BACKENDS_CANDLE

# Datasets with good coverage (must be valid DatasetId names, not "synthetic")
DATASETS = [
    # Real NER
    "WikiGold",
    "Wnut17",
    "MitMovie",
    "MitRestaurant",
    "CoNLL2003Sample",
    "BC5CDR",
    "NCBIDisease",
    "MultiNERD",
    "FewNERD",
    # Coreference
    "GAP",
    "PreCo",
    "LitBank",
]

# Seeds for statistical significance
DEFAULT_SEEDS = [42, 123, 456, 789, 999]


def generate_tasks(
    backends: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    max_examples: int = 500,
    task_type: str = "ner",
) -> Iterator[dict]:
    """Generate (backend, dataset, seed) task combinations."""
    backends = backends or BACKENDS
    datasets = datasets or DATASETS
    seeds = seeds or DEFAULT_SEEDS
    
    for backend in backends:
        for dataset in datasets:
            for seed in seeds:
                yield {
                    "backend": backend,
                    "dataset": dataset,
                    "seed": seed,
                    "max_examples": max_examples,
                    "task": task_type,
                }


def enqueue_tasks(config: Config, tasks: list[dict]) -> int:
    """Send tasks to SQS queue."""
    sqs = boto3.client("sqs", region_name=config.region)
    
    # Get queue URL if not cached
    if not config.queue_url:
        resp = sqs.get_queue_url(QueueName=config.queue_name)
        config.queue_url = resp["QueueUrl"]
    
    # Batch send (max 10 per batch)
    sent = 0
    for i in range(0, len(tasks), 10):
        batch = tasks[i:i+10]
        entries = [
            {
                "Id": str(j),
                "MessageBody": json.dumps(task),
                "MessageGroupId": task["backend"] if config.queue_name.endswith(".fifo") else None,
            }
            for j, task in enumerate(batch)
        ]
        # Remove None MessageGroupId for standard queues
        entries = [{k: v for k, v in e.items() if v is not None} for e in entries]
        
        sqs.send_message_batch(QueueUrl=config.queue_url, Entries=entries)
        sent += len(batch)
    
    return sent


def get_backends_for_profile(profile: str, explicit_backends: str | None) -> list[str] | None:
    """Get backend list based on profile or explicit override."""
    if explicit_backends:
        return explicit_backends.split(",")
    if profile == "quick":
        return BACKENDS_ZERO_DEP
    elif profile == "onnx":
        return BACKENDS_ONNX
    else:  # full
        return BACKENDS  # All backends


def cmd_generate(args, config: Config):
    """Generate and enqueue evaluation tasks."""
    backends = get_backends_for_profile(args.profile, args.backends)
    
    tasks = list(generate_tasks(
        backends=backends,
        datasets=args.datasets.split(",") if args.datasets else None,
        seeds=[int(s) for s in args.seeds.split(",")] if args.seeds else None,
        max_examples=args.max_examples,
    ))
    
    console.print(f"[bold]Generated {len(tasks)} tasks[/bold]")
    
    if args.dry_run:
        for task in tasks[:10]:
            console.print(f"  {task}")
        if len(tasks) > 10:
            console.print(f"  ... and {len(tasks) - 10} more")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Enqueuing tasks...", total=None)
        sent = enqueue_tasks(config, tasks)
    
    console.print(f"[green]Enqueued {sent} tasks to {config.queue_name}[/green]")


# ============================================================================
# Fleet Management
# ============================================================================

def launch_fleet(config: Config, target_capacity: int) -> str:
    """Launch spot fleet with given capacity."""
    ec2 = boto3.client("ec2", region_name=config.region)
    
    # Build launch template overrides for multiple instance types
    overrides = [
        {"InstanceType": itype}
        for itype in config.instance_types
    ]
    
    response = ec2.request_spot_fleet(
        SpotFleetRequestConfig={
            "IamFleetRole": f"arn:aws:iam::{get_account_id()}:role/aws-service-role/spotfleet.amazonaws.com/AWSServiceRoleForEC2SpotFleet",
            "TargetCapacity": target_capacity,
            "TerminateInstancesWithExpiration": True,
            "Type": "maintain",
            "AllocationStrategy": "capacityOptimized",
            "LaunchTemplateConfigs": [
                {
                    "LaunchTemplateSpecification": {
                        "LaunchTemplateName": config.launch_template,
                        "Version": "$Latest",
                    },
                    "Overrides": overrides,
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "spot-fleet-request",
                    "Tags": [
                        {"Key": "Name", "Value": "anno-eval-fleet"},
                        {"Key": "Project", "Value": "anno"},
                        {"Key": "Component", "Value": "eval-fleet"},
                        {"Key": "Environment", "Value": "eval"},
                        {"Key": "ManagedBy", "Value": "anno-spot-scripts"},
                        {"Key": "CostCenter", "Value": "ml-eval"},
                        {"Key": "RunId", "Value": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")},
                    ],
                }
            ],
        }
    )
    
    return response["SpotFleetRequestId"]


def get_account_id() -> str:
    """Get current AWS account ID."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def cmd_launch(args, config: Config):
    """Launch spot fleet."""
    fleet_size = args.fleet_size or config.fleet_size
    
    console.print(f"[bold]Launching spot fleet with {fleet_size} instances...[/bold]")
    
    fleet_id = launch_fleet(config, fleet_size)
    
    console.print(f"[green]Fleet launched: {fleet_id}[/green]")
    console.print(f"Monitor with: aws ec2 describe-spot-fleet-instances --spot-fleet-request-id {fleet_id}")
    
    # Save fleet ID
    (Path(__file__).parent / "fleet_id.txt").write_text(fleet_id)


def cmd_status(args, config: Config):
    """Show evaluation progress."""
    sqs = boto3.client("sqs", region_name=config.region)
    ec2 = boto3.client("ec2", region_name=config.region)
    s3 = boto3.client("s3", region_name=config.region)
    
    # Queue status
    if not config.queue_url:
        resp = sqs.get_queue_url(QueueName=config.queue_name)
        config.queue_url = resp["QueueUrl"]
    
    attrs = sqs.get_queue_attributes(
        QueueUrl=config.queue_url,
        AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"]
    )["Attributes"]
    
    pending = int(attrs.get("ApproximateNumberOfMessages", 0))
    in_flight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
    
    # Fleet status
    fleet_id_path = Path(__file__).parent / "fleet_id.txt"
    fleet_info = None
    if fleet_id_path.exists():
        fleet_id = fleet_id_path.read_text().strip()
        try:
            resp = ec2.describe_spot_fleet_requests(SpotFleetRequestIds=[fleet_id])
            if resp["SpotFleetRequestConfigs"]:
                fleet_info = resp["SpotFleetRequestConfigs"][0]
        except Exception:
            pass
    
    # Results count
    result_count = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config.bucket, Prefix="results/"):
            result_count += page.get("KeyCount", 0)
    except Exception:
        pass
    
    # Display table
    table = Table(title="Evaluation Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Tasks Pending", str(pending))
    table.add_row("Tasks In-Flight", str(in_flight))
    table.add_row("Results Uploaded", str(result_count))
    
    if fleet_info:
        table.add_row("Fleet State", fleet_info.get("SpotFleetRequestState", "unknown"))
        table.add_row("Target Capacity", str(fleet_info.get("SpotFleetRequestConfig", {}).get("TargetCapacity", 0)))
        table.add_row("Fulfilled", str(fleet_info.get("SpotFleetRequestConfig", {}).get("FulfilledCapacity", 0)))
    
    console.print(table)
    
    # Completion estimate
    total_tasks = pending + in_flight + result_count
    if total_tasks > 0 and result_count > 0:
        completion = result_count / total_tasks * 100
        console.print(f"\n[bold]Progress: {completion:.1f}% complete[/bold]")


# ============================================================================
# Results Aggregation
# ============================================================================

def aggregate_results(config: Config, output_path: Path) -> dict:
    """Download and aggregate all results from S3."""
    s3 = boto3.client("s3", region_name=config.region)
    
    results = []
    paginator = s3.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=config.bucket, Prefix="results/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") and not key.endswith(".error"):
                try:
                    resp = s3.get_object(Bucket=config.bucket, Key=key)
                    data = json.loads(resp["Body"].read())
                    results.append(data)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to load {key}: {e}[/yellow]")
    
    # Aggregate by backend
    by_backend = {}
    for r in results:
        meta = r.get("_meta", {})
        backend = meta.get("backend", "unknown")
        if backend not in by_backend:
            by_backend[backend] = []
        by_backend[backend].append(r)
    
    # Compute summary statistics
    summary = {
        "total_results": len(results),
        "backends": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    for backend, backend_results in by_backend.items():
        # Extract F1 scores (if present in results)
        f1_scores = []
        for r in backend_results:
            if "f1" in r:
                f1_scores.append(r["f1"])
            elif "metrics" in r and "f1" in r["metrics"]:
                f1_scores.append(r["metrics"]["f1"])
        
        summary["backends"][backend] = {
            "count": len(backend_results),
            "datasets": list({r.get("_meta", {}).get("dataset", "?") for r in backend_results}),
            "mean_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
            "min_f1": min(f1_scores) if f1_scores else None,
            "max_f1": max(f1_scores) if f1_scores else None,
        }
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    
    # Also write raw results
    raw_path = output_path.with_suffix(".raw.json")
    raw_path.write_text(json.dumps(results, indent=2))
    
    return summary


def cmd_results(args, config: Config):
    """Aggregate and display results."""
    output_path = Path(args.output)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Downloading and aggregating results...", total=None)
        summary = aggregate_results(config, output_path)
    
    # Display summary table
    table = Table(title="Evaluation Results Summary")
    table.add_column("Backend", style="cyan")
    table.add_column("Runs", style="white")
    table.add_column("Datasets", style="white")
    table.add_column("Mean F1", style="green")
    table.add_column("Range", style="yellow")
    
    for backend, stats in sorted(summary["backends"].items()):
        mean_f1 = f"{stats['mean_f1']:.3f}" if stats["mean_f1"] else "-"
        f1_range = f"{stats['min_f1']:.3f}-{stats['max_f1']:.3f}" if stats["min_f1"] else "-"
        table.add_row(
            backend,
            str(stats["count"]),
            str(len(stats["datasets"])),
            mean_f1,
            f1_range,
        )
    
    console.print(table)
    console.print(f"\n[bold]Total results: {summary['total_results']}[/bold]")
    console.print(f"Output written to: {output_path}")


# ============================================================================
# Cleanup
# ============================================================================

def cmd_teardown(args, config: Config):
    """Cancel fleet and optionally clean up queue."""
    ec2 = boto3.client("ec2", region_name=config.region)
    sqs = boto3.client("sqs", region_name=config.region)
    
    # Cancel fleet
    fleet_id_path = Path(__file__).parent / "fleet_id.txt"
    if fleet_id_path.exists():
        fleet_id = fleet_id_path.read_text().strip()
        console.print(f"[bold]Canceling fleet: {fleet_id}[/bold]")
        
        ec2.cancel_spot_fleet_requests(
            SpotFleetRequestIds=[fleet_id],
            TerminateInstances=True
        )
        fleet_id_path.unlink()
        console.print("[green]Fleet canceled[/green]")
    
    # Optionally purge queue
    if args.purge_queue:
        if not config.queue_url:
            resp = sqs.get_queue_url(QueueName=config.queue_name)
            config.queue_url = resp["QueueUrl"]
        
        console.print(f"[bold]Purging queue: {config.queue_name}[/bold]")
        sqs.purge_queue(QueueUrl=config.queue_url)
        console.print("[green]Queue purged[/green]")


def cmd_full(args, config: Config):
    """Run full evaluation: generate + launch + wait + results."""
    # Generate tasks
    console.print("\n[bold blue]Step 1: Generating tasks...[/bold blue]")
    backends = get_backends_for_profile(args.profile, args.backends)
    
    tasks = list(generate_tasks(
        backends=backends,
        datasets=args.datasets.split(",") if args.datasets else None,
        seeds=[int(s) for s in args.seeds.split(",")] if args.seeds else None,
        max_examples=args.max_examples,
    ))
    sent = enqueue_tasks(config, tasks)
    console.print(f"[green]Enqueued {sent} tasks[/green]")
    
    # Launch fleet
    console.print("\n[bold blue]Step 2: Launching fleet...[/bold blue]")
    fleet_size = args.fleet_size or config.fleet_size
    fleet_id = launch_fleet(config, fleet_size)
    (Path(__file__).parent / "fleet_id.txt").write_text(fleet_id)
    console.print(f"[green]Fleet launched: {fleet_id}[/green]")
    
    # Wait for completion
    console.print("\n[bold blue]Step 3: Waiting for completion...[/bold blue]")
    sqs = boto3.client("sqs", region_name=config.region)
    if not config.queue_url:
        resp = sqs.get_queue_url(QueueName=config.queue_name)
        config.queue_url = resp["QueueUrl"]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting...", total=None)
        
        while True:
            attrs = sqs.get_queue_attributes(
                QueueUrl=config.queue_url,
                AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"]
            )["Attributes"]
            
            pending = int(attrs.get("ApproximateNumberOfMessages", 0))
            in_flight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
            
            if pending == 0 and in_flight == 0:
                break
            
            progress.update(task, description=f"Tasks remaining: {pending} pending, {in_flight} in-flight")
            time.sleep(30)
    
    console.print("[green]All tasks complete![/green]")
    
    # Aggregate results
    console.print("\n[bold blue]Step 4: Aggregating results...[/bold blue]")
    output_path = Path(args.output)
    summary = aggregate_results(config, output_path)
    console.print(f"[green]Results written to {output_path}[/green]")
    
    # Show summary
    console.print(f"\n[bold]Evaluation complete: {summary['total_results']} results across {len(summary['backends'])} backends[/bold]")


# ============================================================================
# Local Evaluation (no AWS)
# ============================================================================

def cmd_local(args, config: Config):
    """Run evaluation locally (no AWS required).
    
    Useful for:
    - Development and testing
    - CI environments without AWS credentials
    - Quick single-backend/dataset runs
    """
    import subprocess
    from datetime import datetime
    
    backends = get_backends_for_profile(args.profile, args.backends)
    datasets = args.datasets.split(",") if args.datasets else ["WikiGold"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [42]
    max_cases = args.max_examples
    
    tasks = list(generate_tasks(
        backends=backends,
        datasets=datasets,
        seeds=seeds,
        max_examples=max_cases,
    ))
    
    console.print(f"[bold]Running {len(tasks)} evaluations locally...[/bold]")
    
    results = []
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find anno binary
    anno_bin = Path("target/release/anno")
    if not anno_bin.exists():
        anno_bin = Path("target/debug/anno")
    if not anno_bin.exists():
        console.print("[red]Error: anno binary not found. Run 'cargo build --release -p anno-cli --features eval-advanced' first.[/red]")
        return
    
    for i, task in enumerate(tasks, 1):
        backend = task["backend"]
        dataset = task["dataset"]
        seed = task["seed"]
        
        console.print(f"\n[{i}/{len(tasks)}] [bold]{backend}[/bold] × [cyan]{dataset}[/cyan] (seed={seed})")
        
        start = datetime.now()
        try:
            result = subprocess.run(
                [
                    str(anno_bin), "dataset", "eval",
                    "--dataset", dataset,
                    "--model", backend,
                    "--task", "ner",
                    "--max-cases", str(max_cases),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per task
            )
            
            duration_ms = int((datetime.now() - start).total_seconds() * 1000)
            
            # Parse output for P/R/F1
            import re
            prf = re.search(r'P:\s*([\d.]+)%\s*R:\s*([\d.]+)%\s*F1:\s*([\d.]+)%', result.stdout)
            n_match = re.search(r'Sentences:\s*(\d+)', result.stdout)
            
            if prf:
                r = {
                    "timestamp": datetime.now().isoformat(),
                    "backend": backend,
                    "dataset": dataset,
                    "seed": seed,
                    "f1": float(prf.group(3)),
                    "precision": float(prf.group(1)),
                    "recall": float(prf.group(2)),
                    "n": int(n_match.group(1)) if n_match else 0,
                    "duration_ms": duration_ms,
                }
                console.print(f"  [green]F1={r['f1']:.1f}% P={r['precision']:.1f}% R={r['recall']:.1f}% ({duration_ms}ms)[/green]")
            else:
                r = {
                    "timestamp": datetime.now().isoformat(),
                    "backend": backend,
                    "dataset": dataset,
                    "seed": seed,
                    "f1": 0, "precision": 0, "recall": 0, "n": 0,
                    "duration_ms": duration_ms,
                    "error": "parse_failed",
                    "stderr": result.stderr[:500] if result.stderr else None,
                }
                console.print(f"  [yellow]No results parsed (exit={result.returncode})[/yellow]")
                if result.stderr:
                    console.print(f"  [dim]{result.stderr[:200]}[/dim]")
            
            results.append(r)
            
        except subprocess.TimeoutExpired:
            console.print(f"  [red]Timeout after 600s[/red]")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "backend": backend, "dataset": dataset, "seed": seed,
                "f1": 0, "precision": 0, "recall": 0, "n": 0,
                "duration_ms": 600000, "error": "timeout",
            })
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "backend": backend, "dataset": dataset, "seed": seed,
                "f1": 0, "precision": 0, "recall": 0, "n": 0,
                "duration_ms": 0, "error": str(e),
            })
    
    # Write results
    output_path = Path(args.output)
    with output_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    console.print(f"\n[bold green]Results written to {output_path}[/bold green]")
    
    # Summary
    successful = [r for r in results if r.get("f1", 0) > 0]
    if successful:
        best = max(successful, key=lambda r: r["f1"])
        console.print(f"[bold]Best: {best['backend']}/{best['dataset']} F1={best['f1']:.1f}%[/bold]")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Orchestrate distributed spot evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # generate
    gen = subparsers.add_parser("generate", help="Generate and enqueue tasks")
    gen.add_argument("--profile", choices=["quick", "full", "onnx"], default="full",
                     help="Backend profile: quick (zero-dep), onnx (ML only), full (all)")
    gen.add_argument("--backends", help="Comma-separated backends (overrides --profile)")
    gen.add_argument("--datasets", help="Comma-separated datasets")
    gen.add_argument("--seeds", help="Comma-separated seeds")
    gen.add_argument("--max-examples", type=int, default=500)
    gen.add_argument("--dry-run", action="store_true")
    
    # launch
    launch = subparsers.add_parser("launch", help="Launch spot fleet")
    launch.add_argument("--fleet-size", type=int)
    
    # status
    subparsers.add_parser("status", help="Check progress")
    
    # results
    res = subparsers.add_parser("results", help="Aggregate results")
    res.add_argument("--output", default="reports/spot-eval-results.json")
    
    # teardown
    td = subparsers.add_parser("teardown", help="Clean up resources")
    td.add_argument("--purge-queue", action="store_true")
    
    # full
    full = subparsers.add_parser("full", help="Full evaluation pipeline")
    full.add_argument("--profile", choices=["quick", "full", "onnx"], default="full",
                      help="Backend profile: quick (zero-dep), onnx (ML only), full (all)")
    full.add_argument("--backends", help="Comma-separated backends (overrides --profile)")
    full.add_argument("--datasets", help="Comma-separated datasets")
    full.add_argument("--seeds", help="Comma-separated seeds")
    full.add_argument("--max-examples", type=int, default=500)
    full.add_argument("--fleet-size", type=int)
    full.add_argument("--output", default="reports/spot-eval-results.json")
    
    # local (no AWS)
    local = subparsers.add_parser("local", help="Run evaluation locally (no AWS)")
    local.add_argument("--profile", choices=["quick", "full", "onnx"], default="quick",
                       help="Backend profile: quick (zero-dep), onnx (ML only), full (all)")
    local.add_argument("--backends", help="Comma-separated backends (overrides --profile)")
    local.add_argument("--datasets", help="Comma-separated datasets", default="WikiGold")
    local.add_argument("--seeds", help="Comma-separated seeds", default="42")
    local.add_argument("--max-examples", type=int, default=50,
                       help="Max examples per dataset (lower for faster local runs)")
    local.add_argument("--output", default="reports/local-eval-results.jsonl")
    
    args = parser.parse_args()
    config = Config()
    
    commands = {
        "generate": cmd_generate,
        "launch": cmd_launch,
        "status": cmd_status,
        "results": cmd_results,
        "teardown": cmd_teardown,
        "full": cmd_full,
        "local": cmd_local,
    }
    
    commands[args.command](args, config)


if __name__ == "__main__":
    main()

