# AWS Spot Instance Evaluation Infrastructure

Parallel NER/Coref evaluation across many backends and datasets using AWS Spot instances.

## Overview

This infrastructure uses:
- **Spot Fleet** for distributed execution
- **Persistent EBS volume** for cargo/sccache caching
- **Parallel work distribution** via SQS queue
- **S3** for results aggregation

## Quick Start

### Local (no AWS, free)

```bash
# Run evaluation locally (no AWS credentials needed)
just eval-local-quick

# Custom backends/datasets
just eval-local BACKENDS="heuristic,stacked" DATASETS="WikiGold,Wnut17" MAX=100
```

### AWS Spot (distributed)

```bash
# 1. Initial setup (one-time)
./scripts/spot/setup.sh

# 2. Launch evaluation fleet
just spot-eval

# 3. Check progress
just spot-status

# 4. Collect results
just spot-results
```

## Architecture

```
┌─────────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Orchestrator   │────▶│  SQS Queue  │◀────│  Spot Workers   │
│  (local/lambda) │     │  (tasks)    │     │  (c7i.xlarge)   │
└─────────────────┘     └─────────────┘     └────────┬────────┘
        │                                            │
        │                                            ▼
        │              ┌─────────────────────────────────────┐
        └─────────────▶│          S3 Bucket                  │
                       │  - datasets/ (cached)               │
                       │  - models/ (cached)                 │
                       │  - results/ (output)                │
                       └─────────────────────────────────────┘
                                     ▲
                                     │
                       ┌─────────────────────────────────────┐
                       │      Persistent EBS Volume          │
                       │  - /mnt/cache/cargo (CARGO_HOME)    │
                       │  - /mnt/cache/sccache               │
                       │  - /mnt/cache/target (build cache)  │
                       └─────────────────────────────────────┘
```

## Work Distribution

Each task in the queue is a (backend, dataset, seed) triple:

```json
{
  "backend": "gliner",
  "dataset": "WikiGold",
  "seed": 42,
  "max_examples": 500,
  "task": "ner"
}
```

Workers pull tasks, run evaluation, write results to S3.

## Costs and runtime

Costs and runtime vary materially with instance types, fleet size, dataset volume, cache warmth, and region.
Treat this directory as a runner, not a benchmark; measure your own spend and wall-clock time for your setup.

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | One-time infrastructure setup (VPC, roles, bucket) |
| `launch.sh` | Launch spot fleet with specified capacity |
| `worker.sh` | Runs on each spot instance (pulls tasks, runs eval) |
| `orchestrate.py` | Generate tasks, monitor progress, aggregate results |
| `monitor.py` | Live worker monitoring via SSM |
| `aggregate.py` | Local aggregation utilities for results |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANNO_SPOT_REGION` | AWS region | us-east-1 |
| `ANNO_SPOT_BUCKET` | S3 bucket for data | arc-anno-data |
| `ANNO_SPOT_QUEUE` | SQS queue name | anno-eval-tasks |
| `ANNO_SPOT_FLEET_SIZE` | Number of spot instances | 4 |
| `ANNO_SPOT_INSTANCE_TYPES` | Comma-separated types | c7i.xlarge,c7a.xlarge,m7i.xlarge |

## Interruption Handling

Spot instances may be reclaimed with 2-minute notice:
1. Worker detects interruption via instance metadata
2. Current task marked as incomplete in SQS (returns to queue)
3. Fleet automatically requests replacement instance
4. New instance picks up the task (idempotent)

## Caching Strategy

### Build Caching (EBS)

1. **EBS volume** mounted at `/mnt/cache`:
   - `CARGO_HOME=/mnt/cache/cargo` - toolchains, registry
   - `SCCACHE_DIR=/mnt/cache/sccache` - compiled artifacts
   - `CARGO_TARGET_DIR=/mnt/cache/target/anno` - build output

2. First instance builds; subsequent instances reuse via shared EBS.

### Data Caching (S3)

On-demand fetch with local caching:
- Datasets: `ANNO_S3_CACHE=1` enables S3 fallback
- Models: `HF_HOME=/mnt/cache/models` for ONNX models

### Prediction caching

Removed: prediction caching was implementation-sensitive and could silently mask regressions. Spot
workers rely on dataset/model caching (S3 + local) and recompute predictions as needed.

## Manual Commands

```bash
# Check fleet status
aws ec2 describe-spot-fleet-requests --region us-east-1

# View queue depth
aws sqs get-queue-attributes --queue-url $QUEUE_URL --attribute-names ApproximateNumberOfMessages

# Cancel fleet (graceful)
aws ec2 cancel-spot-fleet-requests --spot-fleet-request-ids $FLEET_ID --terminate-instances

# List results
aws s3 ls s3://arc-anno-data/results/
```

## Relationship to CI Tests

The spot evaluation complements the CI randomized matrix tests:

| Aspect | CI (`matrix_muxer_ci::test_randomized_matrix_sample`) | Spot Evaluation |
|--------|----------------------------|-----------------|
| **When** | Every PR, every push | On-demand, nightly, pre-release |
| **Scope** | Small cached-only slice | Many backends × datasets × seeds |
| **Examples** | Small cap per dataset | Larger caps per dataset |
| **Duration** | Bounded by CI timeouts | Depends on fleet size and workload |
| **Cost** | Covered by CI | Billed by AWS usage |
| **Goal** | Catch regressions fast | Comprehensive coverage |

### Integration Points

1. **Muxer history (optional)**:
   The CI matrix sampler can read/write muxer history via `ANNO_HISTORY_FILE` (JSON). Spot runs
   can share artifacts in `reports/`, but a direct “spot → muxer history” export is not wired up
   by default in this directory.

2. **CI Seeds Trace Back to Spot**:
   When CI finds a regression, reproduce it in spot:
   ```bash
   # CI failed with seed 12345 on gliner×WikiGold
   just spot-eval-custom "gliner" "WikiGold" 1
   ```

3. **Pre-Release Checklist**:
   ```bash
   # Before release, run comprehensive spot evaluation
   just spot-upload-src
   just spot-eval
   
   # Verify no regressions vs previous release
   just spot-results --compare-baseline reports/v0.1.0-baseline.json
   ```

### Shared Components

Both use the same evaluation harness:
- `anno::eval::task_evaluator::TaskEvaluator`
- `anno::eval::backend_factory::BackendFactory`  
- `anno::eval::loader::DatasetLoader`

The spot worker script calls the same `anno benchmark` CLI that the CI tests use internally.

## Just Commands Reference

### Local (no AWS)

| Command | Description |
|---------|-------------|
| `just eval-local-quick` | Quick local eval (zero-dep backends) |
| `just eval-local BACKENDS DATASETS MAX` | Custom local eval |

### AWS Spot

| Command | Description |
|---------|-------------|
| `just spot-setup` | One-time AWS infrastructure setup |
| `just spot-upload-src` | Upload source code to S3 |
| `just spot-eval-quick` | Quick distributed smoke run |
| `just spot-eval` | Full distributed evaluation |
| `just spot-eval-ml` | ML backends only (ONNX/Candle) |
| `just spot-monitor` | Monitor workers via SSM |
| `just spot-status` | Check fleet and queue status |
| `just spot-results` | Download and aggregate results |
| `just spot-teardown` | Cancel fleet, cleanup |
| `just ci-matrix-local` | Run CI sampler locally |

## trainctl Integration (Optional)

If [trainctl](../../../trainctl) is available, additional features are unlocked:

```bash
just spot-dash              # Interactive ratatui dashboard
just spot-ps                # Process list on worker
just spot-top               # Interactive top on worker
just spot-cost              # Show fleet costs
just spot-sync-datasets     # Fast parallel dataset download
```

The `trainctl-bridge.sh` script provides the integration. Build trainctl with:
```bash
cd ../trainctl && cargo build --release
```

The self-contained approach (`just spot-eval`) works without trainctl.

## Troubleshooting

**Instances not launching:**
- Check Spot capacity in selected AZs
- Try adding more instance types
- Check IAM role has required permissions

**Builds failing:**
- Verify EBS volume is attached and mounted
- Check `/var/log/cloud-init-output.log`
- Ensure CARGO_HOME points to persistent storage

**Tasks not completing:**
- Check CloudWatch logs: `/aws/anno-eval/workers`
- Verify SQS visibility timeout > max task duration
- Check for OOM (increase instance size)

