#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Pre-download datasets and upload to S3 for spot evaluation.

This script downloads problematic datasets (those prone to HuggingFace API errors)
and uploads them to S3 so spot instances can use them without hitting API rate limits.

Usage:
    uv run scripts/prepare_datasets_s3.py
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Datasets prone to HuggingFace API errors or slow downloads
PROBLEMATIC_DATASETS = [
    "FewNERD",      # Fine-grained, large dataset
    "MultiNERD",    # Multilingual, large dataset
    "BC5CDR",       # Biomedical, may require authentication
    "NCBIDisease",  # Biomedical, may require authentication
    "GAP",          # Coreference dataset
    "PreCo",        # Coreference dataset
    "LitBank",      # Coreference dataset
]

def check_aws_credentials():
    """Check if AWS credentials are configured."""
    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info("AWS credentials configured")
            return True
        else:
            logger.error("AWS credentials not configured: %s", result.stderr)
            return False
    except FileNotFoundError:
        logger.error("AWS CLI not found. Install with: pip install awscli")
        return False
    except subprocess.TimeoutExpired:
        logger.error("AWS credentials check timed out")
        return False

def check_anno_binary():
    """Find the anno binary."""
    # Check environment variable first
    if "ANNO_BIN" in os.environ:
        bin_path = Path(os.environ["ANNO_BIN"])
        if bin_path.exists():
            return bin_path
        logger.warning("ANNO_BIN set but file doesn't exist: %s", bin_path)
    
    # Check common locations
    workspace_root = Path(__file__).parent.parent
    for path in [
        workspace_root / "target" / "release" / "anno",
        workspace_root / "target" / "debug" / "anno",
    ]:
        if path.exists():
            return path
    
    logger.error("anno binary not found. Build with: cargo build --release -p anno-cli --bin anno --features eval")
    return None

def download_dataset(anno_bin: Path, dataset: str, s3_bucket: str) -> bool:
    """Download a dataset using anno CLI, which will upload to S3 if enabled."""
    logger.info("Downloading dataset: %s", dataset)
    
    # Set environment variables for S3 upload
    env = os.environ.copy()
    env["ANNO_S3_CACHE"] = "1"
    env["ANNO_S3_BUCKET"] = s3_bucket
    
    # Use 'dataset info' command which triggers download if not cached
    cmd = [
        str(anno_bin),
        "dataset",
        "info",
        "--dataset", dataset,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per dataset
        )
        
        if result.returncode == 0:
            logger.info("✓ Successfully downloaded %s", dataset)
            # Check if dataset was loaded (has statistics)
            if "Sentences:" in result.stdout or "Loaded Statistics:" in result.stdout:
                logger.info("  Dataset loaded successfully")
                return True
            else:
                logger.warning("  Dataset info retrieved but may not be fully loaded")
                return True
        else:
            logger.error("✗ Failed to download %s: %s", dataset, result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("✗ Timeout downloading %s (exceeded 10 minutes)", dataset)
        return False
    except Exception as e:
        logger.error("✗ Error downloading %s: %s", dataset, e)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Pre-download datasets and upload to S3 for spot evaluation"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=PROBLEMATIC_DATASETS,
        help="Datasets to download (default: problematic datasets)"
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("ANNO_SPOT_BUCKET", "arc-anno-data"),
        help="S3 bucket name (default: from ANNO_SPOT_BUCKET or arc-anno-data)"
    )
    parser.add_argument(
        "--skip-aws-check",
        action="store_true",
        help="Skip AWS credentials check (for testing)"
    )
    args = parser.parse_args()
    
    # Check prerequisites
    if not args.skip_aws_check and not check_aws_credentials():
        logger.error("AWS credentials required. Configure with: aws configure")
        sys.exit(1)
    
    anno_bin = check_anno_binary()
    if not anno_bin:
        sys.exit(1)
    
    logger.info("Using anno binary: %s", anno_bin)
    logger.info("S3 bucket: %s", args.bucket)
    logger.info("Datasets to download: %s", ", ".join(args.datasets))
    logger.info("")
    
    # Download each dataset
    success_count = 0
    failed_datasets = []
    
    for dataset in args.datasets:
        if download_dataset(anno_bin, dataset, args.bucket):
            success_count += 1
        else:
            failed_datasets.append(dataset)
        logger.info("")  # Blank line between datasets
    
    # Summary
    logger.info("=" * 60)
    logger.info("Summary: %d/%d datasets downloaded successfully", success_count, len(args.datasets))
    if failed_datasets:
        logger.warning("Failed datasets: %s", ", ".join(failed_datasets))
        sys.exit(1)
    else:
        logger.info("All datasets downloaded and uploaded to S3 successfully!")
        logger.info("Spot instances can now use these datasets without API calls")

if __name__ == "__main__":
    main()

