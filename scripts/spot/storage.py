"""Read the operator's bucket choice and its saved Spot configuration."""

import os
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.env")


def saved_settings(config_file=CONFIG_PATH):
    if not config_file.is_file():
        return {}
    settings = dict(
        line.split("=", 1) for line in config_file.read_text().splitlines()
        if line.startswith("ANNO_SPOT_") and "=" in line
    )
    # Older setup versions saved a hard-coded bucket without an operator choice.
    return settings if settings.get("ANNO_SPOT_CONFIG_VERSION") == "2" else {}


def bucket_name(value):
    name = (value or "").removeprefix("s3://").removesuffix("/")
    if ".." in name or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", name):
        raise ValueError(
            "Set ANNO_SPOT_BUCKET or ANNO_S3_BUCKET to a bucket name "
            "or s3://bucket/ without an object path."
        )
    return name


def selected_bucket(override=None, config_file=CONFIG_PATH):
    if override is None:
        override = os.environ.get("ANNO_SPOT_BUCKET") or os.environ.get("ANNO_S3_BUCKET")
        if not override:
            override = saved_settings(config_file).get("ANNO_SPOT_BUCKET")
    return bucket_name(override)


def fleet_settings(bucket, config_file=CONFIG_PATH):
    """Use saved queue and region only when they belong to the selected bucket."""
    saved = saved_settings(config_file)
    if not saved.get("ANNO_SPOT_BUCKET") or bucket_name(saved["ANNO_SPOT_BUCKET"]) != bucket:
        saved = {}
    region = os.getenv("ANNO_SPOT_REGION") or saved.get("ANNO_SPOT_REGION") or "us-east-1"
    queue = os.getenv("ANNO_SPOT_QUEUE") or saved.get("ANNO_SPOT_QUEUE")
    same_queue = not queue or queue == saved.get("ANNO_SPOT_QUEUE")
    url = os.getenv("ANNO_SPOT_QUEUE_URL") or (saved.get("ANNO_SPOT_QUEUE_URL") if same_queue else "")
    return region, queue or ("anno-eval-" + bucket.replace(".", "_") if bucket else ""), url or ""
