"""Validate the configured CI destination before including it in an IAM policy."""
import ipaddress
import os
import re


def validate(bucket, role):
    if not role.strip():
        raise ValueError("Set the ANNO_AWS_ROLE_ARN repository variable to an authorized read role.")
    if not bucket:
        raise ValueError("Supply a bucket or set the ANNO_S3_BUCKET repository variable.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) or ".." in bucket:
        raise ValueError("Expected a plain S3 bucket name for the dataset cache.")
    try:
        ipaddress.ip_address(bucket)
    except ValueError:
        return
    raise ValueError("Expected an S3 bucket name, received an IP address.")


if __name__ == "__main__":
    try:
        validate(os.environ.get("BUCKET", ""), os.environ.get("CACHE_ROLE", ""))
    except ValueError as error:
        raise SystemExit(str(error))
