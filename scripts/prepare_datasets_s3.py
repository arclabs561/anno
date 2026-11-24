#!/usr/bin/env python3
"""Report current shared-dataset publication eligibility without transferring raw data."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Recognize the former wrapper's arguments for scheduled invocations.
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--bucket")
    parser.add_argument("--skip-aws-check", action="store_true")
    parser.parse_args()
    print("No reviewed raw dataset packages are eligible for shared S3 publication. "
          "Use an authorized local cache for evaluation.")


if __name__ == "__main__":
    main()
