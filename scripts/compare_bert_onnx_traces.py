#!/usr/bin/env python3
"""Compare a Rust ``BertNerTrace`` JSON file with the Python ORT reference.

The inputs must be traces of the same text and exact graph/tokenizer/config
artifacts. It checks input preparation and raw logits before reporting the two
decoder outputs separately; decoder differences therefore cannot be mistaken
for an ONNX or tokenizer mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ABS_TOLERANCE = 1e-5
REL_TOLERANCE = 1e-5


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def byte_offsets(text: str, character_offsets: list[list[int]]) -> list[list[int]]:
    """Convert Python fast-tokenizer Unicode scalar offsets to UTF-8 bytes."""

    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    return [[boundaries[start], boundaries[end]] for start, end in character_offsets]


def close(left: float, right: float) -> bool:
    return abs(left - right) <= ABS_TOLERANCE or abs(left - right) <= REL_TOLERANCE * max(
        abs(left), abs(right)
    )


def artifact(trace: dict, role: str) -> dict | None | object:
    artifacts = trace.get("artifacts")
    if not isinstance(artifacts, dict) or role not in artifacts:
        return _MISSING
    value = artifacts[role]
    return value if isinstance(value, dict) or value is None else _MISSING


def artifact_sha(artifact: dict | None | object) -> str | None:
    if not isinstance(artifact, dict):
        return None
    sha256 = artifact.get("sha256")
    return sha256 if isinstance(sha256, str) else None


def valid_sha256(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


_MISSING = object()


def compare_artifact_identity(rust: dict, reference: dict) -> list[str]:
    failures: list[str] = []
    for role in ("graph", "tokenizer"):
        rust_sha = artifact_sha(artifact(rust, role))
        reference_sha = artifact_sha(artifact(reference, role))
        if not (valid_sha256(rust_sha) and valid_sha256(reference_sha)):
            failures.append(f"artifact {role} is missing or has an invalid SHA-256")
        elif rust_sha != reference_sha:
            failures.append(f"artifact {role} differs: rust={rust_sha!r} reference={reference_sha!r}")

    rust_config = artifact(rust, "config")
    reference_config = artifact(reference, "config")
    if rust_config is None and reference_config is None:
        # Exact labels are checked separately below, so a local CoNLL fallback
        # cannot be conflated with a selected config artifact.
        return failures
    rust_sha = artifact_sha(rust_config)
    reference_sha = artifact_sha(reference_config)
    if not (valid_sha256(rust_sha) and valid_sha256(reference_sha)):
        failures.append("artifact config is missing or has an invalid SHA-256")
    elif rust_sha != reference_sha:
        failures.append(f"artifact config differs: rust={rust_sha!r} reference={reference_sha!r}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text", required=True, help="the text used for both traces")
    args = parser.parse_args()

    rust = load(args.rust)
    reference = load(args.reference)
    failures: list[str] = []
    failures.extend(compare_artifact_identity(rust, reference))
    for field in ("input_ids", "attention_mask", "token_type_ids", "tokens", "labels"):
        if rust.get(field) != reference.get(field):
            failures.append(f"{field} differs")

    reference_offsets = byte_offsets(args.text, reference["offsets"])
    if rust.get("offsets") != reference_offsets:
        failures.append("offsets differ after Python-character to Rust-byte conversion")

    rust_logits = rust.get("logits", [])
    reference_logits = reference.get("logits", [])
    if len(rust_logits) != len(reference_logits) or any(
        len(left) != len(right) for left, right in zip(rust_logits, reference_logits)
    ):
        failures.append("logit shape differs")
    else:
        for token, (rust_row, reference_row) in enumerate(
            zip(rust_logits, reference_logits)
        ):
            for label, (rust_value, reference_value) in enumerate(
                zip(rust_row, reference_row)
            ):
                if not (math.isfinite(rust_value) and math.isfinite(reference_value)) or not close(
                    rust_value, reference_value
                ):
                    failures.append(
                        f"logit[{token}][{label}] differs: rust={rust_value} reference={reference_value}"
                    )
                    break
            if failures and failures[-1].startswith("logit["):
                break

    result = {
        "tolerance": {"absolute": ABS_TOLERANCE, "relative": REL_TOLERANCE},
        "artifact_identity_match": not any(
            failure.startswith("artifact ") for failure in failures
        ),
        "stages_match": not failures,
        "failures": failures,
        "rust_decoder_entities": rust.get("entities"),
        "reference_strict_bio_entities": reference.get("strict_bio_entities"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
