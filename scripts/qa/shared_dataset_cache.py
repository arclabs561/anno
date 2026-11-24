"""Print raw-source object exclusions for reads from pre-provisioned S3 caches."""
from pathlib import Path

RULES = Path(__file__).resolve().parents[2] / "crates/anno-eval/src/eval/shared_cache_exclusions.tsv"


def excluded_patterns():
    for row in RULES.read_text().splitlines():
        filename, _, aliases = row.split("\t")
        for name in dict.fromkeys([filename.removesuffix(".cache"), *aliases.split(",")]):
            yield f"*{name}*"


if __name__ == "__main__":
    print("\n".join(excluded_patterns()))
