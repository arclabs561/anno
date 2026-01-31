#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Check DatasetId enum synchronization in anno.

This script validates that:
1. loader.rs re-exports DatasetId from dataset_registry.rs (unified architecture)
2. OR if loader.rs has its own enum, all variants match the registry

Exit codes:
- 0: DatasetId is unified or synced
- 1: There are issues with the DatasetId configuration
"""

import argparse
import json
import re
import sys
from pathlib import Path


def get_registry_ids(json_path: Path) -> set[str]:
    """Extract dataset IDs from the generated JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return {d["id"] for d in data["datasets"]}


def check_loader_architecture(loader_path: Path) -> tuple[str, set[str]]:
    """Check loader.rs architecture and extract any enum variants.
    
    Returns:
        Tuple of (architecture_type, variants)
        - architecture_type: "unified" (re-exports), "duplicate" (own enum), or "unknown"
        - variants: Set of enum variants if duplicate architecture, empty otherwise
    """
    content = loader_path.read_text()
    
    # Check for re-export pattern (unified architecture)
    reexport_pattern = r'pub use .*dataset_registry::DatasetId'
    if re.search(reexport_pattern, content):
        return ("unified", set())
    
    # Check for duplicate enum pattern
    enum_match = re.search(r'pub enum DatasetId\s*\{([^}]+)\}', content, re.DOTALL)
    if enum_match:
        enum_body = enum_match.group(1)
        # Match variants
        pattern = r'^\s+([A-Z][a-zA-Z0-9_]*)\s*,'
        matches = re.findall(pattern, enum_body, re.MULTILINE)
        return ("duplicate", set(matches))
    
    return ("unknown", set())


def main():
    parser = argparse.ArgumentParser(description="Check DatasetId enum sync")
    parser.add_argument(
        "--json", 
        default="generated/datasets_generated.json",
        help="Path to datasets_generated.json"
    )
    parser.add_argument(
        "--loader",
        default="crates/anno-eval/src/eval/loader.rs",
        help="Path to loader.rs"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with error on mismatch"
    )
    args = parser.parse_args()
    
    # Get IDs from registry
    registry_ids = get_registry_ids(Path(args.json))
    
    # Check loader architecture
    arch_type, loader_ids = check_loader_architecture(Path(args.loader))
    
    print(f"Registry datasets: {len(registry_ids)}")
    print(f"Loader architecture: {arch_type}")
    
    if arch_type == "unified":
        print(f"Loader variants:   {len(registry_ids)} (via re-export)")
        print()
        print("All datasets in sync!")
        print("(loader.rs re-exports DatasetId from dataset_registry.rs)")
        return 0
    
    if arch_type == "duplicate":
        print(f"Loader variants:   {len(loader_ids)}")
        print()
        
        # Find differences
        missing_in_loader = registry_ids - loader_ids
        orphaned_in_loader = loader_ids - registry_ids
        
        if missing_in_loader:
            print(f"Missing in loader ({len(missing_in_loader)}):")
            for id in sorted(missing_in_loader)[:20]:
                print(f"  - {id}")
            if len(missing_in_loader) > 20:
                print(f"  ... and {len(missing_in_loader) - 20} more")
            print()
        
        if orphaned_in_loader:
            print(f"Orphaned in loader ({len(orphaned_in_loader)}):")
            for id in sorted(orphaned_in_loader)[:20]:
                print(f"  - {id}")
            if len(orphaned_in_loader) > 20:
                print(f"  ... and {len(orphaned_in_loader) - 20} more")
            print()
        
        if not missing_in_loader and not orphaned_in_loader:
            print("All datasets in sync!")
            return 0
        
        # Summary
        total_diff = len(missing_in_loader) + len(orphaned_in_loader)
        print(f"Total discrepancies: {total_diff}")
        print()
        print("RECOMMENDATION: Consider unifying to re-export architecture:")
        print("  Replace `pub enum DatasetId { ... }` in loader.rs with:")
        print("  `pub use super::dataset_registry::DatasetId;`")
        
        if args.ci:
            print("\nCI mode: failing due to dataset sync issues")
            return 1
        
        return 0
    
    # Unknown architecture
    print(f"Loader variants:   unknown")
    print()
    print("WARNING: Could not determine loader.rs architecture")
    print("Expected either:")
    print("  - Re-export: `pub use super::dataset_registry::DatasetId;`")
    print("  - Or enum: `pub enum DatasetId { ... }`")
    
    if args.ci:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
