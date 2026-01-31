#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Generate loader.rs stubs from dataset_registry metadata.

This script reads generated/datasets_generated.json and generates Rust code snippets
that can be added to loader.rs to bring it in sync with the registry.

Usage:
    uv run scripts/generate_loader_stubs.py [--output FILE] [--missing-only]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def to_rust_variant(name: str) -> str:
    """Convert dataset name to Rust enum variant."""
    # Handle special cases
    if name.upper() == name:  # All caps like "BBQ", "GAP"
        return name
    # Convert snake_case or kebab-case to PascalCase
    parts = name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts)


def to_rust_string(s: str) -> str:
    """Escape a string for Rust."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def infer_format_enum(fmt: str) -> str:
    """Map format string to Rust DatasetFormat enum."""
    mapping = {
        "CoNLL": "DatasetFormat::Conll",
        "CoNLLU": "DatasetFormat::Conllu", 
        "CoNLL-U": "DatasetFormat::Conllu",
        "BIO": "DatasetFormat::Bio",
        "IOB2": "DatasetFormat::Bio",
        "JSONL": "DatasetFormat::Jsonl",
        "JSON": "DatasetFormat::Json",
        "CSV": "DatasetFormat::Csv",
        "TSV": "DatasetFormat::Tsv",
        "XML": "DatasetFormat::Xml",
        "TXT": "DatasetFormat::Text",
        "Custom": "DatasetFormat::Custom",
    }
    return mapping.get(fmt, "DatasetFormat::Custom")


def generate_enum_variants(datasets: list, existing: set) -> str:
    """Generate enum variant declarations."""
    lines = []
    for ds in sorted(datasets, key=lambda x: x.get("id", x.get("name", ""))):
        name = ds.get("id", ds.get("name", ""))
        variant = to_rust_variant(name)
        if variant in existing:
            continue
        
        desc = ds.get("description", name)[:60]  # Truncate long descriptions
        lines.append(f"    /// {to_rust_string(desc)}")
        lines.append(f"    {variant},")
    
    return "\n".join(lines)


def generate_download_url_matches(datasets: list, existing: set) -> str:
    """Generate download_url() match arms."""
    lines = []
    for ds in sorted(datasets, key=lambda x: x.get("id", x.get("name", ""))):
        name = ds.get("id", ds.get("name", ""))
        variant = to_rust_variant(name)
        if variant in existing:
            continue
        
        url = ds.get("url", "")
        if not url or "doi.org" in url or "aclanthology" in url:
            url = "\"\"  // TODO: Add download URL"
        else:
            url = f'"{to_rust_string(url)}"'
        
        lines.append(f"            DatasetId::{variant} => {url},")
    
    return "\n".join(lines)


def generate_name_matches(datasets: list, existing: set) -> str:
    """Generate name() match arms."""
    lines = []
    for ds in sorted(datasets, key=lambda x: x.get("id", x.get("name", ""))):
        name = ds.get("id", ds.get("name", ""))
        variant = to_rust_variant(name)
        if variant in existing:
            continue
        
        display_name = ds.get("name", name)
        lines.append(f'            DatasetId::{variant} => "{to_rust_string(display_name)}",')
    
    return "\n".join(lines)


def generate_description_matches(datasets: list, existing: set) -> str:
    """Generate description() match arms."""
    lines = []
    for ds in sorted(datasets, key=lambda x: x.get("id", x.get("name", ""))):
        name = ds.get("id", ds.get("name", ""))
        variant = to_rust_variant(name)
        if variant in existing:
            continue
        
        desc = ds.get("description", f"{name} dataset")
        # Truncate very long descriptions
        if len(desc) > 100:
            desc = desc[:97] + "..."
        lines.append(f'            DatasetId::{variant} => "{to_rust_string(desc)}",')
    
    return "\n".join(lines)


def generate_category_matches(datasets: list, existing: set, category: str) -> str:
    """Generate is_X() match arms for a category."""
    lines = []
    for ds in sorted(datasets, key=lambda x: x.get("id", x.get("name", ""))):
        name = ds.get("id", ds.get("name", ""))
        variant = to_rust_variant(name)
        if variant in existing:
            continue
        
        # Check if dataset has this category
        categories = ds.get("categories", [])
        if category in categories:
            lines.append(f"            DatasetId::{variant} => true,")
    
    return "\n".join(lines) if lines else ""


def load_existing_variants(loader_path: Path) -> set:
    """Parse existing DatasetId variants from loader.rs or dataset_registry.rs."""
    if not loader_path.exists():
        return set()
    
    content = loader_path.read_text()
    variants = set()
    
    # Match lines like "    WikiGold," (loader.rs style)
    for match in re.finditer(r'^\s+([A-Z][A-Za-z0-9_]*),\s*$', content, re.MULTILINE):
        variants.add(match.group(1))
    
    # Match lines like "    WikiGold {" (dataset_registry.rs define_datasets! macro style)
    for match in re.finditer(r'^    ([A-Z][A-Za-z0-9_]*) \{$', content, re.MULTILINE):
        variants.add(match.group(1))
    
    return variants


def main():
    parser = argparse.ArgumentParser(description="Generate loader.rs stubs from registry")
    parser.add_argument(
        "--input",
        default="generated/datasets_generated.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--loader",
        default="crates/anno-eval/src/eval/loader.rs",
        help="Existing loader.rs",
    )
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--missing-only", action="store_true", help="Only show missing datasets")
    parser.add_argument("--section", choices=["enum", "url", "name", "desc", "all"], 
                        default="all", help="Which section to generate")
    args = parser.parse_args()
    
    json_path = Path(args.input)
    loader_path = Path(args.loader)
    
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    datasets = data.get("datasets", [])
    existing = load_existing_variants(loader_path)
    
    # Count missing
    missing = 0
    for ds in datasets:
        name = ds.get("id", ds.get("name", ""))
        variant = to_rust_variant(name)
        if variant not in existing:
            missing += 1
    
    print(f"// Generated loader stubs from {json_path}")
    print(f"// Existing variants: {len(existing)}")
    print(f"// Registry datasets: {len(datasets)}")
    print(f"// Missing variants: {missing}")
    print()
    
    if args.section in ("enum", "all"):
        print("// === ENUM VARIANTS ===")
        print("// Add these to `pub enum DatasetId {`")
        print()
        print(generate_enum_variants(datasets, existing if args.missing_only else set()))
        print()
    
    if args.section in ("url", "all"):
        print("// === DOWNLOAD URL MATCH ARMS ===")
        print("// Add these to `fn download_url(&self)`")
        print()
        print(generate_download_url_matches(datasets, existing if args.missing_only else set()))
        print()
    
    if args.section in ("name", "all"):
        print("// === NAME MATCH ARMS ===")
        print("// Add these to `fn name(&self)`")
        print()
        print(generate_name_matches(datasets, existing if args.missing_only else set()))
        print()
    
    if args.section in ("desc", "all"):
        print("// === DESCRIPTION MATCH ARMS ===")
        print("// Add these to `fn description(&self)`")
        print()
        print(generate_description_matches(datasets, existing if args.missing_only else set()))
        print()
    
    # Generate category matchers
    if args.section == "all":
        categories = ["biomedical", "coref", "historical", "multilingual", 
                      "bias", "nested", "relation_extraction"]
        for cat in categories:
            matches = generate_category_matches(datasets, existing if args.missing_only else set(), cat)
            if matches:
                print(f"// === is_{cat}() MATCH ARMS ===")
                print(matches)
                print()


if __name__ == "__main__":
    main()
