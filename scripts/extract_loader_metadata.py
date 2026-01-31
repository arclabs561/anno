#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Extract dataset metadata from loader.rs and generate registry entries.

Parses loader.rs to extract:
- Dataset names and descriptions (from doc comments)
- Download URLs (from download_url() match arms)
- Categories (from is_X() match arms)

Usage:
    uv run scripts/extract_loader_metadata.py [--missing-only] [--output FILE]
"""

import argparse
import re
import sys
from pathlib import Path


def extract_enum_variants(content: str) -> dict:
    """Extract DatasetId variants with their doc comments."""
    variants = {}
    
    # Match patterns like:
    # /// WikiGold: Wikipedia-based NER
    # WikiGold,
    pattern = r'///\s*(.+?)\n\s*([A-Z][A-Za-z0-9_]+),'
    
    for match in re.finditer(pattern, content):
        doc_comment = match.group(1).strip()
        variant = match.group(2)
        variants[variant] = {"description": doc_comment}
    
    return variants


def extract_download_urls(content: str) -> dict:
    """Extract download URLs from download_url() method."""
    urls = {}
    
    # Find the download_url method
    method_match = re.search(
        r'pub fn download_url\(&self\)[^{]+\{(.*?)^\s{4}\}',
        content, 
        re.MULTILINE | re.DOTALL
    )
    
    if not method_match:
        return urls
    
    method_body = method_match.group(1)
    
    # Match patterns like:
    # DatasetId::WikiGold => "https://..."
    # or DatasetId::WikiGold => {
    #     "https://..."
    # }
    pattern = r'DatasetId::([A-Z][A-Za-z0-9_]+)\s*=>\s*(?:\{[^}]*"([^"]+)"[^}]*\}|"([^"]+)")'
    
    for match in re.finditer(pattern, method_body):
        variant = match.group(1)
        url = match.group(2) or match.group(3)
        if url:
            urls[variant] = url
    
    return urls


def extract_names(content: str) -> dict:
    """Extract human-readable names from name() method."""
    names = {}
    
    method_match = re.search(
        r'pub fn name\(&self\)[^{]+\{(.*?)^\s{4}\}',
        content,
        re.MULTILINE | re.DOTALL
    )
    
    if not method_match:
        return names
    
    method_body = method_match.group(1)
    
    pattern = r'DatasetId::([A-Z][A-Za-z0-9_]+)\s*=>\s*"([^"]+)"'
    
    for match in re.finditer(pattern, method_body):
        variant = match.group(1)
        name = match.group(2)
        names[variant] = name
    
    return names


def extract_category(content: str, category: str) -> set:
    """Extract datasets that belong to a category from is_X() method."""
    datasets = set()
    
    method_match = re.search(
        rf'pub fn is_{category}\(&self\)[^{{]+\{{(.*?)^\s{{4}}\}}',
        content,
        re.MULTILINE | re.DOTALL
    )
    
    if not method_match:
        return datasets
    
    method_body = method_match.group(1)
    
    # Find all DatasetId::X that map to true
    for match in re.finditer(r'DatasetId::([A-Z][A-Za-z0-9_]+)', method_body):
        variant = match.group(1)
        # Check if this is in a true arm (not a false arm or wildcard)
        datasets.add(variant)
    
    return datasets


def generate_registry_entry(variant: str, metadata: dict) -> str:
    """Generate a registry entry for a dataset."""
    desc = metadata.get("description", f"{variant} dataset")
    url = metadata.get("url", "")
    name = metadata.get("name", variant)
    categories = metadata.get("categories", [])
    
    # Infer domain from categories
    domain = "general"
    if "biomedical" in categories:
        domain = "biomedical"
    elif "historical" in categories:
        domain = "historical"
    elif "social_media" in categories:
        domain = "social_media"
    
    # Infer language
    language = "en"
    if "multilingual" in categories:
        language = "multi"
    
    # Build category list
    cat_list = ", ".join(categories) if categories else "core_ner"
    
    entry = f'''    {variant} {{
        name: "{name}",
        description: "{desc[:100]}",
        url: "{url}",
        entity_types: ["ENTITY"],
        language: "{language}",
        domain: "{domain}",
        categories: [{cat_list}]
    }}'''
    
    return entry


def main():
    parser = argparse.ArgumentParser(description="Extract loader metadata for registry")
    parser.add_argument(
        "--loader",
        default="crates/anno-eval/src/eval/loader.rs",
        help="Path to loader.rs",
    )
    parser.add_argument("--missing", help="File with list of missing variants")
    parser.add_argument("--output", help="Output file (default: stdout)")
    args = parser.parse_args()
    
    loader_path = Path(args.loader)
    if not loader_path.exists():
        print(f"Error: {loader_path} not found", file=sys.stderr)
        sys.exit(1)
    
    content = loader_path.read_text()
    
    # Extract all metadata
    variants = extract_enum_variants(content)
    urls = extract_download_urls(content)
    names = extract_names(content)
    
    # Extract categories
    categories_map = {}
    for cat in ["biomedical", "coreference", "historical", "multilingual", 
                "bias_evaluation", "social_media", "relation_extraction",
                "temporal_ner", "dialogue_coref", "discontinuous_ner"]:
        datasets = extract_category(content, cat)
        for d in datasets:
            if d not in categories_map:
                categories_map[d] = []
            categories_map[d].append(cat)
    
    # Merge metadata
    for variant in variants:
        if variant in urls:
            variants[variant]["url"] = urls[variant]
        if variant in names:
            variants[variant]["name"] = names[variant]
        if variant in categories_map:
            variants[variant]["categories"] = categories_map[variant]
    
    # Filter to missing only if specified
    if args.missing:
        missing_path = Path(args.missing)
        if missing_path.exists():
            missing = set(missing_path.read_text().strip().split("\n"))
            variants = {k: v for k, v in variants.items() if k in missing}
    
    # Generate output
    print(f"// Generated registry entries for {len(variants)} datasets")
    print(f"// From: {loader_path}")
    print()
    
    for variant, metadata in sorted(variants.items()):
        print(generate_registry_entry(variant, metadata))
        print(",")
        print()
    
    print(f"// Total: {len(variants)} entries")


if __name__ == "__main__":
    main()

