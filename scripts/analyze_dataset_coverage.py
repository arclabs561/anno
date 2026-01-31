#!/usr/bin/env python3
"""Analyze dataset registry coverage and suggest improvements."""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path.home() / "Library/Caches/anno/datasets"
REGISTRY_FILE = Path(__file__).parent.parent / "crates/anno-eval/src/eval/dataset_registry.rs"

def get_cached_files():
    """Get all files in the local cache."""
    if not CACHE_DIR.exists():
        return set()
    return {f.name for f in CACHE_DIR.iterdir() if f.is_file() and f.name != ".gitkeep"}

def parse_registry():
    """Parse dataset registry to extract dataset info."""
    content = REGISTRY_FILE.read_text()
    
    # Find all dataset definitions
    datasets = {}
    
    # Match dataset blocks
    pattern = r'(\w+)\s*\{[^}]*name:\s*"([^"]+)"[^}]*(?:hf_id:\s*"([^"]+)")?[^}]*(?:s3_path:\s*"([^"]+)")?[^}]*(?:alt_sources:\s*\[([^\]]*)\])?[^}]*categories:\s*\[([^\]]+)\]'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        variant = match.group(1)
        name = match.group(2)
        hf_id = match.group(3)
        s3_path = match.group(4)
        alt_sources_raw = match.group(5)
        categories = match.group(6)
        
        alt_sources = []
        if alt_sources_raw:
            alt_sources = re.findall(r'"([^"]+)"', alt_sources_raw)
        
        datasets[variant] = {
            "name": name,
            "hf_id": hf_id,
            "s3_path": s3_path,
            "alt_sources": alt_sources,
            "categories": [c.strip() for c in categories.split(",")],
        }
    
    return datasets

def main():
    print("=" * 60)
    print("Dataset Coverage Analysis")
    print("=" * 60)
    
    cached_files = get_cached_files()
    datasets = parse_registry()
    
    print(f"\nTotal datasets in registry: {len(datasets)}")
    print(f"Files in local cache: {len(cached_files)}")
    
    # Count fields
    has_hf_id = sum(1 for d in datasets.values() if d["hf_id"])
    has_s3_path = sum(1 for d in datasets.values() if d["s3_path"])
    has_alt_sources = sum(1 for d in datasets.values() if d["alt_sources"])
    
    print(f"\nField coverage:")
    print(f"  - With hf_id: {has_hf_id} ({100*has_hf_id/len(datasets):.1f}%)")
    print(f"  - With s3_path: {has_s3_path} ({100*has_s3_path/len(datasets):.1f}%)")
    print(f"  - With alt_sources: {has_alt_sources} ({100*has_alt_sources/len(datasets):.1f}%)")
    
    # Find cached files that could map to datasets
    print(f"\n{'=' * 60}")
    print("Cached files that could be mapped to s3_path:")
    print("=" * 60)
    
    # Create normalized name mapping
    def normalize(name):
        return re.sub(r'[^a-z0-9]', '', name.lower())
    
    cache_normalized = {normalize(f.rsplit('.', 1)[0]): f for f in cached_files}
    
    suggestions = []
    for variant, info in datasets.items():
        if info["s3_path"]:
            continue  # Already has s3_path
            
        variant_norm = normalize(variant)
        name_norm = normalize(info["name"])
        
        # Try to find matching cache file
        for cache_norm, cache_file in cache_normalized.items():
            if variant_norm in cache_norm or cache_norm in variant_norm:
                suggestions.append((variant, cache_file))
                break
            if name_norm in cache_norm or cache_norm in name_norm:
                suggestions.append((variant, cache_file))
                break
    
    if suggestions[:20]:
        for variant, cache_file in suggestions[:20]:
            print(f"  {variant} -> datasets/{cache_file}")
        if len(suggestions) > 20:
            print(f"  ... and {len(suggestions) - 20} more")
    
    # HuggingFace IDs that could be added as alt_sources
    print(f"\n{'=' * 60}")
    print("Datasets with hf_id but not in alt_sources:")
    print("=" * 60)
    
    hf_suggestions = []
    for variant, info in datasets.items():
        if info["hf_id"] and "huggingface" not in str(info["alt_sources"]).lower():
            hf_url = f"https://huggingface.co/datasets/{info['hf_id']}"
            hf_suggestions.append((variant, hf_url))
    
    for variant, url in hf_suggestions[:15]:
        print(f"  {variant}: {url}")
    if len(hf_suggestions) > 15:
        print(f"  ... and {len(hf_suggestions) - 15} more")
    
    print(f"\n{'=' * 60}")
    print("Summary of Improvements")
    print("=" * 60)
    print(f"  - {len(suggestions)} datasets could have s3_path added")
    print(f"  - {len(hf_suggestions)} datasets could have HF URL in alt_sources")
    
    # Category breakdown
    print(f"\n{'=' * 60}")
    print("Category Breakdown")
    print("=" * 60)
    
    category_counts = defaultdict(int)
    for info in datasets.values():
        for cat in info["categories"]:
            category_counts[cat] += 1
    
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()

