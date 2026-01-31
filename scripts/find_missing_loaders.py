#!/usr/bin/env python3
"""
Find datasets in registry that have format metadata but are missing from loader.

This script helps identify datasets that could be added to the loader by:
1. Checking which datasets have format metadata (CoNLL, JSONL, etc.)
2. Checking which datasets have public URLs
3. Identifying which ones are missing from the loader parse_plan

Usage:
    python3 scripts/find_missing_loaders.py
"""

import json
import subprocess
import sys
from pathlib import Path

def get_loader_datasets():
    """Extract DatasetId variants from loader.rs."""
    result = subprocess.run(
        ["rg", "-o", r"DatasetId::([A-Za-z0-9_]+)", "crates/anno-eval/src/eval/loader.rs"],
        capture_output=True,
        text=True
    )
    
    # Extract unique dataset IDs
    ids = set()
    for line in result.stdout.strip().split('\n'):
        if 'DatasetId::' in line:
            # Extract the ID name
            parts = line.split('DatasetId::')
            if len(parts) > 1:
                id_name = parts[1].split()[0].rstrip(',|)')
                ids.add(id_name)
    
    return ids

def get_registry_datasets():
    """Get datasets from generated JSON."""
    json_path = Path('generated/datasets_generated.json')
    if not json_path.exists():
        print(f"Error: {json_path} not found. Run: cargo test generate_datasets_json -- --ignored")
        sys.exit(1)
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Handle both array and object formats
    if isinstance(data, list):
        return {d.get('name', ''): d for d in data if isinstance(d, dict)}
    elif isinstance(data, dict):
        return data
    else:
        return {}

def main():
    print("Finding datasets missing from loader...")
    print("=" * 60)
    
    # Get datasets from both sources
    loader_ids = get_loader_datasets()
    registry_datasets = get_registry_datasets()
    
    print(f"Loader has: {len(loader_ids)} datasets")
    print(f"Registry has: {len(registry_datasets)} datasets")
    print()
    
    # Find datasets with format but not in loader
    missing = []
    hintable_formats = ['CoNLL', 'CoNLLU', 'CoNLL-U', 'BIO', 'IOB2', 'JSONL', 'TSV', 'CSV']
    
    for name, dataset in registry_datasets.items():
        if not isinstance(dataset, dict):
            continue
        
        # Skip if already in loader
        if name in loader_ids:
            continue
        
        # Check if it has a parseable format
        format_val = dataset.get('format', '')
        if format_val not in hintable_formats:
            continue
        
        # Check if it has a URL
        url = dataset.get('url', '')
        if not url or url.startswith('""'):
            continue
        
        # Check tasks
        tasks = dataset.get('tasks', [])
        is_ner = 'ner' in tasks or 'ner' in dataset.get('categories', [])
        is_re = 're' in tasks or 'relation_extraction' in tasks
        is_coref = 'coref' in tasks or 'coreference' in tasks
        
        if is_ner or is_re or is_coref:
            missing.append({
                'name': name,
                'format': format_val,
                'tasks': tasks,
                'url': url[:80] + '...' if len(url) > 80 else url,
                'has_url': bool(url),
            })
    
    print(f"Found {len(missing)} datasets with format + URL but missing from loader:")
    print()
    
    # Group by format
    by_format = {}
    for d in missing:
        fmt = d['format']
        if fmt not in by_format:
            by_format[fmt] = []
        by_format[fmt].append(d)
    
    for fmt in sorted(by_format.keys()):
        datasets = by_format[fmt]
        print(f"{fmt} format ({len(datasets)} datasets):")
        for d in sorted(datasets, key=lambda x: x['name'])[:10]:
            tasks_str = ', '.join(d['tasks'][:2]) if d['tasks'] else 'unknown'
            print(f"  - {d['name']:30} tasks=[{tasks_str:20}] url={d['url'][:50]}")
        if len(datasets) > 10:
            print(f"  ... and {len(datasets) - 10} more")
        print()
    
    print("=" * 60)
    print(f"Total: {len(missing)} datasets could potentially be added")
    print()
    print("Recommendation:")
    print("1. Check if these have format metadata in registry")
    print("2. If format is clear, they should be auto-detected by registry_hint_plan()")
    print("3. If not auto-detected, add explicit match in parse_plan()")

if __name__ == '__main__':
    main()

