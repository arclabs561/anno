#!/usr/bin/env python3
"""
Enrich dataset registry with:
1. s3_path from local cache mapping
2. HuggingFace URLs in alt_sources
3. URL health check results

Run with: uv run python scripts/enrich_dataset_registry.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error
import ssl

CACHE_DIR = Path.home() / "Library/Caches/anno/datasets"
REGISTRY_FILE = Path(__file__).parent.parent / "crates/anno-eval/src/eval/dataset_registry.rs"
S3_BUCKET = "arc-anno-data"
OUTPUT_FILE = Path(__file__).parent.parent / "scripts/registry_enrichment.json"

# Timeout for URL health checks (seconds)
URL_TIMEOUT = 10

def get_cached_files():
    """Get all files in the local cache with sizes."""
    if not CACHE_DIR.exists():
        return {}
    result = {}
    for f in CACHE_DIR.iterdir():
        if f.is_file() and f.name != ".gitkeep":
            result[f.name] = {
                "path": f"datasets/{f.name}",
                "size": f.stat().st_size,
            }
    return result

def normalize_name(name: str) -> str:
    """Normalize a name for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def parse_registry():
    """Parse dataset registry to extract dataset info."""
    content = REGISTRY_FILE.read_text()
    
    datasets = {}
    
    # Find all dataset definitions using a more robust pattern
    # This captures the variant name and the full block
    pattern = r'(\w+)\s*\{([^}]+categories:\s*\[[^\]]+\][^}]*)\}'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        variant = match.group(1)
        block = match.group(2)
        
        # Extract fields from block
        name_match = re.search(r'name:\s*"([^"]+)"', block)
        url_match = re.search(r'url:\s*"([^"]*)"', block)
        hf_id_match = re.search(r'hf_id:\s*"([^"]+)"', block)
        s3_path_match = re.search(r's3_path:\s*"([^"]+)"', block)
        alt_sources_match = re.search(r'alt_sources:\s*\[([^\]]*)\]', block)
        
        alt_sources = []
        if alt_sources_match:
            alt_sources = re.findall(r'"([^"]+)"', alt_sources_match.group(1))
        
        datasets[variant] = {
            "name": name_match.group(1) if name_match else variant,
            "url": url_match.group(1) if url_match else "",
            "hf_id": hf_id_match.group(1) if hf_id_match else None,
            "s3_path": s3_path_match.group(1) if s3_path_match else None,
            "alt_sources": alt_sources,
        }
    
    return datasets

def check_url_health(url: str) -> dict:
    """Check if a URL is accessible."""
    if not url or url.startswith("s3://"):
        return {"status": "skip", "url": url}
    
    try:
        # Create SSL context that doesn't verify certificates (for speed)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; anno-health-check/1.0)'},
            method='HEAD'
        )
        
        with urllib.request.urlopen(req, timeout=URL_TIMEOUT, context=ctx) as response:
            return {
                "status": "ok",
                "url": url,
                "http_status": response.status,
            }
    except urllib.error.HTTPError as e:
        return {"status": "error", "url": url, "http_status": e.code, "error": str(e)}
    except urllib.error.URLError as e:
        return {"status": "error", "url": url, "error": str(e.reason)}
    except Exception as e:
        return {"status": "error", "url": url, "error": str(e)}

def match_cache_to_registry(datasets: dict, cached_files: dict) -> dict:
    """Match cached files to registry datasets."""
    matches = {}
    
    # Create normalized lookup
    cache_normalized = {}
    for filename, info in cached_files.items():
        base = filename.rsplit('.', 1)[0]
        norm = normalize_name(base)
        cache_normalized[norm] = (filename, info)
    
    for variant, info in datasets.items():
        if info["s3_path"]:
            continue  # Already has s3_path
        
        variant_norm = normalize_name(variant)
        name_norm = normalize_name(info["name"])
        
        # Try exact matches first
        for cache_norm, (filename, cache_info) in cache_normalized.items():
            if variant_norm == cache_norm or name_norm == cache_norm:
                matches[variant] = {
                    "s3_path": cache_info["path"],
                    "filename": filename,
                    "match_type": "exact",
                }
                break
        
        # Try substring matches
        if variant not in matches:
            for cache_norm, (filename, cache_info) in cache_normalized.items():
                if variant_norm in cache_norm or cache_norm in variant_norm:
                    matches[variant] = {
                        "s3_path": cache_info["path"],
                        "filename": filename,
                        "match_type": "substring_variant",
                    }
                    break
                if name_norm in cache_norm or cache_norm in name_norm:
                    matches[variant] = {
                        "s3_path": cache_info["path"],
                        "filename": filename,
                        "match_type": "substring_name",
                    }
                    break
    
    return matches

def generate_hf_alt_sources(datasets: dict) -> dict:
    """Generate HuggingFace URLs for datasets with hf_id."""
    additions = {}
    
    for variant, info in datasets.items():
        if not info["hf_id"]:
            continue
        
        hf_url = f"https://huggingface.co/datasets/{info['hf_id']}"
        
        # Check if already in alt_sources
        if any("huggingface.co" in src for src in info["alt_sources"]):
            continue
        
        additions[variant] = {
            "hf_url": hf_url,
            "hf_id": info["hf_id"],
        }
    
    return additions

def run_health_checks(datasets: dict, max_workers: int = 10) -> dict:
    """Run URL health checks in parallel."""
    results = {}
    urls_to_check = []
    
    for variant, info in datasets.items():
        if info["url"]:
            urls_to_check.append((variant, "primary", info["url"]))
        for i, alt in enumerate(info["alt_sources"]):
            if not alt.startswith("s3://"):
                urls_to_check.append((variant, f"alt_{i}", alt))
    
    print(f"Checking {len(urls_to_check)} URLs...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(check_url_health, url): (variant, url_type, url)
            for variant, url_type, url in urls_to_check
        }
        
        completed = 0
        for future in as_completed(future_to_url):
            variant, url_type, url = future_to_url[future]
            try:
                result = future.result()
                if variant not in results:
                    results[variant] = {}
                results[variant][url_type] = result
            except Exception as e:
                if variant not in results:
                    results[variant] = {}
                results[variant][url_type] = {"status": "error", "url": url, "error": str(e)}
            
            completed += 1
            if completed % 50 == 0:
                print(f"  Checked {completed}/{len(urls_to_check)} URLs...")
    
    return results

def generate_rust_updates(s3_matches: dict, hf_additions: dict) -> str:
    """Generate Rust code snippets for registry updates."""
    lines = []
    lines.append("// Auto-generated s3_path additions")
    lines.append("// Copy these into the appropriate dataset blocks in dataset_registry.rs")
    lines.append("")
    
    for variant, match in sorted(s3_matches.items()):
        lines.append(f"// {variant}: s3_path: \"{match['s3_path']}\",")
    
    lines.append("")
    lines.append("// Auto-generated alt_sources additions (HuggingFace URLs)")
    lines.append("")
    
    for variant, addition in sorted(hf_additions.items()):
        lines.append(f"// {variant}: add \"{addition['hf_url']}\" to alt_sources")
    
    return "\n".join(lines)

def main():
    print("=" * 60)
    print("Dataset Registry Enrichment")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading cached files...")
    cached_files = get_cached_files()
    print(f"  Found {len(cached_files)} cached files")
    
    print("\n[2/5] Parsing registry...")
    datasets = parse_registry()
    print(f"  Found {len(datasets)} datasets")
    
    # Match cache to registry
    print("\n[3/5] Matching cache to registry...")
    s3_matches = match_cache_to_registry(datasets, cached_files)
    print(f"  Matched {len(s3_matches)} datasets to cache files")
    
    # Generate HuggingFace URLs
    print("\n[4/5] Generating HuggingFace alt_sources...")
    hf_additions = generate_hf_alt_sources(datasets)
    print(f"  Found {len(hf_additions)} datasets needing HF URLs")
    
    # URL health checks (optional - takes time)
    health_results = {}
    if "--check-urls" in sys.argv:
        print("\n[5/5] Running URL health checks...")
        health_results = run_health_checks(datasets)
        
        # Summarize results
        ok_count = sum(1 for v in health_results.values() for r in v.values() if r.get("status") == "ok")
        error_count = sum(1 for v in health_results.values() for r in v.values() if r.get("status") == "error")
        print(f"  OK: {ok_count}, Errors: {error_count}")
        
        # List broken URLs
        broken = []
        for variant, checks in health_results.items():
            for url_type, result in checks.items():
                if result.get("status") == "error":
                    broken.append((variant, url_type, result.get("url"), result.get("error")))
        
        if broken:
            print(f"\n  Broken URLs ({len(broken)}):")
            for variant, url_type, url, error in broken[:20]:
                print(f"    {variant} ({url_type}): {error[:50]}...")
            if len(broken) > 20:
                print(f"    ... and {len(broken) - 20} more")
    else:
        print("\n[5/5] Skipping URL health checks (use --check-urls to enable)")
    
    # Generate output
    output = {
        "summary": {
            "total_datasets": len(datasets),
            "cached_files": len(cached_files),
            "s3_path_matches": len(s3_matches),
            "hf_additions": len(hf_additions),
        },
        "s3_matches": s3_matches,
        "hf_additions": hf_additions,
        "health_results": health_results,
    }
    
    # Save JSON output
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] Saved to {OUTPUT_FILE}")
    
    # Generate Rust snippets
    rust_code = generate_rust_updates(s3_matches, hf_additions)
    rust_file = OUTPUT_FILE.with_suffix(".rs")
    with open(rust_file, "w") as f:
        f.write(rust_code)
    print(f"[Output] Rust snippets saved to {rust_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Datasets that can have s3_path added: {len(s3_matches)}")
    print(f"  Datasets that can have HF URL added:  {len(hf_additions)}")
    
    if s3_matches:
        print(f"\n  Sample s3_path matches:")
        for variant, match in list(s3_matches.items())[:5]:
            print(f"    {variant} -> {match['s3_path']}")
    
    if hf_additions:
        print(f"\n  Sample HF URL additions:")
        for variant, addition in list(hf_additions.items())[:5]:
            print(f"    {variant} -> {addition['hf_url']}")

if __name__ == "__main__":
    main()

