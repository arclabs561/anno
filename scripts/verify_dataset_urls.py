#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "tomli; python_version < '3.11'"]
# ///
"""
Verify that dataset download URLs are accessible.

Reads from generated/download_configs_generated.json (preferred) or
falls back to legacy datasets.toml if JSON not found.

Usage:
    uv run scripts/verify_dataset_urls.py [--all] [--fix] [--timeout SECONDS]
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import requests
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class URLResult(NamedTuple):
    dataset: str
    url: str
    status: str  # "ok", "error", "redirect", "missing"
    code: int | None
    message: str


def check_url(dataset: str, url: str, timeout: int = 10) -> URLResult:
    """Check if a URL is accessible."""
    if not url:
        return URLResult(dataset, "", "missing", None, "No URL defined")
    
    try:
        # Use HEAD request first (faster), fall back to GET if needed
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 200:
            return URLResult(dataset, url, "ok", 200, "OK")
        elif response.status_code in (301, 302, 307, 308):
            return URLResult(dataset, url, "redirect", response.status_code, 
                           f"Redirects to {response.headers.get('Location', 'unknown')}")
        elif response.status_code == 405:
            # HEAD not allowed, try GET
            response = requests.get(url, timeout=timeout, stream=True)
            response.close()
            if response.status_code == 200:
                return URLResult(dataset, url, "ok", 200, "OK (HEAD not allowed)")
            return URLResult(dataset, url, "error", response.status_code, response.reason)
        else:
            return URLResult(dataset, url, "error", response.status_code, response.reason)
    except requests.Timeout:
        return URLResult(dataset, url, "error", None, "Timeout")
    except requests.ConnectionError as e:
        return URLResult(dataset, url, "error", None, f"Connection error: {e}")
    except Exception as e:
        return URLResult(dataset, url, "error", None, str(e))


def main():
    parser = argparse.ArgumentParser(description="Verify dataset URLs")
    parser.add_argument("--all", action="store_true", help="Check all URLs (default: only check a sample)")
    parser.add_argument("--fix", action="store_true", help="Suggest fixes for broken URLs")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()

    # Load download configs (generated from Rust registry)
    # Prefer generated JSON over legacy TOML
    json_path = Path(__file__).parent.parent / "generated" / "download_configs_generated.json"
    toml_path = Path(__file__).parent.parent / "datasets.toml"
    
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        datasets = data.get("configs", {})
        print(f"Using generated configs from: {json_path}")
    elif toml_path.exists():
        with toml_path.open("rb") as f:
            data = tomllib.load(f)
        datasets = data.get("datasets", {})
        print(f"Warning: Using legacy TOML (generated JSON not found): {toml_path}")
    else:
        print("Error: No dataset configuration found!")
        print(f"  Expected: {json_path}")
        print(f"  Fallback: {toml_path}")
        sys.exit(1)
    
    # Filter to datasets with URLs (skip aliases)
    urls_to_check = [
        (name, entry.get("url", ""))
        for name, entry in datasets.items()
        if entry.get("url") and not entry.get("alias_of")
    ]
    
    if not args.all:
        # Sample: check every 10th URL for quick validation
        urls_to_check = urls_to_check[::10]
        print(f"Checking sample of {len(urls_to_check)} URLs (use --all for complete check)")
    else:
        print(f"Checking all {len(urls_to_check)} URLs")

    print()
    
    results = []
    ok_count = 0
    error_count = 0
    
    # Check URLs in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_url, name, url, args.timeout): (name, url)
            for name, url in urls_to_check
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result.status == "ok":
                ok_count += 1
                print(f"  [OK] {result.dataset}")
            elif result.status == "redirect":
                ok_count += 1  # Redirects usually work
                print(f"  [REDIRECT] {result.dataset}: {result.message}")
            elif result.status == "missing":
                print(f"  [SKIP] {result.dataset}: No URL")
            else:
                error_count += 1
                print(f"  [ERROR] {result.dataset}: {result.code} {result.message}")
                if len(result.url) > 60:
                    print(f"          URL: {result.url[:60]}...")
                else:
                    print(f"          URL: {result.url}")

    print()
    print(f"Summary: {ok_count} OK, {error_count} errors")
    
    # Report broken URLs
    if error_count > 0 and args.fix:
        print("\n=== Broken URLs ===")
        for result in results:
            if result.status == "error":
                print(f"  {result.dataset}:")
                print(f"    Current: {result.url}")
                print(f"    Error: {result.message}")
                print()

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

