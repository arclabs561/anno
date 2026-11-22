#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.25"]
# ///
"""
Validate download URLs in the dataset registry.

Tests that URLs are accessible and returns status codes.

Usage:
    uv run scripts/validate_urls.py [--quick] [--output FILE]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError as e:
    print(f"Error: Missing dependency. Run: uv run scripts/validate_urls.py", file=sys.stderr)
    sys.exit(1)


async def check_url(client: httpx.AsyncClient, name: str, url: str) -> dict:
    """Check if a URL is accessible."""
    if not url or url.startswith("\"\""):
        return {"name": name, "url": url, "status": "no_url", "code": None}
    
    # Skip non-HTTP URLs
    if not url.startswith("http"):
        return {"name": name, "url": url, "status": "invalid", "code": None}
    
    # Skip known problem URLs
    if any(x in url for x in ["doi.org", "aclanthology", "arxiv.org", "catalog.ldc"]):
        return {"name": name, "url": url, "status": "paper_only", "code": None}
    
    try:
        response = await client.head(url, follow_redirects=True, timeout=10.0)
        if response.status_code == 405:  # Method not allowed, try GET
            response = await client.get(url, follow_redirects=True, timeout=10.0)
        
        status = "ok" if response.status_code < 400 else "error"
        return {"name": name, "url": url, "status": status, "code": response.status_code}
    except httpx.TimeoutException:
        return {"name": name, "url": url, "status": "timeout", "code": None}
    except httpx.HTTPError as e:
        return {"name": name, "url": url, "status": "error", "code": str(e)[:50]}
    except Exception as e:
        return {"name": name, "url": url, "status": "exception", "code": str(e)[:50]}


async def validate_all(datasets: list, quick: bool = False) -> list:
    """Validate all URLs concurrently."""
    # Use a semaphore to limit concurrent requests
    sem = asyncio.Semaphore(10)
    
    async with httpx.AsyncClient() as client:
        async def check_with_sem(name: str, url: str) -> dict:
            async with sem:
                return await check_url(client, name, url)
        
        urls_to_check = [(ds.get("id", ds.get("name", "unknown")), ds.get("url", "")) 
                         for ds in datasets]
        
        if quick:
            urls_to_check = urls_to_check[:20]
        
        tasks = [check_with_sem(name, url) for name, url in urls_to_check]
        results = await asyncio.gather(*tasks)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate dataset URLs")
    parser.add_argument(
        "--input",
        default="generated/datasets_generated.json",
        help="Input JSON file",
    )
    parser.add_argument("--quick", action="store_true", help="Only check first 20 URLs")
    parser.add_argument("--output", help="Output results to file")
    args = parser.parse_args()
    
    json_path = Path(args.input)
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, list) or not datasets:
        parser.error(f"{json_path} must contain a nonempty datasets array")
    print(f"Validating URLs for {len(datasets)} datasets...")
    
    results = asyncio.run(validate_all(datasets, quick=args.quick))
    
    # Summarize
    by_status = {}
    for r in results:
        status = r["status"]
        by_status[status] = by_status.get(status, 0) + 1
    
    print("\n=== URL Validation Summary ===")
    for status, count in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")
    
    # Show errors
    errors = [r for r in results if r["status"] in ("error", "timeout", "exception")]
    if errors:
        print(f"\n=== Errors ({len(errors)}) ===")
        for r in errors[:20]:
            print(f"  {r['name']}: {r['status']} ({r['code']})")
            if len(r['url']) < 80:
                print(f"    URL: {r['url']}")
    
    # Show successes
    successes = [r for r in results if r["status"] == "ok"]
    print(f"\n=== Valid URLs: {len(successes)} ===")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
