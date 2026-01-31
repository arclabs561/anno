#!/usr/bin/env python3
"""
Apply registry enrichment updates.
Adds s3_path and HuggingFace alt_sources to dataset registry.

Run with: uv run python scripts/apply_registry_enrichment.py
"""

import json
import re
from pathlib import Path

REGISTRY_FILE = Path(__file__).parent.parent / "crates/anno/eval/dataset_registry.rs"
ENRICHMENT_FILE = Path(__file__).parent / "registry_enrichment.json"
S3_BUCKET = "arc-anno-data"

def load_enrichment():
    """Load enrichment data from JSON."""
    with open(ENRICHMENT_FILE) as f:
        return json.load(f)

def update_registry(content: str, s3_matches: dict, hf_additions: dict) -> tuple[str, int]:
    """Update registry content with enrichment data."""
    
    updates_applied = 0
    
    # Process each dataset block - add s3_path for datasets with cache files
    for variant, match in s3_matches.items():
        s3_path = match["s3_path"]
        s3_url = f"s3://{S3_BUCKET}/{s3_path}"
        
        # Find the dataset block - look for the categories line
        pattern = rf'(\s+{variant}\s*\{{[^}}]*?)(categories:\s*\[[^\]]+\])'
        
        def add_s3_enrichment(m):
            nonlocal updates_applied
            block = m.group(1)
            categories = m.group(2)
            
            # Check if s3_path already exists
            if "s3_path:" in block:
                return m.group(0)
            
            # Check if alt_sources exists and add s3_url there
            if "alt_sources:" in block:
                # Add s3_url to existing alt_sources (before the closing bracket)
                block = re.sub(
                    r'(alt_sources:\s*\[)([^\]]*)\]',
                    lambda x: f'{x.group(1)}{x.group(2).rstrip().rstrip(",")}, "{s3_url}"]',
                    block
                )
            else:
                # Add alt_sources with s3_url before categories
                block = block.rstrip() + f'\n        alt_sources: ["{s3_url}"],\n        '
            
            # Add s3_path before categories
            block = block.rstrip() + f'\n        s3_path: "{s3_path}",\n        '
            
            updates_applied += 1
            return block + categories
        
        content = re.sub(pattern, add_s3_enrichment, content, flags=re.DOTALL)
    
    # Add HuggingFace URLs to alt_sources for datasets with hf_id
    for variant, addition in hf_additions.items():
        hf_url = addition["hf_url"]
        
        # Find the dataset block
        pattern = rf'(\s+{variant}\s*\{{[^}}]*?)(categories:\s*\[[^\]]+\])'
        
        def add_hf_url(m):
            nonlocal updates_applied
            block = m.group(1)
            categories = m.group(2)
            
            # Check if HF URL already exists
            if "huggingface.co" in block:
                return m.group(0)
            
            # Check if alt_sources exists
            if "alt_sources:" in block:
                # Add HF URL to existing alt_sources (before the closing bracket)
                block = re.sub(
                    r'(alt_sources:\s*\[)([^\]]*)\]',
                    lambda x: f'{x.group(1)}{x.group(2).rstrip().rstrip(",")}, "{hf_url}"]',
                    block
                )
            else:
                # Add alt_sources with HF URL before categories
                block = block.rstrip() + f'\n        alt_sources: ["{hf_url}"],\n        '
            
            updates_applied += 1
            return block + categories
        
        content = re.sub(pattern, add_hf_url, content, flags=re.DOTALL)
    
    return content, updates_applied

def main():
    print("=" * 60)
    print("Applying Registry Enrichment")
    print("=" * 60)
    
    # Load enrichment data
    print("\n[1/3] Loading enrichment data...")
    data = load_enrichment()
    s3_matches = data.get("s3_matches", {})
    hf_additions = data.get("hf_additions", {})
    print(f"  S3 path matches: {len(s3_matches)}")
    print(f"  HF URL additions: {len(hf_additions)}")
    
    # Load registry
    print("\n[2/3] Loading registry...")
    content = REGISTRY_FILE.read_text()
    original_len = len(content)
    print(f"  Original size: {original_len:,} bytes")
    
    # Apply updates
    print("\n[3/3] Applying updates...")
    updated_content, updates_applied = update_registry(content, s3_matches, hf_additions)
    
    # Write back
    REGISTRY_FILE.write_text(updated_content)
    new_len = len(updated_content)
    
    print(f"\n  Updates applied: {updates_applied}")
    print(f"  New size: {new_len:,} bytes (+{new_len - original_len:,})")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nRun 'cargo check -p anno' to verify the changes compile.")

if __name__ == "__main__":
    main()

