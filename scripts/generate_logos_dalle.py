#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=1.0.0",
#     "pillow>=10.0.0",
#     "requests>=2.31.0",
# ]
# ///

"""
Generate logo variations using DALL-E 3 for iterative refinement.

Usage:
    OPENAI_API_KEY=your_key uv run scripts/generate_logos_dalle.py
"""

import os
import sys
from pathlib import Path
from time import sleep

try:
    from openai import OpenAI
    from PIL import Image
    import requests
except ImportError as e:
    print(f"Error: Missing dependency. Install with: uv pip install {e.name}")
    sys.exit(1)

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "assets" / "logo_variations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompt variations for batch generation
PROMPTS = [
    {
        "name": "minimal_abstract",
        "prompt": """Create a minimalist, brutalist logo for "anno" - an information extraction library. 
The logo should visually represent the pipeline: Extract → Coalesce → Stratify.

Design requirements:
- Dark background (#0f172a or similar)
- Three distinct visual elements:
  1. Extract: Scattered blue dots/circles (representing individual entities)
  2. Coalesce: Connected green nodes/cluster (representing merged entities)
  3. Stratify: Layered purple ellipses/strata (representing hierarchical organization)
- Text: "anno" in clean, sans-serif, white lowercase
- Style: Brutalist, minimal, technical, no gradients or shadows
- Format: Square aspect ratio, suitable for favicon/icon use
- Color palette: Blue (#60a5fa), Green (#34d399), Purple (#a78bfa), White (#f1f5f9)""",
    },
    {
        "name": "geometric_clean",
        "prompt": """Design a clean, geometric logo for "anno" information extraction library.

Visual concept: Extract. Coalesce. Stratify.
- Extract: Blue geometric shapes (squares/circles) scattered
- Coalesce: Green connected network/graph structure
- Stratify: Purple horizontal layers/strata stacked vertically
- Background: Dark (#0f172a)
- Text: "anno" in white, modern sans-serif
- Style: Geometric, precise, technical, brutalist
- Square format, high contrast""",
    },
    {
        "name": "icon_focused",
        "prompt": """Create an icon-style logo for "anno" that works at small sizes.

Key elements:
- Extract: 3-5 small blue dots (top-left)
- Coalesce: 3 green circles connected by lines (center-right)
- Stratify: 3-4 purple horizontal ellipses stacked (bottom-center)
- "anno" text at bottom in white
- Dark background (#0f172a)
- Minimal detail, high contrast, scalable
- Brutalist aesthetic, no decorative elements""",
    },
    {
        "name": "pipeline_flow",
        "prompt": """Design a logo showing the flow: Extract → Coalesce → Stratify.

Layout:
- Left: Blue scattered entities (Extract)
- Center: Green cluster with connections (Coalesce)
- Right: Purple layered structure (Stratify)
- Text: "anno" centered at bottom
- Background: Dark slate (#0f172a)
- Style: Brutalist, minimal, technical
- Show progression/flow from left to right
- Square format""",
    },
    {
        "name": "layered_depth",
        "prompt": """Create a logo with depth through layering for "anno".

Visual hierarchy:
- Foreground: Blue extraction dots (scattered)
- Midground: Green coalesced cluster (connected)
- Background: Purple stratified layers (stacked ellipses)
- Text: "anno" in white, bottom
- Dark background (#0f172a)
- Use opacity/variation to show depth
- Brutalist, minimal, technical aesthetic
- Square format""",
    },
]


def setup_openai(api_key: str | None = None) -> OpenAI:
    """Initialize OpenAI client."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Get your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    return OpenAI(api_key=api_key)


def generate_image(client: OpenAI, prompt: str, output_path: Path, variant_name: str) -> bool:
    """Generate a single image using DALL-E 3."""
    try:
        print(f"Generating: {variant_name}...")
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",  # Square format
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        
        # Download the image
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        # Save as PNG
        output_path.write_bytes(img_response.content)
        
        print(f"  ✅ Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Generate batch of logo variations."""
    print("🎨 Generating logo variations with DALL-E 3\n")
    
    client = setup_openai()
    
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    generated = 0
    for i, variant in enumerate(PROMPTS, 1):
        output_path = OUTPUT_DIR / f"{variant['name']}.png"
        
        if generate_image(client, variant["prompt"], output_path, variant["name"]):
            generated += 1
        
        # Rate limiting: DALL-E 3 has rate limits
        if i < len(PROMPTS):
            print("  ⏳ Waiting 5s for rate limit...\n")
            sleep(5)
    
    print(f"\n📊 Generated {generated}/{len(PROMPTS)} variations")
    print(f"📁 Review images in: {OUTPUT_DIR}")
    print("\n💡 Next steps:")
    print("   1. Review the generated PNGs")
    print("   2. Provide feedback on favorites")
    print("   3. We'll refine using Rocchio-style iterative improvement")


if __name__ == "__main__":
    main()

