#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-generativeai>=0.8.0",
#     "pillow>=10.0.0",
#     "requests>=2.31.0",
#     "python-dotenv>=1.0.0",
#     "cairosvg>=2.7.0",
# ]
# ///

"""
Generate logo variations using Gemini Pro 3 for iterative refinement.

Usage:
    uv run scripts/generate_logos.py
"""

import os
import sys
from pathlib import Path

try:
    import google.generativeai as genai
    from PIL import Image, ImageDraw, ImageFont
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing dependency. Install with: uv pip install {e.name}")
    sys.exit(1)

# Load .env file
load_dotenv()

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
    {
        "name": "zen_cairn",
        "prompt": """Create a zen rock cairn logo for "anno" - user loved this aesthetic.

Key elements:
- Stack of 5 smooth, flat oval stones (zen cairn style) in upper-middle
- Stones: Dark purple at bottom, progressively lighter purple/lavender to top
- Each stone: Flattened oval/disc with rounded edges, polished surface
- Subtle 3D shading: Soft shadows beneath each stone, gentle highlights on top surfaces
- Background: Dark blue-purple (twilight/deep space) with speckled light blue-green dots (starry effect)
- Text: "anno" in bold white, modern sans-serif, rounded edges, with soft gray drop shadow for 3D effect
- Composition: Centered, balanced, calm and serene
- Style: Clean, modern, minimal, cosmic wonder
- Square format""",
    },
    {
        "name": "zen_cairn_minimal",
        "prompt": """Ultra-minimal zen cairn logo for "anno".

Elements:
- 4-5 stacked purple stones (dark to light gradient)
- Dark blue-purple starry background
- White "anno" text, clean and simple
- No decorative elements, pure minimalism
- Square format""",
    },
    {
        "name": "zen_cairn_with_pipeline",
        "prompt": """Zen cairn logo incorporating all three pipeline stages.

Main focus:
- Stacked purple stones (zen cairn) representing Stratify - center/upper
- Subtle blue dots scattered (Extract) - background/foreground
- Green connected cluster (Coalesce) - integrated naturally
- Dark starry background (blue-purple with light specks)
- White "anno" text with 3D effect
- Balanced composition showing Extract → Coalesce → Stratify flow
- Square format""",
    },
    {
        "name": "zen_cairn_compact",
        "prompt": """Compact zen cairn logo - tighter composition.

Elements:
- 3-4 stacked purple stones (smaller, more compact stack)
- Dark blue-purple starry background (denser star field)
- White "anno" text, smaller, positioned closer to stones
- More intimate, focused composition
- Square format""",
    },
    {
        "name": "zen_cairn_colorful",
        "prompt": """Zen cairn with colorful gradient stones.

Stones:
- Bottom: Deep purple
- Middle: Blue-purple transition
- Top: Light lavender/white
- Each stone has subtle color variation and glow
- Dark starry background (indigo with teal specks)
- White "anno" text with soft shadow
- Square format""",
    },
    {
        "name": "zen_cairn_glow",
        "prompt": """Zen cairn with ethereal glow effect.

Stones:
- Stacked purple stones (dark to light gradient)
- Soft inner glow/aura around each stone
- Subtle light emission from between stones
- Dark blue-purple background with bright starry specks
- White "anno" text with gentle glow
- Mystical, ethereal quality
- Square format""",
    },
    {
        "name": "zen_cairn_geometric",
        "prompt": """Geometric zen cairn - more structured stones.

Stones:
- Stacked oval/disc shapes with crisp edges
- Perfect alignment, geometric precision
- Purple gradient (dark to light)
- Minimal shadows, clean lines
- Dark starry background
- Bold white "anno" text, geometric sans-serif
- Modern, technical aesthetic
- Square format""",
    },
    {
        "name": "zen_cairn_organic",
        "prompt": """Organic, natural zen cairn - irregular stones.

Stones:
- 5 naturally shaped stones (not perfect ovals, more organic)
- Each stone slightly irregular, like real river stones
- Purple tones with natural variation
- Soft, natural shadows
- Dark blue-purple starry background
- White "anno" text, friendly rounded font
- Natural, approachable aesthetic
- Square format""",
    },
    {
        "name": "zen_cairn_tall",
        "prompt": """Tall zen cairn - more stones, vertical emphasis.

Stones:
- 6-7 stacked stones creating taller cairn
- Narrower stones, more vertical composition
- Purple gradient throughout
- Dark starry background
- White "anno" text below
- Elegant, vertical flow
- Square format""",
    },
    {
        "name": "zen_cairn_wide",
        "prompt": """Wide zen cairn - horizontal stone arrangement.

Stones:
- 3-4 wider, flatter stones stacked
- More horizontal emphasis
- Purple gradient
- Dark starry background
- White "anno" text, wider spacing
- Balanced, grounded composition
- Square format""",
    },
]


def setup_gemini(api_key: str | None = None) -> tuple[genai.GenerativeModel, bool]:
    """Initialize Gemini API client."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file or environment.")
        print("Add GEMINI_API_KEY=your_key to .env file")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    
    # List available models first
    print("🔍 Checking available Gemini models...")
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_id = m.name.replace('models/', '')
                available_models.append(model_id)
                print(f"  ✓ {model_id}")
    except Exception as e:
        print(f"  ⚠️  Could not list models: {e}")
    
    # Prefer image generation models first
    image_models = [m for m in available_models if "image" in m.lower() or "banana" in m.lower()]
    text_models = [m for m in available_models if m not in image_models]
    
    # Try image generation models first, then text models
    model_names = image_models + text_models if available_models else [
        "nano-banana-pro-preview",
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest", 
        "gemini-pro",
    ]
    
    model = None
    is_image_model = False
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            is_image_model = "image" in model_name.lower() or "banana" in model_name.lower()
            # Test with a simple call (skip test for image models as they may have different API)
            if not is_image_model:
                _ = model.generate_content("test")
            print(f"✅ Using model: {model_name} ({'image generation' if is_image_model else 'text-to-SVG'})\n")
            break
        except Exception as e:
            print(f"  ⚠️  {model_name} failed: {e}")
            continue
    
    if model is None:
        print("❌ Error: Could not initialize any working Gemini model.")
        sys.exit(1)
    
    return model, is_image_model


def generate_image_direct(model: genai.GenerativeModel, prompt: str, is_image_model: bool) -> bytes | None:
    """Generate PNG image directly using image generation model."""
    try:
        if is_image_model:
            # Use image generation API
            response = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "image/png",
                }
            )
            # Extract image bytes from response
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return part.inline_data.data
            return None
        else:
            # Fallback: generate SVG code
            return None
    except Exception as e:
        print(f"  ⚠️  Error generating image: {e}")
        return None

def generate_svg_code(model: genai.GenerativeModel, prompt: str) -> str | None:
    """Use Gemini to generate SVG code for the logo."""
    try:
        full_prompt = f"""Generate SVG code for this logo design:

{prompt}

Requirements:
- Return ONLY valid SVG code, no markdown, no explanations
- SVG should be 200x200 viewBox
- Use the exact colors specified
- Include all three elements: Extract (blue), Coalesce (green), Stratify (purple)
- Include "anno" text at bottom
- Dark background (#0f172a)
- Valid, renderable SVG XML"""

        response = model.generate_content(full_prompt)
        svg_code = response.text.strip()
        
        # Clean up if wrapped in markdown code blocks
        if svg_code.startswith("```"):
            lines = svg_code.split("\n")
            svg_code = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        
        return svg_code
        
    except Exception as e:
        print(f"  ⚠️  Error generating SVG: {e}")
        return None


def svg_to_png(svg_code: str, output_path: Path) -> bool:
    """Convert SVG code to PNG using cairosvg."""
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_code.encode('utf-8'),
            write_to=str(output_path),
            output_width=1024,
            output_height=1024
        )
        return True
    except ImportError:
        print(f"  ⚠️  Install cairosvg: uv pip install cairosvg")
        return False
    except Exception as e:
        print(f"  ⚠️  Error converting to PNG: {e}")
        return False


def generate_image(model: genai.GenerativeModel, prompt: str, output_path: Path, variant_name: str, is_image_model: bool) -> bool:
    """Generate a logo using Gemini (PNG directly or SVG->PNG)."""
    try:
        print(f"Generating: {variant_name}...")
        
        # Try direct image generation first
        if is_image_model:
            image_bytes = generate_image_direct(model, prompt, is_image_model)
            if image_bytes:
                output_path.write_bytes(image_bytes)
                print(f"  ✅ PNG saved: {output_path}")
                return True
        
        # Fallback: Generate SVG code
        svg_code = generate_svg_code(model, prompt)
        if not svg_code:
            return False
        
        # Save SVG
        svg_path = output_path.with_suffix('.svg')
        svg_path.write_text(svg_code)
        print(f"  ✅ SVG saved: {svg_path}")
        
        # Convert to PNG
        if svg_to_png(svg_code, output_path):
            print(f"  ✅ PNG saved: {output_path}")
            return True
        else:
            print(f"  ⚠️  PNG conversion skipped (install cairosvg for PNG output)")
            return True  # Still count as success since SVG was generated
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Generate batch of logo variations."""
    print("🎨 Generating logo variations with Gemini\n")
    
    model, is_image_model = setup_gemini()
    
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    generated = 0
    for variant in PROMPTS:
        output_path = OUTPUT_DIR / f"{variant['name']}.png"
        
        if generate_image(model, variant["prompt"], output_path, variant["name"], is_image_model):
            generated += 1
        print()  # Blank line between variants
    
    print(f"\n📊 Generated {generated}/{len(PROMPTS)} variations")
    print(f"📁 Review images in: {OUTPUT_DIR}")
    print("\n💡 Next steps:")
    print("   1. Review the generated PNGs")
    print("   2. Provide feedback on favorites")
    print("   3. We'll refine using Rocchio-style iterative improvement")


if __name__ == "__main__":
    main()

