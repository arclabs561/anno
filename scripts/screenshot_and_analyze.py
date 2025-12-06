#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "playwright>=1.40.0",
#     "google-generativeai>=0.3.0",
#     "python-dotenv>=1.0.0",
#     "pillow>=10.0.0",
# ]
# ///

"""
Screenshot README preview and analyze with Gemini 3 Pro Vision.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from playwright.async_api import async_playwright

# Load environment variables
load_dotenv()

async def take_screenshot(url: str, output_path: str):
    """Take a screenshot of the URL using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set viewport to match typical browser size
        await page.set_viewport_size({"width": 1280, "height": 720})
        
        print(f"Loading {url}...")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except Exception as e:
            print(f"Warning: {e}, trying with load event...")
            await page.goto(url, wait_until="load", timeout=15000)
        
        # Wait a bit for any dynamic content
        await page.wait_for_timeout(2000)
        
        print(f"Taking screenshot...")
        await page.screenshot(path=output_path, full_page=True)
        await browser.close()
        
        print(f"✅ Screenshot saved to {output_path}")

def analyze_with_gemini(image_path: str, prompt: str) -> str:
    """Analyze screenshot with Gemini 3 Pro Vision."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    genai.configure(api_key=api_key)
    
    # List available models and find one that supports vision
    print("Listing available models...")
    models = genai.list_models()
    vision_model = None
    
    for model in models:
        if "gemini" in model.name.lower():
            methods = str(model.supported_generation_methods)
            if "generateContent" in methods:
                # Check if it's a vision-capable model
                model_name_lower = model.name.lower()
                if any(x in model_name_lower for x in ["pro", "flash", "vision"]):
                    vision_model = model.name
                    print(f"Found vision model: {vision_model}")
                    break
    
    if not vision_model:
        # Fallback: try common model names
        fallback_names = [
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.5-flash-latest",
            "models/gemini-pro-vision",
        ]
        for name in fallback_names:
            try:
                test_model = genai.GenerativeModel(name)
                vision_model = name
                print(f"Using fallback model: {vision_model}")
                break
            except:
                continue
    
    if not vision_model:
        raise ValueError("No vision model found. Available models: " + ", ".join([m.name for m in models[:10]]))
    
    model = genai.GenerativeModel(vision_model)
    
    # Read image
    import PIL.Image
    img = PIL.Image.open(image_path)
    
    print("Analyzing with Gemini...")
    response = model.generate_content([prompt, img])
    
    return response.text

async def main():
    # Try custom renderer first, fallback to grip
    url = "http://localhost:8001/README_github_style.html"
    screenshot_path = "README_preview_screenshot.png"
    
    # Check if custom renderer is available
    import subprocess
    result = subprocess.run(
        ["lsof", "-ti:8001"],
        capture_output=True,
        text=True
    )
    if not result.stdout.strip():
        # Fallback to grip
        url = "http://localhost:8000"
        result = subprocess.run(
            ["lsof", "-ti:8000"],
            capture_output=True,
            text=True
        )
        if not result.stdout.strip():
            print("❌ Error: Neither custom renderer (port 8001) nor grip (port 8000) is running")
            print("Start custom renderer with: python3 -m http.server 8001")
            print("Or start grip with: grip README.md :8000")
            sys.exit(1)
    
    # Check if grip is running
    import subprocess
    result = subprocess.run(
        ["lsof", "-ti:8000"],
        capture_output=True,
        text=True
    )
    if not result.stdout.strip():
        print("❌ Error: Grip is not running on port 8000")
        print("Start it with: grip README.md :8000")
        sys.exit(1)
    
    # Take screenshot
    await take_screenshot(url, screenshot_path)
    
    # Analyze with Gemini
    prompt = """Compare this README.md rendering to how GitHub would render it. 
    
Specifically analyze:
1. Typography and font rendering
2. Code block styling
3. Table formatting
4. Link styling
5. Overall layout and spacing
6. Color scheme and theme
7. Any visual differences from GitHub's actual rendering

Provide specific, actionable feedback on what's different and how to make it match GitHub's rendering more closely."""
    
    analysis = analyze_with_gemini(screenshot_path, prompt)
    
    print("\n" + "="*80)
    print("GEMINI ANALYSIS:")
    print("="*80)
    print(analysis)
    print("="*80)
    
    # Save analysis to file
    analysis_path = "README_preview_analysis.txt"
    with open(analysis_path, "w") as f:
        f.write(analysis)
    print(f"\n✅ Analysis saved to {analysis_path}")
    print(f"✅ Screenshot saved to {screenshot_path}")

if __name__ == "__main__":
    asyncio.run(main())

