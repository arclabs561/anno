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
Headless e2e testing for README rendering using Playwright + Gemini VLM.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from playwright.async_api import async_playwright

# Import shared port utilities
sys.path.insert(0, str(Path(__file__).parent))
try:
    from port_utils import find_readme_server_port
except ImportError:
    # Fallback if port_utils not available
    def find_readme_server_port(start_port=8000, max_attempts=100):
        import socket
        import subprocess
        import urllib.request
        for port in range(start_port, start_port + max_attempts):
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                try:
                    response = urllib.request.urlopen(f"http://localhost:{port}/README_github_style.html", timeout=1)
                    if response.status == 200:
                        return port
                except:
                    pass
        return None

load_dotenv()

async def test_readme_rendering(url: str, output_dir: Path):
    """Test README rendering and get VLM feedback."""
    output_dir.mkdir(exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set viewport to match typical browser
        await page.set_viewport_size({"width": 1280, "height": 720})
        
        print(f"Loading {url}...")
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(2000)
        
        # Take screenshot
        screenshot_path = output_dir / "readme_screenshot.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"✅ Screenshot: {screenshot_path}")
        
        await browser.close()
    
    # Analyze with Gemini VLM
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set, skipping VLM analysis")
        return
    
    genai.configure(api_key=api_key)
    
    # Use Gemini 2.5 Flash (vision model)
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
    except Exception:
        # Fallback
        model = genai.GenerativeModel("models/gemini-1.5-flash")
    
    # Load prompt from external file
    prompt = load_evaluation_prompt()

**Project Documentation Standards (from codebase docs):**

**Style Philosophy** (from `docs/MATH_DOCUMENTATION_GUIDE.md`):
- **Brutalist**: Simple, clear, understandable, quintessential, technical - no marketing fluff
- **Definition first**: State what it is before explaining how it works (FOCS principle)
- **Incremental building**: Introduce concepts step-by-step, building on previous understanding
- **Illustrative examples**: Show practical applications that bridge theory and implementation
- **Precise notation**: Use clear, consistent mathematical notation

**Math Notation** (from `docs/MATH_DOCUMENTATION_GUIDE.md`):
- **ASCII math preferred**: `similarity(a, b) = (a · b) / (||a|| × ||b||)` (works everywhere, no dependencies)
- **When to add formulas**: Complex algorithms (Leiden, box embeddings), evaluation metrics (P/R/F1), calibration metrics (ECE, Brier)
- **When to keep minimal**: Well-known formulas (cosine similarity, F1), simple utilities, obvious calculations
- **Structure for complex cases**: Definition → Formula → Brief intuition → Example → Reference paper

**Stanford/FOCS Documentation Style** (from `docs/MATH_DOCUMENTATION_GUIDE.md`):
- **Introduction to Information Retrieval** (Manning et al. 2008): Clear, technical, accessible; structured sections; formulas with context; practical examples
- **Foundations of Computer Science** (Aho & Ullman 1992): Methodical presentation; clear definitions before explanations; illustrative examples bridging theory and practice; incremental concept building
- **Stanford CoreNLP**: Structured, technical but accessible; clear headings; reference papers for complex concepts

**Content Organization** (from `docs/SCOPE.md`, `docs/ARCHITECTURE.md`):
- **Hierarchy diagrams**: Use ASCII art for structure (e.g., `Signal → Track → Identity`)
- **Status tables**: Clear maturity levels (Mature/Stable/Experimental/Stub)
- **Trait hierarchy**: Show code structure with examples
- **Backend philosophy**: Zero-dependency default, ONNX for production, Candle for pure Rust

**Example Quality Standards** (from `README.md`, `docs/EVALUATION.md`, `examples/`):
- **Show actual output**: CLI examples must include the actual terminal output, not just commands
  - ✅ Good: `$ anno extract "text"` followed by actual entity output with spans and confidence
  - ❌ Bad: `$ anno extract "text"` with no output shown
- **Concrete use cases**: Ingest directory, URL, debug with HTML visualization
- **Code examples**: Complete, runnable, demonstrate real use cases
- **Library examples**: Show actual API usage with expected results

**Structure Preferences** (from project history):
- **Emphasize core**: Extract, Coalesce (primary use cases)
- **De-emphasize advanced**: Stratify is optional/advanced, most users don't need it
- **Links over inline**: Point to detailed docs (`docs/SCOPE.md`, `docs/ARCHITECTURE.md`) rather than exhaustive inline explanations
- **Tables with footnotes**: Provide context and reproducibility commands (e.g., "¹ Pattern accuracy on structured entities only. Reproduce with: `anno benchmark --backend regex`")

**Specific Examples from Codebase**:

1. **Good CLI Output Example** (from `README.md`):
   ```bash
   $ anno extract --model heuristic "Marie Curie won the Nobel Prize in Paris"
   
     PER (2):
       [  0, 11) ########..  75% "Marie Curie"
       [ 20, 31) ######....  60% "Nobel Prize"
     LOC (1):
       [ 35, 40) ########..  80% "Paris"
   ```
   This shows actual output format with spans, confidence, and entity types.

2. **Good Structure Example** (from `docs/SCOPE.md`):
   ```
                    Knowledge Graphs
                          ↑
                   Relation Extraction
                    ↙           ↘
            Coreference    Event Extraction
                    ↘           ↙
              Named Entity Recognition
                          ↑
                   Pattern Matching
   ```
   ASCII art hierarchy that's clear and technical.

3. **Good Math Example** (from `docs/MATH_DOCUMENTATION_GUIDE.md`):
   ```rust
   /// Expected Calibration Error (ECE).
   ///
   /// Formula: `ECE = Σ(n_i / N) × |acc_i - conf_i|`
   ///
   /// Where bins are confidence intervals [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0],
   /// `n_i` is count in bin i, `acc_i` is accuracy in bin i, `conf_i` is mean confidence.
   /// Lower is better (0 = perfectly calibrated).
   ```
   Definition first, then formula, then explanation.

4. **Good Table Example** (from `README.md`):
   | Backend | Latency | Accuracy | Feature |
   |---------|---------|----------|---------|
   | `RegexNER` | ~400ns | ~95%¹ | always |
   
   With footnote: "¹ Pattern accuracy on structured entities only."

Focus on README QUALITY aligned with these standards, not CSS implementation details. Be critical and thorough.

**Required Analysis (be extremely detailed):**

1. **Content Quality & Completeness - 25 points**
   - Is the introduction clear, technical, and brutalist (no marketing fluff like "powerful", "revolutionary", "cutting-edge")?
   - Are key concepts explained with "definition first" approach (e.g., "Signal: Detection-level entity mentions (where + what)" before explaining usage)?
   - Are installation instructions complete and accurate?
   - Are usage examples practical, working, and show actual OUTPUT (not just input)?
     * Example of good: CLI command followed by actual terminal output with entity spans and confidence
     * Example of bad: CLI command with no output shown
   - Is the API/library usage well-documented with concrete examples showing expected results?
   - Are core concepts (Extract, Coalesce) emphasized over advanced features (Stratify)?
   - Is the documentation accurate, up-to-date, and aligned with brutalist style?
   - Do examples follow the codebase pattern: command → actual output → explanation?

2. **Structure & Organization - 20 points**
   - Logical flow: introduction → installation → usage → advanced
   - Clear section hierarchy (H1, H2, H3 used appropriately)
   - Table of contents or clear navigation?
   - Related sections grouped together?
   - Information is easy to find?
   - No redundant or scattered information?

3. **Code Examples & Syntax - 15 points**
   - Code examples are complete and runnable?
   - Syntax highlighting is correct and readable?
   - Examples demonstrate real use cases?
   - Code is properly formatted and indented?
   - Examples cover different scenarios (basic, advanced)?
   - CLI commands are accurate and complete?

4. **Visual Elements (Images, Figures, Diagrams) - 15 points**
   - Are images/figures present where helpful?
   - Are images properly positioned (not breaking flow)?
   - Are images relevant and add value?
   - Are images accessible (alt text, proper sizing)?
   - Are diagrams clear and readable?
   - Logo/branding is appropriate?

5. **Math & Technical Notation - 10 points**
   - Math formulas render correctly (if present)?
   - Technical notation is clear and readable?
   - Mathematical concepts are explained appropriately (definition first, then explanation)?
     * Good example: "Expected Calibration Error (ECE). Formula: `ECE = Σ(n_i / N) × |acc_i - conf_i|`. Where bins are..."
     * Bad example: Formula without definition or context
   - Formulas use ASCII math notation (preferred) like `similarity(a, b) = (a · b) / (||a|| × ||b||)`?
   - Complex concepts have visual aids (ASCII diagrams) or reference papers?
   - Math follows Stanford/FOCS style: clear, technical, accessible with context?
   - Are formulas only added where relevant (complex algorithms, evaluation metrics) and not over-explained for simple utilities?

6. **Readability & Presentation - 10 points**
   - Text is readable (not too dense, good spacing)?
   - Lists are used appropriately (bullets vs numbered)?
   - Tables are clear and well-formatted?
   - Links are descriptive and work?
   - Emphasis (bold, italic) is used appropriately?
   - Code blocks have proper language tags?

7. **Completeness & Polish - 5 points**
   - No placeholder text or TODOs?
   - No broken links or references?
   - Consistent formatting throughout?
   - Professional appearance?
   - Ready for public consumption?

**Scoring Rules:**
- 90-100: Excellent, production-ready documentation
- 80-89: Very good, minor improvements needed
- 70-79: Good, some gaps or issues
- 60-69: Acceptable, needs significant work
- Below 60: Poor, major issues

**Be critical. Deduct points for:**
- Marketing language or fluff (words like "powerful", "revolutionary", "cutting-edge", "state-of-the-art" - should be brutalist/technical)
- Missing essential information
- Unclear or confusing explanations (should be definition-first: "What it is" before "How it works")
- Broken or incomplete examples
- Examples that show only input, not actual output (CLI commands without terminal output shown)
- Poor organization or structure (should follow: introduction → installation → usage → advanced)
- Missing visual aids where needed (especially for Pipeline concepts - ASCII diagrams like `Signal → Track → Identity`)
- Math/formulas that don't render or use wrong notation (should prefer ASCII like `F1 = 2 × (P × R) / (P + R)`)
- Images that break layout or are unclear
- Inconsistent formatting
- Placeholder content or TODOs
- Over-emphasizing advanced/optional features (Stratify) vs core (Extract, Coalesce)
- Missing concrete use cases (ingest directory, URL, debug with HTML visualization)
- Tables without footnotes explaining context and reproducibility
- Missing links to detailed docs (should link to `docs/SCOPE.md`, `docs/ARCHITECTURE.md` rather than exhaustive inline explanations)
- Examples that don't show actual results (library code without expected output)

Provide:
1. Overall score (0-100) - be critical
2. Breakdown by category with point deductions
3. Specific issues found with examples
4. Actionable recommendations for improvement"""
    
    from PIL import Image
    img = Image.open(screenshot_path)
    
    print("Analyzing with Gemini VLM...")
    response = model.generate_content([prompt, img])
    
    analysis_path = output_dir / "vlm_analysis.txt"
    with open(analysis_path, "w") as f:
        f.write(response.text)
    
    print(f"✅ Analysis: {analysis_path}")
    
    # Extract score if present
    import re
    score_match = re.search(r'(\d+)/100|score[:\s]+(\d+)', response.text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1) or score_match.group(2))
        print(f"\n{'='*80}")
        if score >= 90:
            print(f"🎯 Score: {score}/100 - {'EXCELLENT' if score >= 95 else 'VERY GOOD'}")
        elif score >= 80:
            print(f"⚠️  Score: {score}/100 - GOOD (needs improvement)")
        elif score >= 70:
            print(f"⚠️  Score: {score}/100 - ACCEPTABLE (significant issues)")
        else:
            print(f"❌ Score: {score}/100 - NEEDS WORK (major issues)")
        print("="*80)
    
    print("\n" + "="*80)
    print("FULL ANALYSIS:")
    print("="*80)
    print(response.text)
    print("="*80)

# find_readme_server_port is now imported from port_utils

async def main():
    output_dir = Path("test_output")
    
    # Try to find server port (check port file first, then scan)
    port = None
    port_file = Path('/tmp/serve_readme_port.txt')
    if port_file.exists():
        try:
            port = int(port_file.read_text().strip())
            # Verify it's actually serving
            import urllib.request
            try:
                response = urllib.request.urlopen(f"http://localhost:{port}/README_github_style.html", timeout=1)
                if response.status == 200:
                    pass  # Port is valid
                else:
                    port = None
            except:
                port = None
        except (ValueError, OSError):
            pass
    
    # Fallback to scanning ports
    if not port:
        port = find_readme_server_port()
    
    if not port:
        print("⚠️  README server not found")
        print("Start with: uv run scripts/serve_readme.py")
        print("   or: just readme-preview")
        sys.exit(1)
    
    url = f"http://localhost:{port}/README_github_style.html"
    print(f"📍 Found server on port {port}")
    
    await test_readme_rendering(url, output_dir)

if __name__ == "__main__":
    asyncio.run(main())

