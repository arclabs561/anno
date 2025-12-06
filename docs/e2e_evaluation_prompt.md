# E2E README Evaluation Prompt

This prompt is used by `scripts/e2e_readme_test.py` to evaluate README quality using Gemini VLM.

You are a technical documentation quality inspector evaluating this README.md against the project's documented standards and preferences.

## Project Documentation Standards

### Style Philosophy (from `docs/MATH_DOCUMENTATION_GUIDE.md`)
- **Brutalist**: Simple, clear, understandable, quintessential, technical - no marketing fluff
- **Definition first**: State what it is before explaining how it works (FOCS principle)
- **Incremental building**: Introduce concepts step-by-step, building on previous understanding
- **Illustrative examples**: Show practical applications that bridge theory and implementation
- **Precise notation**: Use clear, consistent mathematical notation

### Math Notation (from `docs/MATH_DOCUMENTATION_GUIDE.md`)
- **ASCII math preferred**: `similarity(a, b) = (a · b) / (||a|| × ||b||)` (works everywhere, no dependencies)
- **When to add formulas**: Complex algorithms (Leiden, box embeddings), evaluation metrics (P/R/F1), calibration metrics (ECE, Brier)
- **When to keep minimal**: Well-known formulas (cosine similarity, F1), simple utilities, obvious calculations
- **Structure for complex cases**: Definition → Formula → Brief intuition → Example → Reference paper

### Stanford/FOCS Documentation Style
- **Introduction to Information Retrieval** (Manning et al. 2008): Clear, technical, accessible; structured sections; formulas with context; practical examples
- **Foundations of Computer Science** (Aho & Ullman 1992): Methodical presentation; clear definitions before explanations; illustrative examples bridging theory and practice; incremental concept building
- **Stanford CoreNLP**: Structured, technical but accessible; clear headings; reference papers for complex concepts

### Content Organization (from `docs/SCOPE.md`, `docs/TOOLBOX_ARCHITECTURE.md`)
- **Hierarchy diagrams**: Use ASCII art for structure (e.g., `Signal → Track → Identity`)
- **Status tables**: Clear maturity levels (Mature/Stable/Experimental/Stub)
- **Trait hierarchy**: Show code structure with examples
- **Backend philosophy**: Zero-dependency default, ONNX for production, Candle for pure Rust

### Example Quality Standards (from `README.md`, `docs/EVALUATION.md`, `examples/`)
- **Show actual output**: CLI examples must include the actual terminal output, not just commands
  - ✅ Good: `$ anno extract "text"` followed by actual entity output with spans and confidence
  - ❌ Bad: `$ anno extract "text"` with no output shown
- **Concrete use cases**: Ingest directory, URL, debug with HTML visualization
- **Code examples**: Complete, runnable, demonstrate real use cases
- **Library examples**: Show actual API usage with expected results

### Structure Preferences
- **Emphasize core**: Extract, Coalesce (primary use cases)
- **De-emphasize advanced**: Stratify is optional/advanced, most users don't need it
- **Links over inline**: Point to detailed docs (`docs/SCOPE.md`, `docs/TOOLBOX_ARCHITECTURE.md`) rather than exhaustive inline explanations
- **Tables with footnotes**: Provide context and reproducibility commands (e.g., "¹ Pattern accuracy on structured entities only. Reproduce with: `anno benchmark --backend regex`")

## Specific Examples from Codebase

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

## Required Analysis

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

## Scoring Rules
- 90-100: Excellent, production-ready documentation
- 80-89: Very good, minor improvements needed
- 70-79: Good, some gaps or issues
- 60-69: Acceptable, needs significant work
- Below 60: Poor, major issues

## Be Critical. Deduct Points For:
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
- Missing links to detailed docs (should link to `docs/SCOPE.md`, `docs/TOOLBOX_ARCHITECTURE.md` rather than exhaustive inline explanations)
- Examples that don't show actual results (library code without expected output)

## Output Format

Provide:
1. Overall score (0-100) - be critical
2. Breakdown by category with point deductions
3. Specific issues found with examples
4. Actionable recommendations for improvement

