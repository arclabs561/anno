#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "markdown>=3.5.0",
#     "pygments>=2.16.0",
#     "pillow>=10.0.0",
# ]
# ///

"""
Render README.md with GitHub-style CSS using markdown library.
"""

import sys
from pathlib import Path

try:
    import markdown
    from markdown.extensions import codehilite, tables, fenced_code, toc
    from pygments.formatters import HtmlFormatter
except ImportError as e:
    print(f"Error: Missing dependency. Install with: uv pip install {e.name}")
    sys.exit(1)

def render_readme_github_style(readme_path: str, output_path: str):
    """Render README.md with GitHub-style CSS."""
    
    # Read README
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Configure markdown extensions
    extensions = [
        'codehilite',
        'fenced_code',
        'tables',
        'toc',
        'nl2br',
        'sane_lists',
    ]
    
    # Render markdown to HTML
    md = markdown.Markdown(extensions=extensions, extension_configs={
        'codehilite': {
            'css_class': 'highlight',
            'use_pygments': True,
            'noclasses': False,
        }
    })
    
    html_content = md.convert(readme_content)
    
    # Get Pygments CSS (use default style, then override with GitHub colors)
    formatter = HtmlFormatter(style='default')
    pygments_css = formatter.get_style_defs('.highlight')
    
    # Override with GitHub's exact color scheme
    pygments_css += """
    .highlight { background-color: #f6f8fa; }
    /* GitHub's syntax highlighting colors - MUST match exactly */
    .highlight .c { color: #6a737d !important; } /* Comments */
    .highlight .c1 { color: #6a737d !important; } /* Single-line comments */
    .highlight .cm { color: #6a737d !important; } /* Multi-line comments */
    .highlight .cp { color: #6a737d !important; } /* Comment preproc */
    .highlight .cs { color: #6a737d !important; } /* Comment special */
    .highlight .go { color: #6a737d !important; } /* Generic output */
    .highlight .gp { color: #6a737d !important; } /* Generic prompt */
    .highlight .k { color: #d73a49; } /* Keywords (use, let, fn, etc.) */
    .highlight .kt { color: #d73a49; } /* Keyword types */
    .highlight .kd { color: #d73a49; } /* Keyword declaration */
    .highlight .s { color: #032f62; } /* Strings */
    .highlight .s1 { color: #032f62; } /* Single-quoted strings */
    .highlight .s2 { color: #032f62; } /* Double-quoted strings */
    .highlight .sx { color: #032f62; } /* Other strings */
    .highlight .n { color: #005cc5; } /* Names (variables, functions) */
    .highlight .na { color: #005cc5; } /* Name attribute */
    .highlight .nb { color: #005cc5; } /* Name builtin */
    .highlight .nc { color: #005cc5; } /* Name class */
    .highlight .no { color: #005cc5; } /* Name constant */
    .highlight .nd { color: #6f42c1; } /* Name decorator */
    .highlight .ni { color: #005cc5; } /* Name entity */
    .highlight .ne { color: #005cc5; } /* Name exception */
    .highlight .nf { color: #6f42c1; } /* Name function */
    .highlight .nl { color: #005cc5; } /* Name label */
    .highlight .nn { color: #005cc5; } /* Name namespace */
    .highlight .nt { color: #22863a; } /* Name tag */
    .highlight .nv { color: #e36209; } /* Name variable */
    .highlight .o { color: #d73a49; } /* Operators */
    .highlight .ow { color: #d73a49; } /* Operator word */
    .highlight .m { color: #005cc5; } /* Numbers */
    .highlight .mf { color: #005cc5; } /* Float numbers */
    .highlight .mh { color: #005cc5; } /* Hex numbers */
    .highlight .mi { color: #005cc5; } /* Integer numbers */
    .highlight .mo { color: #005cc5; } /* Octal numbers */
    .highlight .p { color: #24292e; } /* Punctuation */
    .highlight .cp { color: #6a737d; } /* Comment preproc */
    .highlight .cs { color: #6a737d; } /* Comment special */
    .highlight .err { color: #d73a49; } /* Error */
    .highlight .gd { color: #d73a49; background-color: #ffeef0; } /* Generic deleted */
    .highlight .ge { font-style: italic; } /* Generic emph */
    .highlight .gr { color: #d73a49; } /* Generic error */
    .highlight .gh { color: #005cc5; font-weight: 600; } /* Generic heading */
    .highlight .gi { color: #22863a; background-color: #f0fff4; } /* Generic inserted */
    .highlight .go { color: #6a737d; } /* Generic output */
    .highlight .gp { color: #6a737d; } /* Generic prompt */
    .highlight .gs { font-weight: 600; } /* Generic strong */
    .highlight .gu { color: #6f42c1; } /* Generic subheading */
    .highlight .gt { color: #d73a49; } /* Generic traceback */
    """
    
    # Read GitHub style CSS
    css_path = Path(__file__).parent / 'github_style_css.css'
    if css_path.exists():
        with open(css_path, 'r') as f:
            github_css = f.read()
    else:
        github_css = ""
    
    # Combine into full HTML
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README.md - GitHub Style</title>
    <style>
        /* GitHub Style CSS */
        {github_css}
        
        /* Pygments Syntax Highlighting */
        {pygments_css}
        
        /* Container for GitHub-like width - MUST be exactly 32px padding */
        .container {{
            max-width: 1012px;
            margin: 0 auto;
            padding: 32px;
            padding-left: 32px;
            padding-right: 32px;
            box-sizing: border-box;
        }}
        
        /* Additional GitHub-specific styles */
        .highlight {{
            background-color: #f6f8fa;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 16px;
            overflow: auto;
        }}
        
        .highlight pre {{
            background-color: transparent;
            padding: 0;
            margin: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(full_html)
    
    print(f"✅ Rendered README to {output_path}")

if __name__ == "__main__":
    readme_path = Path(__file__).parent.parent / "README.md"
    output_path = Path(__file__).parent.parent / "README_github_style.html"
    
    render_readme_github_style(str(readme_path), str(output_path))

