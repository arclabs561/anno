#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "watchdog>=3.0.0",
#     "markdown>=3.5.0",
#     "pygments>=2.16.0",
# ]
# ///

"""
Serve README with auto-reload when README.md changes.
Single stable URL: http://localhost:8002
"""

import http.server
import socketserver
import threading
import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import renderer and port utilities
sys.path.insert(0, str(Path(__file__).parent))
try:
    from render_readme_github_style import render_readme_github_style
    from port_utils import find_free_port
except ImportError:
    # Fallback if port_utils not available
    def find_free_port(start_port=8000, max_attempts=100):
        import socket
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")
    
    try:
        from render_readme_github_style import render_readme_github_style
    except ImportError:
        print("Error: Could not import render_readme_github_style")
        sys.exit(1)

class READMEHandler(FileSystemEventHandler):
    def __init__(self, readme_path, output_path):
        self.readme_path = readme_path
        self.output_path = output_path
    
    def on_modified(self, event):
        if event.src_path == str(self.readme_path):
            print(f"🔄 README.md changed, re-rendering...")
            try:
                render_readme_github_style(str(self.readme_path), str(self.output_path))
                print(f"✅ Re-rendered to {self.output_path}")
            except Exception as e:
                print(f"❌ Error re-rendering: {e}")

# find_free_port is now imported from port_utils

def serve_readme(start_port=8000):
    """Serve README with auto-reload. Auto-finds free port."""
    readme_path = Path(__file__).parent.parent / "README.md"
    output_path = Path(__file__).parent.parent / "README_github_style.html"
    
    # Find free port
    port = find_free_port(start_port)
    
    # Initial render
    print("Rendering README...")
    render_readme_github_style(str(readme_path), str(output_path))
    
    # Watch for changes
    event_handler = READMEHandler(readme_path, output_path)
    observer = Observer()
    observer.schedule(event_handler, str(readme_path.parent), recursive=False)
    observer.start()
    print(f"👀 Watching {readme_path} for changes...")
    
    # Serve
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_path.parent), **kwargs)
        
        def end_headers(self):
            # Add auto-reload script
            if self.path.endswith('.html'):
                self.send_header('Content-Type', 'text/html')
            super().end_headers()
            if self.path.endswith('.html'):
                # Inject auto-reload script
                pass  # Will be handled in do_GET
    
        def do_GET(self):
            if self.path == '/' or self.path == '/README_github_style.html':
                self.path = '/README_github_style.html'
                # Read and inject auto-reload script
                try:
                    file_path = Path(self.directory) / 'README_github_style.html'
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    # Inject auto-reload script before </body>
                    reload_script = b'''
                    <script>
                    // Auto-reload every 2 seconds if file changes
                    let lastModified = null;
                    setInterval(async () => {
                        const response = await fetch('/README_github_style.html?t=' + Date.now());
                        const text = await response.text();
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(text, 'text/html');
                        const newModified = doc.lastModified || response.headers.get('last-modified');
                        if (lastModified && newModified && newModified !== lastModified) {
                            location.reload();
                        }
                        lastModified = newModified || Date.now();
                    }, 2000);
                    </script>
                    '''
                    content = content.replace(b'</body>', reload_script + b'</body>')
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.send_header('Content-length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_error(500, str(e))
                return
            
            super().do_GET()
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}/README_github_style.html"
        print(f"🌐 Serving at {url}")
        print(f"   Auto-reload enabled (watches README.md)")
        print(f"   Press Ctrl+C to stop")
        # Write port to file for justfile/test to read (more reliable than log parsing)
        port_file = Path('/tmp/serve_readme_port.txt')
        with open(port_file, 'w') as f:
            f.write(str(port))
        httpd.serve_forever()

if __name__ == "__main__":
    import sys
    serve_readme()

