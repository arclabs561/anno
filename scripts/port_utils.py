#!/usr/bin/env python3
"""Shared port utilities for README server and test."""

import socket
import subprocess
import urllib.request

def find_free_port(start_port=8000, max_attempts=100):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")

def find_readme_server_port(start_port=8000, max_attempts=100):
    """Find which port the README server is running on."""
    for port in range(start_port, start_port + max_attempts):
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            # Check if it's serving our HTML
            try:
                response = urllib.request.urlopen(f"http://localhost:{port}/README_github_style.html", timeout=1)
                if response.status == 200:
                    return port
            except:
                pass
    return None

