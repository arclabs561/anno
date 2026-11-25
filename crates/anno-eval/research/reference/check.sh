#!/usr/bin/env bash
# Run from any directory after installing the research-only NumPy/SciPy dependencies.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m unittest test_entity_memory_replay
python3 -m unittest discover -s bookscale -p 'test_*.py'
python3 -m unittest discover -s anno_adapter -p 'test_*.py'
