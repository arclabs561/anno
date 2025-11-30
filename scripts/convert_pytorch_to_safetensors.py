#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch>=2.0.0",
#     "safetensors>=0.4.0",
#     "packaging>=21.0",
# ]
# ///
"""Convert PyTorch model.bin to safetensors format.

This script converts a PyTorch state dict (.bin file) to safetensors format,
which is required by Candle backends.

Usage:
    uv run scripts/convert_pytorch_to_safetensors.py <input.bin> <output.safetensors>
    # Or with uvx:
    uvx --from scripts convert_pytorch_to_safetensors.py <input.bin> <output.safetensors>
"""

import sys
import torch
from safetensors.torch import save_file
from pathlib import Path


def convert_pytorch_to_safetensors(input_path: str, output_path: str) -> None:
    """Convert PyTorch state dict to safetensors format.
    
    Args:
        input_path: Path to pytorch_model.bin file
        output_path: Path where safetensors file will be written
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load PyTorch state dict
        print(f"Loading PyTorch model from: {input_path}")
        state_dict = torch.load(input_path, map_location="cpu")
        
        # Save as safetensors
        print(f"Saving safetensors to: {output_path}")
        save_file(state_dict, output_path)
        
        # Verify output file exists
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"SUCCESS: Converted {len(state_dict)} tensors ({size_mb:.1f} MB)")
        else:
            print(f"ERROR: Output file was not created: {output_path}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.safetensors>", file=sys.stderr)
        sys.exit(1)
    
    convert_pytorch_to_safetensors(sys.argv[1], sys.argv[2])

