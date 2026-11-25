"""ONNX-compatible sliding attention windows for the NuNER Zero-4k exporter."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch


def _windows(tensor, width, step):
    """Return [heads, windows, width, features] indexed by logical sequence positions."""
    import torch

    starts = torch.arange((tensor.shape[1] - width) // step + 1, device=tensor.device) * step
    positions = starts[:, None] + torch.arange(width, device=tensor.device)[None, :]
    return tensor.index_select(1, positions.flatten()).reshape(
        tensor.shape[0], -1, width, tensor.shape[2]
    )


def _query_key_windows(states, window_overlap, onnx_export=False):
    # Upstream _chunk uses as_strided on non-contiguous head projections.
    # TorchScript ONNX flattens those inputs before applying the storage strides.
    return _windows(states, 2 * window_overlap, window_overlap)


def _weighted_values(layer, attn_probs, values, window_overlap):
    """Use the same padded and diagonalized blocks as LongformerSelfAttention."""
    import torch
    import torch.nn.functional as F

    batch, length, heads, depth = values.shape
    w = window_overlap
    values = values.transpose(1, 2).reshape(batch * heads, length, depth)
    padded = F.pad(values, (0, 0, w, w), value=-1)
    # The eager as_strided kernel also embeds the example sequence length in
    # its head stride. Compute the window positions from the runtime shape.
    blocks = _windows(padded, 3 * w, w)
    probs = attn_probs.transpose(1, 2).reshape(batch * heads, -1, w, 2 * w + 1)
    probs = layer._pad_and_diagonalize(probs)
    output = torch.einsum("bcwd,bcdh->bcwh", probs, blocks)
    return output.reshape(batch, heads, length, depth).transpose(1, 2)


@contextmanager
def export_windows(model):
    """Patch only the loaded Longformer attention layers and only during tracing."""
    from functools import partial
    from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

    layers = [m for m in model.modules() if isinstance(m, LongformerSelfAttention)]
    if not layers:
        raise ValueError("NuNER Zero-4k export requires a Longformer attention module")
    with ExitStack() as stack:
        for layer in layers:
            stack.enter_context(patch.object(layer, "_chunk", _query_key_windows))
            stack.enter_context(
                patch.object(layer, "_sliding_chunks_matmul_attn_probs_value", partial(_weighted_values, layer))
            )
        yield


def parity_cases():
    """Texts and ordered prompt labels used to compare Zero-4k runtimes."""
    return [
        ("Marie Curie worked in Paris.", ["person", "location"]),
        ("Éva joined Acme in Zürich.", ["organization", "location", "person"]),
        ("work " * 498 + "Marie Curie worked in Paris.", ["person", "location"]),
        ("Work continued. " * 220 + "Marie Curie arrived in Paris.", ["person", "location"]),
        # With the pinned tokenizer this is exactly 4096 IDs, prompt included.
        (
            "electroencephalogram " * 680 + "work work Marie Curie worked in Paris.",
            ["person", "location"],
        ),
    ]


def reference_cases(model):
    """Capture all native PyTorch logits before installing any export patches."""
    import torch

    references = []
    with torch.inference_mode():
        for text, labels in parity_cases():
            wrapper, inputs, spec = model._prepare_export_graph(text=text, labels=labels)
            feed = {
                key: value.numpy()
                for key, value in zip(spec["input_names"], inputs, strict=True)
            }
            references.append((feed, wrapper(*inputs).numpy()))
    return references


def verify_logits(path, references):
    """Check dynamic prompt width, context length and the 0.5 decision boundary."""
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for feed, expected in references:
        actual = session.run(None, feed)[0]
        if actual.shape != expected.shape:
            raise ValueError(f"Output shape {actual.shape} differs from PyTorch {expected.shape}")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3, equal_nan=False)
        np.testing.assert_array_equal(actual > 0, expected > 0)
        print(
            f"{feed['input_ids'].shape[1]} tokens, {actual.shape[-1]} labels, "
            f"maximum absolute logit difference {np.abs(actual - expected).max():.6g}",
            flush=True,
        )
