# SPDX-License-Identifier: MIT OR Apache-2.0
"""Project printed names and caller-supplied bindings into one GroundedDocument."""


def grounded_names(document_id, occurrences, *, catalog="bioguide"):
    """Preserve source order and Unicode scalar offsets in the newline-joined text."""
    names, signals, tracks, identities = [], [], {}, {}
    by_accession = {}
    position = 0
    for i, (name, accession) in enumerate(occurrences):
        if not name or "\n" in name:
            raise ValueError("each occurrence must be one nonempty printed name")
        end = position + len(name)
        names.append(name)
        signals.append({
            "id": i, "location": {"Text": {"start": position, "end": end}},
            "surface": name, "label": "PER", "confidence": 1.0,
            "modality": "Symbolic", "negated": False,
        })
        position = end + 1
        identity = None
        if accession is not None:
            if accession not in by_accession:
                identity = len(identities)
                by_accession[accession] = identity
                identities[str(identity)] = {
                    "id": identity, "canonical_name": name, "aliases": [],
                    "confidence": 0.0, "kb_name": catalog, "kb_id": accession,
                    "source": {"KnowledgeBase": {"kb_name": catalog, "kb_id": accession}},
                }
            identity = by_accession[accession]
        tracks[str(i)] = {
            "id": i, "signals": [{"signal_id": i, "position": 0}],
            "canonical_surface": name, "identity_id": identity, "cluster_confidence": 1.0,
        }
    return {
        "id": document_id, "text": "\n".join(names), "signals": signals,
        "tracks": tracks, "identities": identities,
    }
