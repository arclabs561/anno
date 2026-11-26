"""Check plain-text source coordinates on multilingual inputs with local CPU models.

The language describes the fixture. These synthetic witnesses measure source
integrity only. Pinned neural runs and labeled corpora have separate workflows.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


PREFIX = {
    "ar": "وصلت ليلى", "ca": "La Mercè arriba", "cs": "Přišla Věra",
    "cu": "Въ начѧло", "de": "Grüße aus Köln", "en": "A note from Zoë",
    "es": "Mañana en León", "fr": "À côté de Léa", "grc": "Ἐν Ἀθήναις",
    "hbo": "בְּרֵאשִׁית", "he": "שלום מיעל", "hi": "दिल्ली में",
    "hu": "Győr városa", "ja": "東京から", "ko": "서울에서",
    "la": "Rōmae", "lt": "Žinutė iš Šiaulių", "nl": "Groet uit België",
    "no": "Hilsen fra Tromsø", "pl": "Łódź dzisiaj", "ru": "Письмо из Москвы",
    "tr": "İzmir’den", "zh": "来自北京",
}
LOCAL = {"pattern", "heuristic", "stacked", "ensemble", "crf", "hmm",
         "heuristic_crf", "heuristic_fr", "tplinker"}


def verify(binary, model, prefix):
    email = "hi@example.org"
    text = f"{prefix} 📨 e\u0301 — {email} / {email}."
    raw = text.encode()
    proc = subprocess.run([binary, "extract", "--model", model.replace("_", "-"),
                           "--text", text, "--format", "json"], capture_output=True,
                          text=True, timeout=45, check=True)
    out = json.loads(proc.stdout)
    digest = hashlib.sha256(raw).hexdigest()
    for key in ("source_sha256", "analyzed_sha256"):
        if out["source_map"][key] != digest:
            raise ValueError(f"{key} differs from original bytes")
    provenance = out["provenance"]
    runtime = provenance["runtime_model"]
    if runtime in (None, "", "unknown") or provenance["model"].replace("-", "_") != model:
        raise ValueError("missing runtime or wrong requested model")
    emails = []
    for entity in out["entities"]:
        start, end = entity["start"], entity["end"]
        if not (0 <= start < end <= len(text) and text[start:end] == entity["text"]):
            raise ValueError("invalid character offsets")
        expected = [{"start": len(text[:start].encode()), "end": len(text[:end].encode())}]
        if entity["analyzed_byte_ranges"] != expected:
            raise ValueError("byte offsets differ from exact occurrence")
        if entity["text"] == email:
            emails.append(start)
    if model in {"pattern", "stacked"} and sorted(emails) != [text.index(email), text.rindex(email)]:
        raise ValueError("lost a repeated email occurrence")
    if model == "tplinker" and runtime == "tplinker" and "heuristic" in provenance.get("entity_sources", []):
        raise ValueError("heuristic fallback reported as neural")
    entities = out["entities"]
    return {"status": "spans_verified" if entities else "empty", "entities": len(entities),
            "runtime_model": runtime, "entity_sources": provenance.get("entity_sources", [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    listing = subprocess.check_output([binary, "models", "list"], text=True)
    inventory = dict((name, status) for status, name in re.findall(
        r"^\s*(not compiled|compiled) ([\w-]+)\s*$", listing, re.MULTILINE))
    if inventory.get("pattern") != "compiled":
        raise RuntimeError("missing compiled pattern backend in inventory")
    rows = []
    failures = 0
    for model, build in inventory.items():
        for lang, prefix in PREFIX.items():
            if build == "not compiled":
                result = {"status": "not_compiled"}
            elif model not in LOCAL:
                result = {"status": "outside_local_panel"}
            else:
                try:
                    result = verify(binary, model, prefix)
                except (ValueError, KeyError, TypeError, subprocess.SubprocessError) as err:
                    result = {"status": "error", "error": str(err)}
                    print(model, lang, err)
                    failures += 1
            rows.append({"requested": model, "language": lang, **result})
    report = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "task": "entity_source_coordinates", "domain": "synthetic_plain_text",
              "inventory": inventory, "cells": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
