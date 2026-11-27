#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Download and convert the Elgold entity linking dataset.

Elgold is a gold standard, multi-genre dataset for NER and entity linking.
Reference: Olewniczak & Szymanski (2025), Scientific Data
Paper: https://doi.org/10.1038/s41597-025-05274-4
Dataset: https://doi.org/10.34808/f3q9-9k64

Features:
- 7 text categories: News, Job offers, Movie reviews, Automotive blogs,
  Amazon reviews, Scientific abstracts (5 subcategories), Historic blogs
- 14 entity types: Modified OntoNotes 5.0 + DISEASE, SUBSTANCE, SPECIE
- 276 documents, ~3,559 mentions (3,106 linked, 453 NIL)
- Wikipedia KB (2023-11-21 snapshot)
- CC-BY-4.0 license

Usage:
    uv run scripts/download_elgold.py              # Download and convert
    uv run scripts/download_elgold.py --stats      # Show dataset statistics
    uv run scripts/download_elgold.py --to-jsonl   # Convert to JSONL format
    uv run scripts/download_elgold.py --validate   # Validate annotations
"""

import argparse
import json
import os
import platform
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

class ElgoldAnnotation(NamedTuple):
    """A single Elgold annotation."""
    mention: str
    entity_class: str
    target: str  # Wikipedia article title, empty for NIL
    start_char: int
    end_char: int


class ElgoldDocument(NamedTuple):
    """A parsed Elgold document."""
    doc_id: str
    category: str
    text: str  # Plain text without annotations
    raw_text: str  # Original text with annotations
    annotations: list[ElgoldAnnotation]


# Entity class mapping to standardized labels
ELGOLD_ENTITY_CLASSES = {
    "PERSON": "PER",
    "ORG": "ORG", 
    "LOC": "LOC",
    "GPE": "GPE",
    "FAC": "FAC",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW": "LAW",
    "LANGUAGE": "LANGUAGE",
    "NORP": "NORP",
    # Extended classes (beyond OntoNotes 5.0)
    "DISEASE": "DISEASE",
    "SUBSTANCE": "SUBSTANCE",
    "SPECIE": "SPECIE",
}

# Category ID to name mapping
CATEGORY_NAMES = {
    "1": "News",
    "2": "Job offers",
    "3": "Movie reviews",
    "4": "Automotive blogs",
    "5": "Amazon product reviews",
    "6a": "Scientific - Biomedicine",
    "6b": "Scientific - Life Sciences",
    "6c": "Scientific - Mathematics",
    "6d": "Scientific - Medicine & Public Health",
    "6e": "Scientific - Humanities & Social Sciences",
    "8": "Historic blogs",
}


def get_cache_dir() -> Path:
    """Get the anno cache directory."""
    if custom := os.environ.get("ANNO_CACHE_DIR"):
        return Path(custom)
    
    if platform.system() == "Darwin":
        return Path.home() / "Library/Caches/anno"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(xdg_cache) / "anno"


def download_elgold(cache_dir: Path, force: bool = False) -> Path:
    """Find staged Elgold documents or print the publisher download instructions."""
    dataset_dir = cache_dir / "elgold"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    data_dir = dataset_dir / "data"
    if data_dir.exists() and not force:
        txt_files = list(data_dir.glob("*.txt"))
        if txt_files:
            print(f"[INFO] Elgold already downloaded: {len(txt_files)} files in {data_dir}")
            return data_dir
    
    print("""
================================================================================
Elgold Dataset Download Instructions
================================================================================

Elgold v1.1 is hosted by Gdańsk University of Technology. To prepare its
English annotated documents for conversion:

1. Visit the versioned release: https://doi.org/10.34808/8zbz-1e66
   (or directly: https://mostwiedzy.pl/en/open-research-data/elgold-gold-standard-multi-genre-dataset-for-named-entity-recognition-and-linking,202503241553453028951-0)

2. Click "Download" to get the ZIP file

3. Extract the contents to:
   {dataset_dir}

4. The directory should contain a 'data' folder with .txt files like:
   - 1_1.txt, 1_2.txt, ... (News)
   - 2_1.txt, 2_2.txt, ... (Job offers)
   - etc.

The publisher ZIP has SHA-256
4c2afa2eb6622307da4bf279aa9d0420fc75abec13f2dc143d49a9ec8e307aec.

After downloading, run this script again to process the dataset.
================================================================================
""".format(dataset_dir=dataset_dir))
    
    return dataset_dir


def parse_elgold_file(filepath: Path) -> ElgoldDocument | None:
    """Parse a single Elgold annotated file."""
    try:
        raw_text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = filepath.read_text(encoding="latin-1")
    return parse_elgold_text(filepath.stem, raw_text)


def parse_elgold_text(filename: str, raw_text: str) -> ElgoldDocument | None:
    """Parse a publisher document by its category/serial stem."""
    parts = filename.split("_")
    if len(parts) != 2:
        print(f"[WARN] Unexpected filename format: {filename}")
        return None
    
    category_id, serial = parts
    category = CATEGORY_NAMES.get(category_id, f"Unknown ({category_id})")
    
    # Parse annotations: {{mention|class|target}}
    annotation_pattern = re.compile(r"\{\{([^|{}]+)\|([^|{}]+)\|([^{}]*)\}\}")
    
    annotations = []
    plain_text = raw_text
    offset_adjustment = 0
    
    for match in annotation_pattern.finditer(raw_text):
        mention = match.group(1)
        entity_class = match.group(2).upper()
        target = match.group(3).strip()
        
        # Calculate position in plain text
        orig_start = match.start()
        plain_start = orig_start - offset_adjustment
        plain_end = plain_start + len(mention)
        
        annotations.append(ElgoldAnnotation(
            mention=mention,
            entity_class=entity_class,
            target=target,
            start_char=plain_start,
            end_char=plain_end,
        ))
        
        # Track how much text we're removing
        removed_chars = len(match.group(0)) - len(mention)
        offset_adjustment += removed_chars
    
    # Create plain text by removing annotation markup
    plain_text = annotation_pattern.sub(r"\1", raw_text)
    
    return ElgoldDocument(
        doc_id=filename,
        category=category,
        text=plain_text,
        raw_text=raw_text,
        annotations=annotations,
    )


def parse_elgold_dataset(data_dir: Path) -> list[ElgoldDocument]:
    """Parse all Elgold files in a directory."""
    documents = []
    
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No .txt files found in {data_dir}")
        return documents
    
    for filepath in txt_files:
        doc = parse_elgold_file(filepath)
        if doc:
            documents.append(doc)
    
    return documents


def compute_statistics(documents: list[ElgoldDocument]) -> dict:
    """Compute dataset statistics."""
    stats = {
        "total_documents": len(documents),
        "total_mentions": 0,
        "linked_mentions": 0,
        "nil_mentions": 0,
        "categories": Counter(),
        "entity_classes": Counter(),
        "mentions_per_category": defaultdict(int),
        "unique_targets": set(),
    }
    
    for doc in documents:
        stats["categories"][doc.category] += 1
        for ann in doc.annotations:
            stats["total_mentions"] += 1
            stats["entity_classes"][ann.entity_class] += 1
            stats["mentions_per_category"][doc.category] += 1
            
            if ann.target:
                stats["linked_mentions"] += 1
                stats["unique_targets"].add(ann.target)
            else:
                stats["nil_mentions"] += 1
    
    stats["unique_targets"] = len(stats["unique_targets"])
    return stats


def print_statistics(stats: dict):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Elgold Dataset Statistics")
    print("=" * 60)
    
    print(f"\nDocuments: {stats['total_documents']}")
    print(f"Total mentions: {stats['total_mentions']}")
    print(f"  - Linked to Wikipedia: {stats['linked_mentions']}")
    print(f"  - NIL (no article): {stats['nil_mentions']}")
    print(f"Unique Wikipedia targets: {stats['unique_targets']}")
    
    print("\nBy Category:")
    for cat, count in sorted(stats["categories"].items()):
        mentions = stats["mentions_per_category"][cat]
        print(f"  {cat}: {count} docs, {mentions} mentions")
    
    print("\nBy Entity Class:")
    for cls, count in stats["entity_classes"].most_common():
        mapped = ELGOLD_ENTITY_CLASSES.get(cls, cls)
        print(f"  {cls} -> {mapped}: {count}")


def convert_to_jsonl(documents: list[ElgoldDocument], output_path: Path):
    """Convert Elgold to JSONL format for Anno."""
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            # Convert to spans format
            entities = []
            for ann in doc.annotations:
                entities.append({
                    "text": ann.mention,
                    "start": ann.start_char,
                    "end": ann.end_char,
                    "label": ELGOLD_ENTITY_CLASSES.get(ann.entity_class, ann.entity_class),
                    "wikipedia_target": ann.target or None,
                })
            
            record = {
                "doc_id": doc.doc_id,
                "category": doc.category,
                "text": doc.text,
                "entities": entities,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"[INFO] Wrote {len(documents)} documents to {output_path}")


def convert_to_conll(documents: list[ElgoldDocument], output_path: Path):
    """Convert Elgold to CoNLL BIO format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            # Simple whitespace tokenization
            tokens = doc.text.split()
            char_offset = 0
            
            for token in tokens:
                # Find token position
                pos = doc.text.find(token, char_offset)
                if pos == -1:
                    tag = "O"
                else:
                    token_start = pos
                    token_end = pos + len(token)
                    
                    # Check for overlap with annotations
                    tag = "O"
                    for ann in doc.annotations:
                        if token_start < ann.end_char and token_end > ann.start_char:
                            prefix = "B" if token_start <= ann.start_char else "I"
                            label = ELGOLD_ENTITY_CLASSES.get(ann.entity_class, ann.entity_class)
                            tag = f"{prefix}-{label}"
                            break
                    
                    char_offset = token_end
                
                f.write(f"{token}\t{tag}\n")
            
            f.write("\n")  # Blank line between documents
    
    print(f"[INFO] Wrote CoNLL format to {output_path}")


def validate_annotations(documents: list[ElgoldDocument]) -> bool:
    """Validate annotation consistency."""
    errors = []
    
    for doc in documents:
        for ann in doc.annotations:
            # Check entity class is known
            if ann.entity_class not in ELGOLD_ENTITY_CLASSES:
                errors.append(f"{doc.doc_id}: Unknown entity class '{ann.entity_class}'")
            
            # Check mention matches text
            extracted = doc.text[ann.start_char:ann.end_char]
            if extracted != ann.mention:
                errors.append(
                    f"{doc.doc_id}: Mention mismatch at {ann.start_char}-{ann.end_char}: "
                    f"'{ann.mention}' vs '{extracted}'"
                )
    
    if errors:
        print(f"\n[ERROR] Found {len(errors)} validation errors:")
        for err in errors[:20]:  # Show first 20
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False
    
    print("[OK] All annotations validated successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and process the Elgold entity linking dataset"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--to-jsonl", action="store_true", help="Convert to JSONL format")
    parser.add_argument("--to-conll", action="store_true", help="Convert to CoNLL BIO format")
    parser.add_argument("--validate", action="store_true", help="Validate annotations")
    parser.add_argument("--data-dir", type=Path, help="Path to Elgold data directory")
    parser.add_argument("--output", type=Path, help="Output path for conversions")
    
    args = parser.parse_args()
    
    cache_dir = get_cache_dir() / "datasets"
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        dataset_dir = download_elgold(cache_dir, args.force)
        data_dir = dataset_dir / "data" if (dataset_dir / "data").exists() else dataset_dir
    
    # Check if data exists
    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"\n[INFO] No data found in {data_dir}")
        print("Please download the dataset first. See instructions above.")
        return 1
    
    # Parse the dataset
    print(f"\n[INFO] Parsing {len(txt_files)} files from {data_dir}")
    documents = parse_elgold_dataset(data_dir)
    
    if not documents:
        print("[ERROR] No documents parsed")
        return 1
    
    # Show statistics
    if args.stats or not any([args.to_jsonl, args.to_conll, args.validate]):
        stats = compute_statistics(documents)
        print_statistics(stats)
    
    # Validate
    if args.validate:
        if not validate_annotations(documents):
            return 1
    
    # Convert to JSONL
    if args.to_jsonl:
        output = args.output or (cache_dir / "elgold.jsonl")
        convert_to_jsonl(documents, output)
    
    # Convert to CoNLL
    if args.to_conll:
        output = args.output or (cache_dir / "elgold.conll")
        convert_to_conll(documents, output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

