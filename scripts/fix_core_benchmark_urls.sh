#!/bin/bash
#
# Script to fix broken URLs for core NER benchmarks
# Based on URL health check results and web research
#
# Usage: ./scripts/fix_core_benchmark_urls.sh
#
# This script updates dataset_registry.rs with working URLs for:
# - CoNLL-2003
# - OntoNotes 5.0
# - WikiANN
# - MultiCoNER
# - GENIA
# - BC5CDR

set -e

echo "Fixing core benchmark URLs..."
echo "================================"

# Backup original file
cp crates/anno-eval/src/eval/dataset_registry.rs crates/anno-eval/src/eval/dataset_registry.rs.backup

# Note: This script identifies URLs to update. Manual editing required.
# The URLs below are research-backed alternatives from HuggingFace and official sources.

cat << 'EOF'
# Core Benchmark URL Updates Needed:

## CoNLL-2003
Current: (check registry)
Recommended: https://huggingface.co/datasets/tner/conll2003
Alternative: https://huggingface.co/datasets/eriktks/conll2003

## OntoNotes 5.0
Current: https://catalog.ldc.upenn.edu/LDC2013T19 (LDC license required)
Recommended: https://huggingface.co/datasets/tner/ontonotes5 (sample)
Note: Full dataset requires LDC membership

## WikiANN
Current: (check registry)
Recommended: https://huggingface.co/datasets/SEACrowd/wikiann
Alternative: https://huggingface.co/datasets/unimelb-nlp/wikiann

## MultiCoNER
Current: (check registry)
Recommended: https://huggingface.co/datasets/Babelscape/multiconer
Alternative: https://github.com/Babelscape/multiconer

## GENIA
Current: (check registry)
Recommended: https://huggingface.co/datasets/tner/genia
Alternative: https://www.geniaproject.org/genia-corpus

## BC5CDR
Current: (check registry)
Recommended: https://huggingface.co/datasets/bc5cdr
Alternative: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/

EOF

echo ""
echo "URL research complete. Manual updates required in dataset_registry.rs"
echo "Backup created at: crates/anno-eval/src/eval/dataset_registry.rs.backup"
