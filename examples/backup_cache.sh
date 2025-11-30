#!/bin/bash
# Backup/mirror script for anno caches (models + datasets)
#
# This script creates a backup of all downloaded models and datasets
# for easy transfer to other machines or offline use.
#
# Usage:
#   ./examples/backup_cache.sh [backup-dir]
#
# Default backup location: ./anno-cache-backup-$(date +%Y%m%d)

set -euo pipefail

# Default backup directory
BACKUP_DIR="${1:-./anno-cache-backup-$(date +%Y%m%d)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Anno Cache Backup/Mirror ==="
echo ""

# Detect cache directories
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
ANNO_CACHE="${HOME}/.cache/anno/datasets"

# Fallback if dirs crate unavailable
if [ ! -d "$ANNO_CACHE" ]; then
    ANNO_CACHE="./.anno/datasets"
fi

echo "Source directories:"
echo "  Models:    $HF_CACHE"
echo "  Datasets:  $ANNO_CACHE"
echo "Backup to:   $BACKUP_DIR"
echo ""

# Check if sources exist
if [ ! -d "$HF_CACHE" ] && [ ! -d "$ANNO_CACHE" ]; then
    echo -e "${RED}Error: No cache directories found!${NC}"
    echo "Have you downloaded any models or datasets?"
    exit 1
fi

# Create backup directory structure
mkdir -p "$BACKUP_DIR/models"
mkdir -p "$BACKUP_DIR/datasets"

# Backup models (HuggingFace cache)
if [ -d "$HF_CACHE" ]; then
    echo -e "${GREEN}Backing up models...${NC}"
    rsync -av --progress "$HF_CACHE/" "$BACKUP_DIR/models/" || {
        echo -e "${YELLOW}Warning: rsync not available, using cp${NC}"
        cp -r "$HF_CACHE"/* "$BACKUP_DIR/models/" 2>/dev/null || true
    }
    MODEL_SIZE=$(du -sh "$BACKUP_DIR/models" 2>/dev/null | cut -f1 || echo "unknown")
    echo "  Models: $MODEL_SIZE"
else
    echo -e "${YELLOW}No models cache found (skipping)${NC}"
fi

# Backup datasets
if [ -d "$ANNO_CACHE" ]; then
    echo -e "${GREEN}Backing up datasets...${NC}"
    rsync -av --progress "$ANNO_CACHE/" "$BACKUP_DIR/datasets/" || {
        echo -e "${YELLOW}Warning: rsync not available, using cp${NC}"
        cp -r "$ANNO_CACHE"/* "$BACKUP_DIR/datasets/" 2>/dev/null || true
    }
    DATASET_SIZE=$(du -sh "$BACKUP_DIR/datasets" 2>/dev/null | cut -f1 || echo "unknown")
    echo "  Datasets: $DATASET_SIZE"
else
    echo -e "${YELLOW}No datasets cache found (skipping)${NC}"
fi

# Create restore script
cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Restore anno cache from backup
#
# Usage: ./restore.sh [target-dir]
#
# Default restore location: ~/.cache/

set -euo pipefail

TARGET="${1:-$HOME/.cache}"

echo "=== Restoring Anno Cache ==="
echo ""

# Restore models
if [ -d "models" ]; then
    echo "Restoring models to: $TARGET/huggingface/hub/"
    mkdir -p "$TARGET/huggingface/hub"
    rsync -av --progress "models/" "$TARGET/huggingface/hub/" || {
        cp -r models/* "$TARGET/huggingface/hub/" 2>/dev/null || true
    }
fi

# Restore datasets
if [ -d "datasets" ]; then
    echo "Restoring datasets to: $TARGET/anno/datasets/"
    mkdir -p "$TARGET/anno/datasets"
    rsync -av --progress "datasets/" "$TARGET/anno/datasets/" || {
        cp -r datasets/* "$TARGET/anno/datasets/" 2>/dev/null || true
    }
fi

echo ""
echo "Restore complete!"
EOF

chmod +x "$BACKUP_DIR/restore.sh"

# Create README
cat > "$BACKUP_DIR/README.md" << EOF
# Anno Cache Backup

This directory contains a backup of all anno models and datasets.

## Contents

- \`models/\` - HuggingFace model cache (ONNX, Candle models)
- \`datasets/\` - Anno evaluation datasets cache
- \`restore.sh\` - Script to restore this backup

## Restore

To restore this backup on another machine:

\`\`\`bash
cd $(basename "$BACKUP_DIR")
./restore.sh
\`\`\`

Or restore to a custom location:

\`\`\`bash
./restore.sh /path/to/cache
\`\`\`

## Manual Restore

If the restore script doesn't work, manually copy:

- \`models/\` → \`~/.cache/huggingface/hub/\`
- \`datasets/\` → \`~/.cache/anno/datasets/\`

## Backup Date

Created: $(date)
EOF

# Summary
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "unknown")
echo ""
echo -e "${GREEN}=== Backup Complete ===${NC}"
echo "Total size: $TOTAL_SIZE"
echo "Location: $BACKUP_DIR"
echo ""
echo "To restore on another machine:"
echo "  cd $BACKUP_DIR"
echo "  ./restore.sh"
echo ""

