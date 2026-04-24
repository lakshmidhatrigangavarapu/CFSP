#!/bin/bash
# ============================================================================
# Copy Training Data to DGX Package
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PACKAGE_DIR/training_data"

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/source/training_data"
    echo ""
    echo "Expected source directory structure:"
    echo "  training_data/"
    echo "    ├── train.jsonl"
    echo "    ├── val.jsonl"
    echo "    ├── test.jsonl"
    echo "    └── extraction_schema.json"
    exit 1
fi

SOURCE_DIR="$1"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Check required files
REQUIRED_FILES=("train.jsonl" "val.jsonl" "test.jsonl" "extraction_schema.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SOURCE_DIR/$file" ]; then
        echo "ERROR: Required file not found: $SOURCE_DIR/$file"
        exit 1
    fi
done

echo "Copying training data..."
echo "  Source: $SOURCE_DIR"
echo "  Target: $TARGET_DIR"
echo ""

mkdir -p "$TARGET_DIR"

# Copy with progress
for file in "${REQUIRED_FILES[@]}"; do
    echo "Copying $file..."
    rsync -ah --progress "$SOURCE_DIR/$file" "$TARGET_DIR/"
done

# Copy optional files if they exist
for file in "label_stats.json" "split_stats.json"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        echo "Copying $file..."
        rsync -ah --progress "$SOURCE_DIR/$file" "$TARGET_DIR/"
    fi
done

echo ""
echo "Data copy complete!"
echo ""
ls -lh "$TARGET_DIR/"
