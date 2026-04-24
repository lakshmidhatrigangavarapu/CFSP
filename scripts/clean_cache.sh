#!/bin/bash

# Script to delete model cache and pycache directories

echo "Starting cache cleanup..."

# Remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove .ipynb_checkpoints directories
echo "Removing .ipynb_checkpoints directories..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# Remove .cache directory (huggingface models, torch, etc.)
echo "Removing .cache directory..."
rm -rf ~/.cache 2>/dev/null || true

# Remove model-specific cache directories
echo "Removing model-specific cache directories..."
rm -rf ~/.huggingface 2>/dev/null || true
rm -rf ~/.torch 2>/dev/null || true
rm -rf ~/.transformers_cache 2>/dev/null || true

# Remove local .cache in project
echo "Removing local .cache directory..."
rm -rf .cache 2>/dev/null || true

# Remove Python egg-info directories (optional)
echo "Removing .egg-info directories..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

echo "Cache cleanup completed!"
