#!/bin/bash
# Script to train the TinyStories model with enhanced options

# --- Configuration Variables ---
CONFIG_FILE="experiments/configs/tinystories_512d_4L_16H.yaml"

# --- Run Training ---
echo "Starting TinyStories model training..."

uv run python train.py --config experiments/configs/tinystories_512d_4L_16H.yaml

echo "Training script finished."
