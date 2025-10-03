#!/bin/bash
# Script to train the TinyStories model with enhanced options

# --- Configuration Variables ---
CONFIG_FILE="experiments/configs/tinystories_512d_4L_16H.yaml"
TRAIN_DATA="/Users/cyrilwong/github/mini_lm/data/TinyStories_train_token_ids.npy"
VAL_DATA="/Users/cyrilwong/github/mini_lm/data/TinyStories_valid_token_ids.npy"
NUM_ITERATIONS=10
CHECKPOINT_DIR="checkpoints/tinystories_512d_4L_16H"
DTYPE="int64"
DEVICE="cpu"

# --- Run Training ---
echo "Starting enhanced TinyStories model training..."

uv run python trainer.py \
    --config "${CONFIG_FILE}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --num-iterations "${NUM_ITERATIONS}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --use-mmap \
    --dtype "${DTYPE}" \
    --device "${DEVICE}"

echo "Training script finished."