#!/bin/bash
# Script to train the TinyStories model with specified hyperparameters

# Token calculation verification:
# batch_size (64) * gradient_accumulation_steps (4) * context_length (256) * num_iterations (20000)
# = 64 * 4 * 256 * 20000 = 1,310,720,000 tokens
# This is too many! Let's adjust to get exactly 327,680,000 tokens

# Correct calculation:
# 327,680,000 / (64 * 4 * 256) = 5,000 iterations

echo "Starting TinyStories model training..."
echo "Target: 327,680,000 tokens"
echo "Model: 512d, 4 layers, 16 heads"
echo "Dataset paths:"
echo "  Train: /Users/cyrilwong/github/mini_lm/data/TinyStories_train_token_ids.npy"
echo "  Valid: /Users/cyrilwong/github/mini_lm/data/TinyStories_valid_token_ids.npy"

# Run training with corrected iterations
uv run python train_enhanced.py \
    --config experiments/configs/tinystories_512d_4L_16H.yaml \
    --train-data /Users/cyrilwong/github/mini_lm/data/TinyStories_train_token_ids.npy \
    --val-data /Users/cyrilwong/github/mini_lm/data/TinyStories_valid_token_ids.npy \
    --num-iterations 5000 \
    --checkpoint-dir checkpoints/tinystories_512d_4L_16H \
    --use-mmap \
    --dtype int64