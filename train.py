#!/usr/bin/env python3
"""
Training script for mini-LM transformer model.

This script provides a flexible training loop with support for:
- Configurable hyperparameters via command-line arguments
- Memory-efficient data loading with np.memmap
- Checkpoint saving and loading
- Training and validation logging
- Optional Weights & Biases integration
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from mini_lm.nn import (
    Transformer,
    AdamW,
    CrossEntropyLoss,
    get_batch,
    save_checkpoint,
    load_checkpoint,
    get_cosine_schedule_with_warmup,
    clip_grad_norm_
)
from mini_lm.config.logging_config import get_logger, configure_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a mini-LM transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture arguments
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--vocab-size", type=int, default=50257,
                            help="Vocabulary size")
    model_group.add_argument("--context-length", type=int, default=1024,
                            help="Maximum sequence length")
    model_group.add_argument("--d-model", type=int, default=768,
                            help="Model dimension")
    model_group.add_argument("--num-layers", type=int, default=12,
                            help="Number of transformer layers")
    model_group.add_argument("--num-heads", type=int, default=12,
                            help="Number of attention heads")
    model_group.add_argument("--d-ff", type=int, default=3072,
                            help="Feed-forward dimension")
    model_group.add_argument("--rope-theta", type=float, default=10000.0,
                            help="RoPE theta parameter")
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--train-data", type=str, required=True,
                            help="Path to training data (numpy array)")
    train_group.add_argument("--val-data", type=str, required=True,
                            help="Path to validation data (numpy array)")
    train_group.add_argument("--batch-size", type=int, default=32,
                            help="Training batch size")
    train_group.add_argument("--val-batch-size", type=int, default=32,
                            help="Validation batch size")
    train_group.add_argument("--num-iterations", type=int, default=100000,
                            help="Number of training iterations")
    train_group.add_argument("--gradient-accumulation-steps", type=int, default=1,
                            help="Number of gradient accumulation steps")
    train_group.add_argument("--mixed-precision", action="store_true",
                            help="Use mixed precision training")
    
    # Optimizer arguments
    optim_group = parser.add_argument_group("Optimizer")
    optim_group.add_argument("--learning-rate", type=float, default=3e-4,
                            help="Maximum learning rate")
    optim_group.add_argument("--min-learning-rate", type=float, default=3e-5,
                            help="Minimum learning rate")
    optim_group.add_argument("--warmup-steps", type=int, default=2000,
                            help="Number of warmup steps")
    optim_group.add_argument("--weight-decay", type=float, default=0.01,
                            help="Weight decay")
    optim_group.add_argument("--beta1", type=float, default=0.9,
                            help="Adam beta1")
    optim_group.add_argument("--beta2", type=float, default=0.999,
                            help="Adam beta2")
    optim_group.add_argument("--eps", type=float, default=1e-8,
                            help="Adam epsilon")
    optim_group.add_argument("--grad-clip", type=float, default=1.0,
                            help="Gradient clipping threshold")
    
    # Checkpointing arguments
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                                 help="Directory to save checkpoints")
    checkpoint_group.add_argument("--checkpoint-interval", type=int, default=1000,
                                 help="Save checkpoint every N iterations")
    checkpoint_group.add_argument("--resume-from", type=str, default=None,
                                 help="Path to checkpoint to resume from")
    checkpoint_group.add_argument("--keep-last-n", type=int, default=5,
                                 help="Keep only the last N checkpoints")
    
    # Logging arguments
    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument("--log-interval", type=int, default=10,
                              help="Log training metrics every N iterations")
    logging_group.add_argument("--val-interval", type=int, default=100,
                              help="Evaluate on validation set every N iterations")
    logging_group.add_argument("--val-iterations", type=int, default=50,
                              help="Number of validation iterations")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                              choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                              help="Logging level")
    logging_group.add_argument("--log-format", type=str, default="console",
                              choices=["console", "json"],
                              help="Logging format")
    logging_group.add_argument("--log-file", type=str, default=None,
                              help="Log file path")
    
    # Weights & Biases arguments
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--use-wandb", action="store_true",
                            help="Use Weights & Biases for logging")
    wandb_group.add_argument("--wandb-project", type=str, default="mini-lm",
                            help="W&B project name")
    wandb_group.add_argument("--wandb-entity", type=str, default=None,
                            help="W&B entity (username or team)")
    wandb_group.add_argument("--wandb-name", type=str, default=None,
                            help="W&B run name")
    wandb_group.add_argument("--wandb-tags", type=str, nargs="+", default=[],
                            help="W&B tags")
    
    # Device arguments
    device_group = parser.add_argument_group("Device")
    device_group.add_argument("--device", type=str, default="cuda",
                             help="Device to use (cuda/cpu)")
    device_group.add_argument("--compile", action="store_true",
                             help="Use torch.compile for optimization")
    
    # Data loading arguments
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument("--use-mmap", action="store_true",
                           help="Use memory-mapped arrays for data loading")
    data_group.add_argument("--dtype", type=str, default="int64",
                           choices=["int16", "int32", "int64"],
                           help="Data type for token arrays")
    
    return parser.parse_args()


def setup_wandb(args: argparse.Namespace, model: nn.Module) -> Optional[Any]:
    """Initialize Weights & Biases if requested."""
    if not args.use_wandb:
        return None
    
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "Weights & Biases is not installed. "
            "Install it with: pip install wandb"
        )
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Initialize W&B
    config = {
        # Model architecture
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        # Training
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_iterations": args.num_iterations,
        # Optimizer
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "grad_clip": args.grad_clip,
        # Model stats
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        tags=args.wandb_tags,
        config=config
    )
    
    # Watch the model
    wandb.watch(model, log="all", log_freq=100)
    
    return wandb


def get_dtype(dtype_str: str):
    """Convert string dtype to numpy dtype."""
    dtype_map = {
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64
    }
    return dtype_map[dtype_str]


def validate(
    model: nn.Module,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    num_iterations: int
) -> Dict[str, float]:
    """Run validation and return metrics."""
    model.eval()
    criterion = CrossEntropyLoss()
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(num_iterations):
            x, y = get_batch(
                val_data,
                args.val_batch_size,
                args.context_length,
                device,
                mmap_mode='r' if args.use_mmap else None,
                dtype=get_dtype(args.dtype)
            )
            
            if args.mixed_precision:
                with autocast():
                    logits = model(x, return_logits=True)
                    # Reshape for loss calculation
                    logits = logits.view(-1, logits.size(-1))
                    targets = y.view(-1)
                    loss = criterion(logits, targets)
            else:
                logits = model(x, return_logits=True)
                # Reshape for loss calculation
                logits = logits.view(-1, logits.size(-1))
                targets = y.view(-1)
                loss = criterion(logits, targets)
            
            batch_tokens = x.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    model.train()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity
    }


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    """Remove old checkpoints, keeping only the last N."""
    # Filter out checkpoint_final.pt and only get numbered checkpoints
    numbered_checkpoints = [
        p for p in checkpoint_dir.glob("checkpoint_*.pt")
        if p.stem != "checkpoint_final" and p.stem.split("_")[1].isdigit()
    ]
    
    checkpoints = sorted(
        numbered_checkpoints,
        key=lambda p: int(p.stem.split("_")[1])
    )
    
    if len(checkpoints) > keep_last_n:
        for checkpoint in checkpoints[:-keep_last_n]:
            checkpoint.unlink()


def train(args: argparse.Namespace):
    """Main training loop."""
    # Configure logging
    configure_logging(
        level=args.log_level,
        format=args.log_format,
        log_file=args.log_file
    )
    logger = get_logger(__name__)
    
    # Set device
    device = torch.device(args.device)
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info("Using CPU")
    
    # Create model
    logger.info("Creating model...")
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    
    # Compile model if requested
    if args.compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if resuming
    start_iteration = 0
    if args.resume_from:
        logger.info(f"Loading checkpoint from {args.resume_from}")
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"Resumed from iteration {start_iteration}")
    
    # Setup W&B if requested
    wandb = setup_wandb(args, model)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_data = args.train_data
    val_data = args.val_data
    
    # Verify data can be loaded
    if args.use_mmap:
        # Test loading with memmap
        test_batch = get_batch(
            train_data, 1, args.context_length, device,
            mmap_mode='r', dtype=get_dtype(args.dtype)
        )
        logger.info(f"Successfully loaded training data with memmap")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup mixed precision
    scaler = GradScaler() if args.mixed_precision else None
    
    # Loss function
    criterion = CrossEntropyLoss()
    
    # Training metrics
    train_losses = []
    train_start_time = time.time()
    tokens_processed = 0
    
    logger.info("Starting training...")
    logger.info(f"Training for {args.num_iterations} iterations")
    
    # Training loop
    model.train()
    for iteration in range(start_iteration, args.num_iterations):
        # Get learning rate for this iteration
        lr = get_cosine_schedule_with_warmup(
            iteration,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_steps,
            args.num_iterations
        )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Accumulate gradients
        accumulated_loss = 0.0
        for micro_step in range(args.gradient_accumulation_steps):
            # Get batch
            x, y = get_batch(
                train_data,
                args.batch_size,
                args.context_length,
                device,
                mmap_mode='r' if args.use_mmap else None,
                dtype=get_dtype(args.dtype)
            )
            
            # Forward pass
            if args.mixed_precision:
                with autocast():
                    logits = model(x, return_logits=True)
                    # Reshape for loss calculation
                    logits = logits.view(-1, logits.size(-1))
                    targets = y.view(-1)
                    loss = criterion(logits, targets)
                    loss = loss / args.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
            else:
                logits = model(x, return_logits=True)
                # Reshape for loss calculation
                logits = logits.view(-1, logits.size(-1))
                targets = y.view(-1)
                loss = criterion(logits, targets)
                loss = loss / args.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            accumulated_loss += loss.item()
            tokens_processed += x.numel()
        
        # Gradient clipping and optimizer step
        if args.mixed_precision:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Track training loss
        train_loss = accumulated_loss * args.gradient_accumulation_steps
        train_losses.append(train_loss)
        
        # Logging
        if iteration % args.log_interval == 0:
            # Calculate metrics
            elapsed_time = time.time() - train_start_time
            tokens_per_sec = tokens_processed / elapsed_time
            perplexity = np.exp(train_loss)
            
            # Log to console
            logger.info(
                f"Iteration {iteration}/{args.num_iterations} | "
                f"Loss: {train_loss:.4f} | "
                f"Perplexity: {perplexity:.2f} | "
                f"LR: {lr:.2e} | "
                f"Grad Norm: {grad_norm:.2f} | "
                f"Tokens/sec: {tokens_per_sec:.0f}"
            )
            
            # Log to W&B
            if wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "train/perplexity": perplexity,
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm,
                    "train/tokens_per_sec": tokens_per_sec,
                    "iteration": iteration
                })
        
        # Validation
        if iteration % args.val_interval == 0 and iteration > 0:
            logger.info("Running validation...")
            val_metrics = validate(
                model, val_data, args, device, args.val_iterations
            )
            
            logger.info(
                f"Validation | "
                f"Loss: {val_metrics['val_loss']:.4f} | "
                f"Perplexity: {val_metrics['val_perplexity']:.2f}"
            )
            
            # Log to W&B
            if wandb:
                wandb.log({
                    "val/loss": val_metrics["val_loss"],
                    "val/perplexity": val_metrics["val_perplexity"],
                    "iteration": iteration
                })
        
        # Checkpointing
        if iteration % args.checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.pt"
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            
            # Cleanup old checkpoints
            cleanup_old_checkpoints(checkpoint_dir, args.keep_last_n)
    
    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    logger.info(f"Saving final checkpoint to {final_checkpoint_path}")
    save_checkpoint(model, optimizer, args.num_iterations, final_checkpoint_path)
    
    # Training complete
    total_time = time.time() - train_start_time
    logger.info(f"Training complete! Total time: {total_time/3600:.2f} hours")
    
    # Close W&B
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)