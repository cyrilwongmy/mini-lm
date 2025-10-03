#!/usr/bin/env python3
"""
Enhanced training script for mini-LM transformer model with experiment tracking.

This script provides comprehensive experiment tracking with:
- Wallclock time tracking for all metrics
- Gradient step counting (not just iterations)
- Enhanced W&B integration for loss curves
- Systematic experiment configuration
"""

import argparse
import sys
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
from mini_lm.tracking import ExperimentTracker, ExperimentConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a mini-LM transformer model with experiment tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture arguments
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--model-size", type=str, default="17M",
                            help="Model size descriptor (e.g., 17M, 125M)")
    model_group.add_argument("--vocab-size", type=int, default=50257,
                            help="Vocabulary size")
    model_group.add_argument("--context-length", type=int, default=1024,
                            help="Maximum sequence length")
    model_group.add_argument("--d-model", type=int, default=512,
                            help="Model dimension")
    model_group.add_argument("--num-layers", type=int, default=8,
                            help="Number of transformer layers")
    model_group.add_argument("--num-heads", type=int, default=8,
                            help="Number of attention heads")
    model_group.add_argument("--d-ff", type=int, default=2048,
                            help="Feed-forward dimension")
    model_group.add_argument("--rope-theta", type=float, default=10000.0,
                            help="RoPE theta parameter")
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--train-data", type=str, required=True,
                            help="Path to training data (numpy array)")
    train_group.add_argument("--val-data", type=str, required=True,
                            help="Path to validation data (numpy array)")
    train_group.add_argument("--dataset-name", type=str, default="tinystories",
                            help="Name of the dataset for tracking")
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
    
    # Experiment tracking arguments
    exp_group = parser.add_argument_group("Experiment Tracking")
    exp_group.add_argument("--experiment-name", type=str, default=None,
                          help="Name for this experiment")
    exp_group.add_argument("--experiment-type", type=str, default="baseline",
                          choices=["baseline", "ablation", "hyperparameter_sweep"],
                          help="Type of experiment")
    exp_group.add_argument("--experiment-tags", type=str, nargs="+", default=[],
                          help="Additional tags for the experiment")
    exp_group.add_argument("--experiment-notes", type=str, default="",
                          help="Notes about this experiment")
    exp_group.add_argument("--track-memory", action="store_true",
                          help="Track GPU and system memory usage")
    exp_group.add_argument("--wandb-project", type=str, default="mini-lm",
                          help="W&B project name")
    exp_group.add_argument("--wandb-entity", type=str, default=None,
                          help="W&B entity (username or team)")
    exp_group.add_argument("--no-wandb", action="store_true",
                          help="Disable W&B tracking (for debugging)")
    
    # Logging arguments
    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument("--log-interval", type=int, default=10,
                              help="Log training metrics every N gradient steps")
    logging_group.add_argument("--val-interval", type=int, default=100,
                              help="Evaluate on validation set every N gradient steps")
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
    
    # Device arguments
    device_group = parser.add_argument_group("Device")
    device_group.add_argument("--device", type=str, default="cuda",
                             help="Device to use (cuda/cpu/mps)")
    device_group.add_argument("--compile", action="store_true",
                             help="Use torch.compile for optimization")
    
    # Data loading arguments
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument("--use-mmap", action="store_true",
                           help="Use memory-mapped arrays for data loading")
    data_group.add_argument("--dtype", type=str, default="int64",
                           choices=["int16", "int32", "int64"],
                           help="Data type for token arrays")
    
    # Config file support
    parser.add_argument("--config", type=str, default=None,
                       help="Path to experiment config file (YAML or JSON)")
    
    return parser.parse_args()


def create_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Create ExperimentConfig from command-line arguments."""
    # If config file is provided, load it first
    if args.config:
        config = ExperimentConfig.load(Path(args.config))

        # Only override values that were explicitly provided on the CLI.
        # Build a set of provided flag names mapped to argparse dest format.
        provided_flags = set()
        for tok in sys.argv[1:]:
            if tok.startswith("--"):
                name = tok[2:].split("=")[0]
                provided_flags.add(name.replace("-", "_"))

        # Map CLI dest names to ExperimentConfig attribute names when they differ
        alias_map = {
            "dataset_name": "dataset",
            "experiment_tags": "tags",
            "experiment_notes": "notes",
        }

        for key, value in vars(args).items():
            if key not in provided_flags:
                continue
            target = alias_map.get(key, key)
            if hasattr(config, target) and value is not None:
                setattr(config, target, value)
    else:
        # Create config from arguments
        config = ExperimentConfig(
            # Model configuration
            model_size=args.model_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            
            # Training configuration
            dataset=args.dataset_name,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            num_iterations=args.num_iterations,
            mixed_precision=args.mixed_precision,
            
            # Experiment metadata
            experiment_type=args.experiment_type,
            experiment_name=args.experiment_name,
            tags=args.experiment_tags,
            notes=args.experiment_notes,
            
            # Tracking configuration
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            val_iterations=args.val_iterations,
            checkpoint_interval=args.checkpoint_interval,
            track_memory=args.track_memory,
            
            # W&B configuration
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
    
    return config


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
    """Main training loop with experiment tracking."""
    # Configure logging
    configure_logging(
        level=args.log_level,
        format=args.log_format,
        log_file=args.log_file
    )
    logger = get_logger(__name__)
    
    # Create experiment configuration
    config = create_experiment_config(args)
    logger.info(f"Experiment configuration:\n{config}")
    
    # Save config for reproducibility
    config_dir = Path("experiments/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config.experiment_name}.yaml"
    config.save(config_path)
    logger.info(f"Config saved to {config_path}")
    
    # Set device
    device = torch.device(args.device)
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info(f"Using {args.device}")
    
    if torch.amp.autocast_mode.is_autocast_available(args.device):
        logger.info(f"{args.device} autocast is available")
    else:
        logger.info(f"{args.device} autocast is not available")
    
    # Create model
    logger.info("Creating model...")
    model = Transformer(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta
    ).to(device)

    
    # Compile model if mps or requested
    if args.device == "mps":
        logger.info("Compiling model with torch.compile for MPS...")
        model = torch.compile(model, backend="aot_eager")
    elif args.compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # Load checkpoint if resuming
    start_iteration = 0
    if args.resume_from:
        logger.info(f"Loading checkpoint from {args.resume_from}")
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"Resumed from iteration {start_iteration}")
    
    # Initialize experiment tracker
    if not args.no_wandb:
        tracker = ExperimentTracker(
            config=config,
            model=model,
            resume_from_step=start_iteration // config.gradient_accumulation_steps
        )
    else:
        tracker = None
        logger.warning("W&B tracking disabled")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_data = args.train_data
    val_data = args.val_data
    
    # Verify data can be loaded
    if args.use_mmap:
        # Test loading with memmap
        test_batch = get_batch(
            train_data, 1, config.context_length, device,
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
    train_start_time = time.time()
    
    logger.info("Starting training...")
    logger.info(f"Training for {config.num_iterations} iterations")
    
    # Training loop
    model.train()
    for iteration in range(start_iteration, config.num_iterations):
        if iteration % 100 == 0:
            logger.info(f"start of iteration: {iteration}")
        # Get learning rate for this iteration
        lr = get_cosine_schedule_with_warmup(
            iteration,
            config.learning_rate,
            config.min_learning_rate,
            config.warmup_steps,
            config.num_iterations
        )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Accumulate gradients
        accumulated_loss = 0.0
        accumulated_tokens = 0
        
        for micro_step in range(config.gradient_accumulation_steps):
            # Get batch
            x, y = get_batch(
                train_data,
                config.batch_size,
                config.context_length,
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
                    loss = loss / config.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
            else:
                logits = model(x, return_logits=True)
                # Reshape for loss calculation
                logits = logits.view(-1, logits.size(-1))
                targets = y.view(-1)
                loss = criterion(logits, targets)
                loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            accumulated_loss += loss.item()
            accumulated_tokens += x.numel()
        
        # Gradient clipping and optimizer step
        if args.mixed_precision:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Track training loss
        train_loss = accumulated_loss * config.gradient_accumulation_steps
        
        # Log to tracker
        if tracker:
            tracker.log_train_step(
                loss=train_loss,
                learning_rate=lr,
                grad_norm=grad_norm,
                tokens=accumulated_tokens,
                iteration=iteration + 1,
                accumulation_step=config.gradient_accumulation_steps - 1
            )
        
        # Validation
        gradient_steps = (iteration + 1) // config.gradient_accumulation_steps
        if gradient_steps % config.val_interval == 0 and gradient_steps > 0:
            logger.info("Running validation...")
            val_start_time = time.time()
            
            val_metrics = validate(
                model, val_data, args, device, config.val_iterations
            )
            
            val_time = time.time() - val_start_time
            
            logger.info(
                f"Validation | "
                f"Loss: {val_metrics['val_loss']:.4f} | "
                f"Perplexity: {val_metrics['val_perplexity']:.2f} | "
                f"Time: {val_time:.1f}s"
            )
            
            # Log to tracker
            if tracker:
                tracker.log_validation(
                    val_loss=val_metrics['val_loss'],
                    val_perplexity=val_metrics['val_perplexity'],
                    num_batches=config.val_iterations
                )
        
        # Checkpointing
        if (iteration + 1) % config.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration + 1}.pt"
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            
            # Log checkpoint to tracker
            if tracker:
                tracker.log_checkpoint(checkpoint_path)
            
            # Cleanup old checkpoints
            cleanup_old_checkpoints(checkpoint_dir, args.keep_last_n)
    
    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    logger.info(f"Saving final checkpoint to {final_checkpoint_path}")
    save_checkpoint(model, optimizer, config.num_iterations, final_checkpoint_path)
    
    if tracker:
        tracker.log_checkpoint(final_checkpoint_path, is_best=True)
    
    # Training complete
    total_time = time.time() - train_start_time
    logger.info(f"Training complete! Total time: {total_time/3600:.2f} hours")
    
    # Finish tracking
    if tracker:
        tracker.finish(save_summary=True)


if __name__ == "__main__":
    args = parse_args()
    train(args)