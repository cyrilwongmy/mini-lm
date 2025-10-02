"""
Utility functions for experiment tracking.
"""

import time
import psutil
import torch
from typing import Dict, Optional, Any
import numpy as np


class Timer:
    """Simple timer for tracking elapsed time."""
    
    def __init__(self):
        self.start_time = time.time()
        self.lap_time = self.start_time
    
    def elapsed(self) -> float:
        """Get total elapsed time in seconds."""
        return time.time() - self.start_time
    
    def elapsed_hours(self) -> float:
        """Get total elapsed time in hours."""
        return self.elapsed() / 3600
    
    def lap(self) -> float:
        """Get time since last lap and reset lap timer."""
        current_time = time.time()
        lap_duration = current_time - self.lap_time
        self.lap_time = current_time
        return lap_duration
    
    def reset(self):
        """Reset the timer."""
        self.start_time = time.time()
        self.lap_time = self.start_time


def get_gpu_memory_stats() -> Dict[str, float]:
    """Get GPU memory statistics in GB."""
    if not torch.cuda.is_available():
        return {}
    
    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        stats.update({
            f"gpu_{i}_allocated_gb": allocated,
            f"gpu_{i}_reserved_gb": reserved,
            f"gpu_{i}_total_gb": total,
            f"gpu_{i}_free_gb": total - reserved,
            f"gpu_{i}_utilization_pct": (allocated / total) * 100,
        })
    
    return stats


def get_system_memory_stats() -> Dict[str, float]:
    """Get system memory statistics in GB."""
    memory = psutil.virtual_memory()
    
    return {
        "system_memory_used_gb": memory.used / 1e9,
        "system_memory_total_gb": memory.total / 1e9,
        "system_memory_percent": memory.percent,
    }


def calculate_tokens_per_second(
    tokens_processed: int,
    elapsed_time: float
) -> float:
    """Calculate training throughput in tokens per second."""
    if elapsed_time <= 0:
        return 0.0
    return tokens_processed / elapsed_time


def calculate_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1
) -> int:
    """Calculate effective batch size accounting for gradient accumulation and data parallelism."""
    return batch_size * gradient_accumulation_steps * world_size


def format_large_number(num: int) -> str:
    """Format large numbers with appropriate suffixes (K, M, B)."""
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num/1000:.1f}K"
    elif num < 1_000_000_000:
        return f"{num/1_000_000:.1f}M"
    else:
        return f"{num/1_000_000_000:.1f}B"


def estimate_remaining_time(
    current_step: int,
    total_steps: int,
    elapsed_time: float
) -> float:
    """Estimate remaining training time in hours."""
    if current_step <= 0:
        return 0.0
    
    steps_remaining = total_steps - current_step
    time_per_step = elapsed_time / current_step
    remaining_seconds = steps_remaining * time_per_step
    
    return remaining_seconds / 3600


def create_experiment_summary(
    config: Any,
    metrics: Dict[str, float],
    elapsed_time: float
) -> Dict[str, Any]:
    """Create a summary of the experiment for logging."""
    return {
        "experiment_name": config.experiment_name,
        "model_size": config.model_size,
        "dataset": config.dataset,
        "total_iterations": config.num_iterations,
        "elapsed_hours": elapsed_time / 3600,
        "final_train_loss": metrics.get("train/loss", None),
        "final_val_loss": metrics.get("val/loss", None),
        "final_train_perplexity": metrics.get("train/perplexity", None),
        "final_val_perplexity": metrics.get("val/perplexity", None),
        "avg_tokens_per_second": metrics.get("train/tokens_per_second", None),
    }


def smooth_metric(
    values: list,
    window_size: int = 100
) -> Optional[float]:
    """Apply exponential moving average smoothing to a metric."""
    if not values:
        return None
    
    if len(values) < window_size:
        return np.mean(values)
    
    # Use exponential moving average
    alpha = 2 / (window_size + 1)
    ema = values[0]
    
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
    
    return ema


def log_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Log model architecture information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get parameter breakdown by layer type
    param_breakdown = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                module_type = type(module).__name__
                if module_type not in param_breakdown:
                    param_breakdown[module_type] = 0
                param_breakdown[module_type] += module_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "parameter_breakdown": param_breakdown,
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
    }


def create_run_tags(config: Any) -> list:
    """Create tags for the W&B run based on configuration."""
    tags = []
    
    # Model size
    tags.append(config.model_size)
    
    # Dataset
    tags.append(config.dataset)
    
    # Experiment type
    tags.append(config.experiment_type)
    
    # Learning rate
    tags.append(f"lr_{config.learning_rate}")
    
    # Batch size
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    tags.append(f"bs_{effective_batch}")
    
    # Special configurations
    if config.mixed_precision:
        tags.append("mixed_precision")
    
    if config.gradient_accumulation_steps > 1:
        tags.append(f"grad_accum_{config.gradient_accumulation_steps}")
    
    # Add any custom tags
    if hasattr(config, 'tags') and config.tags:
        tags.extend(config.tags)
    
    return list(set(tags))  # Remove duplicates