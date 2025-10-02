"""
Centralized experiment tracking with Weights & Biases integration.
"""

import time
from typing import Dict, Optional, Any, List
from pathlib import Path
import torch
import numpy as np

from .config import ExperimentConfig
from .utils import (
    Timer, 
    get_gpu_memory_stats, 
    get_system_memory_stats,
    calculate_tokens_per_second,
    format_large_number,
    estimate_remaining_time,
    create_experiment_summary,
    log_model_info
)
from mini_lm.config.logging_config import get_logger


class ExperimentTracker:
    """Centralized experiment tracking with W&B integration.
    
    This class handles all experiment tracking, including:
    - Wallclock time tracking
    - Gradient step counting
    - Loss curves with proper timing
    - Validation metrics
    - System metrics (GPU/CPU memory)
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[torch.nn.Module] = None,
        resume_from_step: int = 0
    ):
        """Initialize experiment tracker.
        
        Args:
            config: Experiment configuration
            model: Optional model to track
            resume_from_step: Step to resume from (for continued training)
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize timers
        self.global_timer = Timer()
        self.step_timer = Timer()
        self.val_timer = Timer()
        
        # Initialize counters
        self.gradient_steps = resume_from_step
        self.iterations = resume_from_step  # Raw iterations (before gradient accumulation)
        self.tokens_seen = 0
        self.val_runs = 0
        
        # Initialize metric storage
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []
        
        # Initialize W&B
        self._init_wandb(model)
        
        self.logger.info(
            "ExperimentTracker initialized",
            experiment_name=self.config.experiment_name,
            gradient_steps=self.gradient_steps,
            model_size=self.config.model_size
        )
    
    def _init_wandb(self, model: Optional[torch.nn.Module] = None):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "Weights & Biases is not installed. "
                "Install it with: pip install wandb"
            )
        
        # Get W&B configuration
        wandb_config = self.config.get_wandb_config()
        
        # Initialize W&B
        self.run = wandb.init(**wandb_config)
        
        # Log model info if provided
        if model is not None:
            model_info = log_model_info(model)
            wandb.config.update(model_info)
            
            # Watch model for gradient tracking
            wandb.watch(model, log="all", log_freq=100)
            
            self.logger.info(
                "Model registered with W&B",
                total_params=format_large_number(model_info["total_parameters"]),
                model_size_mb=f"{model_info['model_size_mb']:.1f}"
            )
    
    def log_train_step(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        tokens: int,
        iteration: int,
        accumulation_step: int
    ):
        """Log training step metrics.
        
        Args:
            loss: Training loss for this step
            learning_rate: Current learning rate
            grad_norm: L2 norm of gradients
            tokens: Number of tokens in this batch
            iteration: Current iteration (raw, not gradient steps)
            accumulation_step: Current accumulation step (0 to accumulation_steps-1)
        """
        # Update counters
        self.iterations = iteration
        self.tokens_seen += tokens
        
        # Only log after gradient accumulation is complete
        if accumulation_step == self.config.gradient_accumulation_steps - 1:
            self.gradient_steps += 1
            
            # Calculate timing metrics
            wallclock_hours = self.global_timer.elapsed_hours()
            step_time = self.step_timer.lap()
            tokens_per_second = calculate_tokens_per_second(
                tokens * self.config.gradient_accumulation_steps,
                step_time
            )
            
            # Calculate perplexity
            perplexity = np.exp(loss)
            
            # Store metrics for smoothing
            self.train_losses.append(loss)
            self.learning_rates.append(learning_rate)
            
            # Prepare metrics dict
            metrics = {
                # Loss metrics
                "train/loss": loss,
                "train/perplexity": perplexity,
                
                # Optimization metrics
                "train/learning_rate": learning_rate,
                "train/grad_norm": grad_norm,
                
                # Timing metrics
                "time/wallclock_hours": wallclock_hours,
                "time/gradient_steps": self.gradient_steps,
                "time/iterations": self.iterations,
                "time/tokens_seen": self.tokens_seen,
                "time/tokens_seen_formatted": format_large_number(self.tokens_seen),
                
                # Throughput metrics
                "throughput/tokens_per_second": tokens_per_second,
                "throughput/steps_per_hour": 3600 / step_time if step_time > 0 else 0,
                
                # Progress metrics
                "progress/completion_pct": (self.gradient_steps / self.config.num_iterations) * 100,
                "progress/estimated_hours_remaining": estimate_remaining_time(
                    self.gradient_steps,
                    self.config.num_iterations,
                    wallclock_hours * 3600
                ),
            }
            
            # Add memory metrics if requested
            if self.config.track_memory:
                metrics.update({
                    f"memory/{k}": v 
                    for k, v in get_gpu_memory_stats().items()
                })
                metrics.update({
                    f"memory/{k}": v 
                    for k, v in get_system_memory_stats().items()
                })
            
            # Log to W&B
            self.wandb.log(metrics, step=self.gradient_steps)
            
            # Log summary to console if at log interval
            if self.gradient_steps % self.config.log_interval == 0:
                self.logger.info(
                    "Training step",
                    step=self.gradient_steps,
                    loss=f"{loss:.4f}",
                    perplexity=f"{perplexity:.2f}",
                    lr=f"{learning_rate:.2e}",
                    grad_norm=f"{grad_norm:.2f}",
                    tokens_per_sec=f"{tokens_per_second:.0f}",
                    wallclock=f"{wallclock_hours:.2f}h",
                    progress=f"{metrics['progress/completion_pct']:.1f}%"
                )
    
    def log_validation(
        self,
        val_loss: float,
        val_perplexity: float,
        num_batches: int
    ):
        """Log validation metrics.
        
        Args:
            val_loss: Average validation loss
            val_perplexity: Validation perplexity
            num_batches: Number of validation batches evaluated
        """
        self.val_runs += 1
        val_time = self.val_timer.lap()
        wallclock_hours = self.global_timer.elapsed_hours()
        
        # Store validation loss
        self.val_losses.append(val_loss)
        
        # Prepare metrics
        metrics = {
            "val/loss": val_loss,
            "val/perplexity": val_perplexity,
            "val/eval_time_seconds": val_time,
            "val/num_batches": num_batches,
            "val/runs_completed": self.val_runs,
            
            # Also log timing context
            "time/wallclock_hours_at_val": wallclock_hours,
            "time/gradient_steps_at_val": self.gradient_steps,
        }
        
        # Calculate train/val gap
        if self.train_losses:
            recent_train_loss = np.mean(self.train_losses[-100:])
            metrics["val/train_val_gap"] = val_loss - recent_train_loss
        
        # Log to W&B
        self.wandb.log(metrics, step=self.gradient_steps)
        
        self.logger.info(
            "Validation completed",
            step=self.gradient_steps,
            val_loss=f"{val_loss:.4f}",
            val_perplexity=f"{val_perplexity:.2f}",
            eval_time=f"{val_time:.1f}s",
            wallclock=f"{wallclock_hours:.2f}h"
        )
    
    def log_checkpoint(
        self,
        checkpoint_path: Path,
        is_best: bool = False
    ):
        """Log checkpoint save event.
        
        Args:
            checkpoint_path: Path where checkpoint was saved
            is_best: Whether this is the best checkpoint so far
        """
        wallclock_hours = self.global_timer.elapsed_hours()
        
        metrics = {
            "checkpoint/saved": 1,
            "checkpoint/gradient_steps": self.gradient_steps,
            "checkpoint/wallclock_hours": wallclock_hours,
            "checkpoint/is_best": int(is_best),
        }
        
        # Log to W&B
        self.wandb.log(metrics, step=self.gradient_steps)
        
        # Also save checkpoint as W&B artifact
        artifact = self.wandb.Artifact(
            name=f"checkpoint-{self.config.experiment_name}",
            type="model",
            metadata={
                "gradient_steps": self.gradient_steps,
                "wallclock_hours": wallclock_hours,
                "is_best": is_best,
            }
        )
        artifact.add_file(str(checkpoint_path))
        self.run.log_artifact(artifact)
        
        self.logger.info(
            "Checkpoint saved",
            path=checkpoint_path,
            step=self.gradient_steps,
            is_best=is_best
        )
    
    def log_custom_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None
    ):
        """Log a custom metric.
        
        Args:
            name: Metric name (will be prefixed with "custom/")
            value: Metric value
            step: Optional step to log at (defaults to current gradient steps)
        """
        if step is None:
            step = self.gradient_steps
        
        self.wandb.log({f"custom/{name}": value}, step=step)
    
    def finish(self, save_summary: bool = True):
        """Finish experiment tracking and close W&B run.
        
        Args:
            save_summary: Whether to save experiment summary
        """
        total_time = self.global_timer.elapsed()
        
        # Create final summary
        final_metrics = {
            "train/loss": self.train_losses[-1] if self.train_losses else None,
            "val/loss": self.val_losses[-1] if self.val_losses else None,
            "train/perplexity": np.exp(self.train_losses[-1]) if self.train_losses else None,
            "val/perplexity": np.exp(self.val_losses[-1]) if self.val_losses else None,
            "train/tokens_per_second": self.tokens_seen / total_time if total_time > 0 else 0,
        }
        
        summary = create_experiment_summary(
            self.config,
            final_metrics,
            total_time
        )
        
        # Log summary to W&B
        for key, value in summary.items():
            if value is not None:
                self.wandb.summary[key] = value
        
        # Save summary to file if requested
        if save_summary:
            summary_path = Path(f"experiments/{self.config.experiment_name}_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Experiment summary saved to {summary_path}")
        
        # Finish W&B run
        self.wandb.finish()
        
        self.logger.info(
            "Experiment completed",
            name=self.config.experiment_name,
            total_hours=f"{total_time/3600:.2f}",
            gradient_steps=self.gradient_steps,
            tokens_seen=format_large_number(self.tokens_seen)
        )
    
    def state_dict(self) -> Dict[str, Any]:
        """Get tracker state for checkpointing."""
        return {
            "gradient_steps": self.gradient_steps,
            "iterations": self.iterations,
            "tokens_seen": self.tokens_seen,
            "val_runs": self.val_runs,
            "train_losses": self.train_losses[-1000:],  # Keep last 1000
            "val_losses": self.val_losses,
            "start_time": self.global_timer.start_time,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load tracker state from checkpoint."""
        self.gradient_steps = state_dict["gradient_steps"]
        self.iterations = state_dict["iterations"]
        self.tokens_seen = state_dict["tokens_seen"]
        self.val_runs = state_dict["val_runs"]
        self.train_losses = state_dict["train_losses"]
        self.val_losses = state_dict["val_losses"]
        
        # Adjust timer to account for previous training time
        elapsed = time.time() - state_dict["start_time"]
        self.global_timer.start_time = time.time() - elapsed
        
        self.logger.info(
            "Tracker state loaded",
            gradient_steps=self.gradient_steps,
            tokens_seen=format_large_number(self.tokens_seen)
        )