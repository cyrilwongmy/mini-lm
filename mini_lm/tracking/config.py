"""
Experiment configuration for tracking and reproducibility.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import yaml
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and management.
    
    This class handles experiment configuration, naming, and metadata
    for systematic experiment tracking and reproducibility.
    """
    
    # Model configuration
    model_name: str = "mini-lm"
    model_size: str = "17M"  # e.g., "17M", "125M", "350M"
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0
    
    # Training configuration
    dataset: str = "tinystories"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_iterations: int = 100000
    mixed_precision: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Experiment metadata
    experiment_type: str = "baseline"  # e.g., "baseline", "ablation", "hyperparameter_sweep"
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Tracking configuration
    log_interval: int = 10
    val_interval: int = 100
    val_iterations: int = 50
    checkpoint_interval: int = 1000
    track_memory: bool = False
    
    # W&B configuration
    wandb_project: str = "mini-lm"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    
    def __post_init__(self):
        """Generate automatic experiment name if not provided."""
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.model_size}_{self.dataset}_{self.experiment_type}_{timestamp}"
        
        # Add automatic tags based on configuration
        auto_tags = [
            self.model_size,
            self.dataset,
            self.experiment_type,
            f"lr_{self.learning_rate}",
            f"bs_{self.batch_size * self.gradient_accumulation_steps}"
        ]
        
        # Merge with user-provided tags
        self.tags = list(set(self.tags + auto_tags))
        
        # Set W&B group and job type if not provided
        if self.wandb_group is None:
            self.wandb_group = f"{self.model_size}_{self.dataset}"
        
        if self.wandb_job_type is None:
            self.wandb_job_type = self.experiment_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)
    
    def save(self, path: Path):
        """Save configuration to file (JSON or YAML based on extension)."""
        path = Path(path)
        config_dict = self.to_dict()
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "num_iterations": self.num_iterations,
            "mixed_precision": self.mixed_precision,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Get W&B-specific configuration for wandb.init()."""
        # Combine all configs for W&B
        config = {
            **self.get_model_config(),
            **self.get_training_config(),
            "experiment_type": self.experiment_type,
            "model_size": self.model_size,
            "model_name": self.model_name,
        }
        
        return {
            "project": self.wandb_project,
            "entity": self.wandb_entity,
            "name": self.experiment_name,
            "tags": self.tags,
            "group": self.wandb_group,
            "job_type": self.wandb_job_type,
            "config": config,
            "notes": self.notes,
        }
    
    def estimate_model_params(self) -> int:
        """Estimate total model parameters based on configuration."""
        # Embedding parameters
        embedding_params = self.vocab_size * self.d_model
        
        # Transformer block parameters per layer
        # Multi-head attention: Q, K, V, O projections
        attention_params = 4 * self.d_model * self.d_model
        
        # Feed-forward network (SwiGLU has 3 matrices)
        ffn_params = 3 * self.d_model * self.d_ff
        
        # Layer norm parameters (2 per block)
        norm_params = 2 * self.d_model
        
        # Total per layer
        params_per_layer = attention_params + ffn_params + norm_params
        
        # Output layer
        output_params = self.d_model * self.vocab_size
        
        # Total
        total_params = embedding_params + (params_per_layer * self.num_layers) + output_params
        
        return total_params
    
    def __str__(self) -> str:
        """String representation of the config."""
        params_millions = self.estimate_model_params() / 1e6
        return (
            f"ExperimentConfig(\n"
            f"  name: {self.experiment_name}\n"
            f"  model: {self.model_size} (~{params_millions:.1f}M params)\n"
            f"  dataset: {self.dataset}\n"
            f"  type: {self.experiment_type}\n"
            f"  tags: {', '.join(self.tags)}\n"
            f")"
        )