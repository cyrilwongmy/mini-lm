# Experiment Tracking Module

This module provides comprehensive experiment tracking infrastructure for mini-LM training, with a focus on tracking loss curves with respect to gradient steps and wallclock time.

## Components

### `experiment_tracker.py`
The main tracking class that integrates with Weights & Biases to log:
- Training metrics (loss, perplexity, gradient norms)
- Timing information (wallclock hours, gradient steps, tokens processed)
- Validation metrics with separate timing
- System metrics (GPU/CPU memory usage)
- Checkpoint events

### `config.py`
Configuration management with:
- Automatic experiment naming based on model size, dataset, and timestamp
- Tag generation for easy filtering
- Configuration serialization (YAML/JSON)
- Parameter estimation for model size verification

### `utils.py`
Utility functions including:
- Timer class for tracking elapsed time
- GPU and system memory monitoring
- Token throughput calculation
- Experiment summary generation
- Metric smoothing functions

## Usage

### Basic Usage

```python
from mini_lm.tracking import ExperimentTracker, ExperimentConfig

# Create configuration
config = ExperimentConfig(
    model_size="17M",
    dataset="tinystories",
    learning_rate=3e-4,
    num_iterations=50000
)

# Initialize tracker
tracker = ExperimentTracker(config, model)

# During training loop
tracker.log_train_step(
    loss=loss.item(),
    learning_rate=current_lr,
    grad_norm=grad_norm,
    tokens=batch_size * seq_length,
    iteration=step,
    accumulation_step=accum_step
)

# During validation
tracker.log_validation(
    val_loss=val_loss,
    val_perplexity=val_perplexity,
    num_batches=val_batches
)

# When saving checkpoints
tracker.log_checkpoint(checkpoint_path, is_best=False)

# At the end of training
tracker.finish(save_summary=True)
```

### Configuration Files

Create YAML configuration files for reproducible experiments:

```yaml
# config.yaml
model_size: "17M"
vocab_size: 50257
d_model: 512
num_layers: 8
num_heads: 8
learning_rate: 3e-4
dataset: "tinystories"
experiment_type: "baseline"
tags: ["my_experiment", "v1"]
```

Load and use:

```python
config = ExperimentConfig.load("config.yaml")
```

## Metrics Tracked

### Training Metrics
- `train/loss`: Current training loss
- `train/perplexity`: exp(loss)
- `train/learning_rate`: Current LR
- `train/grad_norm`: L2 norm of gradients

### Timing Metrics
- `time/wallclock_hours`: Hours since training start
- `time/gradient_steps`: Number of optimizer steps
- `time/tokens_seen`: Total tokens processed
- `progress/completion_pct`: Training progress

### Throughput Metrics
- `throughput/tokens_per_second`: Training speed
- `throughput/steps_per_hour`: Gradient steps per hour

### Validation Metrics
- `val/loss`: Validation loss
- `val/perplexity`: Validation perplexity
- `val/eval_time_seconds`: Validation duration
- `val/train_val_gap`: Generalization gap

### System Metrics (optional)
- `memory/gpu_*_allocated_gb`: GPU memory usage
- `memory/system_memory_percent`: System RAM usage

## W&B Dashboard Features

When using this tracking module, your W&B dashboard will show:

1. **Loss Curves**: Plotted against both gradient steps and wallclock time
2. **Learning Rate Schedule**: Visualize warmup and cosine annealing
3. **Throughput Metrics**: Monitor training efficiency
4. **Memory Usage**: Track GPU and system memory (if enabled)
5. **Validation Curves**: Compare train/val loss over time
6. **Experiment Comparison**: Easy comparison across runs

## Best Practices

1. **Consistent Naming**: Use the automatic naming or follow the pattern: `{model_size}_{dataset}_{experiment_type}_{timestamp}`

2. **Tagging**: Use tags to organize experiments:
   - Model size: `17M`, `125M`
   - Dataset: `tinystories`, `openwebtext`
   - Experiment type: `baseline`, `ablation`

3. **Configuration Management**: Save all configs in version control

4. **Memory Tracking**: Enable for large models to optimize batch sizes

5. **Checkpointing**: The tracker logs checkpoint saves for recovery tracking

## Troubleshooting

### W&B Connection Issues
```python
# Run offline
import os
os.environ["WANDB_MODE"] = "offline"

# Sync later
# wandb sync wandb/offline-run-*
```

### Memory Tracking Errors
If memory tracking fails, disable it:
```python
config = ExperimentConfig(track_memory=False)
```

### Large History
For very long training runs, the tracker automatically limits history storage to prevent memory issues.

## Advanced Features

### Custom Metrics
```python
tracker.log_custom_metric("my_metric", value=0.95)
```

### State Persistence
The tracker can save/load its state for resuming:
```python
# Save
state = tracker.state_dict()

# Load
tracker.load_state_dict(state)
```

### Experiment Summaries
Summaries are automatically saved to `experiments/{experiment_name}_summary.json` containing final metrics and configuration.