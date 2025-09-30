import math
import torch
from typing import Iterable


def get_cosine_schedule_with_warmup(
    current_step: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.

    Args:
        current_step: Current training step (0-indexed)
        max_learning_rate: Maximum learning rate (α_max)
        min_learning_rate: Minimum learning rate (α_min)
        warmup_steps: Number of warmup steps (T_w)
        total_steps: Total number of cosine annealing steps (T_c)

    Returns:
        Learning rate for the current step
    """
    if current_step < warmup_steps:
        # Linear warmup phase
        return current_step * max_learning_rate / warmup_steps
    elif current_step <= total_steps:
        # Cosine annealing phase
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(progress * math.pi)
        )
    else:
        # Post-annealing phase
        return min_learning_rate


def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6
) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    
    Args:
        parameters: An iterable of torch.nn.Parameter
        max_norm: Maximum L2 norm of the gradients
        eps: Small value for numerical stability (default: 1e-6)
    
    Returns:
        Total norm of the parameters (viewed as a single vector)
    """
    # Filter out parameters without gradients
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    # Compute the L2 norm of all gradients
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(2).item()
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    
    # Clip gradients if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm
