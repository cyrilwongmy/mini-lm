import math


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
