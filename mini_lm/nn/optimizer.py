from typing import Callable, Optional
import torch

class AdamW(torch.optim.Optimizer):
  def __init__(self, params=None, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    # Handle the case when params is None or empty
    if params is None:
      params = []
    
    super().__init__(params, defaults)
    
    # Initialize state for each parameter
    for group in self.param_groups:
      for p in group['params']:
        self.state[p] = {'t': 0, 'm': torch.zeros_like(p), 'v': torch.zeros_like(p)}

  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data

        state = self.state[p]
        state['t'] += 1
        t = state['t']

        m = state['m']
        v = state['v']

        # Avoid creating new tensors by using in-place operations
        beta1, beta2 = group['betas']
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        adjusted_lr = group['lr'] * (1 - beta2**t)**0.5 / (1 - beta1**t)

        p.data.addcdiv_(m, v.sqrt().add_(group['eps']), value=-adjusted_lr)
        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

    return loss