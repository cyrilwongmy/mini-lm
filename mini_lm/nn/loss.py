from torch.nn import Module
import torch
from torch import Tensor
from jaxtyping import Float, Int

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: Float[Tensor, "batch_size vocab_size"], labels: Int[Tensor, "batch_size"]):
        """Given a tensor of inputs and targets, compute the average cross-entropy
        loss across examples.

        Args:
            logits (Float[Tensor, "batch_size vocab_size"]): logits[i][j] is the
                unnormalized logit of jth class for the ith example.
            labels (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
                Each value must be between 0 and `num_classes - 1`.

        Returns:
            Float[Tensor, ""]: The average cross-entropy loss across examples.
        """

        mx = logits.amax(dim=-1, keepdim=True)
        z = (logits - mx).exp()
        sum_exp = z.sum(dim=-1, keepdim=True)
        log_sum_exp = sum_exp.clamp_min(torch.finfo(z.dtype).tiny).log() + mx

        log_softmax = logits - log_sum_exp
        return -log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).mean()
