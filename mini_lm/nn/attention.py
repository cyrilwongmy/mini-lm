from torch.nn import Module
import torch
from torch import Tensor
from typing import Optional, Tuple

from jaxtyping import Float

from . import Softmax

class ScaledDotProductAttention(Module):
    """
    Scaled Dot-Product Attention mechanism.
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, "batch_size ... seq_len_q d_v"]:
        """
        Args:
            query (torch.Tensor): a tensor of shape (batch_size, ..., seq_len_q, d_k)
            key (torch.Tensor): a tensor of shape (batch_size, ..., seq_len_k, d_k)
            value (torch.Tensor): a tensor of shape (batch_size, ..., seq_len_k, d_v)
            mask (torch.Tensor, optional): a boolean tensor of shape (batch_size, ..., seq_len_q, seq_len_k).
                                           Positions with `False` will be masked out.
        """
        d_k = Q.size(-1)
        # scores shape: (batch_size, ..., seq_len_q, seq_len_k)
        scores = torch.einsum("...qd,...kd->...qk", Q, K) / (d_k**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == False, float("-inf"))

        attn = Softmax(dim=-1)(scores)

        output = torch.einsum("...qk,...kd->...qd", attn, V)
        return output