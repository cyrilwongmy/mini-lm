from torch.nn import Module, Linear
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

class MultiHeadAttention(Module):
    def __init__(self, d_model:int, num_heads:int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        # Following Vaswani et al. (2017), d_k = d_v = d_model / num_heads
        self.k_dim = d_model // num_heads
        self.v_dim = d_model // num_heads
        self.w_q = Linear(d_model, d_model, bias=False)
        self.w_k = Linear(d_model, d_model, bias=False)
        self.w_v = Linear(d_model, d_model, bias=False)
        self.w_o = Linear(d_model, d_model, bias=False)
        self.softmax = Softmax(dim=-1)
    
    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, k_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, k_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.v_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, v_dim)

        K = K.transpose(-2, -1)  # (batch_size, num_heads, k_dim, seq_len)

        scores = torch.matmul(Q, K) / (self.k_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(query.device)
        scores = scores.masked_fill(mask == True, float("-inf"))
        attn = self.softmax(scores)  # (batch_size, num_heads, seq_len, seq_len)
        output = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len, v_dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # (batch_size, seq_len, d_model)
        return self.w_o(output)  # Apply output projection