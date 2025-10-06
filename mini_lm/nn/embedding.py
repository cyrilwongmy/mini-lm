from torch.nn import Module, Parameter
import torch
from torch import Tensor

from jaxtyping import Float, Int


class Embedding(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Embedding layer mapping indices to dense vectors.

        Args:
            vocab_size (int): Size of the dictionary of embeddings. a.k.a. num_embeddings.
            d_model (int): Dimension of each embedding vector. a.k.a. embedding_dim.
            device (torch.device | None, optional): Device to store the parameters on.
            dtype (torch.dtype | None, optional): Data type of the parameters.
        """
        super(Embedding, self).__init__()
        self.num_embeddings = vocab_size
        self.embedding_dim = d_model

        mean, variance = 0.0, 1
        std = variance**0.5
        self.weight = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(vocab_size, d_model, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3,
                b=3,
            )
        )

    def forward(self, input: Int[Tensor, " ..."]):
        return self.weight[input]


class RotaryPositionEmbedding(Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """Rotary Position Embedding.

        Args:
            theta (float): RoPE parameter.
            d_k (int): dimension of query and key vectors
            max_seq_len (int): maximum sequence length that will be input to the model
            device (torch.device | None): device to store the buffer on
        """
        super(RotaryPositionEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Create frequency bands for rotary embeddings
        # We need d_k/2 frequency bands since we apply rotation to pairs of dimensions
        # The formula is: freq_i = 1 / (theta^(2i/d_k)) for i in [0, 1, ..., d_k/2-1]
        idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (idx / d_k))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos and sin for all positions up to max_seq_len
        self._update_cache(max_seq_len, device)

    def _update_cache(self, seq_len: int, device: torch.device | None = None):
        """Update the cached cos and sin values for positions up to seq_len."""
        if device is None:
            device = self.inv_freq.device

        # Generate position indices [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # Compute frequencies for each position: [seq_len, d_k/2]
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)

        # For RoPE, each pair of dimensions uses the same frequency
        # So we need to repeat each frequency value twice
        # Shape: [seq_len, d_k/2] -> [seq_len, d_k]
        emb = torch.repeat_interleave(freqs, 2, dim=-1)

        # Cache cos and sin values: [seq_len, d_k]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"],
    ):
        """Apply rotary position embedding to input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)

        Returns:
            Tensor of same shape as x with rotary position embedding applied
        """

        # Handle broadcasting: if token_positions has fewer dimensions than x, expand it
        # This handles the case where pos_ids is 1D but x has batch dimensions
        if token_positions.dim() < x.dim() - 1:
            # Expand token_positions to match x's batch dimensions
            # E.g., if x is [batch, seq, d_k] and token_positions is [seq],
            # expand token_positions to [1, seq] and let broadcasting handle the rest
            for _ in range(x.dim() - 1 - token_positions.dim()):
                token_positions = token_positions.unsqueeze(0)
            # Expand to match batch dimensions
            token_positions = token_positions.expand(*x.shape[:-1])

        # Get cos and sin values for the given positions
        # cos_cached and sin_cached have shape [max_positions, d_k]
        # token_positions has shape [..., seq_len]
        # We need to index into the first dimension of cos/sin_cached
        
        # Clamp token_positions to ensure they're within the cached range
        # This avoids the need for dynamic cache updates during forward pass
        max_cached_pos = self.cos_cached.size(0) - 1
        token_positions = torch.clamp(token_positions, min=0, max=max_cached_pos)
        
        # Ensure token_positions is on the same device as the cached tensors for indexing
        if token_positions.device != self.cos_cached.device:
            token_positions = token_positions.to(self.cos_cached.device)
        
        cos = self.cos_cached[token_positions]  # [..., seq_len, d_k]
        sin = self.sin_cached[token_positions]  # [..., seq_len, d_k]

        # Move cos and sin to the same device as x if needed
        if cos.device != x.device:
            cos = cos.to(x.device)
            sin = sin.to(x.device)

        # Apply rotary embedding
        # RoPE applies rotation to consecutive pairs: (x0,x1), (x2,x3), etc.
        # We need to rearrange x to separate even and odd indices
        x_even = x[..., 0::2]  # indices 0, 2, 4, ...
        x_odd = x[..., 1::2]  # indices 1, 3, 5, ...

        # Apply rotation:
        # x_even_rot = x_even * cos - x_odd * sin
        # x_odd_rot = x_even * sin + x_odd * cos
        cos_half = cos[..., 0::2]
        sin_half = sin[..., 0::2]

        x_even_rot = x_even * cos_half - x_odd * sin_half
        x_odd_rot = x_even * sin_half + x_odd * cos_half

        # Interleave back to original format
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)

        return x_rot
