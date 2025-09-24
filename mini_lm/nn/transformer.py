from torch.nn import Module
from . import MultiHeadAttentionWithRope, SwiGLU, RMSNorm


class TransformerBlock(Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttentionWithRope(
            d_model, num_heads, max_seq_len, theta
        )
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        # Self-attention sub-layer with residual connection and layer normalization

        # pre-norm
        attn_input = self.norm1(x)
        attn_output = self.attention(attn_input, attn_input, attn_input)

        # Residual connection
        x = x + attn_output

        # Feed-forward sub-layer with residual connection and layer normalization
        ff_input = self.norm2(x)
        ff_output = self.ffn(ff_input)
        x = x + ff_output

        return x
