from torch.nn import Module, ModuleList
from . import (
    MultiHeadAttentionWithRope,
    FeedForwardSwiGLU,
    RMSNorm,
    Embedding,
    Linear,
    Softmax,
)


class TransformerBlock(Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttentionWithRope(
            d_model, num_heads, max_seq_len, theta
        )
        self.ffn = FeedForwardSwiGLU(d_model, d_ff)
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


class Transformer(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embedding = Embedding(vocab_size, self.d_model)
        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    max_seq_len=context_length,
                    theta=self.rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(self.d_model)
        self.linear = Linear(self.d_model, vocab_size)
        self.softmax = Softmax(dim=-1)

    def forward(self, x, return_logits=False):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.linear(self.norm(x))
        if return_logits:
            return logits
        return self.softmax(logits)
