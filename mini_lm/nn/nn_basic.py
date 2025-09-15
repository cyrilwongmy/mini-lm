from torch.nn import Module, Parameter
import torch
from torch import Tensor

from jaxtyping import Float, Int


class Linear(Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(Linear, self).__init__()
        self.in_features = d_in
        self.out_features = d_out

        mean, variance = 0.0, 2 / (d_in + d_out)
        std = variance**0.5
        self.weight = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, input: Float[Tensor, " ... d_in"]):
        return torch.einsum("...i, oi -> ...o", input, self.weight)


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Embedding layer mapping indices to dense vectors.

        Args:
            num_embeddings (int): Size of the dictionary of embeddings.
            embedding_dim (int): Dimension of each embedding vector.
            device (torch.device | None, optional): Device to store the parameters on.
            dtype (torch.dtype | None, optional): Data type of the parameters.
        """
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        mean, variance = 0.0, 1
        std = variance**0.5
        self.weight = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3,
                b=3,
            )
        )

    def forward(self, input: Int[Tensor, " ..."]):
        return self.weight[input]


class RMSNorm(Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Root Mean Square Layer Normalization (RMSNorm).

        Args:
            d_model (int): Hidden diemension of the model.
            eps (float): Small epsilon for numerical stability. Defaults to 1e-5.
            device (torch.device | None, optional): Device for parameters. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, " ... d_model"]):
        # Upcast to float32 to prevent overflow when squaring the inputs
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        x_norm = (x / rms) * self.gain
        return x_norm.to(in_dtype)


class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """SwiGLU activation.

        Args:
            d_model (int): Hidden dimension of the model.
            d_ff (int): Hidden dimension of the feedforward layer.
        """
        super(SwiGLU, self).__init__()

        # w1_weight: "d_ff x d_model"
        mean, variance = 0.0, 2 / (d_model + d_ff)
        std = variance**0.5
        self.w1 = Parameter(torch.nn.init.trunc_normal_(
            torch.empty(d_ff, d_model, device=device, dtype=dtype),
            mean=mean,
            a=-3 * std,
            b=3 * std,
        ))

        # w2_weight: "d_model x d_ff"
        self.w2 = Parameter(torch.nn.init.trunc_normal_(
            torch.empty(d_model, d_ff, device=device, dtype=dtype),
            mean=mean,
            a=-3 * std,
            b=3 * std,
        ))

        # w3_weight: "d_ff x d_model"
        self.w3 = Parameter(torch.nn.init.trunc_normal_(
            torch.empty(d_ff, d_model, device=device, dtype=dtype),
            mean=mean,
            a=-3 * std,
            b=3 * std,
        ))

    def forward(self, x: Float[Tensor, " ..."]):
        # formula: FFN(x) = SwiGLU(x1, w1, w2, w3) = w2 matmul (SiLU(w1 matmul x) * (w3 matmul x))

        # SiLU(w1 matmul x)
        left = torch.einsum("...i, oi -> ...o", x, self.w1)
        # left = torch.nn.functional.silu(left)
        left = left * torch.sigmoid(left)
        # w3 matmul x
        right = torch.einsum("...i, oi -> ...o", x, self.w3)
        # result = (left * right) matmul w2

        return torch.einsum("...i, oi -> ...o", left * right, self.w2)