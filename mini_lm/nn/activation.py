from torch.nn import Module, Parameter
import torch
from torch import Tensor
from jaxtyping import Float

class Softmax(Module):
    def __init__(self, dim: int = -1, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Softmax activation.
        
        Args:
            dim (int): Dimension to normalize. Default is -1.
            device (torch.device | None): Device to place the parameters. Default is None.
            dtype (torch.dtype | None): Data type of the parameters. Default is None.
        """
        super(Softmax, self).__init__()
        self.dim = dim
        self.device = device
        self.dtype = dtype

    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        dim = self.dim
        # shift by max for numerical stability 
        mx = x.amax(dim=dim, keepdim=True)
        # normalize
        z = (x - mx).exp()
        den = z.sum(dim=dim, keepdim=True)
        # guard against weird edge cases (e.g. -inf)
        den = den.clamp_min(torch.finfo(z.dtype).tiny)
        return z / den

class SwiGLU(Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """SwiGLU activation.

        Args:
            d_model (int): Hidden dimension of the model.
            d_ff (int): Hidden dimension of the feedforward layer.
        """
        super(SwiGLU, self).__init__()

        # w1_weight: "d_ff x d_model"
        mean, variance = 0.0, 2 / (d_model + d_ff)
        std = variance**0.5
        self.w1 = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype),
                mean=mean,
                a=-3 * std,
                b=3 * std,
            )
        )

        # w2_weight: "d_model x d_ff"
        self.w2 = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_model, d_ff, device=device, dtype=dtype),
                mean=mean,
                a=-3 * std,
                b=3 * std,
            )
        )

        # w3_weight: "d_ff x d_model"
        self.w3 = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype),
                mean=mean,
                a=-3 * std,
                b=3 * std,
            )
        )

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