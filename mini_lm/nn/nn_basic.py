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
