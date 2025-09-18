from torch.nn import Module, Parameter
import torch
from torch import Tensor

from jaxtyping import Float


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