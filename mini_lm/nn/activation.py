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


class SiLU(Module):
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """SiLU (Sigmoid Linear Unit) activation function.
        
        Also known as Swish activation function.
        SiLU(x) = x * sigmoid(x)
        
        Args:
            device (torch.device | None): Device to place the parameters. Default is None.
            dtype (torch.dtype | None): Data type of the parameters. Default is None.
        """
        super(SiLU, self).__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        """Apply SiLU activation.
        
        Args:
            x: Input tensor of any shape.
            
        Returns:
            Output tensor with same shape as input.
        """
        return x * torch.sigmoid(x)