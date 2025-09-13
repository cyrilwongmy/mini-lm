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