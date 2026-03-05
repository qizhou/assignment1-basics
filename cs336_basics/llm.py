import torch
import math
from einops import reduce, einsum, repeat

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super(Linear, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros(out_features, in_features))
        std = math.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        std = 1
        torch.nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.weights = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape is batch_size, squence_length, d_model
        int_dtype = x.dtype
        x = x.to(torch.float32)

        # calculate x^2
        x2 = torch.square(x)

        # sum x^2
        ms = reduce(x2, '... d_model -> ...', 'mean')

        # add eps
        mse = ms + self.eps

        # rms
        rms = torch.sqrt(mse)
        rms_repeat = repeat(rms, "... -> ... d_model", d_model=self.d_model)

        rms_a = x / rms_repeat

        result = rms_a * self.weights

        return result.to(int_dtype)


class SiLU(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.w1 = torch.nn.Parameter(torch.zeros(d_ff, d_model))
        self.w2 = torch.nn.Parameter(torch.zeros(d_model, d_ff))
        self.w3 = torch.nn.Parameter(torch.zeros(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is ... d_model
        w1x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1x * torch.sigmoid(w1x)
        w3x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        inner = silu * w3x
        result = einsum(self.w2, inner, "d_model d_ff, ... d_ff -> ... d_model")
        return result

