import torch
from einops import einsum

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        std = 1
        torch.nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]