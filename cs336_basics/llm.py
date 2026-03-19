import torch
import math
from einops import reduce, einsum, repeat
from torch import Tensor
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from collections.abc import Callable, Iterable
from typing import Optional


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


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE, self).__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        i_s = torch.linspace(0, max_seq_len-1, max_seq_len)
        t_s = 1.0 / (theta ** ((2 * torch.linspace(1, d_k//2, d_k//2) - 2) / d_k))

        freqs = einsum(i_s, t_s, 'i,j->i j')

        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *prefix, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"expected last dim {self.d_k}, got {d_k}"
        assert seq_len <= self.max_seq_len, f"T={seq_len} exceeds max_seq_len={self.max_seq_len}"

        if token_positions is None:
            # (T,)
            token_positions = torch.arange(seq_len)

        cos = self.rope_cos.index_select(0, token_positions.reshape(-1)).view(*token_positions.shape, -1)
        sin = self.rope_sin.index_select(0, token_positions.reshape(-1)).view(*token_positions.shape, -1)

        # We want cos/sin broadcastable to x_even/x_odd which are (..., T, half)
        # So we reshape cos/sin to have leading singleton dims for any prefix dims.
        # token_positions could be (T,) or (B,T). We align the *T* dimension to x's T.
        # while cos.ndim < x.ndim - 1:  # x.ndim-1 because cos lacks the last dim
        #     cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        # Interleaved RoPE: rotate pairs (x0,x1), (x2,x3), ...
        x_even = x[..., 0::2]  # (..., T, half)
        x_odd  = x[..., 1::2]  # (..., T, half)

        # Apply rotation
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # Re-interleave
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)  # (..., T, d_k)
        return out


def softmax(in_features: Float[Tensor, " ..."], dim: int):
    # Subtract max value
    repeat_vec = [1 for _ in range(len(in_features.shape))]
    repeat_vec[dim] = in_features.shape[dim]
    max_v, _ = in_features.max(dim, keepdim=True)
    max_v = max_v.repeat(*repeat_vec)

    # Calculate exp and the sum
    ev = (in_features - max_v).exp()
    sum_ev = ev.sum(dim, keepdim=True)
    sum_ev = sum_ev.repeat(*repeat_vec)

    return ev / sum_ev


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None
) -> Float[Tensor, " ... queries d_v"]:
    # queries, keys, and values == seq_len
    # d_v == d_k
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        # qk[~mask] = -torch.inf # forward works, but not autograde
        qk = qk.masked_fill(~mask, float("-inf"))

    # softmax over keys
    sm = softmax(qk, -1)
    return einsum(sm, V, "... queries keys, ... keys d_v -> ... queries d_v")


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, rope=None):
        super(CausalMultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.d_model = d_model
        self.rope = rope
        self.w_q = torch.nn.Parameter(torch.zeros(d_model, d_model))
        self.w_k = torch.nn.Parameter(torch.zeros(d_model, d_model))
        self.w_v = torch.nn.Parameter(torch.zeros(d_model, d_model))
        self.w_o = torch.nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, in_features: torch.Tensor, token_positions: Tensor | None = None):
        # in_features "batch seq_len d_in=d_model"
        batch, seq_len, _ = in_features.shape
        # q,k,v "batch seq_len d_model"
        q = einsum(in_features, self.w_q, "... seq_len d_model, dd d_model -> ... seq_len dd")
        k = einsum(in_features, self.w_k, "... seq_len d_model, dd d_model -> ... seq_len dd")
        v = einsum(in_features, self.w_v, "... seq_len d_model, dd d_model -> ... seq_len dd")

        # Reshape into heads: "batch, num_heads, seq_len, d_k"
        q = q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Create mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        full_mask = mask.unsqueeze(0).unsqueeze(0)

        # attn "batch, num_heads, seq_len, d_k"
        attn = scaled_dot_product_attention(q, k, v, full_mask)

        # attn "batch, seq_len, num_heads, d_k" => "batch, seq_len, d_model"
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        return einsum(attn, self.w_o, "... seq_len d_model, dd d_model -> ... seq_len dd")


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
    state_prefix: str = ""
):
    ln1 = RMSNorm(d_model, 1e-5)
    ln1.load_state_dict({"weights": weights[state_prefix + "ln1.weight"]})

    d_k = d_model // num_heads
    rope = RoPE(theta, d_k, max_seq_len)
    attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope)
    attn.load_state_dict({
        "w_q": weights[state_prefix + "attn.q_proj.weight"],
        "w_k": weights[state_prefix + "attn.k_proj.weight"],
        "w_v": weights[state_prefix + "attn.v_proj.weight"],
        "w_o": weights[state_prefix + "attn.output_proj.weight"],
    })

    # The first part of the block
    first = in_features + attn.forward(ln1.forward(in_features))

    ln2 = RMSNorm(d_model, 1e-5)
    ln2.load_state_dict({"weights": weights[state_prefix + "ln2.weight"]})
    swiglu = SwiGLU(d_model, d_ff)
    # If your state dict keys match, you can use `load_state_dict()`
    swiglu.load_state_dict({
        "w1": weights[state_prefix + "ffn.w1.weight"],
        "w2": weights[state_prefix + "ffn.w2.weight"],
        "w3": weights[state_prefix + "ffn.w3.weight"]
    })

    # The second part of the block
    second = first + swiglu.forward(ln2.forward(first))

    return second

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        m = None
        v = None
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros(p.data.shape))
                v = state.get("v", torch.zeros(p.data.shape))
                grad = p.grad.data # Get the gradient of loss with respect to p.

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                state["m"] = m
                state["v"] = v
                lr_t = lr * math.sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t))

                p.data -= lr_t * m / (v.sqrt() + eps) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.
                return loss