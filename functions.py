import torch
from typing import Tuple



def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0) -> torch.Tensor:
    """
    Precompute the frequencies for the rotary position embedding.

    Args:
        dim: The last dimension of the input tensor.
        end: The first dimension of the input tensor.
        theta: The temperature in the computation of the frequencies.

    Returns:
        A tensor of shape `(end, dim)` with the precomputed frequencies.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32) # change to torch.bfloat16 when using bfloat16
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape the precomputed frequencies for rotary embeddings to be broadcasted to x.

    The shape of freqs_cis is (seq_len, dim) and the shape of x is (batch_size, seq_len, dim).
    We want to reshape freqs_cis to be (1, seq_len, 1, dim) so that it can be broadcasted to
    x.

    Args:
        freqs_cis: The precomputed frequencies for the rotary embeddings.
        x: The tensor to which the rotary embeddings will be applied.

    Returns:
        The reshaped frequencies.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotary embeddings to the query and key tensors.

    The rotary embeddings are precomputed using the `precompute_freqs_cis` function.
    The query and key tensors are reshaped to have shape (batch_size, seq_len, dim),
    and the precomputed frequencies are reshaped to have shape (1, seq_len, 1, dim)
    so that they can be broadcasted to the query and key tensors.

    The rotary embeddings are applied by element-wise multiplying the query and key
    tensors with the precomputed frequencies.

    Args:
        xq: The query tensor.
        xk: The key tensor.
        freqs_cis: The precomputed frequencies.

    Returns:
        The query and key tensors after applying the rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat the key-value heads of a tensor.

    This function takes an input tensor `x` of shape (batch_size, seq_len, n_kv_heads, head_dim)
    and repeats the key-value heads `n_rep` times, resulting in a new tensor with
    shape (batch_size, seq_len, n_kv_heads * n_rep, head_dim).

    Args:
        x: The input tensor with shape (batch_size, seq_len, n_kv_heads, head_dim).
        n_rep: The number of times to repeat the key-value heads.

    Returns:
        A tensor with shape (batch_size, seq_len, n_kv_heads * n_rep, head_dim) after repeating
        the key-value heads.
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )