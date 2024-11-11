from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gin
from typing import Tuple, Optional


@gin.configurable
@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    multiple_of: int
    ffn_dim_multiplier: float
    norm_eps: float
    rope_theta: float
    max_batch_size: int
    max_seq_len: int


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
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initializes the RMSNorm module.

        Args:
            dim: The dimension of the input tensor.
            eps: The epsilon value used to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Computes the RMSNorm of a tensor.

        Given an input tensor `x`, compute its RMSNorm by dividing it by the root
        mean square of its elements.

        Args:
            x: The input tensor.

        Returns:
            The RMSNorm of the input tensor.
        """
        
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):        
        """
        Computes the RMSNorm of a tensor and applies a learnable scale factor.

        Args:
            x: The input tensor.

        Returns:
            The RMSNorm of the input tensor multiplied by a learnable scale factor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Initializes the Attention module.

        Args:
            args: An instance of ModelArgs containing configuration parameters such as
                dimensions, number of heads, and maximum sequence length.

        Attributes:
            n_kv_heads: The number of key-value heads, derived from args.
            n_rep: The number of repetitions for key-value heads if needed.
            head_dim: The dimension of each head, calculated from args.
            wq: A linear layer for transforming input to query vectors.
            wk: A linear layer for transforming input to key vectors.
            wv: A linear layer for transforming input to value vectors.
            wo: A linear layer for outputting the final result after attention.
            cache_k: A tensor to cache key vectors for attention.
            cache_v: A tensor to cache value vectors for attention.
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # linear layers for queries, keys, and values
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # kv cache to store the key and value tensors 
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):        
        """Computes the forward pass of the attention module.

        Args:
            x: The input tensor with shape (batch_size, seq_len, dim).
            start_pos: The starting position of the current segment in the cache.
            freqs_cis: The precomputed frequencies for the rotary embedding.
            mask: An optional tensor with shape (batch_size, n_local_heads, seq_len, cache_len + seq_len)
                to mask out the scores of invalid positions.

        Returns:
            The output tensor with shape (batch_size, seq_len, dim) after applying the attention mechanism."""
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initializes the FeedForward module.

        Args:
            dim: The input dimension.
            hidden_dim: The hidden dimension.
            multiple_of: The multiple of the hidden dimension.
            ffn_dim_multiplier: An optional float to multiply the hidden dimension by.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        """
        Computes the output of the feed-forward network.

        Given an input tensor `x`, apply two linear layers with the ReLU activation
        function to produce the output.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after applying the feed-forward network.
        """
        
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the TransformerBlock module.

        Args:
            layer_id: The layer ID in the layer stack.
            args: An instance of ModelArgs containing configuration parameters such as
                dimensions, number of heads, and maximum sequence length.

        Attributes:
            n_heads: The number of attention heads.
            dim: The input dimension.
            head_dim: The dimension of each attention head.
            attention: The attention module.
            feed_forward: The feed-forward network module.
            layer_id: The layer ID in the layer stack.
            attention_norm: The normalization module for the attention module.
            ffn_norm: The normalization module for the feed-forward network module.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Computes the output of the transformer block.

        Given an input tensor `x`, starting position `start_pos`, precomputed frequencies
        `freqs_cis` and an optional tensor `mask`, apply the attention module and the
        feed-forward network module to produce the output.

        Args:
            x: The input tensor.
            start_pos: The starting position of the current segment in the cache.
            freqs_cis: The precomputed frequencies for the rotary embedding.
            mask: An optional tensor with shape (batch_size, n_local_heads, seq_len, cache_len + seq_len)
                to mask out the scores of invalid positions.

        Returns:
            The output tensor after applying the transformer block.
        """
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out



class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            params: An instance of ModelArgs containing configuration parameters such as
                dimensions, number of layers, number of heads, vocabulary size, and other
                hyperparameters.

        Attributes:
            params: Stores the configuration parameters.
            vocab_size: The size of the vocabulary.
            n_layers: The number of transformer layers.
            tok_embeddings: The token embedding layer.
            layers: A list of TransformerBlock layers.
            norm: An RMSNorm layer for normalizing the output.
            output: A linear layer for generating output logits.
            freqs_cis: Precomputed frequencies for rotary embeddings.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Given a tensor of tokens, applies the transformer model to generate a
        tensor of output logits.

        The transformer model consists of a sequence of TransformerBlock layers.
        Each TransformerBlock layer applies a multi-head self-attention mechanism
        using the rotary embeddings, followed by a feed-forward network.

        The method takes two arguments: `tokens` and `start_pos`. `tokens` is a
        tensor of shape (batch_size, seq_len) containing the input tokens. `start_pos`
        is an integer indicating the starting position of the current segment in
        the cache.

        The method first embeds the input tokens using the `tok_embeddings` layer.
        It then applies the TransformerBlock layers sequentially to the embedded
        tokens. The output of the last layer is normalized using the `norm` layer,
        and the output logits are generated using the `output` layer.

        Returns:
            A tensor of shape (batch_size, seq_len, vocab_size) containing the
            output logits.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output