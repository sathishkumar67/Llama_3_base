from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gin
from typing import Optional
from functions import precompute_freqs_cis, apply_rotary_emb, repeat_kv, reshape_for_broadcast

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
    attn_dropout: float = 0.0


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
            n_heads: The number of attention heads.
            n_kv_heads: The number of key-value heads (default: same as n_heads).
            n_rep: The number of times to repeat key-value heads if n_kv_heads < n_heads.
            head_dim: The dimension of each attention head.
            wq, wk, wv, wo: Linear layers for queries, keys, values, and output.
        """
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # linear layers for queries, keys, and values
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):        
        """
        Computes the output of the attention module.

        Given an input tensor `x`, precomputed frequencies `freqs_cis`, and
        configuration parameters `args`, apply the attention mechanism to produce
        the output.

        Args:
            x: The input tensor.
            freqs_cis: The precomputed frequencies for the rotary embedding.

        Returns:
            The output of the attention module.
        """
        bsz, seqlen, _ = x.shape

        # linear projections for queries, keys, and values
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # reshape for attention computation
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # compute attention
        y = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True, dropout_p=self.args.attn_dropout)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_heads * self.head_dim)

        # output projection
        return self.wo(y)


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
            mask: An optional tensor with shape (batch_size, n_heads, seq_len, cache_len + seq_len)
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

        # tie the weights of the token embeddings and the output layer
        self.tok_embeddings.weight = self.output.weight

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