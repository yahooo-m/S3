# Modified from PyTorch nn.Transformer

from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from cutie.model.channel_attn import CAResBlock


class SelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False]):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv

    def forward(self,
                x: torch.Tensor,
                pe: torch.Tensor,
                attn_mask: bool = None,
                key_padding_mask: bool = None) -> torch.Tensor:
        x = self.norm(x)
        if any(self.add_pe_to_qkv):
            x_with_pe = x + pe
            q = x_with_pe if self.add_pe_to_qkv[0] else x
            k = x_with_pe if self.add_pe_to_qkv[1] else x
            v = x_with_pe if self.add_pe_to_qkv[2] else x
        else:
            q = k = v = x

        r = x
        x = self.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return r + self.dropout(x)


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
class CrossAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False],
                 residual: bool = True,
                 norm: bool = True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim,
                                                nhead,
                                                dropout=dropout,
                                                batch_first=batch_first)
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv
        self.residual = residual

    def forward(self,
                x: torch.Tensor,
                mem: torch.Tensor,
                x_pe: torch.Tensor,
                mem_pe: torch.Tensor,
                attn_mask: bool = None,
                *,
                need_weights: bool = False) -> (torch.Tensor, torch.Tensor):
        x = self.norm(x)
        if self.add_pe_to_qkv[0]:
            q = x + x_pe
        else:
            q = x

        if any(self.add_pe_to_qkv[1:]):
            mem_with_pe = mem + mem_pe
            k = mem_with_pe if self.add_pe_to_qkv[1] else mem
            v = mem_with_pe if self.add_pe_to_qkv[2] else mem
        else:
            k = v = mem
        r = x
        x, weights = self.cross_attn(q,
                                     k,
                                     v,
                                     attn_mask=attn_mask,
                                     need_weights=need_weights,
                                     average_attn_weights=False)

        if self.residual:
            return r + self.dropout(x), weights
        else:
            return self.dropout(x), weights


class FFN(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_in)
        self.norm = nn.LayerNorm(dim_in)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        x = self.linear2(self.activation(self.linear1(x)))
        x = r + x
        return x


class PixelFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel: torch.Tensor, pixel_flat: torch.Tensor) -> torch.Tensor:
        # pixel: batch_size * num_objects * dim * H * W
        # pixel_flat: (batch_size*num_objects) * (H*W) * dim
        bs, num_objects, _, h, w = pixel.shape
        pixel_flat = pixel_flat.view(bs * num_objects, h, w, self.dim)
        pixel_flat = pixel_flat.permute(0, 3, 1, 2).contiguous()

        x = self.conv(pixel_flat)
        x = x.view(bs, num_objects, self.dim, h, w)
        return x


class OutputFFN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class LocalRepresentation(nn.Module):
    """
    Local Representation module for generating feature vectors from input features.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).

    Attributes:
        to_query_3x3 (nn.Conv2d): 3x3 depth-wise convolutional layer for local feature extraction.
        bn (nn.BatchNorm2d): Batch normalization layer.
        out (nn.Linear): Linear transformation layer.
        d_model (int): The dimensionality of the input and output feature vectors.

    Methods:
        forward(self, x): Forward pass through the LocalRepresentation module.
    """

    def __init__(self, d_model=256):
        super().__init__()

        self.to_query_3x3 = nn.Conv2d(d_model, d_model, 3, groups=d_model, padding=1)
        self.bn = nn.SyncBatchNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

        self.d_model = d_model

    def forward(self, x):
        # Retrieve input tensor shape
        B, C, H, W = x.shape

        # Apply pre-normalisation followed by 3x3 local convolution to extract local features
        x = self.bn(x)
        x_3x3 = self.to_query_3x3(x)

        # Reshape the local features and permute dimensions for linear transformation
        return self.out(x_3x3.view(B, self.d_model, H * W).permute(0, 2, 1))


class DQ_CA(nn.Module):
    """
    Discriminative query Cross-Attention module.

    This module implements a variant of the cross-attention mechanism for use in segmentation heads.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).
        nhead (int): The number of attention heads (default: 8).

    Attributes:
        to_query (LocalRepresentation): Module for converting input to query representations.
        to_key (nn.Sequential): Sequential module for transforming input to key representations.
        proj (nn.Linear): Linear transformation layer.
        final (nn.Linear): Final linear transformation layer.
        alpha (nn.Parameter): Parameter for scaling in the attention mechanism.
        num_heads (int): Number of attention heads.

    Methods:
        with_pos_embed(self, tensor, pos): Adds positional embeddings to the input tensor.
        most_similar_tokens(self, x, q, mask=None): Finds the most similar tokens based on content-based attention.
        forward(self, q, x, memory_mask, pos, query_pos): Forward pass through the PEM_CA module.
    """

    def __init__(self, d_model, nhead,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False],
                 residual: bool = True,
                 norm: bool = True):
        super().__init__()

        if norm:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.Identity()

        self.proj = nn.Linear(d_model, d_model)
        self.final = nn.Linear(d_model, d_model)

        self.alpha = nn.Parameter(torch.ones(1, 1, d_model))
        self.num_heads = nhead

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def most_similar_tokens(self, x, q, mask=None):
        # Retrieve input tensors shapes
        B, N, C = x.shape
        Q, D = q.shape[1], C // self.num_heads

        # Reshape tensors in multi-head fashion
        x = x.view(B, N, self.num_heads, D).permute(0, 2, 1, 3)
        q = q.view(B, Q, self.num_heads, D).permute(0, 2, 1, 3)

        # Compute similarity scores between features and queries
        sim = torch.einsum('bhnc, bhqc -> bhnq', x, q)

        # Apply mask to similarity scores if provided
        if mask is not None:
            mask = mask.permute(0, 2, 1)
            mask = mask.view(B, self.num_heads, *mask.shape[1:])
            #             sim = sim.masked_fill_(mask, float('-1e-10'))
            sim = sim.masked_fill_(mask, float('-inf'))

        # Find indices of most similar tokens
        most_similar_indices = torch.argmax(sim, dim=2)

        #         sim[sim != sim] = 0

        # Gather most similar tokens
        return torch.gather(x, 2, most_similar_indices.unsqueeze(-1).expand(-1, -1, -1, D)).permute(0, 2, 1, 3).reshape(
            B, Q, C), sim

    def forward(self, q, x, query_pos, x_pos, attn_mask, *, need_weights):
        q = self.norm(q)
        res = q

        # Add positional embeddings to input tensors
        x, q = self.with_pos_embed(x, x_pos), self.with_pos_embed(q, query_pos)

        # Convert inputs to query and key representations
        query = x
        key = q

        # Normalize query and key vectors
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        # Find most similar tokens
        query, sim = self.most_similar_tokens(query, key, attn_mask)  # BxQxD

        # Perform attention mechanism with projection and scaling
        out = nn.functional.normalize(self.proj(query * key), dim=1) * self.alpha + query  # BxQxD

        # Final linear transformation
        out = self.final(out)  # BxQxD

        return out + res, sim