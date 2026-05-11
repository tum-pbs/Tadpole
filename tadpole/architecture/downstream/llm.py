from typing import Union, Sequence
from einops import rearrange
from functools import lru_cache
from .hyper_attn.hyper_attn import HyperAttention
from .mamba import create_mamba_block
import torch.utils.checkpoint as checkpoint
import math
import torch
import numpy as np
import torch.nn as nn


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # assert embed_dim % 2 == 0
    embed_dim_ = int(math.ceil(embed_dim / 2))
    omega = np.arange(embed_dim_, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb[:, :embed_dim]  # (M, D), truncate to the desired embedding dimension


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 3D sine-cosine embeddings from a grid.
    Args:
        embed_dim (int): Total embedding dimension.
        grid (numpy.ndarray): Grid positions of shape [3, 1, W, H, D].
    Returns:
        numpy.ndarray: A 3D positional embedding of shape [W*H*D, embed_dim].
    """
    # assert embed_dim % 3 == 0, "Embedding dimension must be divisible by 3 for 3D embedding."
    embed_dim_per_axis = int(math.ceil(embed_dim / 3))

    emb_d = get_1d_sincos_pos_embed_from_grid(
        embed_dim_per_axis, grid[0].flatten()
    )  # Depth
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim_per_axis, grid[1].flatten()
    )  # Height
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim_per_axis, grid[2].flatten()
    )  # Width

    emb = np.concatenate(
        [emb_d, emb_h, emb_w], axis=1
    )  # Combine along embedding dimensions
    return emb[:, :embed_dim]  # Truncate to the desired embedding dimension


@lru_cache(maxsize=32)
def get_3d_sincos_pos_embed(
    embed_dim, grid_size: Union[int, Sequence[int]], cls_token=False, i=0, j=0, k=0
):
    """
    Generate a 3D sine-cosine positional embedding.
    Args:
        embed_dim (int): Total embedding dimension.
        grid_size (int): The size of each spatial dimension of the grid.
        cls_token (bool): If True, prepend a [CLS] token embedding.
        i, j, k (int): Offsets for the grid along each dimension.
    Returns:
        numpy.ndarray: A 3D sine-cosine positional embedding of shape
                       [(1 + grid_size^3) if cls_token else grid_size^3, embed_dim].
    """

    # check if grid_size is a sequence
    if isinstance(grid_size, Sequence):
        assert len(grid_size) == 3, "grid_size must be a sequence of length 3"
        grid_size_w = grid_size[0]
        grid_size_h = grid_size[1]
        grid_size_d = grid_size[2]
    else:
        grid_size_w = grid_size_h = grid_size_d = grid_size

    grid_d = torch.arange(grid_size_d, dtype=torch.float32) + i * grid_size_d
    grid_h = torch.arange(grid_size_h, dtype=torch.float32) + j * grid_size_h
    grid_w = torch.arange(grid_size_w, dtype=torch.float32) + k * grid_size_w
    grid = torch.meshgrid(grid_w, grid_h, grid_d, indexing="ij")  # Shape: [3, W, H, D]
    grid = torch.stack(grid, dim=0).reshape(
        [3, 1, grid_size_w, grid_size_h, grid_size_d]
    )

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_mask=False,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_mask = use_mask
        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HyperAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        inner_dim,
        num_heads,
        causal=False,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.proj = nn.Linear(inner_dim, dim)
        assert inner_dim % num_heads == 0, (inner_dim, num_heads)
        self.num_heads = num_heads

        self.attn = HyperAttention(
            input_dim=inner_dim // num_heads,
            lsh_num_projs=7,
            block_size=256,
            sample_size=256,
            min_seq_len=4096,
        )
        self.causal = causal

    def forward(self, x):
        """
        X: N L H
        """
        B, L, D = x.shape
        q, k, v = (
            self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )  # B H L D // num_heads
        attn_out = self.attn(q, k, v, causal=self.causal).permute(
            0, 2, 1, 3
        )  # B H L D // num_heads
        attn_out = attn_out.reshape(B, L, -1).contiguous()
        attn_out = self.proj(attn_out)

        return attn_out


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.pretraining_tp = 1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LLMLayer(nn.Module):
    def __init__(
        self, dim, inner_dim, num_heads, causal=False, attention_method="hyper"
    ):
        super().__init__()
        # num_heads = inner_dim // dim
        if attention_method == "hyper":
            self.attn = HyperAttentionBlock(dim, dim, num_heads, causal=causal)
        else:
            self.attn = AttentionBlock(dim, dim, num_heads, causal=causal)
        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.mlp = LlamaMLP(dim, inner_dim)
        self.causal = causal

    def forward(self, hidden_states, residual_in=-1):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if residual_in != -1:
            return hidden_states, 0.0
        return hidden_states


class SequentialModel(nn.Module):
    """
    A transformer-like model for 3D data with flexible attention mechanisms and in-context patching.

    This model projects 3D input data to a hidden space, adds positional encoding, and processes it through
    a stack of attention/MLP layers. It supports full-sequence and sliding-window (in-context) processing,
    and can use different attention mechanisms (hyper, naive, mamba).

    Args:
        in_dim (int): Input feature dimension.
        mlp_ratio (int): Ratio for MLP hidden size.
        use_checkpoint (bool): Whether to use gradient checkpointing.
        hidden_size (int): Hidden feature dimension.
        num_heads (int): Number of attention heads.
        n_layers (int): Number of transformer layers.
        attention_method (str): Attention type ('hyper', 'naive', 'mamba').
        in_context_patches (int): Patch size for in-context learning. -1 for full sequence.
        init_zero_proj (bool): Whether to initialize output projection as zeros.
    """
    def __init__(
        self,
        in_dim,
        mlp_ratio=4,
        use_checkpoint=False,
        hidden_size=768,
        num_heads=8,
        n_layers=2,
        attention_method="hyper",
        in_context_patches=-1,
        init_zero_proj=True, ### Please be extremely careful when setting this to True, you need to make sure SequentialModel is the last part of the whole model, otherwise the model will not learn anything
        use_conv_proj=False,
    ):
        """
        Initialize the SequentialModel.

        See class docstring for parameter details.
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.in_context_patches = in_context_patches
        self.use_conv_proj = use_conv_proj
        if use_conv_proj:
            self.input_proj = nn.Conv3d(in_dim, hidden_size, kernel_size=3,padding="same",padding_mode="circular")
        else:
            self.input_proj = nn.Linear(in_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, in_dim)

        self.init_zero_proj = init_zero_proj

        assert attention_method in ["hyper", "naive", "mamba"]

        if attention_method == "mamba":
            ssm_cfg = {"d_state": 16}
            self.layers = nn.Sequential(
                *[
                    create_mamba_block(
                        d_model=hidden_size,
                        ssm_cfg=ssm_cfg,
                        residual_in_fp32=True,
                        drop_rate=0.0,
                        drop_path_rate=0.0,
                        reverse=i % 2 == 0,
                        transpose=False,
                        use_mlp=False,
                        is_2d=False,
                        rms_norm=False,
                        split_head=False,
                        use_nd=False,
                        downsample=False,
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.layers = nn.Sequential(
                *[
                    LLMLayer(
                        hidden_size,
                        hidden_size * mlp_ratio,
                        num_heads,
                        causal=False,
                        attention_method=attention_method,
                    )
                    for _ in range(n_layers)
                ]
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.hidden_size = hidden_size

        self.init_weights()

    def init_weights(self):
        """
        Initialize the class token and output projection layer.
        The class token is used for global context aggregation.
        The output projection can be zero-initialized for stability.
        """
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

        if self.init_zero_proj:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward_context(self, xx, mem, offset=0):
        """
        Process a patch of the sequence with memory for in-context learning.

        Args:
            xx (Tensor): Input patch of shape [N, L, C].
            mem (list or None): Memory states for each layer.
            offset (int): Offset for overlapping patches.

        Returns:
            Tuple[Tensor, Any]: Output tensor and (optionally updated) memory.
        """
        N, _, _ = xx.shape  # Get batch size N
        classification_mode = True  # Always use classification mode (add cls_token)
        if classification_mode:
            # Concatenate cls_token to the end of the sequence for each sample
            xx = torch.cat([xx, self.cls_token.repeat(N, 1, 1)], dim=1)  # N L+1 C
        base_len = xx.shape[1]  # Length of the sequence (including cls_token)
        keep_len = base_len * 2  # Number of tokens to keep in memory
        new_mem = []  # Placeholder for new memory states

        for idx, layer in enumerate(self.layers):
            if not mem:
                # If no memory, use current input
                xx_i = xx
            else:
                # Otherwise, concatenate memory and current input, with offset
                xx_i = torch.cat(
                    [mem[idx][:, : mem[idx].shape - offset], xx[offset:]], dim=1
                )[:, -keep_len:]  # Only keep the last keep_len tokens
            if classification_mode:
                # Store memory (excluding the last token, which is cls_token)
                new_mem.append(xx_i[:-1].detach())
            else:
                new_mem.append(xx_i.detach())
            # Pass through the layer and keep only the last base_len tokens
            xx = layer(xx_i)[:, -base_len:]  # N L D

        return xx, mem  # Return output and memory (memory is not updated here)

    def forward(self, x):
        """
        Forward pass for the SequentialModel.

        Args:
            x (Tensor): Input tensor of shape [n, c, h, w, d].

        Returns:
            Tensor: Output tensor of shape [n, c, h, w, d].
        """
        if self.use_conv_proj:
            # Project input to hidden_size dimension
            x = self.input_proj(x)
        n, _, h, w, d = x.shape  # Get batch size and spatial dimensions
        # Generate 3D positional embedding for the spatial grid
        pos_embed = get_3d_sincos_pos_embed(
            self.hidden_size, (h, w, d), cls_token=False
        )
        # Rearrange input to [n, h*w*d, c] (flatten spatial dims)
        x = rearrange(x, "n c h w d -> n (h w d) c")

        if not self.use_conv_proj:
            # Project input to hidden_size dimension
            x = self.input_proj(x)
        # Add positional embedding
        x = x + torch.tensor(pos_embed).to(x)
        # Concatenate cls_token to the end of the sequence for each sample
        x = torch.cat([x, self.cls_token.repeat(n, 1, 1)], dim=1)
        residual = None  # For storing residuals in checkpoint mode
        
        # If not using in-context patches, or patch size is too large
        if self.in_context_patches <= 0 or self.in_context_patches >= x.shape[1]:
            if self.use_checkpoint:
                # Use gradient checkpointing to save memory
                for i, blk in enumerate(self.layers):
                    x, residual = checkpoint.checkpoint(blk, x, residual)
                    # On last layer, add residual if exists
                    if i == len(self.layers) - 1:
                        x = (x + residual) if residual is not None else x
            else:
                # Standard forward through each layer
                for i, blk in enumerate(self.layers):
                    x, residual = blk(x, residual)
                    # On last layer, add residual if exists
                    if i == len(self.layers) - 1:
                        x = (x + residual) if residual is not None else x
        else:
            # Use sliding window in-context patching
            mem = None  # Memory for each layer
            all_ys = []  # Store outputs for all patches
            start = 0  # Start index for patch
            all_cls = []  # Store cls_token outputs for all patches
            while start < x.shape[1] - 1:
                # Get a patch of the sequence
                xx = x[:, start : start + self.in_context_patches]
                # Forward through context window, update memory
                ss, mem = self.forward_context(
                    xx, mem, offset=self.in_context_patches // 2
                )
                all_ys.append(ss)  # Store all outputs
                all_cls.append(ss[:, -1:])  # Store cls_token output
                start += self.in_context_patches // 2  # Move window forward
            _ = torch.cat(all_ys, dim=1)  # (Unused) Concatenate all outputs
            x = torch.cat(all_cls, dim=1)  # Concatenate all cls_token outputs
            x = x.mean(dim=1, keepdims=True)  # Average over all patches
        # Remove the cls_token before output projection
        x = x[:, :-1]

        # Project back to input dimension
        x = self.out_proj(x)
        # Rearrange back to [n, c, h, w, d]
        x = rearrange(x, "n (h w d) c -> n c h w d", h=h, w=w, d=d)

        # x = self.decoder(x)  # (Optional) Decoder step
        # x = self.cls(x[:, -1]) # (Optional) Classification head
        return x  # Output tensor of shape [n, c, h, w, d]
