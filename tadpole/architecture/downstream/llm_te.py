import torch
import torch.nn as nn
from diffusers.models.embeddings import (
    TimestepEmbedding,
    get_timestep_embedding
)
from einops import rearrange
from .llm import get_3d_sincos_pos_embed

class UniversalPositiveScalarEmbedding(nn.Module):
    
    def __init__(self,
                 embedding_dim:int,
                 proj_dim:int=256,
                 max_value: float = 1e6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_value = max_value
        self.project_dim = proj_dim
        self.scalar_embedder = TimestepEmbedding(
            in_channels=proj_dim, time_embed_dim=embedding_dim
        )
    
    def forward(self, scalar: torch.Tensor, hidden_dtype=None):
        """
        scalar: [batch_size] tensor
        returns: [batch_size, embedding_dim] tensor
        """
        return self.scalar_embedder(
            get_timestep_embedding(
                scalar, 
                self.project_dim, 
                flip_sin_to_cos=True, 
                downscale_freq_shift=1,
                max_period= self.max_value*10
            )
        )

from .llm import LLMLayer

class TimeEmbeddedLLMLayer(LLMLayer):
    
    def __init__(self, dim, inner_dim, num_heads, causal=False, attention_method="hyper"):
        super().__init__(dim, inner_dim, num_heads, causal, attention_method)
        self.t_embedding=UniversalPositiveScalarEmbedding(embedding_dim=dim, proj_dim=256)
        
    def forward(self, 
                hidden_states, 
                t):
        t_embedding=self.t_embedding(t)
        t_embedding = t_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        hidden_states=hidden_states + t_embedding
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
class EmbeddedProjection(nn.Module):
    
    def __init__(self,
                 in_dim,
                 out_dim,):
        super().__init__()
        self.proj=nn.Linear(in_dim,out_dim)
        self.embedding=UniversalPositiveScalarEmbedding(embedding_dim=out_dim, proj_dim=256)
        
    def forward(self, x, t):
        t_embedding=self.embedding(t)
        t_embedding = t_embedding.unsqueeze(1)  # [batch_size, 1, out_dim*2]:
        #alpha,beta=torch.chunk(t_embedding,2,dim=-1)
        x=self.proj(x)+t_embedding
        #x=x*alpha+beta
        return x
    
class TimeEmbeddedSequentialModel(nn.Module):
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
        hidden_size=768,
        num_heads=8,
        n_layers=2,
        in_context_patches=-1,
        init_zero_proj=False, ### Please be extremely careful when setting this to True, you need to make sure SequentialModel is the last part of the whole model, otherwise the model will not learn anything
    ):
        """
        Initialize the SequentialModel.

        See class docstring for parameter details.
        """
        super().__init__()
        self.in_context_patches = in_context_patches
        self.input_proj = EmbeddedProjection(in_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, in_dim)

        self.init_zero_proj = init_zero_proj

        self.layers = nn.Sequential(
            *[
                TimeEmbeddedLLMLayer(
                    hidden_size,
                    hidden_size * mlp_ratio,
                    num_heads,
                    causal=False,
                    attention_method="hyper",
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


    def forward(self, x, t):
        """
        Forward pass for the SequentialModel.

        Args:
            x (Tensor): Input tensor of shape [n, c, h, w, d].

        Returns:
            Tensor: Output tensor of shape [n, c, h, w, d].
        """
        t=t*1000  # Scale time steps to [0, 1000] range
        n, _, h, w, d = x.shape  # Get batch size and spatial dimensions
        # Generate 3D positional embedding for the spatial grid
        pos_embed = get_3d_sincos_pos_embed(
            self.hidden_size, (h, w, d), cls_token=False
        )
        # Rearrange input to [n, h*w*d, c] (flatten spatial dims)
        x = rearrange(x, "n c h w d -> n (h w d) c")
        x = self.input_proj(x,t)
        # Add positional embedding
        x = x + torch.tensor(pos_embed).to(x)
        # Concatenate cls_token to the end of the sequence for each sample
        x = torch.cat([x, self.cls_token.repeat(n, 1, 1)], dim=1)
        
        # Standard forward through each layer
        for i, blk in enumerate(self.layers):
            x = blk(x,t)
        # Remove the cls_token before output projection
        x = x[:, :-1]

        # Project back to input dimension
        x = self.out_proj(x)
        # Rearrange back to [n, c, h, w, d]
        x = rearrange(x, "n (h w d) c -> n c h w d", h=h, w=w, d=d)

        # x = self.decoder(x)  # (Optional) Decoder step
        # x = self.cls(x[:, -1]) # (Optional) Classification head
        return x  # Output tensor of shape [n, c, h, w, d]
