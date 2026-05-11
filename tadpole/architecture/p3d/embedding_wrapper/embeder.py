from diffusers.models.embeddings import (
    LabelEmbedding,
)
import torch
import torch.nn as nn
from typing import Optional
from torch.nn import init

class AdaLayerNormZero(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        class_dropout: float = 0.0,
        chunk_size: int = 2,
        norm_type="layer_norm",
        bias=True,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.emb_impl = LabelEmbedding(num_classes, embedding_dim, class_dropout)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, chunk_size * embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        # Critical fix: tiny scaling for gradient flow

        # Optionally: zero init still okay
        nn.init.normal_(self.linear.weight, std=1e-2)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, class_labels: Optional[torch.LongTensor] = None):
        emb = self.emb_impl(class_labels)
        emb = self.linear(self.silu(emb))
        return emb.chunk(self.chunk_size, dim=1)
