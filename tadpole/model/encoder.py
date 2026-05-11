import torch
from torch.nn import Module
from typing import Union, Literal,Dict,Optional,Literal
from collections import OrderedDict
from einops import rearrange
from ..architecture.p3d import _KLP3DEncoder, _P3DDecoder
from ..architecture.p3d.kl import DiagonalGaussianDistribution
from ..utils import load_weights


class TadpoleEncoder(Module):

    def __init__(self, 
                 size: Literal["S", "B", "L"],
                weight_encoder: Optional[Union[str,Dict,OrderedDict]] = None,
                latent_type: Literal["sample", "mode"] = "sample",
                encoder_crop_size: int = 64,
                max_internal_batchsize: Optional[int] = None,
                ):
        """
        Tadpole Encoder Model.
        This model encodes a 3D input into a latent representation.
        Input shape: (B, C, X, Y, Z);
        
        Args:
            size (Literal["S", "B", "L"]): Size of the model, one of "S", "B", or "L".
            weight_encoder (Optional[Union[str,Dict,OrderedDict]]): Path to encoder weights or state dict for encoder. If None, encoder will be randomly initialized. Default is None.
            latent_type (Literal["sample", "mode"]): How to sample from the latent distribution, either "sample" or "mode". Default is "sample".
            encoder_crop_size (int): Size to crop input for encoder. If None, no cropping will be applied and the entire input will be processed as a single crop. Default is 64.
            max_internal_batchsize (Optional[int]): Maximum batch size for internal processing. If None, all crops will be processed in a single batch. Default is None.
        """
        
        super().__init__()
        assert size in ["S", "B", "L"], "size must be one of 'S', 'B', 'L'"
        self.encoder = _KLP3DEncoder(size)
        if weight_encoder is not None:
            self.encoder.load_state_dict(load_weights(weight_encoder, "encoder"))
        assert latent_type in ["sample", "mode"], "latent_type must be one of 'sample' or 'mode'"
        self.latent_type = latent_type
        self.encoder_crop_size = encoder_crop_size
        self.max_internal_batchsize = max_internal_batchsize
        
    def latent_sample(self, dist: DiagonalGaussianDistribution) -> torch.Tensor:
        if self.latent_type == "sample":
            return dist.sample()
        elif self.latent_type == "mode":
            return dist.mode()
        else:
            raise ValueError(f"Unknown latent_type: {self.latent_type}")

    def forward(self, 
                x: torch.Tensor,
                return_kl_element: bool = False,) -> torch.Tensor:
        # x: (B, C, X, Y, Z)
        kl_elem = None
        b, c, u, v, w = (
            x.shape[0],
            x.shape[1],
            max(x.shape[2] // self.encoder_crop_size,1),
            max(x.shape[3] // self.encoder_crop_size,1),
            max(x.shape[4] // self.encoder_crop_size,1),
        )
        x = rearrange(
            x, "B C (U Xc) (V Yc) (W Zc) -> (B C U V W) 1 Xc Yc Zc", U=u, V=v, W=w
        )
        if (
            self.max_internal_batchsize is None
            or x.shape[0] <= self.max_internal_batchsize
        ):
            dist = self.encoder(x, "distribution")
            x = self.latent_sample(dist)
            if return_kl_element:
                kl_elem=dist.kl_elem()
        else:
            x_chunks = torch.chunk(
                x, chunks=(x.shape[0] // self.max_internal_batchsize) + 1, dim=0
            )
            x_out_chunks = []
            kl_elem_chunks = []
            for x_chunk in x_chunks:
                x_out_chunk_dist = self.encoder(x_chunk, "distribution")
                x_out_chunk = self.latent_sample(x_out_chunk_dist)
                if return_kl_element:
                    kl_elem_chunks.append(x_out_chunk_dist.kl_elem())
                x_out_chunks.append(x_out_chunk)
            x = torch.cat(x_out_chunks, dim=0)
            if return_kl_element:
                kl_elem = torch.cat(kl_elem_chunks, dim=0)
        x = rearrange(
            x,
            "(B C U V W) Cl Xc Yc Zc -> B (C Cl) (U Xc) (V Yc) (W Zc)",
            B=b,
            C=c,
            U=u,
            V=v,
            W=w,
        )
        if return_kl_element:
            return x, kl_elem
        return x