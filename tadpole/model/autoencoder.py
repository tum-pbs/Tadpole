import torch
from torch.nn import Module
from typing import Union, Literal,Dict,Optional,Literal
from collections import OrderedDict
from einops import rearrange
from ..architecture.p3d import _KLP3DEncoder, _P3DDecoder
from ..architecture.p3d.kl import DiagonalGaussianDistribution
from ..utils import load_weights
from GIFt.strategies.lora import LoRAAllFineTuningStrategy
from GIFt import enable_fine_tuning

class TadpoleAutoencoder(Module):

    def __init__(self, 
                 size: Literal["S", "B", "L"],
                weight_encoder: Optional[Union[str,Dict,OrderedDict]] = None,
                weight_decoder: Optional[Union[str,Dict,OrderedDict]] = None,
                encoder_ft_state: Union[Literal["frozen","FPFT"],int] = "FPFT",
                decoder_ft_state: Union[Literal["frozen","FPFT"],int] = "FPFT",
                latent_type: Literal["sample", "mode"] = "sample",
                encoder_crop_size: int = 64,
                max_internal_batchsize: Optional[int] = None,
                ):
        super().__init__()
        
        """
        Tadpole Autoencoder Model.
        Input shape: (B, C, X, Y, Z); Output shape: (B, C, X, Y, Z)
        
        Args:
            size (Literal["S", "B", "L"]): Size of the model, one of "S", "B", or "L".
            weight_encoder (Optional[Union[str,Dict,OrderedDict]]): Path to encoder weights or state dict for encoder. If None, encoder will be randomly initialized. Default is None.
            weight_decoder (Optional[Union[str,Dict,OrderedDict]]): Path to decoder weights or state dict for decoder. If None, decoder will be randomly initialized. Default is None.
            encoder_ft_state (Union[Literal["frozen","FPFT"],int]): Fine-tuning state for encoder. Can be a positive integer indicating the rank for LoRA fine-tuning, "frozen" to freeze the encoder weights, or "FPFT" to enable full-parameter fine-tuning. Default is "FPFT".
            decoder_ft_state (Union[Literal["frozen","FPFT"],int]): Fine-tuning state for decoder. Can be a positive integer indicating the rank for LoRA fine-tuning, "frozen" to freeze the decoder weights, or "FPFT" to enable full-parameter fine-tuning. Default is "FPFT".
            latent_type (Literal["sample", "mode"]): How to sample from the latent distribution, either "sample" or "mode". Default is "sample".
            encoder_crop_size (int): Size to crop input for encoder. If None, no cropping will be applied and the entire input will be processed as a single crop. Default is 64.
            max_internal_batchsize (Optional[int]): Maximum batch size for internal processing. If None, all crops will be processed in a single batch. Default is None.
        """
        
        assert size in ["S", "B", "L"], "size must be one of 'S', 'B', 'L'"
        self.encoder = _KLP3DEncoder(size)
        self.decoder = _P3DDecoder(size)
        if weight_encoder is not None:
            self.encoder.load_state_dict(load_weights(weight_encoder, "encoder"))
        if weight_decoder is not None:
            self.decoder.load_state_dict(load_weights(weight_decoder, "decoder"))
        assert latent_type in ["sample", "mode"], "latent_type must be one of 'sample' or 'mode'"
        # set fine-tuning states for encoder and decoder
        if isinstance(encoder_ft_state, int):
            assert encoder_ft_state > 0, "encoder_ft_state must be a positive integer or 'eval' or 'FPFT'"
            enable_fine_tuning(self.encoder, LoRAAllFineTuningStrategy(encoder_ft_state,large_rank_warning=False))
        else:
            assert encoder_ft_state in ["frozen", "FPFT"], "encoder_ft_state must be a positive integer or 'eval' or 'FPFT'"
            if encoder_ft_state == "frozen":
                for param in self.encoder.parameters():
                    param.requires_grad = False
        if isinstance(decoder_ft_state, int):
            assert decoder_ft_state > 0, "decoder_ft_state must be a positive integer or 'eval' or 'FPFT'"
            enable_fine_tuning(self.decoder, LoRAAllFineTuningStrategy(decoder_ft_state,large_rank_warning=False))
        else:
            assert decoder_ft_state in ["frozen", "FPFT"], "decoder_ft_state must be a positive integer or 'eval' or 'FPFT'"
            if decoder_ft_state == "frozen":
                for param in self.decoder.parameters():
                    param.requires_grad = False
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
            x = self.decoder(x)
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
                x_out_chunk = self.decoder(x_out_chunk)
                x_out_chunks.append(x_out_chunk)
            x = torch.cat(x_out_chunks, dim=0)
            if return_kl_element:
                kl_elem = torch.cat(kl_elem_chunks, dim=0)
        x = rearrange(
            x,
            "(B C U V W) 1 Xc Yc Zc -> B C (U Xc) (V Yc) (W Zc)",
            B=b,
            C=c,
            U=u,
            V=v,
            W=w,
        )
        if return_kl_element:
            return x, kl_elem
        return x
    
    def save_separate_weights(self, encoder_path: str, decoder_path: str):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)