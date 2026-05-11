import torch
from torch import nn
from torch.nn import Module
from torch.utils.checkpoint import checkpoint
from typing import Union, Literal,Dict,Optional,Literal
from collections import OrderedDict
from ..architecture.p3d import _KLP3DEncoder,_P3DDecoder
from ..architecture.p3d.skip_wrapper  import KLP3DEncoderSkip,P3DDecoderSkip
from ..architecture.p3d.kl import DiagonalGaussianDistribution
from ..architecture.downstream import SequentialModel
from ..utils import load_weights
from GIFt.strategies.lora import LoRAAllFineTuningStrategy
from GIFt import enable_fine_tuning
from einops import rearrange

def default_subnetwork(size: Literal["S", "B", "L"],n_input_channels: int) -> Module:
    if size == "S":
        return SequentialModel(
            in_dim=n_input_channels*256,
            n_layers=4,
            attention_method="hyper",
            num_heads=8,
            hidden_size=144,
        )
    elif size == "B":
        return SequentialModel(
            in_dim=n_input_channels*512,
            n_layers=6,
            attention_method="hyper",
            num_heads=8,
            hidden_size=176,
        )
    elif size == "L":
        return SequentialModel(
            in_dim=n_input_channels*1024,
            n_layers=8,
            attention_method="hyper",
            num_heads=8,
            hidden_size=224,
        )
    else:
        raise ValueError(f"Unknown size: {size}")
        
class TadpoleDFT(Module):

    def __init__(self, 
                 size: Literal["S", "B", "L"],
                 input_channels: int = 3,
                 subnetwork: Union[Literal["default"], None, Module] = "default",
                weight_encoder: Optional[Union[str,Dict,OrderedDict]] = None,
                weight_decoder: Optional[Union[str,Dict,OrderedDict]] = None,
                encoder_ft_state: Union[Literal["frozen","FPFT"],int] = 32,
                decoder_ft_state: Union[Literal["frozen","FPFT"],int] = 32,
                latent_type: Literal["sample", "mode"] = "sample",
                encoder_crop_size: Optional[int] = None,
                max_internal_batchsize: Optional[int] = None,
                encoder_activation_ckpt: bool = False,
                decoder_activation_ckpt: bool = False, 
                subnetwork_activation_ckpt: bool = False,
                ):
        """
        Tadpole Dynamic Fine-Tuning (DFT) Model
        Input shape: (B, C, X, Y, Z); Output shape: (B, C, X, Y, Z)
        
        
        Args:
            size (Literal["S", "B", "L"]): Size of the model, one of "S", "B", or "L".
            input_channels (int): Number of input channels. Default is 3.
            subnetwork (Union[Literal["default"], None, Module]): Subnetwork to apply in the latent space. Can be "default" to use a default subnetwork based on the model size. If None, no subnetwork will be applied. If a torch.nn.Module is provided, it will be used as the subnetwork. Default is "default".
            weight_encoder (Optional[Union[str,Dict,OrderedDict]]): Path to encoder weights or state dict for encoder. If None, encoder will be randomly initialized. Default is None.
            weight_decoder (Optional[Union[str,Dict,OrderedDict]]): Path to decoder weights or state dict for decoder. If None, decoder will be randomly initialized. Default is None.
            encoder_ft_state (Union[Literal["frozen","FPFT"],int]): Fine-tuning state for encoder. Can be a positive integer indicating the rank for LoRA fine-tuning, "frozen" to freeze the encoder weights, or "FPFT" to enable full-parameter fine-tuning. Default is 32.
            decoder_ft_state (Union[Literal["frozen","FPFT"],int]): Fine-tuning state for decoder. Can be a positive integer indicating the rank for LoRA fine-tuning, "frozen" to freeze the decoder weights, or "FPFT" to enable full-parameter fine-tuning. Default is 32.
            latent_type (Literal["sample", "mode"]): How to sample from the latent distribution, either "sample" or "mode". Default is "sample".
            encoder_crop_size (Optional[int]): Size to crop input for encoder. If None, no cropping will be applied and the entire input will be processed as a single crop. Default is None.
            max_internal_batchsize (Optional[int]): Maximum batch size for internal processing. If None, all crops will be processed in a single batch. Default is None.
            encoder_activation_ckpt (bool): Whether to apply activation checkpointing to the encoder. Default is False.
            decoder_activation_ckpt (bool): Whether to apply activation checkpointing to the decoder. Default is False.
            subnetwork_activation_ckpt (bool): Whether to apply activation checkpointing to the subnetwork. Default is False.
        """
        super().__init__()
        
        # build encoder and decoder
        assert size in ["S", "B", "L"], "size must be one of 'S', 'B', 'L'"
        self.encoder = _KLP3DEncoder(size)
        self.decoder = _P3DDecoder(size)
        if weight_encoder is not None:
            self.encoder.load_state_dict(load_weights(weight_encoder, "encoder"))
        if weight_decoder is not None:
            self.decoder.load_state_dict(load_weights(weight_decoder, "decoder"))
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
        # add skip connections and activation checkpointing for encoder and decoder
        self.encoder = KLP3DEncoderSkip(self.encoder, encoder_activation_ckpt)
        self.decoder = P3DDecoderSkip(self.decoder, decoder_activation_ckpt)  
        # build subnetwork
        if subnetwork == "default":
            self.subnetwork = default_subnetwork(size, input_channels)
        else:
            assert subnetwork is None or isinstance(subnetwork, Module), "subnetwork must be 'default', None, or a torch.nn.Module"
            self.subnetwork = subnetwork
        if subnetwork is not None:
            self.latent_residual_scale = nn.Parameter(torch.tensor(1.0))
        # set other parameters
        assert latent_type in ["sample", "mode"], "latent_type must be one of 'sample' or 'mode'"
        self.latent_type = latent_type
        self.encoder_crop_size = encoder_crop_size if encoder_crop_size is not None else 1e6
        self.max_internal_batchsize = max_internal_batchsize
        self.subnetwork_activation_ckpt = subnetwork_activation_ckpt
        
    def latent_sample(self, dist: DiagonalGaussianDistribution) -> torch.Tensor:
        if self.latent_type == "sample":
            return dist.sample()
        elif self.latent_type == "mode":
            return dist.mode()
        else:
            raise ValueError(f"Unknown latent_type: {self.latent_type}")
        

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x: (B, C, X, Y, Z)
        c = x.shape[1]
        u = x.shape[2] // self.encoder_crop_size
        v = x.shape[3] // self.encoder_crop_size
        w = x.shape[4] // self.encoder_crop_size
        u = max(u, 1)
        v = max(v, 1)
        w = max(w, 1)
        x = rearrange(
            x,
            "B C (U Xc) (V Yc) (W Zc) -> (B C U V W) 1 Xc Yc Zc",
            U=u,
            V=v,
            W=w,
        )
        if self.max_internal_batchsize is not None and x.shape[0] > self.max_internal_batchsize:
            x_chunks = torch.chunk(x, chunks=(x.shape[0] // self.max_internal_batchsize) + 1, dim=0)
            encoded_chunks = []
            res_chunks = []
            for x_chunk in x_chunks:
                encoded_chunk, res_chunk = self.encoder(x_chunk, latent_type=self.latent_type)
                encoded_chunks.append(encoded_chunk)
                res_chunks.append(res_chunk)
            x = torch.cat(encoded_chunks, dim=0)
        else:
            x, res = self.encoder(x,latent_type=self.latent_type)
        if self.subnetwork is not None:
            x = rearrange(
                x,
                "(B C U V W) Cl Xl Yl Zl -> B (C Cl) (U Xl) (V Yl) (W Zl)",
                C=c,
                U=u,
                V=v,
                W=w,
            )
            if self.subnetwork_activation_ckpt is not None:
                if self.subnetwork_activation_ckpt:
                    x = self.latent_residual_scale * x + checkpoint(self.subnetwork, x, *args, **kwargs)
                else:
                    x = self.latent_residual_scale * x + self.subnetwork(x, *args, **kwargs)
            x = rearrange(
                x,
                "B (C Cl) (U Xl) (V Yl) (W Zl) -> (B C U V W) Cl Xl Yl Zl",
                C=c,
                U=u,
                V=v,
                W=w,
            )
        if self.max_internal_batchsize is not None and x.shape[0] > self.max_internal_batchsize:
            x_chunks = torch.chunk(x, chunks=(x.shape[0] // self.max_internal_batchsize) + 1, dim=0)
            decoded_chunks = []
            for x_chunk in x_chunks:
                decoded_chunk = self.decoder(x_chunk, res_chunks.pop(0))
                decoded_chunks.append(decoded_chunk)
            x = torch.cat(decoded_chunks, dim=0)
        else:
            x = self.decoder(x, res)
        x = rearrange(
            x,
            "(B C U V W) 1 Xc Yc Zc -> B C (U Xc) (V Yc) (W Zc)",
            C=c,
            U=u,
            V=v,
            W=w,
        )
        return x