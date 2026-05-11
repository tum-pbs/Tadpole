import torch
from collections import OrderedDict
from typing import Optional, Sequence, Union
from diffusers import ModelMixin
from .conv import ConditionedEncoder3D, ConditionedDecoder3D
from .transformer import P3DTransformerEncoder, P3DTransformerDecoder


class P3DEncoder(ModelMixin):

    def __init__(
        self,
        window_size: Union[int, Sequence[int]] = 8,
        hidden_size: int = 1152,
        max_hidden_size: int = 2048,
        depth=(2, 4, 4, 6, 4, 4, 2),
        num_heads: Union[int, Sequence[int]] = 16,
        mlp_ratio: float = 4.0,
        periodic: bool = False,
        shift: bool = False,
        feature_embedding_dim: Union[int, Sequence[int]] = 64,
        num_downsampling_layers: int = 3,
        time_embedding_dim: int = 64,
        num_groups: int = 32,
        repetitions: int = 1,
        ckpt_path: Optional[str] = None,
        ckpt_prefix: str = "model.encoder.",
        in_channels: int = 1,
    ):
        super().__init__()
        # "hidden_size must be equal to the last element of feature_embedding_dim"
        self.transformer_encoder = P3DTransformerEncoder(
            window_size=window_size,
            hidden_size=hidden_size,
            max_hidden_size=max_hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            periodic=periodic,
            shift=shift,
        )
        self.num_downsampling_layers = num_downsampling_layers
        self.conv_encoder = ConditionedEncoder3D(
            in_channels=in_channels,
            feature_embedding_dim=feature_embedding_dim,
            num_downsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            num_groups=num_groups,
            repetitions=repetitions,
        )
        self.latent_size = self.transformer_encoder.latent_size
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, prefix=ckpt_prefix)

    def forward(
        self,
        x: torch.Tensor,
    ):
        ## Conv encoding
        x = self.conv_encoder(x)
        x = self.transformer_encoder(x)
        ## Sequence Modeling
        return x

    def init_from_ckpt(self, 
                       pretrained_weights: Union[str,OrderedDict],
                       prefix: str = "model.encoder."
                       ):
        if isinstance(pretrained_weights,str):
            pretrained_weights = torch.load(pretrained_weights, map_location="cpu")["state_dict"]
        if prefix != "":
            encoder_weights = {k.replace(prefix, ""): v for k, v in pretrained_weights.items() if prefix in k}
        self.load_state_dict(encoder_weights, strict=True)

class P3DDecoder(ModelMixin):

    def __init__(
        self,
        window_size: Union[int, Sequence[int]] = 8,
        hidden_size: int = 1152,
        max_hidden_size: int = 2048,
        depth=(2, 4, 4, 6, 4, 4, 2),
        num_heads: Union[int, Sequence[int]] = 16,
        mlp_ratio: float = 4.0,
        periodic: bool = False,
        shift: bool = False,
        feature_embedding_dim: Union[int, Sequence[int]] = 64,
        num_downsampling_layers: int = 3,
        time_embedding_dim: int = 64,
        num_groups: int = 32,
        repetitions: int = 1,
        ckpt_path: Optional[str] = None,
        ckpt_prefix: str = "model.decoder.",
        out_channels: int = 1,
    ):
        super().__init__()
        self.transformer_decoder = P3DTransformerDecoder(
            window_size=window_size,
            hidden_size=hidden_size,
            max_hidden_size=max_hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            periodic=periodic,
            shift=shift,
        )
        self.num_downsampling_layers = num_downsampling_layers
        self.latent_size = hidden_size * 2 ** (len(depth) // 2)
        self.conv_decoder = ConditionedDecoder3D(
            out_channels=out_channels,
            feature_embedding_dim=feature_embedding_dim[::-1],
            num_upsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            features_first_layer=feature_embedding_dim[-1],
            num_groups=num_groups,
            repetitions=repetitions,
        )
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, prefix=ckpt_prefix)

    def forward(self, x):
        # region partition
        x = self.transformer_decoder(x)
        reconstructed = self.conv_decoder(x)
        return reconstructed
    
    def init_from_ckpt(self, 
                       pretrained_weights: Union[str,OrderedDict],
                       prefix: str = "model.decoder."
                       ):
        if isinstance(pretrained_weights,str):
            pretrained_weights = torch.load(pretrained_weights, map_location="cpu")["state_dict"]
        if prefix != "":
            decoder_weights = {k.replace(prefix, ""): v for k, v in pretrained_weights.items() if prefix in k}
        self.load_state_dict(decoder_weights, strict=True)