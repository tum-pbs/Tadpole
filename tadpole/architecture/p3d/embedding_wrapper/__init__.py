from ..core import P3DEncoder, P3DDecoder
from ..kl import KLP3DEncoder, DiagonalGaussianDistribution
from .conv import ConvEncoder3DEmbedded, ConvDecoder3DEmbedded
from .transformer import P3DTransformerEncoderEmbedded, P3DTransformerDecoderEmbedded
from torch import nn
from typing import Literal
import torch

class P3DEncoderEmbedded(nn.Module):
    
    def __init__(
        self,
        p3d_encoder: P3DEncoder,
        num_classes: int,
    ):
        super().__init__()
        self.p3d_encoder = p3d_encoder
        self.p3d_encoder.conv_encoder = ConvEncoder3DEmbedded(
            self.p3d_encoder.conv_encoder,
            num_classes=num_classes,
        )
        self.p3d_encoder.transformer_encoder = P3DTransformerEncoderEmbedded(
            self.p3d_encoder.transformer_encoder,
            num_classes=num_classes,
        )
    def forward(self, x: torch.Tensor, class_labels: torch.Tensor):
        x = self.p3d_encoder.conv_encoder(x, class_labels)
        x = self.p3d_encoder.transformer_encoder(x, class_labels)
        return x

class KLP3DEncoderEmbedded(nn.Module):
    
    def __init__(
        self,
        kl_p3d_encoder: KLP3DEncoder,
        num_classes: int,
    ):
        super().__init__()
        self.p3d_encoder = kl_p3d_encoder
        self.p3d_encoder.conv_encoder = ConvEncoder3DEmbedded(
            self.p3d_encoder.conv_encoder,
            num_classes=num_classes,
        )
        self.p3d_encoder.transformer_encoder = P3DTransformerEncoderEmbedded(
            self.p3d_encoder.transformer_encoder,
            num_classes=num_classes,
        )
        
    def forward(self, x: torch.Tensor, 
                class_labels: torch.Tensor,
                latent_type: Literal["sample", 
                                     "mode", 
                                     "distribution"
                                      "mean_std"
                                     ] = "sample"):
        x = self.p3d_encoder.conv_encoder(x, class_labels)
        x = self.p3d_encoder.transformer_encoder(x, class_labels)
        x = self.p3d_encoder.to_latent(x) 
        if latent_type == "mean_std":
            return x
        dist = DiagonalGaussianDistribution(x)
        if latent_type == "sample":
            return dist.sample()
        elif latent_type == "mode":
            return dist.mode()
        elif latent_type == "distribution":
            return dist
        else:
            raise ValueError(f"Unknown latent_type: {latent_type}")
    
class P3DDecoderEmbedded(nn.Module):
    
    def __init__(
        self,
        p3d_decoder: P3DDecoder,
        num_classes: int,
    ):
        super().__init__()
        self.p3d_decoder = p3d_decoder
        self.p3d_decoder.transformer_decoder = P3DTransformerDecoderEmbedded(
            self.p3d_decoder.transformer_decoder,
            num_classes=num_classes,
        )
        self.p3d_decoder.conv_decoder = ConvDecoder3DEmbedded(
            self.p3d_decoder.conv_decoder,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor):
        x = self.p3d_decoder.transformer_decoder(x, class_labels)
        x = self.p3d_decoder.conv_decoder(x, class_labels)
        return x
         
class P3DEncoderConvEmbedded(nn.Module):
    
    def __init__(
        self,
        p3d_encoder: P3DEncoder,
        num_classes: int,
    ):
        super().__init__()
        self.p3d_encoder = p3d_encoder
        self.p3d_encoder.conv_encoder = ConvEncoder3DEmbedded(
            self.p3d_encoder.conv_encoder,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor):
        x = self.p3d_encoder.conv_encoder(x, class_labels)
        x = self.p3d_encoder.transformer_encoder(x)
        return x

class KLP3DEncoderConvEmbedded(nn.Module):
    
    def __init__(
        self,
        kl_p3d_encoder: KLP3DEncoder,
        num_classes: int,
    ):
        super().__init__()
        self.p3d_encoder = kl_p3d_encoder
        self.p3d_encoder.conv_encoder = ConvEncoder3DEmbedded(
            self.p3d_encoder.conv_encoder,
            num_classes=num_classes,
        )
        
    def forward(self, x: torch.Tensor, 
                class_labels: torch.Tensor,
                latent_type: Literal["sample", "mode", "distribution","mean_std"] = "sample"):
        x = self.p3d_encoder.conv_encoder(x, class_labels)
        x = self.p3d_encoder.transformer_encoder(x)
        x = self.p3d_encoder.to_latent(x)
        if latent_type == "mean_std":
            return x
        dist = DiagonalGaussianDistribution(x)
        if latent_type == "sample":
            return dist.sample()
        elif latent_type == "mode":
            return dist.mode()
        elif latent_type == "distribution":
            return dist
        else:
            raise ValueError(f"Unknown latent_type: {latent_type}")

class P3DDecoderConvEmbedded(nn.Module):
    
    def __init__(
        self,
        p3d_decoder: P3DDecoder,
        num_classes: int,
    ):
        super().__init__()
        self.p3d_decoder = p3d_decoder
        self.p3d_decoder.conv_decoder = ConvDecoder3DEmbedded(
            self.p3d_decoder.conv_decoder,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor):
        x = self.p3d_decoder.transformer_decoder(x)
        x = self.p3d_decoder.conv_decoder(x, class_labels)
        return x