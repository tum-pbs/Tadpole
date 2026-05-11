from torch import nn
from .conv import ConditionedEncoder3D, ConditionedDecoder3D
from .transformer import P3DTransformerEncoder, P3DTransformerDecoder
from .core import P3DEncoder, P3DDecoder
from .kl import KLP3DEncoder, DiagonalGaussianDistribution
from typing import Union, Literal
import torch
from torch.utils.checkpoint import checkpoint


class ConvEncoderSkip(nn.Module):

    def __init__(
        self,
        conv_encoder: ConditionedEncoder3D,
    ):
        super().__init__()
        self.conv_encoder = conv_encoder

    def forward(self, x):
        x = self.conv_encoder.feature_embed(x)
        res_list = [x]
        x = self.conv_encoder.downsampling_layers[0](x)
        for i in range(self.conv_encoder.num_downsampling_layers - 1):
            for j in range(self.conv_encoder.repetitions):
                x = self.conv_encoder.blocks[i * self.conv_encoder.repetitions + j](x)
            res_list.append(x)
            x = self.conv_encoder.downsampling_layers[i + 1](x)
        # res_list.append(x)
        return x, res_list


class ConvDecoderSkip(nn.Module):

    def __init__(
        self,
        conv_decoder: ConditionedDecoder3D,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.conv_decoder = conv_decoder
        self.scales = nn.Parameter(torch.zeros(self.conv_decoder.num_upsampling_layers))
        self.use_checkpoint = use_checkpoint

    def forward(self, x, encoder_outputs):
        x = self.conv_decoder.upsampling_layers[0](x)
        x += encoder_outputs[::-1][0] * self.scales[0]
        for i in range(self.conv_decoder.num_upsampling_layers - 1):
            for j in range(self.conv_decoder.repetitions):
                if self.use_checkpoint:
                    x = checkpoint(self.conv_decoder.blocks[i * self.conv_decoder.repetitions + j], x)
                else:
                    x = self.conv_decoder.blocks[i * self.conv_decoder.repetitions + j](x)
            x = self.conv_decoder.upsampling_layers[i + 1](x)
            x += encoder_outputs[::-1][i + 1] * self.scales[i + 1]
        x = self.conv_decoder.decompress(x)
        return x


class P3DTransformerEncoderSkip(nn.Module):

    def __init__(
        self,
        transformer: P3DTransformerEncoder,
    ):
        super().__init__()
        self.transformer = transformer

    def forward(self, x):
        """
        Encoding of P3D
        x: (N, C, H, W, D) tensor of spatial inputs (images or latent representations of images)
        """
        residuals_list = []
        for encoder_level, downsample in zip(
            self.transformer.encoder_levels, self.transformer.downsamples
        ):
            #           print(f"encoder_level input shape: {x.shape}")
            out_enc_level = encoder_level(x)
            residuals_list.append(out_enc_level)
            #            print(f"encoder_level output shape: {out_enc_level.shape}")
            x = downsample(out_enc_level)
        #        print(f"latent input shape: {x.shape}")
        x = self.transformer.latent(x)
        return x, residuals_list


class P3DTransformerDecoderSkip(nn.Module):

    def __init__(
        self,
        transformer: P3DTransformerDecoder,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.scales = nn.Parameter(torch.zeros(len(self.transformer.upsamples)))
        self.use_checkpoint = use_checkpoint

    def forward(self, x, encoder_outputs):
        # decoder: reverse order except the first (which is up1_0 and decoder_level_0)
        for i, (upsample, decoder_level) in enumerate(
            zip(self.transformer.upsamples, self.transformer.decoder_levels)
        ):
            #            print(f"decoder_level input shape: {x.shape}")
            x = upsample(x)
            x += encoder_outputs[::-1][i] * self.scales[i]
            #            print(f"upsample output shape: {x.shape}")
            if self.use_checkpoint:
                x = checkpoint(decoder_level, x)
            else:
                x = decoder_level(x)
        # output
        #        print(f"final_layer input shape: {x.shape}")
        x = self.transformer.final_layer(x)
        return x


class P3DEncoderSkip(nn.Module):

    def __init__(
        self,
        encoder: P3DEncoder,
    ):
        super().__init__()
        self.conv_encoder = ConvEncoderSkip(encoder.conv_encoder)
        self.transformer_encoder = P3DTransformerEncoderSkip(encoder.transformer_encoder)
    def forward(
        self,
        x: torch.Tensor,
    ):
        ## Conv encoding
        x, res_conv = self.conv_encoder(x)
        x, res_transformer = self.transformer_encoder(x)
        ## Sequence Modeling
        return x, [res_conv, res_transformer]


class P3DDecoderSkip(nn.Module):

    def __init__(
        self,
        decoder: P3DDecoder,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.conv_decoder = ConvDecoderSkip(decoder.conv_decoder,use_checkpoint=use_checkpoint)
        self.transformer_decoder = P3DTransformerDecoderSkip(decoder.transformer_decoder,use_checkpoint=use_checkpoint)

    def forward(
        self,
        x: torch.Tensor,
        encoder_residuals: list,
    ):
        ## Sequence Modeling
        x = self.transformer_decoder(x, encoder_residuals[1])
        ## Conv decoding
        x = self.conv_decoder(x, encoder_residuals[0])
        return x


class KLP3DEncoderSkip(nn.Module):

    def __init__(
        self,
        encoder: KLP3DEncoder,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.conv_encoder = ConvEncoderSkip(encoder.conv_encoder)
        self.transformer_encoder = P3DTransformerEncoderSkip(encoder.transformer_encoder)
        self.to_latent=encoder.to_latent
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        x: torch.Tensor,
        latent_type: Literal["sample", 
                            "mode", 
                            "distribution",
                            "mean_std"] = "sample"
    ) -> Union[torch.Tensor, DiagonalGaussianDistribution]:
        ## Conv encoding
        if self.use_checkpoint:
            x, res_conv = checkpoint(self.conv_encoder, x)
            x, res_transformer = checkpoint(self.transformer_encoder, x)
        else:
            x, res_conv = self.conv_encoder(x)
            x, res_transformer = self.transformer_encoder(x)
        if latent_type=="mean_std":
            return x, [res_conv, res_transformer]
        x = self.to_latent(x)
        x = DiagonalGaussianDistribution(x)
        if latent_type == "sample":
            return x.sample(), [res_conv, res_transformer]
        elif latent_type == "mode":
            return x.mode(), [res_conv, res_transformer]
        elif latent_type == "distribution":
            return x, [res_conv, res_transformer]
        else:
            raise ValueError(f"Unknown latent_type: {latent_type}")
