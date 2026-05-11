import torch.nn as nn
from typing import Optional, Sequence, Union
from .modules import PixelShuffle3d


class ConditionedEncoder3DBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.gn_1 = nn.GroupNorm(num_groups, in_channels)
        self.activation_1 = nn.GELU()
        self.conv_1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)

        self.gn_2 = nn.GroupNorm(num_groups, in_channels)
        self.activation_2 = nn.GELU()
        self.conv_2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x_res = x
        x = self.gn_1(x)
        x = self.activation_1(x)
        x = self.conv_1(x)
        x = self.gn_2(x)
        x = self.activation_2(x)
        x = self.conv_2(x)
        x = x + x_res
        return x


class ConditionedEncoder3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        feature_embedding_dim: Union[int, Sequence[int]],
        num_downsampling_layers: int,
        embedding_dim: int, #need to remove, useless
        repetitions: int = 1,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels

        if isinstance(feature_embedding_dim, Sequence):
            self.feature_embedding_dim = feature_embedding_dim
            if len(self.feature_embedding_dim) != num_downsampling_layers + 1:
                raise ValueError(
                    "Length of feature_embedding_dim sequence must be equal to num_downsampling_layers + 1"
                )
        else:
            self.feature_embedding_dim = [
                feature_embedding_dim * 2**i for i in range(num_downsampling_layers + 1)
            ]

        self.repetitions = repetitions
        self.num_downsampling_layers = num_downsampling_layers
        self.embedding_dim = embedding_dim
        self.feature_embed = nn.Conv3d(
            in_channels, self.feature_embedding_dim[0], 3, 1, 1
        )
        self.downsampling_layers = nn.ModuleList()
        for i in range(num_downsampling_layers):
            self.downsampling_layers.append(
                nn.Conv3d(
                    self.feature_embedding_dim[i],
                    self.feature_embedding_dim[i + 1],
                    3,
                    2,
                    1,
                )
            )
        self.blocks = nn.ModuleList()
        for i in range(num_downsampling_layers - 1):
            self.blocks.extend(
                [
                    ConditionedEncoder3DBlock(
                        self.feature_embedding_dim[i + 1],
                        num_groups=num_groups,
                    )
                    for _ in range(repetitions)
                ]
            )

    def forward(self, x):
        x = self.feature_embed(x)
        x = self.downsampling_layers[0](x)
        for i in range(self.num_downsampling_layers - 1):
            for j in range(self.repetitions):
                x = self.blocks[i * self.repetitions + j](x)
            x = self.downsampling_layers[i + 1](x)
        return x


ConditionedDecoder3DBlock = ConditionedEncoder3DBlock


class DecoderUpsamplingBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, factor: Optional[int] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear_conv = nn.Conv3d(in_channels, out_channels * 8, 1)
        self.shuffle = PixelShuffle3d(2)

    def forward(self, x):
        x = self.linear_conv(x)
        x = self.shuffle(x)
        return x


class ConditionedDecoder3D(nn.Module):

    def __init__(
        self,
        out_channels: int,
        feature_embedding_dim: Union[int, Sequence[int]],
        num_upsampling_layers: int,
        embedding_dim: int,# need to remove, useless
        repetitions: int = 1,
        features_first_layer: int = None,
        num_groups: int = 32,
    ):
        super().__init__()
        self.out_channels = out_channels

        if isinstance(feature_embedding_dim, Sequence):
            self.feature_embedding_dim = feature_embedding_dim
            if len(self.feature_embedding_dim) != num_upsampling_layers + 1:
                raise ValueError(
                    "Length of feature_embedding_dim sequence must be equal to num_upsampling_layers + 1"
                )
        else:
            self.feature_embedding_dim = [
                feature_embedding_dim * 2 ** (num_upsampling_layers - i)
                for i in range(num_upsampling_layers + 1)
            ]

        self.num_upsampling_layers = num_upsampling_layers
        self.embedding_dim = embedding_dim
        self.repetitions = repetitions

        self.decompress = nn.Conv3d(
            self.feature_embedding_dim[-1], out_channels, 3, 1, 1
        )

        self.blocks = nn.ModuleList()
        for i in range(num_upsampling_layers - 1):
            self.blocks.extend(
                [
                    ConditionedDecoder3DBlock(
                        self.feature_embedding_dim[i + 1],
                        num_groups=num_groups,
                    )
                    for _ in range(self.repetitions)
                ]
            )

        if features_first_layer is None:
            features_first_layer = self.feature_embedding_dim[0]

        self.upsampling_layers = nn.ModuleList()

        local_feature_dim = self.feature_embedding_dim[1]
        self.upsampling_layers.append(
            DecoderUpsamplingBlock(
                features_first_layer, local_feature_dim
            )
        )
        for i in range(num_upsampling_layers - 1):
            local_feature_dim = self.feature_embedding_dim[i + 1]
            self.upsampling_layers.append(
                DecoderUpsamplingBlock(
                    local_feature_dim,
                    self.feature_embedding_dim[i + 2],
                )
            )

    def forward(self, x):

        x = self.upsampling_layers[0](x)

        for i in range(self.num_upsampling_layers - 1):
            for j in range(self.repetitions):
                x = self.blocks[i * self.repetitions + j](x)
            x = self.upsampling_layers[i + 1](x)

        x = self.decompress(x)

        return x
