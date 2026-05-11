from ..conv import ConditionedEncoder3D,ConditionedEncoder3DBlock,ConditionedDecoder3D
from torch import nn
from .embeder import AdaLayerNormZero

class ConvEncoder3DBlockEmbedded(nn.Module):
    
    def __init__(self, 
                 conv_encoder_block: ConditionedEncoder3DBlock,
                 num_classes: int,
                 ):
        super().__init__()    
        self.conv_encoder_block = conv_encoder_block
        self.embedder = AdaLayerNormZero(
            embedding_dim=self.conv_encoder_block.in_channels,
            num_classes=num_classes,
            chunk_size=2,
        )
        
    def forward(self, x, classes):
        scale, shift = self.embedder(classes)       
        x_res = x
        x = self.conv_encoder_block.gn_1(x)
        x = self.conv_encoder_block.activation_1(x)
        x = self.conv_encoder_block.conv_1(x)
        x = self.conv_encoder_block.gn_2(x)
        x = x * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        x = self.conv_encoder_block.activation_2(x)
        x = self.conv_encoder_block.conv_2(x)
        x = x + x_res
        return x

class ConvEncoder3DEmbedded(nn.Module):
    
    def __init__(self, 
                 conv_encoder: ConditionedEncoder3D,
                 num_classes: int,
                 ):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.conv_encoder.blocks = nn.ModuleList([
            ConvEncoder3DBlockEmbedded(block,num_classes) for block in self.conv_encoder.blocks
        ])
        
    def forward(self, x, classes):
        x = self.conv_encoder.feature_embed(x)
        x = self.conv_encoder.downsampling_layers[0](x)
        for i in range(self.conv_encoder.num_downsampling_layers - 1):
            for j in range(self.conv_encoder.repetitions):
                x = self.conv_encoder.blocks[i * self.conv_encoder.repetitions + j](x, classes)
            x = self.conv_encoder.downsampling_layers[i + 1](x)
        return x

ConvDecoder3DBlockEmbedded=ConvEncoder3DBlockEmbedded


class ConvDecoder3DEmbedded(nn.Module):
    
    def __init__(self, 
                 conv_decoder: ConditionedDecoder3D,
                 num_classes: int,
                 ):
        super().__init__()
        self.conv_decoder = conv_decoder
        self.conv_decoder.blocks = nn.ModuleList([
            ConvDecoder3DBlockEmbedded(block,num_classes) for block in self.conv_decoder.blocks
        ])
        
    def forward(self, x, classes):
        x = self.conv_decoder.upsampling_layers[0](x)

        for i in range(self.conv_decoder.num_upsampling_layers - 1):
            for j in range(self.conv_decoder.repetitions):
                x = self.conv_decoder.blocks[i * self.conv_decoder.repetitions + j](x, classes)
            x = self.conv_decoder.upsampling_layers[i + 1](x)

        x = self.conv_decoder.decompress(x)

        return x