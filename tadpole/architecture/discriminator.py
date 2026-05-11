import torch.nn as nn
from typing import Literal
from .p3d import _P3DEncoder

class PatchDiscriminator(nn.Module):

    def __init__(
        self,
        backbone_network: nn.Module,
        latent_size: int,
    ):
        super().__init__()
        self.model = nn.Sequential(
            backbone_network,
            nn.Conv3d(latent_size, 1, kernel_size=1),
        )

    def forward(self, x, *args): # need to change
        x= self.model[0](x, *args)
        return self.model[1](x)
        

class P3DDiscriminator(PatchDiscriminator):

    def __init__(self, size: Literal["S", "B", "L", "XL"], in_channels: int = 1):
        backbone_network = _P3DEncoder(size, in_channels=in_channels)
        super().__init__(backbone_network, backbone_network.latent_size)


