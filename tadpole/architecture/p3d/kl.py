import torch
import numpy as np
from typing import Literal, Union
from .core import P3DEncoder


class DiagonalGaussianDistribution:
    """

    Modified from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py#L24
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl_elem(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * (
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
                )
            else:
                return 0.5 * (
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar
                )
    
    def kl_dim(self, other=None):
        """
        kl per dimension
        """
        dims = list(range(1, self.mean.ndim))
        return self.kl_elem(other).mean(dims)
    
    def kl_sample(self,other=None):
        dims = list(range(1, self.mean.ndim))
        return self.kl_elem(other).sum(dims)

    def nll(self, sample):
        dims = list(range(1, sample.ndim))
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class KLP3DEncoder(P3DEncoder):

    def __init__(
        self,
        window_size=8,
        hidden_size=1152,
        max_hidden_size=2048,
        depth=...,
        num_heads=16,
        mlp_ratio=4,
        periodic=False,
        shift=False,
        feature_embedding_dim=64,
        num_downsampling_layers=3,
        time_embedding_dim=64,
        num_groups=32,
        repetitions=1,
        ckpt_path=None,
        ckpt_prefix="model.encoder.",
    ):
        super().__init__(
            window_size,
            hidden_size,
            max_hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            periodic,
            shift,
            feature_embedding_dim,
            num_downsampling_layers,
            time_embedding_dim,
            num_groups,
            repetitions,
            ckpt_path,
            ckpt_prefix,
        )
        self.to_latent = torch.nn.Conv3d(
            self.latent_size, self.latent_size * 2, kernel_size=1
        )

    def forward(self, x: torch.Tensor, latent_type: Literal["sample", 
                                                            "mode", 
                                                            "distribution",
                                                            "mean_std"] = "sample") -> Union[torch.Tensor, DiagonalGaussianDistribution]:
        x=self.to_latent(super().forward(x))
        if latent_type=="mean_std":
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
