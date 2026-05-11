from .core import P3DEncoder, P3DDecoder
from .kl import KLP3DEncoder
import copy

_P3D_TRANSFORMER_CONFIGS = {
    "S": dict(
        hidden_size=64,
        window_size=[4, 4, 4, 4, 4],
        num_heads=[4, 4, 4, 4, 4],
        depth=[2, 2, 2, 2, 2],
        feature_embedding_dim=[32, 32, 64],
        num_downsampling_layers=2,
        time_embedding_dim=64,
        num_groups=16,
        repetitions=2,
    ),
    "B": dict(
        hidden_size=128,
        window_size=[4, 4, 4, 4, 4],
        num_heads=[4, 4, 4, 4, 4],
        depth=[2, 2, 2, 2, 2],
        feature_embedding_dim=[64, 128, 128],
        num_downsampling_layers=2,
        time_embedding_dim=64,
        num_groups=32,
        repetitions=2,
    ),
    "L": dict(
        hidden_size=256,
        window_size=[4, 4, 4, 4, 4],
        num_heads=[8, 8, 8, 8, 8],
        depth=[2, 2, 2, 2, 2],
        feature_embedding_dim=[128, 256, 256],
        num_downsampling_layers=2,
        time_embedding_dim=64,
        num_groups=32,
        repetitions=2,
    ),
    "XL": dict(
        hidden_size=256,
        window_size=[4, 4, 4, 4, 4],
        num_heads=[8, 8, 8, 8, 8],
        depth=[2, 2, 2, 2, 2],
        feature_embedding_dim=[256, 256, 256],
        num_downsampling_layers=2,
        time_embedding_dim=64,
        num_groups=64,
        repetitions=4,
    ),
}


def P3D_Configs(
    size: str = "S",
):
    return copy.deepcopy(_P3D_TRANSFORMER_CONFIGS[size])


def _P3DEncoder(size, **kwargs):
    config = P3D_Configs(size)
    config.update(kwargs)
    return P3DEncoder(**config)


def P3DEncoder_S(**kwargs):
    return _P3DEncoder("S", **kwargs)


def P3DEncoder_B(**kwargs):
    return _P3DEncoder("B", **kwargs)


def P3DEncoder_L(**kwargs):
    return _P3DEncoder("L", **kwargs)


def P3DEncoder_XL(**kwargs):
    return _P3DEncoder("XL", **kwargs)


def _KLP3DEncoder(size, **kwargs):
    config = P3D_Configs(size)
    config.update(kwargs)
    return KLP3DEncoder(**config)


def KLP3DEncoder_S(**kwargs):
    return _KLP3DEncoder("S", **kwargs)


def KLP3DEncoder_B(**kwargs):
    return _KLP3DEncoder("B", **kwargs)


def KLP3DEncoder_L(**kwargs):
    return _KLP3DEncoder("L", **kwargs)


def KLP3DEncoder_XL(**kwargs):
    return _KLP3DEncoder("XL", **kwargs)


def _P3DDecoder(size, **kwargs):
    config = P3D_Configs(size)
    config.update(kwargs)
    return P3DDecoder(**config)


def P3DDecoder_S(**kwargs):
    return _P3DDecoder("S", **kwargs)


def P3DDecoder_B(**kwargs):
    return _P3DDecoder("B", **kwargs)


def P3DDecoder_L(**kwargs):
    return _P3DDecoder("L", **kwargs)


def P3DDecoder_XL(**kwargs):
    return _P3DDecoder("XL", **kwargs)
