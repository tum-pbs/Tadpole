import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Tuple
from typing import Optional, Sequence, Union, Tuple
from timm.layers import DropPath

from .utils import window_partition, window_reverse
from .modules import PixelShuffle3d


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x


class LayerNorm3d(nn.LayerNorm):
    def __init__(self, norm_shape, eps=1e-6, affine=True):
        super().__init__(norm_shape, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class PosEmbMLPSwinv3D(nn.Module):
    def __init__(
        self,
        window_size,
        pretrained_window_size,
        num_heads,
        ct_correct=False,
        no_log=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_d = torch.arange(
            -(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(
                torch.meshgrid(
                    [relative_coords_h, relative_coords_w, relative_coords_d],
                    indexing="ij",
                )
            )
            .permute(1, 2, 3, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2*Wd-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
            relative_coords_table[:, :, :, 2] /= pretrained_window_size[2] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
            relative_coords_table[:, :, :, 2] /= self.window_size[2] - 1

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / np.log2(8)
            )

        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        seq_length = self.window_size[0] * self.window_size[1] * self.window_size[2]
        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.seq_length = seq_length
        self.register_buffer("relative_bias", relative_bias)
        self.ct_correct = ct_correct

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor, local_window_size):
        if self.deploy:
            input_tensor += self.relative_bias
            return input_tensor
        else:
            self.grid_exists = False

        if not self.grid_exists:
            self.grid_exists = True

            num_positions = (
                self.window_size[0] * self.window_size[1] * self.window_size[2]
            )
            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table
            ).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(num_positions, num_positions, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

            self.pos_emb = relative_position_bias.unsqueeze(0)
            self.relative_bias = self.pos_emb

        input_tensor += self.pos_emb
        return input_tensor



class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv3d(
                n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False
            ),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv3d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            PixelShuffle3d(2),
            nn.Conv3d(
                n_feat//4, n_feat //2, kernel_size=3, stride=1, padding=1, bias=False
            )
        )

    def forward(self, x):
        return self.body(x)


class FinalLayer(nn.Module):
    """
    The final layer of IPT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm3d(hidden_size, affine=False, eps=1e-6)
        self.out_proj = nn.Conv3d(
            hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x):
        x = self.out_proj(self.norm_final(x))
        return x


class WindowAttention3D(nn.Module):
    """
    Window attention based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        resolution: int = 0,
        attn_type: str = "v2",
        seq_length: int = 0,
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            resolution: feature resolution.
            seq_length: sequence length.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv3D(
            window_size=[resolution, resolution, resolution],
            pretrained_window_size=[resolution, resolution, resolution],
            num_heads=num_heads,
        )

        self.attn_type = attn_type

        assert self.attn_type in [
            "v1",
            "v2",
        ], f"attn_type {self.attn_type} not supported. Use 'v1' or 'v2'."

        if attn_type == "v2":
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

        self.resolution = resolution

    def forward(self, x, attn_mask=None):

        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_type == "v1":
            attn = (q @ k.transpose(-2, -1)) * self.scale

        elif self.attn_type == "v2":
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
            attn = attn * logit_scale

        attn = self.pos_emb_funct(attn, self.resolution**2)

        if attn_mask is not None:
            # Apply the attention mask is (precomputed for all layers in P3D forward() function)
            mask_shape = attn_mask.shape[0]
            attn = attn.view(
                B // mask_shape, mask_shape, self.num_heads, N, N
            ) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        return x


class PDEDiTBlock(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention

    Modifications:
        - 2D + time
        - AdaIN for diffusion conditioning
        - Conditioning via context tokens
        - Sliding window
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=8,
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
        """
        self.dim = dim
        self.norm1 = norm_layer(dim)

        self.cr_window = 1
        self.attn = WindowAttention3D(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            resolution=window_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.window_size = window_size

    def forward(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,
    ):

        x_msa = self.norm1(x)
        x_msa = self.attn(x_msa, attn_mask=attn_mask)
        x = x + self.drop_path(x_msa)
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        x = x + self.drop_path(x_mlp)
        return x


class P3DStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        periodic: bool = False,
        mlp_ratio: float = 4.0,
        shift: bool = False,
    ):
        super().__init__()

        self.dim = dim
        blocks = []
        for i in range(depth):
            block = PDEDiTBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.periodic = periodic
        self.window_size = window_size
        self.shift = shift

        self.shift_size = window_size // 2 if shift else 0

    def maybe_pad(self, hidden_states, height, width, depth):
        pad_depth = (self.window_size - depth % self.window_size) % self.window_size
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_depth, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def get_attn_mask(self, height, width, depth, dtype, device):

        if (
            height < self.window_size
            or width < self.window_size
            or depth < self.window_size
        ):
            return None

        if self.shift_size > 0 and not self.periodic:
            # calculate attention mask for shifted window multihead self attention

            padded_height = (
                height
                + (self.window_size - height % self.window_size) % self.window_size
            )
            padded_width = (
                width + (self.window_size - width % self.window_size) % self.window_size
            )
            padded_depth = (
                depth + (self.window_size - depth % self.window_size) % self.window_size
            )

            img_mask = torch.zeros(
                (1, padded_height, padded_width, padded_depth, 1),
                dtype=dtype,
                device=device,
            )
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            depth_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    for depth_slice in depth_slices:
                        img_mask[:, height_slice, width_slice, depth_slice, :] = count
                        count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size * self.window_size
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):

        B, C, H, W, D = hidden_states.shape

        # precompute attention mask
        attn_mask_precomputed = self.get_attn_mask(
            H, W, D, hidden_states.dtype, hidden_states.device
        )
        # the below loop can be optimized by remove the window_partition and window_reverse when shift_size=0, but for simplicity we keep the same code for both cases
        for n, block in enumerate(self.blocks):

            shift_size = 0 if n % 2 == 0 else self.shift_size

            # channels last
            hidden_states = torch.permute(hidden_states, (0, 2, 3, 4, 1))

            if shift_size > 0:
                attn_mask = attn_mask_precomputed
                shifted_hidden_states = torch.roll(
                    hidden_states,
                    shifts=(-shift_size, -shift_size, -shift_size),
                    dims=(1, 2, 3),
                )
            else:
                attn_mask = None
                shifted_hidden_states = hidden_states

            shifted_hidden_states, pad_values = self.maybe_pad(
                shifted_hidden_states, H, W, D
            )
            _, height_pad, width_pad, depth_pad, _ = shifted_hidden_states.shape

            hidden_states = window_partition(shifted_hidden_states, self.window_size)

            hidden_states = block(
                hidden_states,
                attn_mask=attn_mask,
            )

            hidden_states = window_reverse(
                hidden_states, self.window_size, height_pad, width_pad, depth_pad, B
            )

            if height_pad > 0 or width_pad > 0 or depth_pad > 0:
                hidden_states = hidden_states[:, :H, :W, :D, :].contiguous()

            if shift_size > 0:
                hidden_states = torch.roll(
                    hidden_states,
                    shifts=(shift_size, shift_size, shift_size),
                    dims=(1, 2, 3),
                )

            hidden_states = torch.permute(hidden_states, (0, 4, 1, 2, 3))

        return hidden_states


class P3DTransformerEncoder(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

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
    ):
        super().__init__()

        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."

        if num_heads is not None and isinstance(num_heads, int):
            num_heads = [num_heads] * len(depth)
        if window_size is not None and isinstance(window_size, int):
            window_size = [window_size] * len(depth)

        self.hidden_size = hidden_size

        self.latent_size = hidden_size * 2 ** (len(depth) // 2)

        self.shift = shift
        self.num_encoder_layers = len(depth) // 2

        self.num_heads = num_heads
        self.periodic = periodic

        self.max_hidden_size = max_hidden_size

        assert (
            self.max_hidden_size >= hidden_size
        ), f"max_hidden_size {max_hidden_size} must be greater than or equal to hidden_size {hidden_size}."

        dit_stage_args = {
            "periodic": periodic,
            "mlp_ratio": mlp_ratio,
            "shift": shift,
        }

        # encoder: use ModuleList for encoder levels and downsample layers
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_encoder_layers):
            hidden_size_layer = min(hidden_size * 2**i, max_hidden_size)
            self.encoder_levels.append(
                P3DStage(
                    dim=hidden_size_layer,
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    depth=depth[i],
                    **dit_stage_args,
                )
            )
            self.downsamples.append(Downsample(hidden_size_layer))

        # latent
        hidden_size_latent = min(
            hidden_size * 2**self.num_encoder_layers, 2*max_hidden_size
        )
        self.latent = P3DStage(
            dim=hidden_size_latent,
            num_heads=num_heads[self.num_encoder_layers],
            window_size=window_size[self.num_encoder_layers],
            depth=depth[self.num_encoder_layers],
            **dit_stage_args,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x):
        """
        Encoding of P3D
        x: (N, C, H, W, D) tensor of spatial inputs (images or latent representations of images)
        """
        for encoder_level, downsample in zip(self.encoder_levels, self.downsamples):
#           print(f"encoder_level input shape: {x.shape}")
            out_enc_level = encoder_level(x)
#            print(f"encoder_level output shape: {out_enc_level.shape}")
            x = downsample(out_enc_level)
#        print(f"latent input shape: {x.shape}")
        x = self.latent(x)
        return x


class P3DTransformerDecoder(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

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
    ):
        super().__init__()

        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."

        if num_heads is not None and isinstance(num_heads, int):
            num_heads = [num_heads] * len(depth)
        if window_size is not None and isinstance(window_size, int):
            window_size = [window_size] * len(depth)

        self.hidden_size = hidden_size

        self.shift = shift
        self.num_encoder_layers = len(depth) // 2

        self.num_heads = num_heads
        self.periodic = periodic

        self.max_hidden_size = max_hidden_size

        assert (
            self.max_hidden_size >= hidden_size
        ), f"max_hidden_size {max_hidden_size} must be greater than or equal to hidden_size {hidden_size}."

        dit_stage_args = {
            "periodic": periodic,
            "mlp_ratio": mlp_ratio,
            "shift": shift,
        }

        hidden_size_layer0 = min(hidden_size * 2, max_hidden_size)
        # double hidden size for last decoder layer 0
        self.upsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()

        self.upsamples.append(Upsample(hidden_size_layer0*2))
        self.decoder_levels.append(
            P3DStage(
                dim=hidden_size_layer0,
                num_heads=num_heads[self.num_encoder_layers + 1],
                window_size=window_size[self.num_encoder_layers + 1],
                depth=depth[self.num_encoder_layers + 1],
                **dit_stage_args,
            )
        )

        # decoder layers 1 - num_encoder_layers
        for i in range(1, self.num_encoder_layers):
            hidden_size_layer = min(hidden_size**i, max_hidden_size)
            if 2* hidden_size_layer >= max_hidden_size:
                hidden_size_upsample = max_hidden_size
            else:
                hidden_size_upsample = 2 * hidden_size_layer
            self.upsamples.append(Upsample(hidden_size_upsample))
            self.decoder_levels.append(
                P3DStage(
                    dim=hidden_size_layer,
                    num_heads=num_heads[self.num_encoder_layers + i + 1],
                    window_size=window_size[self.num_encoder_layers + i + 1],
                    depth=depth[self.num_encoder_layers + i + 1],
                    **dit_stage_args,
                )
            )

        self.final_layer = FinalLayer(self.hidden_size, self.hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def forward(self, x):
        # decoder: reverse order except the first (which is up1_0 and decoder_level_0)
        for upsample, decoder_level in zip(
            self.upsamples, self.decoder_levels
        ):  
#            print(f"decoder_level input shape: {x.shape}")
            x = upsample(x)
#            print(f"upsample output shape: {x.shape}")
            x = decoder_level(x)
        # output
#        print(f"final_layer input shape: {x.shape}")
        x = self.final_layer(x)
        return x
