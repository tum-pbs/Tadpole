from ..transformer import PDEDiTBlock, P3DStage,window_partition, window_reverse,P3DTransformerEncoder,P3DTransformerDecoder
from torch.nn import Module
from .embeder import AdaLayerNormZero
import torch
from typing import Optional

class PDEDiTBlockEmbedded(Module):
    
    def __init__(self, pde_dit_block: PDEDiTBlock,
                 num_classes:int
                 ):
        super().__init__()
        self.pde_dit_block = pde_dit_block
        self.embedder = AdaLayerNormZero(
            embedding_dim=pde_dit_block.dim,
            num_classes=num_classes,
            chunk_size=6,)
        
    def forward(
        self,
        x,
        classes,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        B, W, N = x.shape
        Bc = classes.shape[0]

        ########### DiT block with MSA, MLP, and AdaIN ############
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.embedder(classes)
        num_windows_total = int(B // Bc)
        msa_shift = msa_shift.repeat_interleave(num_windows_total, dim=0)
        msa_scale = msa_scale.repeat_interleave(num_windows_total, dim=0)
        msa_gate = msa_gate.repeat_interleave(num_windows_total, dim=0)
        mlp_shift = mlp_shift.repeat_interleave(num_windows_total, dim=0)
        mlp_scale = mlp_scale.repeat_interleave(num_windows_total, dim=0)
        mlp_gate = mlp_gate.repeat_interleave(num_windows_total, dim=0)


        x_msa = self.pde_dit_block.norm1(x)
        x_msa = x_msa * (1 + msa_scale[:, None]) + msa_shift[:, None]
        x_msa = self.pde_dit_block.attn(x_msa, attn_mask=attn_mask)
        x_msa = x_msa * (1 + msa_gate[:, None])
        x = x + self.pde_dit_block.drop_path(x_msa)
        x_mlp = self.pde_dit_block.norm2(x)
        x_mlp = x_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
        x_mlp = self.pde_dit_block.mlp(x_mlp)
        x_mlp = x_mlp * (1 + mlp_gate[:, None])
        x = x + self.pde_dit_block.drop_path(x_mlp)
        
        
        return x
    
class P3DStageEmbedded(Module):
    
    def __init__(self, 
                 p3d_stage: P3DStage,
                 num_classes: int,
                 ):
        super().__init__()
        self.p3d_stage = p3d_stage
        self.p3d_stage.blocks = torch.nn.ModuleList(
            [
                PDEDiTBlockEmbedded(block, num_classes)
                for block in p3d_stage.blocks
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        classes: Optional[torch.LongTensor],
    ):

        B, C, H, W, D = hidden_states.shape

        # precompute attention mask
        attn_mask_precomputed = self.p3d_stage.get_attn_mask(
            H, W, D, hidden_states.dtype, hidden_states.device
        )

        for n, block in enumerate(self.p3d_stage.blocks):

            shift_size = 0 if n % 2 == 0 else self.p3d_stage.shift_size

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

            shifted_hidden_states, pad_values = self.p3d_stage.maybe_pad(
                shifted_hidden_states, H, W, D
            )
            _, height_pad, width_pad, depth_pad, _ = shifted_hidden_states.shape

            hidden_states = window_partition(shifted_hidden_states, self.p3d_stage.window_size)

            hidden_states = block(
                hidden_states,
                attn_mask=attn_mask,
                classes=classes,
            )

            hidden_states = window_reverse(
                hidden_states, self.p3d_stage.window_size, height_pad, width_pad, depth_pad, B
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

class P3DTransformerEncoderEmbedded(Module):
    
    def __init__(self,
                 p3d_transformer_encoder: P3DTransformerEncoder,
                 num_classes: int,
                 ):
        super().__init__()
        self.p3d_transformer_encoder = p3d_transformer_encoder
        self.p3d_transformer_encoder.encoder_levels = torch.nn.ModuleList(
            [
                P3DStageEmbedded(stage, num_classes)
                for stage in p3d_transformer_encoder.encoder_levels
            ]
        )
        self.p3d_transformer_encoder.latent=P3DStageEmbedded(
            p3d_transformer_encoder.latent, num_classes
        )
        
    def forward(self, x, classes):
        """
        Encoding of P3D
        x: (N, C, H, W, D) tensor of spatial inputs (images or latent representations of images)
        """
        for encoder_level, downsample in zip(self.p3d_transformer_encoder.encoder_levels, self.p3d_transformer_encoder.downsamples):
#           print(f"encoder_level input shape: {x.shape}")
            out_enc_level = encoder_level(x,classes)
#            print(f"encoder_level output shape: {out_enc_level.shape}")
            x = downsample(out_enc_level)
#        print(f"latent input shape: {x.shape}")
        x = self.p3d_transformer_encoder.latent(x,classes)
        return x
    
class P3DTransformerDecoderEmbedded(Module):
    
    def __init__(self,
                 p3d_transformer_decoder: P3DTransformerDecoder,
                 num_classes: int,
                 ):
        super().__init__()
        self.p3d_transformer_decoder = p3d_transformer_decoder
        self.p3d_transformer_decoder.decoder_levels = torch.nn.ModuleList(
            [
                P3DStageEmbedded(stage, num_classes)
                for stage in p3d_transformer_decoder.decoder_levels
            ]
        )
        
    def forward(self, x, classes):
        # decoder: reverse order except the first (which is up1_0 and decoder_level_0)
        for upsample, decoder_level in zip(
            self.p3d_transformer_decoder.upsamples, self.p3d_transformer_decoder.decoder_levels
        ):  
#            print(f"decoder_level input shape: {x.shape}")
            x = upsample(x)
#            print(f"upsample output shape: {x.shape}")
            x = decoder_level(x,classes)
        # output
#        print(f"final_layer input shape: {x.shape}")
        x = self.p3d_transformer_decoder.final_layer(x)
        return x
