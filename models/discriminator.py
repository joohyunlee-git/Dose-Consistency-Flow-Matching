from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from models.nn import (
    conv_nd,
    normalization,
)


@dataclass(eq=False)
class PatchDiscriminator(nn.Module):
    """
    The full Patch Discriminator model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    in_channels: int = 2
    model_channels: int = 128
    out_channels: int = 1
    num_res_blocks: int = 2
    dropout: float = 0.13
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    dims: int = 3
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = False

    def __post_init__(self):
        super().__init__()

        kw = 4
        padw = 1

        ch = int(self.channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList(
            nn.Sequential(conv_nd(self.dims, self.in_channels, ch, 3, padding=1))
        )
        self._feature_size = ch
        input_block_chans = [ch]
        for level, mult in enumerate(self.channel_mult):
            if level == 0:
                layers = [
                    conv_nd(self.dims, ch, mult * self.model_channels, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                layers = [
                    conv_nd(self.dims, ch, mult * self.model_channels, kernel_size=kw, stride=2, padding=padw),
                    normalization(mult * self.model_channels),
                    nn.LeakyReLU(0.2, True),
                ]
            
            ch = int(mult * self.model_channels)
            self.input_blocks.append(nn.Sequential(*layers))
            self._feature_size += ch
            input_block_chans.append(ch)

            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(conv_nd(self.dims, ch, ch, kernel_size=kw, stride=1, padding=padw))
                input_block_chans.append(ch)
                self._feature_size += ch

    def forward(self, x, y):    
        h = torch.cat((x, y), dim=1)

        for block in self.input_blocks:
            h = block(h)
        
        result = h.type(x.dtype)
        
        return result
