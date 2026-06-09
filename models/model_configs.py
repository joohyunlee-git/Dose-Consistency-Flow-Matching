# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel
from models.unet_film import UNetModel_FILM
from models.discriminator import PatchDiscriminator


MODEL_CONFIGS = {
    "time_independent": {
        "in_channels": 1,
        "model_channels": 64,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "dropout": 0.05,
        "channel_mult": [1, 1, 2, 4],
        "conv_resample": True,
        "dims": 3,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
        "ignore_time":True,
    },
    "time_dependent": {
        "in_channels": 1,
        "model_channels": 64,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "dropout": 0.05,
        "channel_mult": [1, 1, 2, 4],
        "conv_resample": True,
        "dims": 3,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
        "ignore_time":False,
    },
    "time_dependent_concat": {
        "in_channels": 2,
        "model_channels": 64,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "dropout": 0.05,
        "channel_mult": [1, 1, 2, 4],
        "conv_resample": True,
        "dims": 3,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
        "ignore_time":False,
    },
    "time_independent_film": {
        "in_channels": 1,
        "model_channels": 64,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "dropout": 0.05,
        "channel_mult": [1, 1, 2, 4],
        "conv_resample": True,
        "dims": 3,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
        "ignore_time": True,
        "use_dose_film": True,
    },
    "time_dependent_film": {
        "in_channels": 1,
        "model_channels": 64,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "dropout": 0.05,
        "channel_mult": [1, 1, 2, 4],
        "conv_resample": True,
        "dims": 3,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
        "ignore_time": False,
        "use_dose_film": True,
    },
    "time_dependent_film_concat": {
        "in_channels": 2,
        "model_channels": 64,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "dropout": 0.05,
        "channel_mult": [1, 1, 2, 4],
        "conv_resample": True,
        "dims": 3,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
        "ignore_time": False,
        "use_dose_film": True,
    },
    "discriminator": {
        "in_channels": 2,
        "model_channels": 64,
        "out_channels": 1,
        "dropout": 0.13,
        "channel_mult": [1, 2, 2, 2],
        "dims": 3,
    },
    "autoencoder": {
        "in_channels": 2,
        "model_channels": 64,
        "out_channels": 1,
        "dropout": 0.13,
        "compression_ratio": 4,
    },
}


def instantiate_model(
    architechture: str, use_ema: bool
) -> Union[UNetModel, DiscreteUNetModel]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    model = UNetModel(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model


def instantiate_discriminator(
    use_ema: bool, architechture: str = "discriminator",
) -> Union[PatchDiscriminator]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    model = PatchDiscriminator(**MODEL_CONFIGS[architechture])

    return model


def instantiate_film(
    architechture: str, use_ema: bool
) -> UNetModel_FILM:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    model = UNetModel_FILM(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model
