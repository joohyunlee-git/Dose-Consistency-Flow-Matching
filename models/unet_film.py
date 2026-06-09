# unet_film.py

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.nn import timestep_embedding
from models.unet import UNetModel


# ─────────────────────────────────────────────
# Helper: log sinusoidal dose embedding
# ─────────────────────────────────────────────

def log_sinusoidal_embedding(
    d: torch.Tensor,
    dim: int,
    log_min: float = -4.6052,  # log(0.01) = log(1/100)
    log_max: float =  0.0,     # log(1.0)  = log(1/1)
) -> torch.Tensor:
    log_d  = torch.log(d.clamp(min=1e-4).float())
    d_norm = (log_d - log_min) / (log_max - log_min)   # [0, 1]

    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) *
        torch.arange(half, device=d.device, dtype=torch.float32) / half
    )
    h = d_norm[:, None] * freqs[None]                   # [B, half]
    return torch.cat([h.sin(), h.cos()], dim=-1)         # [B, dim]


# ─────────────────────────────────────────────
# UNetModel_FILM
# ─────────────────────────────────────────────
 
@dataclass(eq=False)
class UNetModel_FILM(UNetModel):
    """
    UNetModel + dose scalar GlobalFiLM conditioning
    """
 
    use_dose_film: bool = True
 
    def __post_init__(self):
        super().__post_init__()
 
        if not self.use_dose_film:
            return
 
        time_embed_dim = self.model_channels * 4
 
        # dose sinusoidal → time_embed_dim 으로 projection
        self.dose_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.model_channels, time_embed_dim),
        )
 
    def forward(self, x, timesteps, extra={}):
        if self.with_fourier_features:
            from models.unet import base2_fourier_features
            z_f = base2_fourier_features(x, start=6, stop=8, step=1)
            x = torch.cat([x, z_f], dim=1)
 
        hs  = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x))
 
        # ── dose embedding ────────────────────────────────────────── 
        if (self.use_dose_film
                and "dose" in extra and extra["dose"] is not None):
 
            d_src = extra["dose"]
            if d_src.dim() > 1:
                d_src = d_src[:, 0]
 
            dose_emb = self.dose_proj(
                log_sinusoidal_embedding(d_src, self.model_channels).to(x.device)
            )                                    # [B, time_embed_dim]
            emb      = emb + dose_emb
 
        # ── label embedding ─────────────────────────────────────────
        if self.num_classes is not None:
            if "label" not in extra:
                extra["label"] = torch.full(
                    (x.size(0),), self.num_classes,
                    dtype=torch.long, device=x.device
                )
            emb = emb + self.label_emb(extra["label"])
 
        # ── encoder path ────────────────────────────────────────────
        h = x
        if "concat_conditioning" in extra:
            h = torch.cat([x, extra["concat_conditioning"]], dim=1)
 
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
 
        # ── decoder path ────────────────────────────────────────────
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
 
        h = h.type(x.dtype)
        result = self.out(h)
        return result

