"""ComfyUI node: seamless non-uniform scale (latent)."""

import numpy as np
import torch

from .non_uniform_scaler import scale_latent


class LatentNonUniformScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "scaling_mode": (["factor", "targeted_dimension"],),
                "direction": (["vertical", "horizontal"],),
                "crop_mode": (["both", "first", "second"],),
                "scale_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.01},
                ),
                "targeted_dimension": (
                    "INT",
                    {"default": 64, "min": 1, "max": 2048, "step": 1},
                ),
                "affected_start": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "step": 1},
                ),
                "affected_end": (
                    "INT",
                    {"default": 64, "min": 0, "max": 2048, "step": 1},
                ),
                "downsampling_factor_be_8": (
                    "INT",
                    {"default": 8, "min": 1, "max": 64, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "scale"
    CATEGORY = "latent/transform"

    def scale(
        self,
        latent,
        scaling_mode,
        direction,
        crop_mode,
        scale_factor,
        targeted_dimension,
        affected_start,
        affected_end,
        downsampling_factor_be_8,
    ):
        samples = latent["samples"]
        arr = samples.detach().cpu().numpy()
        out = scale_latent(
            arr,
            scaling_mode,
            direction,
            crop_mode,
            float(scale_factor),
            int(targeted_dimension),
            int(affected_start),
            int(affected_end),
            empty_span=False,
            downsampling_factor=int(downsampling_factor_be_8),
        )
        tensor = torch.from_numpy(np.ascontiguousarray(out)).to(
            device=samples.device, dtype=samples.dtype
        )
        return ({"samples": tensor},)