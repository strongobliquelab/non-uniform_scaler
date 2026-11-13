"""ComfyUI node: second-order central-difference normalizer."""

import numpy as np
import torch

from .non_uniform_scaler import normalize_socd_image


class ImageSOCDNormalizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["vertical", "horizontal"],),
                "crop_mode": (["both", "first", "second"],),
                "smooth_factor": (
                    "FLOAT",
                    {"default": 100.0, "min": 0.0, "max": 8192.0, "step": 0.01},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalize"
    CATEGORY = "image/transform"

    def normalize(self, image, direction, crop_mode, smooth_factor, strength):
        arr = image.detach().cpu().numpy()
        out = normalize_socd_image(
            arr,
            direction,
            crop_mode,
            float(smooth_factor),
            float(strength),
        )
        tensor = torch.from_numpy(np.ascontiguousarray(out)).to(
            device=image.device, dtype=image.dtype
        )
        return (tensor,)
