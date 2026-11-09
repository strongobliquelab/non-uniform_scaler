"""ComfyUI node: seamless non-uniform scale."""

import numpy as np
import torch

from .non_uniform_scaler import mask_bounds, scale_image


class ImageNonUniformScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scaling_mode": (["factor", "targeted_dimension"],),
                "direction": (["vertical", "horizontal"],),
                "crop_mode": (["both", "first", "second"],),
                "scale_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.01},
                ),
                "targeted_dimension": (
                    "INT",
                    {"default": 512, "min": 1, "max": 16384, "step": 1},
                ),
                "affected_start": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 1},
                ),
                "affected_end": (
                    "INT",
                    {"default": 512, "min": 0, "max": 16384, "step": 1},
                ),
                "mask_input_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"
    CATEGORY = "image/transform"

    def scale(
        self,
        image,
        scaling_mode,
        direction,
        crop_mode,
        scale_factor,
        targeted_dimension,
        affected_start,
        affected_end,
        mask_input_mode,
        mask=None,
    ):
        empty_span = False
        if mask_input_mode and mask is not None:
            bounds = mask_bounds(
                mask.detach().cpu().numpy(),
                direction,
                int(image.shape[1]),
                int(image.shape[2]),
            )
            if bounds is None:
                empty_span = True
            else:
                affected_start, affected_end = bounds
        arr = image.detach().cpu().numpy()
        out = scale_image(
            arr,
            scaling_mode,
            direction,
            crop_mode,
            float(scale_factor),
            int(targeted_dimension),
            int(affected_start),
            int(affected_end),
            empty_span=empty_span,
        )
        tensor = torch.from_numpy(np.ascontiguousarray(out)).to(
            device=image.device, dtype=image.dtype
        )
        return (tensor,)