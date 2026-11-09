from .imagenode import ImageNonUniformScale
from .latentnode import LatentNonUniformScale

NODE_CLASS_MAPPINGS = {
    "ImageNonUniformScale": ImageNonUniformScale,
    "LatentNonUniformScale": LatentNonUniformScale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageNonUniformScale": "Non-Uniform Scaler (Image)",
    "LatentNonUniformScale": "Non-Uniform Scaler (Latent)",
}

__all__ = [
    "ImageNonUniformScale",
    "LatentNonUniformScale",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]