from .imagenode import ImageNonUniformScale
from .latentnode import LatentNonUniformScale
from .socdnode import ImageSOCDNormalizer

NODE_CLASS_MAPPINGS = {
    "ImageNonUniformScale": ImageNonUniformScale,
    "LatentNonUniformScale": LatentNonUniformScale,
    "ImageSOCDNormalizer": ImageSOCDNormalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageNonUniformScale": "Non-Uniform Scaler (Image)",
    "LatentNonUniformScale": "Non-Uniform Scaler (Latent)",
    "ImageSOCDNormalizer": "SOCD Normalizer (Image)",
}

__all__ = [
    "ImageNonUniformScale",
    "LatentNonUniformScale",
    "ImageSOCDNormalizer",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
