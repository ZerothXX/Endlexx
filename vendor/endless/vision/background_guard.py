"""Keep source pixels outside a valid subject mask; no reference accessory pasting.

Mask validity checks detect empty/full masks, not incorrect semantic selection.
The caller must audit automatic segmentation on its intended input distribution.
"""
import numpy as np
from PIL import Image, ImageFilter


def validate_subject_mask(mask, size):
    if not isinstance(mask, Image.Image) or mask.size != size:
        raise ValueError("Background protection requires a same-sized subject mask")
    binary = np.asarray(mask.convert("L")) > 127
    fraction = float(binary.mean())
    if not 0.01 <= fraction <= 0.90:
        raise ValueError("Empty, tiny or full-image mask cannot protect the background")
    return Image.fromarray(binary.astype(np.uint8) * 255), fraction


def preserve_background(source, generated, mask, feather=1.5):
    if source.size != generated.size:
        raise ValueError("Source and generated images must have identical dimensions")
    binary, fraction = validate_subject_mask(mask, source.size)
    if feather < 0:
        raise ValueError("Feather radius must be nonnegative")
    softened = binary.filter(ImageFilter.GaussianBlur(feather)) if feather else binary
    # Feather inward only: pixels outside the binary mask are exactly unchanged.
    alpha = np.minimum(np.asarray(binary), np.asarray(softened)).astype(np.uint8)
    result = Image.composite(generated.convert("RGB"), source.convert("RGB"), Image.fromarray(alpha))
    return result, {"mask_fraction": fraction, "feather_px": feather,
                    "outside_mask_policy": "exact_source_pixels",
                    "semantic_mask_verified": False}
