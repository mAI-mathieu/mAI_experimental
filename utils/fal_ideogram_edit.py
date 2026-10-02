from io import BytesIO
from typing import Any

IDEOGRAM_EDIT_ENDPOINT = "ideogram/v4.5/edit"
EDIT_PRECISIONS = ("high", "regular")
QUALITY_TIERS = ("very_low", "low", "medium", "high")


def build_ideogram_edit_arguments(
    prompt: str, seed: int, edit_precision: str, quality: str,
) -> dict[str, Any]:
    """Build masked-edit settings using source geometry and one result."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if edit_precision not in EDIT_PRECISIONS:
        raise ValueError(f"Unknown edit_precision: {edit_precision}")
    if quality not in QUALITY_TIERS:
        raise ValueError(f"Unknown quality: {quality}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < -1:
        raise ValueError("seed must be -1 (random) or a non-negative integer")

    arguments: dict[str, Any] = {
        "prompt": prompt,
        "edit_precision": edit_precision,
        "quality": quality,
        "image_size": "auto",
        "num_images": 1,
    }
    if seed >= 0:
        arguments["seed"] = seed
    return arguments


def prepare_ideogram_edit_inputs(image: Any, mask: Any) -> tuple[bytes, bytes]:
    """Encode one source and an inverted binary ComfyUI mask without resizing."""
    import numpy as np
    from PIL import Image

    from .fal_image import image_batch_to_png_bytes

    def as_array(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    pixels = as_array(image)
    if (
        pixels.ndim != 4 or pixels.shape[0] != 1
        or pixels.shape[-1] not in (1, 3, 4)
        or min(pixels.shape[1:3]) < 1
    ):
        raise ValueError(
            "image must contain exactly one ComfyUI IMAGE "
            "[1, height, width, channels]"
        )
    if not np.isfinite(pixels).all():
        raise ValueError("image must contain only finite pixel values")

    mask_pixels = as_array(mask)
    if mask_pixels.ndim == 3 and mask_pixels.shape[0] == 1:
        mask_pixels = mask_pixels[0]
    if mask_pixels.ndim != 2:
        raise ValueError("mask must contain exactly one MASK [1, height, width]")
    if mask_pixels.shape != pixels.shape[1:3]:
        raise ValueError(
            "mask dimensions must match the source image. "
            "Resize the mask to match before connecting it."
        )
    if not np.isfinite(mask_pixels).all():
        raise ValueError("mask must contain only finite values")

    edit_region = mask_pixels >= 0.5
    if not edit_region.any() or edit_region.all():
        raise ValueError(
            "mask must include both an area to edit and an area to preserve "
            "after thresholding at 0.5. White edits; black preserves in ComfyUI."
        )

    # fal expects black to edit and white to preserve, opposite to ComfyUI.
    fal_mask = np.where(edit_region, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(fal_mask).save(buffer, format="PNG")
    return image_batch_to_png_bytes(pixels)[0], buffer.getvalue()
