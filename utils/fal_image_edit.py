from typing import Any

from .fal_ideogram_edit import (
    IDEOGRAM_EDIT_ENDPOINT,
    build_ideogram_edit_arguments,
    prepare_ideogram_edit_inputs,
)
from .fal_image import image_batch_to_png_bytes
from .fal_nano_banana_edit import (
    NANO_BANANA_21_ASPECT_RATIOS,
    NANO_BANANA_21_EDIT_ENDPOINT,
    build_nano_banana_21_arguments,
    prepare_nano_banana_21_uploads,
)

FLUX_3_TEXT_ENDPOINT = "blackforestlabs/flux-3/text-to-image"
FLUX_3_EDIT_ENDPOINT = "blackforestlabs/flux-3/edit-image"
IMAGE_EDIT_ENDPOINTS = (
    IDEOGRAM_EDIT_ENDPOINT, FLUX_3_TEXT_ENDPOINT, FLUX_3_EDIT_ENDPOINT,
    NANO_BANANA_21_EDIT_ENDPOINT,
)
FLUX_3_RESOLUTIONS = ("512sq", "768sq", "1k", "2k", "4k")
FLUX_3_ASPECT_RATIOS = (
    "auto", "21:9", "2:1", "16:9", "3:2", "7:5", "4:3", "5:4",
    "1:1", "4:5", "3:4", "5:7", "2:3", "9:16", "1:2",
)
IMAGE_EDIT_ASPECT_RATIOS = FLUX_3_ASPECT_RATIOS + tuple(
    ratio for ratio in NANO_BANANA_21_ASPECT_RATIOS if ratio not in FLUX_3_ASPECT_RATIOS
)


def build_image_edit_arguments(
    endpoint: str, prompt: str, seed: int = -1, edit_precision: str = "high",
    quality: str = "medium", resolution: str = "1k", aspect_ratio: str = "auto",
    output_format: str = "png", enable_prompt_expansion: bool = False,
    safety_tolerance: int = 2, masked: bool = False,
) -> dict[str, Any]:
    """Send only the selected endpoint's supported settings."""
    if endpoint not in IMAGE_EDIT_ENDPOINTS:
        raise ValueError(f"Unsupported model_endpoint: {endpoint}")
    if endpoint == IDEOGRAM_EDIT_ENDPOINT:
        return build_ideogram_edit_arguments(prompt, seed, edit_precision, quality)
    if endpoint == NANO_BANANA_21_EDIT_ENDPOINT:
        return build_nano_banana_21_arguments(
            prompt, seed, resolution, aspect_ratio, output_format, masked,
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if resolution not in FLUX_3_RESOLUTIONS:
        raise ValueError(f"Unknown resolution: {resolution}")
    if aspect_ratio not in FLUX_3_ASPECT_RATIOS:
        raise ValueError(f"Unknown aspect_ratio: {aspect_ratio}")
    if output_format not in ("png", "jpeg"):
        raise ValueError(f"Unknown output_format: {output_format}")
    if not isinstance(enable_prompt_expansion, bool):
        raise ValueError("enable_prompt_expansion must be a boolean")
    if (
        not isinstance(safety_tolerance, int) or isinstance(safety_tolerance, bool)
        or not 0 <= safety_tolerance <= 4
    ):
        raise ValueError("safety_tolerance must be an integer between 0 and 4")
    return {
        "prompt": prompt,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
        "enable_prompt_expansion": enable_prompt_expansion,
        "safety_tolerance": safety_tolerance,
    }


def prepare_image_edit_uploads(
    endpoint: str, image: Any = None, mask: Any = None,
) -> list[tuple[str, bytes]]:
    """Validate and encode inputs before initializing a fal client."""
    if endpoint not in IMAGE_EDIT_ENDPOINTS:
        raise ValueError(f"Unsupported model_endpoint: {endpoint}")
    if endpoint == FLUX_3_TEXT_ENDPOINT:
        if image is not None or mask is not None:
            raise ValueError(
                "FLUX 3 text-to-image accepts only a prompt. Disconnect image and mask, "
                "or select an edit endpoint."
            )
        return []
    if image is None:
        raise ValueError(f"'{endpoint}' requires a connected image.")
    if endpoint == NANO_BANANA_21_EDIT_ENDPOINT:
        return prepare_nano_banana_21_uploads(image, mask)
    if endpoint == IDEOGRAM_EDIT_ENDPOINT and mask is not None:
        source_png, mask_png = prepare_ideogram_edit_inputs(image, mask)
        return [("comfy_source.png", source_png), ("comfy_mask.png", mask_png)]
    if mask is not None:
        raise ValueError(
            "FLUX 3 edit-image does not support a mask. Disconnect mask and describe "
            "the edit in prompt, or select Ideogram for masked editing."
        )

    import numpy as np

    pixels = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if (
        pixels.ndim != 4 or pixels.shape[-1] not in (1, 3, 4)
        or pixels.shape[0] < 1 or min(pixels.shape[1:3]) < 1
    ):
        raise ValueError("image must have ComfyUI IMAGE shape [batch, height, width, channels]")
    if not np.isfinite(pixels).all():
        raise ValueError("image must contain only finite pixel values")
    if endpoint == IDEOGRAM_EDIT_ENDPOINT:
        if pixels.shape[0] != 1:
            raise ValueError("Ideogram requires exactly one source image")
        return [("comfy_source.png", image_batch_to_png_bytes(pixels)[0])]

    if pixels.shape[0] > 10:
        raise ValueError("FLUX 3 edit-image accepts at most 10 reference images")
    height, width = pixels.shape[1:3]
    if min(height, width) < 256 or height * width > 4_000_000:
        raise ValueError(
            "FLUX 3 reference images must be at least 256 pixels per dimension "
            "and at most 4 megapixels. Resize the inputs before connecting them."
        )
    return [
        (f"comfy_reference_{index}.png", png)
        for index, png in enumerate(image_batch_to_png_bytes(pixels), start=1)
    ]
