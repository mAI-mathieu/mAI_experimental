from io import BytesIO
from typing import Any

from .fal_image import image_batch_to_png_bytes

NANO_BANANA_21_EDIT_ENDPOINT = "google/nano-banana-2.1/edit"
NANO_BANANA_21_RESOLUTIONS = ("1k", "2k", "4k")
NANO_BANANA_21_ASPECT_RATIOS = (
    "auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5",
    "3:4", "2:3", "9:16", "4:1", "1:4", "8:1", "1:8",
)


def build_nano_banana_21_arguments(
    prompt: str, seed: int, resolution: str, aspect_ratio: str,
    output_format: str, masked: bool = False,
) -> dict[str, Any]:
    """Build the documented edit payload; masks are supplied as references."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < -1:
        raise ValueError("seed must be -1 (random) or a non-negative integer")
    if resolution not in NANO_BANANA_21_RESOLUTIONS:
        raise ValueError("Nano Banana 2.1 resolution must be 1k, 2k, or 4k")
    if aspect_ratio not in NANO_BANANA_21_ASPECT_RATIOS:
        raise ValueError(f"Unsupported Nano Banana 2.1 aspect_ratio: {aspect_ratio}")
    if output_format not in ("png", "jpeg"):
        raise ValueError(f"Unknown output_format: {output_format}")
    if masked and aspect_ratio != "auto":
        raise ValueError("Nano Banana masked editing requires aspect_ratio=auto to preserve source geometry")
    if masked:
        prompt = (
            "Image 1 is the source photograph. Image 2 is a grayscale editing mask, "
            "not content to reproduce. Edit only the white region in image 2; "
            "black regions must remain unchanged. Gray regions are transition edges. "
            "Keep image 1's exact composition, framing, geometry and aspect ratio. "
            "Return only the edited image 1, without the mask, overlays or borders. "
            "Apply this edit within the selected region:\n" + prompt
        )
    arguments = {
        "prompt": prompt,
        "num_images": 1,
        "resolution": resolution.upper(),
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
        "limit_generations": True,
    }
    if seed >= 0:
        arguments["seed"] = seed
    return arguments


def _as_array(value):
    import numpy as np

    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _source_pixels(image):
    import numpy as np

    pixels = _as_array(image)
    if (
        pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[-1] != 3
        or min(pixels.shape[1:3]) < 1
    ):
        raise ValueError("Nano Banana 2.1 requires exactly one RGB image [1, height, width, 3]")
    if not np.isfinite(pixels).all():
        raise ValueError("image must contain only finite pixel values")
    return pixels


def _mask_pixels(mask, source_shape):
    import numpy as np

    pixels = _as_array(mask)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2 or pixels.shape != source_shape[1:3]:
        raise ValueError("mask must contain one mask with dimensions matching the source image")
    if not np.isfinite(pixels).all() or ((pixels < 0) | (pixels > 1)).any():
        raise ValueError("mask values must be finite and between 0 and 1")
    if not pixels.any():
        raise ValueError("mask contains no area to edit; paint a white region before queuing")
    return pixels


def prepare_nano_banana_21_uploads(image: Any, mask: Any = None) -> list[tuple[str, bytes]]:
    """Encode the source followed by a white-edits mask reference, when present."""
    import numpy as np
    from PIL import Image

    pixels = _source_pixels(image)
    mask_pixels = _mask_pixels(mask, pixels.shape) if mask is not None else None
    uploads = [("comfy_source.png", image_batch_to_png_bytes(pixels)[0])]
    if mask_pixels is not None:
        buffer = BytesIO()
        Image.fromarray((mask_pixels * 255).round().astype(np.uint8)).save(buffer, format="PNG")
        uploads.append(("comfy_mask_reference.png", buffer.getvalue()))
    return uploads


def composite_nano_banana_edit(image: Any, generated: Any, mask: Any):
    """Blend at source dimensions and preserve original pixels wherever mask=0."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    source = _source_pixels(image)
    weights = _mask_pixels(mask, source.shape)
    edited = _source_pixels(generated)
    original_tensor = torch.from_numpy(np.array(source, dtype=np.float32, copy=True))
    edited_tensor = torch.from_numpy(np.array(edited, dtype=np.float32, copy=True))
    if edited.shape[1:3] != source.shape[1:3]:
        edited_tensor = functional.interpolate(
            edited_tensor.permute(0, 3, 1, 2), size=source.shape[1:3],
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)
    weight_tensor = torch.from_numpy(np.array(weights, dtype=np.float32, copy=True))[None, :, :, None]
    blended = original_tensor * (1 - weight_tensor) + edited_tensor * weight_tensor
    return torch.where(weight_tensor == 0, original_tensor, blended)
