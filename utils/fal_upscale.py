"""Request adapters for fal image upscalers; no ComfyUI dependencies."""

from math import isfinite
from typing import Any


SEEDVR_UPSCALE_ENDPOINT = "fal-ai/seedvr/upscale/image"
TOPAZ_UPSCALE_ENDPOINT = "fal-ai/topaz/upscale/image"
CLARITY_UPSCALE_ENDPOINT = "fal-ai/clarity-upscaler"
CRYSTAL_UPSCALE_ENDPOINT = "clarityai/crystal-upscaler"
AURA_SR_ENDPOINT = "fal-ai/aura-sr"
UPSCALE_ENDPOINTS = (
    SEEDVR_UPSCALE_ENDPOINT,
    TOPAZ_UPSCALE_ENDPOINT,
    CLARITY_UPSCALE_ENDPOINT,
    CRYSTAL_UPSCALE_ENDPOINT,
    AURA_SR_ENDPOINT,
)
DEFAULT_TOPAZ_MODEL = "High Fidelity V2"
TOPAZ_MODELS = (
    DEFAULT_TOPAZ_MODEL,
    "Standard V2",
    "Low Resolution V2",
    "CGI",
    "Text Refine",
    "Wonder 3",
    "Wonder",
    "Standard MAX",
    "Redefine",
    "Recovery V2",
    "Recovery",
)


def validate_upscale_images(image_batches: list[Any]) -> None:
    """Require one frame before encoding, uploading, or billing anything."""
    if len(image_batches) != 1:
        raise ValueError("Upscalers require exactly one input image across image_1, image_2, image_3.")
    shape = getattr(image_batches[0], "shape", ())
    if len(shape) != 4 or shape[-1] not in (1, 3, 4) or min(shape[1:3]) < 1:
        raise ValueError(
            "Connected image must have ComfyUI IMAGE shape [batch, height, width, channels]"
        )
    if shape[0] != 1:
        raise ValueError("Upscalers require exactly one input image; select one frame from the batch.")


def build_upscale_arguments(
    endpoint: str,
    upscale_factor: float = 4.0,
    prompt: str = "",
    seed: int = -1,
    output_format: str = "png",
    topaz_model: str = DEFAULT_TOPAZ_MODEL,
) -> dict[str, Any]:
    """Send only fields supported by each endpoint's public fal schema."""
    if endpoint not in UPSCALE_ENDPOINTS:
        raise ValueError(f"Unknown upscale endpoint: {endpoint}")
    if (
        isinstance(upscale_factor, bool)
        or not isinstance(upscale_factor, (int, float))
        or not isfinite(upscale_factor)
    ):
        raise ValueError("upscale_factor must be a finite number")
    # The node exposes up to 10x; Crystal's API allows larger factors.
    maximum = 4 if endpoint in (TOPAZ_UPSCALE_ENDPOINT, CLARITY_UPSCALE_ENDPOINT) else 10
    if not 1 <= upscale_factor <= maximum:
        raise ValueError(f"'{endpoint}' upscale_factor must be between 1 and {maximum}.")
    if output_format not in ("png", "jpeg"):
        raise ValueError(f"Unknown output_format: {output_format}")

    if endpoint == AURA_SR_ENDPOINT:
        if upscale_factor != 4:
            raise ValueError("AuraSR supports only upscale_factor 4.")
        return {"upscale_factor": 4, "checkpoint": "v2", "overlapping_tiles": True}

    if endpoint == CRYSTAL_UPSCALE_ENDPOINT:
        return {
            "scale_factor": upscale_factor,
            "creativity": 0,
            "output_format": "jpg" if output_format == "jpeg" else "png",
        }

    arguments: dict[str, Any] = {"upscale_factor": upscale_factor}
    if endpoint == TOPAZ_UPSCALE_ENDPOINT:
        if topaz_model not in TOPAZ_MODELS:
            raise ValueError(f"Unknown topaz_model: {topaz_model}")
        arguments.update(
            model=topaz_model,
            output_format=output_format,
            crop_to_fill=False,
            face_enhancement=False,
        )
        if topaz_model == "Redefine":
            if not isinstance(prompt, str) or len(prompt) > 1024:
                raise ValueError("Topaz Redefine prompt must be a string of at most 1024 characters.")
            arguments["creativity"] = 1
            if prompt.strip():
                arguments["prompt"] = prompt
        return arguments

    if seed >= 0:
        arguments["seed"] = seed
    if endpoint == SEEDVR_UPSCALE_ENDPOINT:
        arguments.update(
            upscale_mode="factor",
            output_format="jpg" if output_format == "jpeg" else "png",
            noise_scale=0.1,
        )
    elif endpoint == CLARITY_UPSCALE_ENDPOINT:
        if not isinstance(prompt, str):
            raise ValueError("Clarity prompt must be a string.")
        arguments.update(creativity=0.2, resemblance=0.8)
        if prompt.strip():
            arguments["prompt"] = prompt
    return arguments
