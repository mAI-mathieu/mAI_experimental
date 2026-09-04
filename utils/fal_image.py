import base64
import binascii
import json
from io import BytesIO
from typing import Any
from urllib.parse import unquote_to_bytes, urlparse
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "fal-ai/flux/dev/image-to-image"
NANO_BANANA_2_EDIT_ENDPOINT = "fal-ai/nano-banana-2/edit"
SEEDREAM_5_LITE_EDIT_ENDPOINT = "bytedance/seedream/v5/lite/edit"
SEEDREAM_5_PRO_EDIT_ENDPOINT = "bytedance/seedream/v5/pro/edit"
SUPPORTED_ENDPOINTS = (
    DEFAULT_ENDPOINT,
    "fal-ai/flux-2/edit",
    "fal-ai/flux-2-max/edit",
    "fal-ai/flux-2/lora/edit",
    "fal-ai/flux-2/klein/9b/edit",
    "fal-ai/flux-2/klein/9b/edit/lora",
    "openai/gpt-image-2/edit",
    NANO_BANANA_2_EDIT_ENDPOINT,
    SEEDREAM_5_LITE_EDIT_ENDPOINT,
    SEEDREAM_5_PRO_EDIT_ENDPOINT,
)
CUSTOM_IMAGE_SIZE_ENDPOINTS = frozenset(
    {
        "fal-ai/flux-2/edit",
        "fal-ai/flux-2-max/edit",
        "fal-ai/flux-2/lora/edit",
        "fal-ai/flux-2/klein/9b/edit",
        "fal-ai/flux-2/klein/9b/edit/lora",
        "openai/gpt-image-2/edit",
        SEEDREAM_5_LITE_EDIT_ENDPOINT,
        SEEDREAM_5_PRO_EDIT_ENDPOINT,
    }
)
RESOLUTION_MODES = ("preset", "custom")
IMAGE_SIZE_PRESETS = (
    "landscape_4_3",
    "landscape_16_9",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
)
NANO_BANANA_2_ASPECT_RATIOS = {
    "landscape_4_3": "4:3",
    "landscape_16_9": "16:9",
    "square_hd": "1:1",
    "square": "1:1",
    "portrait_4_3": "3:4",
    "portrait_16_9": "9:16",
}
IMAGE_URL_LIST_ENDPOINTS = frozenset(
    {
        NANO_BANANA_2_EDIT_ENDPOINT,
        SEEDREAM_5_LITE_EDIT_ENDPOINT,
        SEEDREAM_5_PRO_EDIT_ENDPOINT,
        "openai/gpt-image-2/edit",
    }
)
REQUIRED_IMAGE_ENDPOINTS = frozenset(SUPPORTED_ENDPOINTS)


def _validate_image_size_argument(image_size: Any) -> None:
    if isinstance(image_size, str):
        if image_size not in IMAGE_SIZE_PRESETS:
            raise ValueError(f"Unknown image_size preset: {image_size}")
        return

    if not isinstance(image_size, dict):
        raise TypeError("image_size must be a preset name or a width/height object")
    if set(image_size) != {"width", "height"}:
        raise ValueError("Custom image_size must contain only width and height")

    for name in ("width", "height"):
        value = image_size[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Custom image_size {name} must be a positive integer")


def resolve_image_size_argument(
    endpoint: str,
    image_size: str,
    resolution_mode: str,
    custom_width: int,
    custom_height: int,
) -> Any:
    """Return a preset or custom fal image_size value for an endpoint."""
    if resolution_mode not in RESOLUTION_MODES:
        raise ValueError(f"Unknown resolution_mode: {resolution_mode}")
    if resolution_mode == "preset":
        _validate_image_size_argument(image_size)
        return image_size

    if endpoint not in CUSTOM_IMAGE_SIZE_ENDPOINTS:
        if endpoint == NANO_BANANA_2_EDIT_ENDPOINT:
            detail = "it only accepts aspect_ratio and resolution tiers"
        elif endpoint == DEFAULT_ENDPOINT:
            detail = "its fal schema does not expose an image_size input"
        else:
            detail = "this endpoint is not configured for custom dimensions"
        raise ValueError(
            f"'{endpoint}' does not support exact width and height because {detail}. "
            "Select preset resolution mode."
        )

    custom_size = {"width": custom_width, "height": custom_height}
    _validate_image_size_argument(custom_size)

    if endpoint in {"fal-ai/flux-2/edit", "fal-ai/flux-2/lora/edit"}:
        if not (512 <= custom_width <= 2048 and 512 <= custom_height <= 2048):
            raise ValueError(
                f"'{endpoint}' custom width and height must each be between 512 and 2048."
            )

    if endpoint == SEEDREAM_5_LITE_EDIT_ENDPOINT:
        pixel_count = custom_width * custom_height
        if not (2560 * 1440 <= pixel_count <= 4096 * 4096):
            raise ValueError(
                f"'{endpoint}' exact custom size must contain between 2560x1440 and "
                "4096x4096 total pixels; fal would otherwise rescale it."
            )

    if endpoint == SEEDREAM_5_PRO_EDIT_ENDPOINT:
        pixel_count = custom_width * custom_height
        aspect_ratio = custom_width / custom_height
        if not (1024 * 1024 <= pixel_count <= 2048 * 2048):
            raise ValueError(
                f"'{endpoint}' custom size must contain between 1024x1024 and "
                "2048x2048 total pixels."
            )
        if not (1 / 16 <= aspect_ratio <= 16):
            raise ValueError(
                f"'{endpoint}' custom size aspect ratio must be between 1:16 and 16:1."
            )

    return custom_size


def normalize_endpoint(endpoint: str) -> str:
    """Return a fal endpoint ID and reject URLs or malformed IDs."""
    if not isinstance(endpoint, str):
        raise TypeError("model_endpoint must be a string")

    normalized = endpoint.strip().strip("/")
    if not normalized:
        raise ValueError("model_endpoint cannot be empty")
    if "://" in normalized:
        raise ValueError(
            "model_endpoint must be an endpoint ID such as "
            f"'{DEFAULT_ENDPOINT}', not a URL"
        )
    if "/" not in normalized:
        raise ValueError(
            "model_endpoint must include its owner, for example "
            f"'{DEFAULT_ENDPOINT}'"
        )
    return normalized


def build_arguments(
    prompt: str,
    image_size: Any,
    seed: int,
    output_format: str,
    extra_arguments_json: str,
) -> dict[str, Any]:
    """Build the request body, allowing endpoint-specific JSON overrides."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    _validate_image_size_argument(image_size)
    if output_format not in ("png", "jpeg"):
        raise ValueError(f"Unknown output_format: {output_format}")

    try:
        extra = json.loads(extra_arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_arguments_json is not valid JSON: {exc.msg}") from exc
    if not isinstance(extra, dict):
        raise ValueError("extra_arguments_json must contain a JSON object")

    arguments: dict[str, Any] = {
        "prompt": prompt,
        "image_size": image_size,
        "num_images": 1,
        "output_format": output_format,
    }
    if seed >= 0:
        arguments["seed"] = seed
    arguments.update(extra)
    return arguments


def build_nano_banana_2_edit_arguments(
    prompt: str,
    image_size: str,
    seed: int,
    output_format: str,
) -> dict[str, Any]:
    """Build the request body for fal's Nano Banana 2 edit endpoint."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if image_size not in NANO_BANANA_2_ASPECT_RATIOS:
        raise ValueError(f"Unknown image_size preset: {image_size}")
    if output_format not in ("png", "jpeg"):
        raise ValueError(f"Unknown output_format: {output_format}")

    arguments: dict[str, Any] = {
        "prompt": prompt,
        "num_images": 1,
        "aspect_ratio": NANO_BANANA_2_ASPECT_RATIOS[image_size],
        "output_format": output_format,
        "resolution": "1K",
        "limit_generations": True,
    }
    if seed >= 0:
        arguments["seed"] = seed
    return arguments


def build_seedream_5_pro_edit_arguments(
    prompt: str,
    image_size: Any,
    output_format: str,
) -> dict[str, Any]:
    """Build the request body for fal's Seedream 5.0 Pro edit endpoint."""
    return build_arguments(
        prompt,
        image_size,
        -1,
        output_format,
        "{}",
    )


def build_seedream_5_lite_edit_arguments(
    prompt: str,
    image_size: Any,
) -> dict[str, Any]:
    """Build the request body for fal's Seedream 5.0 Lite edit endpoint."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    _validate_image_size_argument(image_size)

    return {
        "prompt": prompt,
        "image_size": image_size,
        "num_images": 1,
        "max_images": 1,
    }


def add_image_urls(
    arguments: dict[str, Any], image_input_name: str, image_urls: list[str]
) -> dict[str, Any]:
    """Add uploaded image URLs using a singular or plural fal input field."""
    if not image_urls:
        return arguments
    if not isinstance(image_input_name, str) or not image_input_name.strip():
        raise ValueError("image_input_name cannot be empty when an image is connected")

    input_name = image_input_name.strip()
    updated = dict(arguments)
    if input_name.endswith("urls"):
        updated[input_name] = image_urls
    elif len(image_urls) == 1:
        updated[input_name] = image_urls[0]
    else:
        raise ValueError(
            f"'{input_name}' is a singular image field, but {len(image_urls)} images "
            "were connected. Use a plural field such as 'image_urls'."
        )
    return updated


def resolve_image_input_name(endpoint: str, image_count: int) -> str:
    """Choose the documented image field shape for supported fal endpoints."""
    if image_count < 1:
        raise ValueError("image_count must be at least 1")
    is_flux_2_edit = endpoint.startswith("fal-ai/flux-2") and "/edit" in endpoint
    if (
        image_count > 1
        or is_flux_2_edit
        or endpoint in IMAGE_URL_LIST_ENDPOINTS
    ):
        return "image_urls"
    return "image_url"


def extract_image_url(result: Any) -> str:
    """Extract the first image URL from common fal image response shapes."""
    if not isinstance(result, dict):
        raise ValueError("fal returned an unexpected response instead of a JSON object")

    data = result.get("data")
    if isinstance(data, dict):
        result = data

    images = result.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict) and isinstance(first.get("url"), str):
            return first["url"]
        if isinstance(first, str):
            return first

    image = result.get("image")
    if isinstance(image, dict) and isinstance(image.get("url"), str):
        return image["url"]
    if isinstance(image, str):
        return image

    raise ValueError(
        "fal response did not contain an image URL in 'images[0]' or 'image'. "
        "Check that the selected endpoint returns an image."
    )


def get_result_seed(result: Any) -> int:
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, dict):
            result = data
        seed = result.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
    return -1


def download_image_bytes(image_url: str, timeout_seconds: int = 120) -> bytes:
    """Download an HTTP(S) image or decode a fal data URI."""
    if image_url.startswith("data:"):
        try:
            header, encoded = image_url.split(",", 1)
        except ValueError as exc:
            raise ValueError("fal returned a malformed image data URI") from exc
        try:
            if ";base64" in header:
                return base64.b64decode(encoded, validate=True)
            return unquote_to_bytes(encoded)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("fal returned invalid image data") from exc

    scheme = urlparse(image_url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported image URL scheme: {scheme or 'missing'}")

    request = Request(image_url, headers={"User-Agent": "mAI-ComfyUI-fal/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def image_bytes_to_tensor(image_bytes: bytes):
    """Convert encoded image bytes into a ComfyUI IMAGE batch tensor."""
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(pixels).unsqueeze(0)
    except (OSError, ValueError) as exc:
        raise ValueError("fal returned data that could not be decoded as an image") from exc


def image_batch_to_png_bytes(image_batch: Any) -> list[bytes]:
    """Encode a ComfyUI IMAGE batch as individual PNG byte strings."""
    import numpy as np
    from PIL import Image

    if hasattr(image_batch, "detach"):
        pixels = image_batch.detach().cpu().numpy()
    else:
        pixels = np.asarray(image_batch)

    if pixels.ndim != 4 or pixels.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Connected image must have ComfyUI IMAGE shape [batch, height, width, channels]"
        )

    encoded_images: list[bytes] = []
    for frame in pixels:
        byte_pixels = np.clip(frame * 255.0, 0, 255).round().astype(np.uint8)
        if byte_pixels.shape[-1] == 1:
            byte_pixels = byte_pixels[:, :, 0]
            mode = "L"
        else:
            mode = "RGB" if byte_pixels.shape[-1] == 3 else "RGBA"

        buffer = BytesIO()
        Image.fromarray(byte_pixels, mode=mode).save(buffer, format="PNG")
        encoded_images.append(buffer.getvalue())
    return encoded_images
