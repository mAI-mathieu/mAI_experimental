import json
import os

from ..utils.fal_ideogram_edit import EDIT_PRECISIONS, IDEOGRAM_EDIT_ENDPOINT, QUALITY_TIERS
from ..utils.fal_image_edit import (
    FLUX_3_ASPECT_RATIOS,
    FLUX_3_EDIT_ENDPOINT,
    FLUX_3_RESOLUTIONS,
    IMAGE_EDIT_ENDPOINTS,
    build_image_edit_arguments,
    prepare_image_edit_uploads,
)
from ..utils.fal_image import (
    download_image_bytes, extract_image_url, get_result_seed, image_bytes_to_tensor,
)


class MAIFalImageEdit:
    """Generate or edit images using explicit fal endpoint adapters."""

    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("image", "image_url", "seed", "response_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_endpoint": (list(IMAGE_EDIT_ENDPOINTS), {"default": IDEOGRAM_EDIT_ENDPOINT}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "Leave blank to use the server's FAL_KEY environment variable."},
                ),
                "edit_precision": (
                    list(EDIT_PRECISIONS),
                    {"default": "high", "tooltip": "Ideogram only. High restores unchanged pixels."},
                ),
                "quality": (
                    list(QUALITY_TIERS), {"default": "medium", "tooltip": "Ideogram only."},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF, "tooltip": "Ideogram only. -1 chooses automatically."},
                ),
                "resolution": (
                    list(FLUX_3_RESOLUTIONS), {"default": "1k", "tooltip": "FLUX 3 only."},
                ),
                "aspect_ratio": (
                    list(FLUX_3_ASPECT_RATIOS),
                    {"default": "auto", "tooltip": "FLUX 3 only. Auto follows the first reference when editing."},
                ),
                "output_format": (["png", "jpeg"], {"default": "png", "tooltip": "FLUX 3 only."}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": False, "tooltip": "FLUX 3 only."}),
                "safety_tolerance": (
                    "INT", {"default": 2, "min": 0, "max": 4, "tooltip": "FLUX 3 only. 0 is strictest."},
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Edit endpoints only. FLUX 3 accepts a batch of up to 10 references."}),
                "mask": ("MASK", {"tooltip": "Ideogram only. White edits, black preserves; must match image dimensions."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Each queue intentionally submits a fresh API request.
        return float("nan")

    def run(
        self, image=None, mask=None, prompt="", api_key="", edit_precision="high",
        quality="medium", seed=-1, model_endpoint=IDEOGRAM_EDIT_ENDPOINT,
        resolution="1k", aspect_ratio="auto", output_format="png",
        enable_prompt_expansion=False, safety_tolerance=2,
    ):
        key = api_key.strip() if isinstance(api_key, str) else ""
        key = key or os.environ.get("FAL_KEY", "").strip()
        if not key:
            raise ValueError("Enter api_key or set FAL_KEY in the ComfyUI server environment.")

        arguments = build_image_edit_arguments(
            model_endpoint, prompt, seed, edit_precision, quality, resolution,
            aspect_ratio, output_format, enable_prompt_expansion, safety_tolerance,
        )
        uploads = prepare_image_edit_uploads(model_endpoint, image, mask)

        try:
            import fal_client
        except ImportError as exc:
            raise ImportError(
                "fal-client is required. Install this node pack's requirements and restart ComfyUI."
            ) from exc

        try:
            client = fal_client.SyncClient(key=key)
            urls = [
                client.upload(png, content_type="image/png", file_name=name)
                for name, png in uploads
            ]
        except Exception as exc:
            raise RuntimeError(f"Could not upload the image or mask to fal: {exc}") from exc
        if model_endpoint == IDEOGRAM_EDIT_ENDPOINT:
            arguments["image_url"] = urls[0]
            if mask is not None:
                arguments["mask_url"] = urls[1]
        elif model_endpoint == FLUX_3_EDIT_ENDPOINT:
            arguments["image_urls"] = urls

        try:
            result = client.subscribe(model_endpoint, arguments=arguments)
        except Exception as exc:
            raise RuntimeError(f"fal request to '{model_endpoint}' failed: {exc}") from exc

        image_url = extract_image_url(result)
        try:
            output_image = image_bytes_to_tensor(download_image_bytes(image_url))
        except Exception as exc:
            raise RuntimeError(f"Could not download or decode the image returned by fal: {exc}") from exc
        return (
            output_image, image_url, get_result_seed(result),
            json.dumps(result, ensure_ascii=False, default=str),
        )
