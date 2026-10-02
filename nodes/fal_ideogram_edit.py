import json
import os

from ..utils.fal_ideogram_edit import (
    EDIT_PRECISIONS,
    IDEOGRAM_EDIT_ENDPOINT,
    QUALITY_TIERS,
    build_ideogram_edit_arguments,
    prepare_ideogram_edit_inputs,
)
from ..utils.fal_image import (
    download_image_bytes,
    extract_image_url,
    get_result_seed,
    image_bytes_to_tensor,
)


class MAIFalIdeogramEdit:
    """Edit a masked image through fal's Ideogram 4.5 endpoint."""

    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("image", "image_url", "seed", "response_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": (
                    "MASK",
                    {"tooltip": "White edits, black preserves. Must match the image size."},
                ),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "", "multiline": False,
                        "tooltip": "Leave blank to use the ComfyUI server's FAL_KEY environment variable.",
                    },
                ),
                "edit_precision": (
                    list(EDIT_PRECISIONS),
                    {
                        "default": "high",
                        "tooltip": "High restores unchanged pixels; regular uses standard editing.",
                    },
                ),
                "quality": (list(QUALITY_TIERS), {"default": "medium"}),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Each queue intentionally submits a fresh API request.
        return float("nan")

    def run(
        self, image, mask, prompt, api_key="", edit_precision="high",
        quality="medium", seed=-1,
    ):
        key = api_key.strip() if isinstance(api_key, str) else ""
        key = key or os.environ.get("FAL_KEY", "").strip()
        if not key:
            raise ValueError("Enter api_key or set FAL_KEY in the ComfyUI server environment.")

        arguments = build_ideogram_edit_arguments(prompt, seed, edit_precision, quality)
        image_png, mask_png = prepare_ideogram_edit_inputs(image, mask)

        try:
            import fal_client
        except ImportError as exc:
            raise ImportError(
                "fal-client is required. Install this node pack's requirements "
                "and restart ComfyUI."
            ) from exc

        try:
            client = fal_client.SyncClient(key=key)
            arguments["image_url"] = client.upload(
                image_png, content_type="image/png", file_name="comfy_source.png",
            )
            arguments["mask_url"] = client.upload(
                mask_png, content_type="image/png", file_name="comfy_mask.png",
            )
        except Exception as exc:
            raise RuntimeError(f"Could not upload the image or mask to fal: {exc}") from exc

        try:
            result = client.subscribe(IDEOGRAM_EDIT_ENDPOINT, arguments=arguments)
        except Exception as exc:
            raise RuntimeError(f"fal request to '{IDEOGRAM_EDIT_ENDPOINT}' failed: {exc}") from exc

        image_url = extract_image_url(result)
        try:
            output_image = image_bytes_to_tensor(download_image_bytes(image_url))
        except Exception as exc:
            raise RuntimeError(f"Could not download or decode the image returned by fal: {exc}") from exc

        return (
            output_image, image_url, get_result_seed(result),
            json.dumps(result, ensure_ascii=False, default=str),
        )
