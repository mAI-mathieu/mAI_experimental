import json

from ..utils.fal_image import (
    DEFAULT_ENDPOINT,
    IMAGE_SIZE_PRESETS,
    NANO_BANANA_2_EDIT_ENDPOINT,
    REQUIRED_IMAGE_ENDPOINTS,
    SEEDREAM_5_PRO_EDIT_ENDPOINT,
    add_image_urls,
    build_arguments,
    build_nano_banana_2_edit_arguments,
    build_seedream_5_pro_edit_arguments,
    download_image_bytes,
    extract_image_url,
    get_result_seed,
    image_batch_to_png_bytes,
    image_bytes_to_tensor,
    normalize_endpoint,
    resolve_image_input_name,
)


class MAIFalImage:
    """Generate an image through a fal text-to-image endpoint."""

    CATEGORY = "mAI / Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("image", "image_url", "seed", "response_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "model_endpoint": (
                    "STRING",
                    {"default": DEFAULT_ENDPOINT, "multiline": False},
                ),
                "prompt": (
                    "STRING",
                    {"default": "A cinematic photograph of a mountain lake", "multiline": True},
                ),
                "image_size": (list(IMAGE_SIZE_PRESETS), {"default": "landscape_4_3"}),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "output_format": (["png", "jpeg"], {"default": "png"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # API calls are intentional side effects and should run for every queue.
        return float("nan")

    def generate(
        self,
        api_key,
        model_endpoint,
        prompt,
        image_size,
        seed,
        output_format,
        image_1=None,
        image_2=None,
        image_3=None,
    ):
        key = api_key.strip() if isinstance(api_key, str) else ""
        if not key:
            raise ValueError(
                "api_key cannot be empty. Paste your fal API key into the node."
            )

        try:
            import fal_client
        except ImportError as exc:
            raise ImportError(
                "fal-client is required. Install this node pack's requirements and "
                "restart ComfyUI."
            ) from exc

        endpoint = normalize_endpoint(model_endpoint)
        image_batches = [image for image in (image_1, image_2, image_3) if image is not None]
        if endpoint in REQUIRED_IMAGE_ENDPOINTS and not image_batches:
            raise ValueError(
                f"'{endpoint}' requires at least one connected input image."
            )

        if endpoint == NANO_BANANA_2_EDIT_ENDPOINT:
            arguments = build_nano_banana_2_edit_arguments(
                prompt,
                image_size,
                seed,
                output_format,
            )
        elif endpoint == SEEDREAM_5_PRO_EDIT_ENDPOINT:
            arguments = build_seedream_5_pro_edit_arguments(
                prompt,
                image_size,
                output_format,
            )
        else:
            arguments = build_arguments(
                prompt,
                image_size,
                seed,
                output_format,
                "{}",
            )

        uploaded_urls = []
        try:
            client = fal_client.SyncClient(key=key)
            image_index = 0
            for image_batch in image_batches:
                for png_bytes in image_batch_to_png_bytes(image_batch):
                    image_index += 1
                    uploaded_urls.append(
                        client.upload(
                            png_bytes,
                            content_type="image/png",
                            file_name=f"comfy_input_{image_index}.png",
                        )
                    )
        except Exception as exc:
            raise RuntimeError(f"Could not upload the input image to fal: {exc}") from exc

        if uploaded_urls:
            image_input_name = resolve_image_input_name(endpoint, len(uploaded_urls))
            arguments = add_image_urls(arguments, image_input_name, uploaded_urls)

        try:
            result = client.subscribe(endpoint, arguments=arguments)
        except Exception as exc:
            raise RuntimeError(f"fal request to '{endpoint}' failed: {exc}") from exc

        image_url = extract_image_url(result)
        try:
            image_bytes = download_image_bytes(image_url)
        except Exception as exc:
            raise RuntimeError(f"Could not download the image returned by fal: {exc}") from exc

        image = image_bytes_to_tensor(image_bytes)
        response_json = json.dumps(result, ensure_ascii=False, default=str)
        return (image, image_url, get_result_seed(result), response_json)
