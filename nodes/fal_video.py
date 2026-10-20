import json
from io import BytesIO

from ..utils.fal_image import get_result_seed, image_batch_to_png_bytes
from ..utils.fal_video import (
    MINIMAX_H3_ENDPOINT,
    PROMPT_EXPANSION_MODES,
    VIDEO_ENDPOINTS,
    VIDEO_RESOLUTIONS,
    WAN_ASPECT_RATIOS,
    add_video_frame_urls,
    build_video_arguments,
    download_video_bytes,
    extract_video_url,
    validate_video_frames,
)


class MAIFalVideo:
    """Generate a fal video from a first frame and an optional last frame."""

    CATEGORY = "mAI / Video"
    FUNCTION = "generate"
    RETURN_TYPES = ("VIDEO", "STRING", "INT", "STRING")
    RETURN_NAMES = ("video", "video_url", "seed", "response_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Enter your fal API key directly here.",
                }),
                "model_endpoint": (list(VIDEO_ENDPOINTS), {"default": MINIMAX_H3_ENDPOINT}),
                "first_frame": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "The camera slowly moves through the scene.", "multiline": True,
                }),
                "duration": ("INT", {
                    "default": 5, "min": 2, "max": 30,
                    "tooltip": "Seconds. MiniMax H3: 5-15; Wan: 2-30.",
                }),
                "resolution": (list(VIDEO_RESOLUTIONS), {
                    "default": "model default",
                    "tooltip": (
                        "MiniMax: 480P, 768P, 2K, 4K. Wan: 480p, 720p, 1080p. "
                        "Model default follows the selected model."
                    ),
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "last_frame": ("IMAGE",),
                "prompt_expansion_mode": (list(PROMPT_EXPANSION_MODES), {
                    "default": "balanced", "tooltip": "MiniMax H3 only.",
                }),
                "aspect_ratio": (list(WAN_ASPECT_RATIOS), {
                    "default": "adaptive", "tooltip": "Wan only. MiniMax follows the first frame's canvas.",
                }),
                "audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wan only: include generated audio. MiniMax uses its service default.",
                }),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True, "tooltip": "Wan only."}),
                "enable_thinking": ("BOOLEAN", {"default": False, "tooltip": "Wan only."}),
                "smart_duration": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Wan only: choose duration automatically and ignore duration. "
                        "Turn off for MiniMax."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Each explicit queue requests a new generation, as with mAI fal image.
        return float("nan")

    def generate(
        self, api_key, model_endpoint, first_frame, prompt, duration=5,
        resolution="model default", seed=-1, last_frame=None,
        prompt_expansion_mode="balanced", aspect_ratio="adaptive", audio=True,
        enable_prompt_expansion=True, enable_thinking=False, smart_duration=False,
    ):
        key = api_key.strip() if isinstance(api_key, str) else ""
        if not key:
            raise ValueError("api_key cannot be empty. Paste your fal API key into the node.")
        validate_video_frames(first_frame, last_frame)
        arguments = build_video_arguments(
            model_endpoint, prompt, duration, resolution, seed, prompt_expansion_mode,
            aspect_ratio, audio, enable_prompt_expansion, enable_thinking, smart_duration,
        )
        try:
            import fal_client
        except ImportError as exc:
            raise ImportError(
                "fal-client is required. Install this node pack's requirements and restart ComfyUI."
            ) from exc
        try:
            from comfy_api.input_impl import VideoFromFile
        except ImportError as exc:
            raise ImportError(
                "mAI fal video requires ComfyUI with native VIDEO support. Update ComfyUI and restart."
            ) from exc

        # Encode both inputs before any upload or billable generation request.
        first_png = image_batch_to_png_bytes(first_frame)[0]
        last_png = image_batch_to_png_bytes(last_frame)[0] if last_frame is not None else None
        try:
            client = fal_client.SyncClient(key=key)
            first_url = client.upload(first_png, content_type="image/png", file_name="first_frame.png")
            last_url = (
                client.upload(last_png, content_type="image/png", file_name="last_frame.png")
                if last_png is not None else None
            )
        except Exception as exc:
            raise RuntimeError(f"Could not upload video frames to fal: {exc}") from exc
        arguments = add_video_frame_urls(model_endpoint, arguments, first_url, last_url)
        try:
            result = client.subscribe(model_endpoint, arguments=arguments)
        except Exception as exc:
            raise RuntimeError(f"fal video request to '{model_endpoint}' failed: {exc}") from exc

        video_url = extract_video_url(result)
        try:
            video = VideoFromFile(BytesIO(download_video_bytes(video_url)))
            # Validate the downloaded container without decoding the whole video.
            video.get_dimensions()
        except Exception as exc:
            raise RuntimeError(
                f"Could not download or open the generated video at {video_url}: {exc}. "
                "The generation already completed; retrieve this URL before queuing again."
            ) from exc
        response_json = json.dumps(result, ensure_ascii=False, default=str)
        return (video, video_url, get_result_seed(result), response_json)
