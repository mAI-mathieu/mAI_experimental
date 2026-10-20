"""Model-specific fal video arguments and response handling, without ComfyUI imports."""

from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MINIMAX_H3_ENDPOINT = "minimax/h3/image-to-video"
WAN_ENDPOINT = "alibaba/wan-3.0/image-to-video"
VIDEO_ENDPOINTS = (MINIMAX_H3_ENDPOINT, WAN_ENDPOINT)
VIDEO_RESOLUTIONS = ("model default", "480P", "768P", "2K", "4K", "480p", "720p", "1080p")
PROMPT_EXPANSION_MODES = ("disabled", "fast", "balanced", "quality")
WAN_ASPECT_RATIOS = ("adaptive", "16:9", "4:3", "1:1", "3:4", "9:16")


def validate_video_frames(first_frame: Any, last_frame: Any = None) -> None:
    """Require exactly one image per connected socket; never discard batch frames."""
    if first_frame is None:
        raise ValueError("Connect an IMAGE to first_frame.")
    for name, frame in (("first_frame", first_frame), ("last_frame", last_frame)):
        if frame is None:
            continue
        shape = getattr(frame, "shape", ())
        if (
            len(shape) != 4 or shape[0] != 1 or shape[1] < 1
            or shape[2] < 1 or shape[3] not in (1, 3, 4)
        ):
            raise ValueError(
                f"{name} must contain exactly one IMAGE with shape [1, height, width, channels] "
                "and 1, 3, or 4 channels. Select a single frame from video/batch inputs first."
            )


def build_video_arguments(
    endpoint: str,
    prompt: str,
    duration: int = 5,
    resolution: str = "model default",
    seed: int = -1,
    prompt_expansion_mode: str = "balanced",
    aspect_ratio: str = "adaptive",
    audio: bool = True,
    enable_prompt_expansion: bool = True,
    enable_thinking: bool = False,
    smart_duration: bool = False,
) -> dict[str, Any]:
    """Validate before uploads/submission and send only fields supported by the model."""
    if endpoint not in VIDEO_ENDPOINTS:
        raise ValueError(f"Unsupported video model_endpoint: {endpoint}")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")
    minimax = endpoint == MINIMAX_H3_ENDPOINT
    prompt_limit = 50000 if minimax else 20000
    if len(prompt) > prompt_limit or (minimax and not prompt.strip()):
        raise ValueError(
            f"prompt must contain {'1' if minimax else '0'}-{prompt_limit} "
            f"characters for {endpoint}."
        )
    if not isinstance(seed, int) or isinstance(seed, bool) or not -1 <= seed <= 2147483647:
        raise ValueError("seed must be -1 (random) or an integer from 0 to 2147483647.")
    for name, value in (
        ("audio", audio), ("enable_prompt_expansion", enable_prompt_expansion),
        ("enable_thinking", enable_thinking), ("smart_duration", smart_duration),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")
    if minimax and smart_duration:
        raise ValueError("smart_duration is only supported by Wan; turn it off for MiniMax H3.")
    min_duration, max_duration = (5, 15) if minimax else (2, 30)
    if not smart_duration:
        if (
            not isinstance(duration, int) or isinstance(duration, bool)
            or not min_duration <= duration <= max_duration
        ):
            raise ValueError(
                f"duration must be {min_duration}-{max_duration} whole seconds for {endpoint}."
            )

    resolutions = ("480P", "768P", "2K", "4K") if minimax else ("480p", "720p", "1080p")
    if resolution == "model default":
        resolution = "2K" if minimax else "1080p"
    if resolution not in resolutions:
        raise ValueError(
            f"resolution for {endpoint} must be one of: {', '.join(resolutions)}, model default."
        )
    arguments = {
        "prompt": prompt,
        "duration": None if smart_duration else duration,
        "resolution": resolution,
    }
    if seed >= 0:
        arguments["seed"] = seed
    if minimax:
        if prompt_expansion_mode not in PROMPT_EXPANSION_MODES:
            raise ValueError(f"Unknown prompt_expansion_mode: {prompt_expansion_mode}")
        arguments["prompt_expansion_mode"] = prompt_expansion_mode
    else:
        if aspect_ratio not in WAN_ASPECT_RATIOS:
            raise ValueError(f"Unknown Wan aspect_ratio: {aspect_ratio}")
        arguments.update(
            aspect_ratio=aspect_ratio, audio=audio,
            enable_prompt_expansion=enable_prompt_expansion, enable_thinking=enable_thinking,
        )
    return arguments


def add_video_frame_urls(
    endpoint: str, arguments: dict[str, Any], first_url: str, last_url: str | None = None,
) -> dict[str, Any]:
    if endpoint not in VIDEO_ENDPOINTS:
        raise ValueError(f"Unsupported video model_endpoint: {endpoint}")
    result = dict(arguments)
    result["image_url" if endpoint == MINIMAX_H3_ENDPOINT else "start_image_url"] = first_url
    if last_url is not None:
        result["end_image_url"] = last_url
    return result


def extract_video_url(result: Any) -> str:
    if isinstance(result, dict):
        if isinstance(result.get("data"), dict):
            result = result["data"]
        video = result.get("video")
        if isinstance(video, dict):
            url = video.get("url")
            if isinstance(url, str) and url.strip():
                if urlparse(url).scheme.lower() not in ("http", "https"):
                    raise ValueError("fal returned a video URL that is not HTTP(S).")
                return url
    raise ValueError("fal response did not contain a video.url.")


def download_video_bytes(video_url: str, timeout_seconds: int = 300) -> bytes:
    """Fetch the encoded video; frames and audio remain in the original container."""
    if urlparse(video_url).scheme.lower() not in ("http", "https"):
        raise ValueError("Video URL must use HTTP(S).")
    request = Request(video_url, headers={"User-Agent": "mAI-ComfyUI-fal/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        data = response.read()
    if not data:
        raise ValueError("fal returned an empty video download.")
    return data
