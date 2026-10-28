"""Pure validation and timeline conversion for MiniMax H3 still-image guides."""

import json
import math

FPS = 24
MAX_IMAGES = 9
NO_IMAGE = "(none)"
REFERENCE_MODES = ("default", "yes", "no")


def resolve_reference_flag(value, default):
    if not isinstance(default, bool):
        raise ValueError("use_as_references must be true or false")
    if value is None or value == "default":
        return default
    if isinstance(value, bool):
        return value
    if value == "yes":
        return True
    if value == "no":
        return False
    raise ValueError("use_as_reference must be 'default', 'yes', 'no', or a boolean")


def validate_settings(width, height, duration, reference_count, ref_image_size):
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 32 <= value <= 16384 or value % 32:
            raise ValueError(f"{name} must be a multiple of 32 between 32 and 16384")
    if isinstance(reference_count, bool) or not isinstance(reference_count, int) or not 1 <= reference_count <= MAX_IMAGES:
        raise ValueError(f"reference_count must be between 1 and {MAX_IMAGES}")
    if ref_image_size not in ("match", "max"):
        raise ValueError("ref_image_size must be 'match' or 'max'")
    return duration_to_length(duration)


def duration_to_length(duration):
    """Match the example workflow: round seconds at 24 fps, then snap up to 17k+5."""
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 15:
        raise ValueError("duration must be finite and greater than 0, up to 15 seconds")
    frames = max(5, round(duration * FPS))
    return frames + (5 - frames) % 17


def parse_reference_json(reference_json):
    """Return local file/time/reference triples, or None for widget/socket inputs."""
    if not isinstance(reference_json, str):
        raise ValueError("reference_json must be a JSON string")
    if not reference_json.strip():
        return None
    try:
        entries = json.loads(reference_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid reference_json: {exc.msg} (line {exc.lineno}, column {exc.colno})") from exc
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_IMAGES:
        raise ValueError(f"reference_json must be a list of 1 to {MAX_IMAGES} image/time objects")
    result = []
    for index, entry in enumerate(entries, 1):
        if (not isinstance(entry, dict) or "image" not in entry
                or set(entry) - {"image", "time", "use_as_reference"}):
            raise ValueError(f"JSON entry {index} needs 'image', with optional 'time' and 'use_as_reference'")
        path = entry["image"]
        if not isinstance(path, str) or not path.strip() or path == NO_IMAGE:
            raise ValueError(f"JSON entry {index}: image must be a local file path")
        if "://" in path or path.startswith(("\\\\", "//")):
            raise ValueError(f"JSON entry {index}: use local file paths, not URLs or network shares")
        flag = entry.get("use_as_reference")
        if "use_as_reference" in entry and not isinstance(flag, bool):
            raise ValueError(f"JSON entry {index}: use_as_reference must be true or false")
        result.append((path, entry.get("time"), flag))
    return result


def resolve_picture_roles(entries, default):
    """Separate reference-only images from timed guides; validate guide times only."""
    if not entries:
        raise ValueError("Add at least one reference image or provide reference_json")
    flags = [resolve_reference_flag(flag, default) for _, _, flag in entries]
    guide_times = [time for (_, time, _), is_reference in zip(entries, flags) if not is_reference]
    return flags, guide_times


def resolve_guide_frames(times, length):
    """Validate all times before encoding. Keep input order for stable Picture tags."""
    frames = []
    seen = set()
    for index, seconds in enumerate(times, 1):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Image {index}: time must be finite, non-negative seconds")
        frame = round(seconds * FPS)
        if frame >= length:
            raise ValueError(f"Image {index}: time {seconds}s maps to frame {frame}, outside {length} frames (last frame at {(length - 1) / FPS:.4f}s)")
        if frame in seen:
            raise ValueError(f"Image {index}: another image already uses frame {frame}; choose distinct times")
        seen.add(frame)
        frames.append(frame)
    if not frames:
        raise ValueError("Add at least one reference image or provide reference_json")
    return frames
