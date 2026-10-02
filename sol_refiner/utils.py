"""Testable tensor, geometry, and configuration rules for the H3 checkpoint."""

import json
import math
from pathlib import Path

DEFAULT_MODEL = "Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3"
DEFAULT_FOLDER = DEFAULT_MODEL.split("/")[-1]
DEFAULT_SIGMA = 0.9093750119
WEIGHT_COMPONENTS = ("vae", "transformer", "connectors", "latent_upsampler", "diffusion_decoder", "text_encoder")


def read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read SoL configuration {path}: {exc}") from exc


def validate_package(directory):
    directory = Path(directory)
    index = read_json(directory / "model_index.json")
    if not isinstance(index, dict) or index.get("_class_name") != "SoLRefinerH3Pipeline":
        raise ValueError("Select the released SoL-Refiner-LTX-2.5-for-MiniMax-H3 package (SoLRefinerH3Pipeline).")
    for name in WEIGHT_COMPONENTS:
        config = read_json(directory / name / "config.json")
        if not isinstance(config, dict) or not any((directory / name).glob("*.safetensors")):
            raise ValueError(f"Incomplete SoL package: {name} needs config.json and safetensors weights.")
        for shard_index in (directory / name).glob("*.safetensors.index.json"):
            shards = read_json(shard_index).get("weight_map", {})
            if not shards:
                raise ValueError(f"Invalid checkpoint shard index: {shard_index}")
            for shard in set(shards.values()):
                path = (directory / name / shard).resolve()
                if not path.is_relative_to((directory / name).resolve()) or not path.is_file():
                    raise ValueError(f"Incomplete SoL package: missing or invalid shard {name}/{shard}.")
    if not (directory / "tokenizer" / "tokenizer_config.json").is_file():
        raise ValueError("Incomplete SoL package: tokenizer/tokenizer_config.json is missing.")
    scheduler = read_json(directory / "scheduler" / "scheduler_config.json")
    if (scheduler.get("_class_name") != "FlowMatchEulerDiscreteScheduler"
            or scheduler.get("use_dynamic_shifting", False) or scheduler.get("shift", 1) != 1
            or any(scheduler.get(k, False) for k in ("invert_sigmas", "stochastic_sampling", "use_beta_sigmas", "use_karras_sigmas", "use_exponential_sigmas"))
            or scheduler.get("shift_terminal") is not None):
        raise ValueError("SoL H3 requires an unshifted deterministic FlowMatch Euler scheduler ending at zero.")
    decoder = read_json(directory / "diffusion_decoder" / "config.json")
    if decoder.get("decoder_num_inference_steps") != 1 or decoder.get("decoder_model_output_type") != "x0":
        raise ValueError("Expected the released LTX-2.5 one-step x0 diffusion decoder.")
    return directory


def compatible_frames(frames, minimum=1, ratio=8):
    if not isinstance(frames, int) or frames < 1:
        raise ValueError("Provide at least one RGB video frame.")
    return 1 + math.ceil((max(frames, minimum) - 1) / ratio) * ratio


def decoder_minimum_frames(config):
    """Derive the minimum from the installed decoder's sample-space stage-5 kernel.

    Stages 1-4 have trailing ghost latents; stage 5 sees only actual sample frames.
    The released 11-frame kernel therefore needs 17 sample frames on an 8k+1 grid.
    """
    return compatible_frames(int(config["decoder_stage5_kernel"][0]), ratio=int(config["temporal_compression_ratio"]))


def canvas_geometry(width, height):
    for name, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool) or not 224 <= value <= 4096 or value % 2:
            raise ValueError(f"{name} must be an even integer between 224 and 4096.")
    return math.ceil(width / 64) * 64, math.ceil(height / 64) * 64


def validate_sigma(sigma):
    if not math.isfinite(sigma) or not 0 < sigma <= 1:
        raise ValueError("sigma must be finite and in (0, 1].")


def validate_images(images):
    import torch

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a ComfyUI IMAGE tensor [B, H, W, 3].")
    if images.shape[-1] != 3:
        raise ValueError("SoL H3 requires RGB images with exactly 3 channels.")
    if images.shape[0] < 1 or images.shape[1] < 1 or images.shape[2] < 1:
        raise ValueError("Provide a nonempty batch of nonempty video frames.")
    if not images.is_floating_point() or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("images must contain finite floating point RGB values in 0..1.")


def images_to_video(images, frames):
    """Comfy BHWC -> video processor BFCHW (processor returns BCFHW)."""
    import torch

    validate_images(images)
    if frames < images.shape[0]:
        raise ValueError("Internal frame count must not truncate the input.")
    if frames > images.shape[0]:
        images = torch.cat((images, images[-1:].expand(frames - images.shape[0], -1, -1, -1)))
    return images.permute(0, 3, 1, 2).unsqueeze(0)


def decoded_to_images(decoded, frames, width, height):
    if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[1] != 3:
        raise ValueError("LTX decoder must return RGB video [1, 3, F, H, W].")
    if decoded.shape[2] < frames or decoded.shape[3] < height or decoded.shape[4] < width:
        raise ValueError("LTX decoder returned fewer frames or pixels than requested.")
    top, left = (decoded.shape[3] - height) // 2, (decoded.shape[4] - width) // 2
    pixels = decoded[0, :, :frames, top:top + height, left:left + width]
    if not pixels.isfinite().all():
        raise ValueError("LTX decoder returned non-finite pixels; check precision and checkpoint compatibility.")
    return ((pixels.float() + 1) / 2).clamp(0, 1).permute(1, 2, 3, 0).contiguous().cpu()


def seed_generator(seed, device):
    import torch

    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Seeds must be integers in 0..18446744073709551615.")
    return torch.Generator(device=device).manual_seed(seed)


def tiling_config(size, frames, stride):
    if size not in (256, 384, 512, 768, 1024) or frames not in (16, 32, 64, 128):
        raise ValueError("Select a supported decoder tile size and frame count.")
    if not isinstance(stride, int) or stride % 8 or not 8 <= stride < size:
        raise ValueError("decoder_tile_stride must be a multiple of 8, smaller than decoder_tile_size.")
    return dict(tile_sample_min_height=size, tile_sample_min_width=size,
                tile_sample_stride_height=stride, tile_sample_stride_width=stride,
                tile_sample_min_num_frames=frames,
                tile_sample_stride_num_frames=frames * 5 // 8 // 2 * 2)


def select_backend(requested, natten_available, flex_available):
    if requested not in ("auto", "flex", "natten"):
        raise ValueError(f"Unknown decoder backend: {requested}")
    if requested == "natten" or (requested == "auto" and natten_available):
        if not natten_available:
            raise ValueError("NATTEN backend selected but NATTEN is missing or incompatible. Use decoder_backend=flex or install a compatible NATTEN build.")
        return "natten"
    if not flex_available:
        raise ValueError("Flex Attention is unavailable. Install a compatible Linux PyTorch/Triton environment or select a compatible NATTEN backend.")
    return "flex"
