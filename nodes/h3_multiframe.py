"""One-node MiniMax H3 reference conditioning and timed still-image guides."""

from ..utils.h3_multiframe import (
    MAX_IMAGES, NO_IMAGE, REFERENCE_MODES, parse_reference_json, resolve_guide_frames,
    resolve_picture_roles, validate_settings,
)
from ..utils.h3_reference_images import H3ImageLoader


def get_image_loader():
    return H3ImageLoader


def get_h3_nodes():
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
    except ImportError as exc:
        raise RuntimeError(
            "mAI H3 multiframe requires ComfyUI's native MiniMax H3 Reference to Video "
            "and Add Guide nodes. Update ComfyUI and check its H3 dependencies."
        ) from exc
    return MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide


def file_entries(reference_json, reference_count, inputs):
    entries = parse_reference_json(reference_json)
    if entries is not None:
        return [(path, time, None, flag) for path, time, flag in entries]
    entries = []
    for index in range(1, reference_count + 1):
        image = inputs.get(f"image_{index}")
        path = inputs.get(f"image_file_{index}", NO_IMAGE)
        if image is not None or path not in (None, "", NO_IMAGE):
            entries.append((path, inputs.get(f"time_{index}", (index - 1) * 1.5), image,
                            inputs.get(f"use_as_reference_{index}", "default")))
    return entries


class MAIH3Multiframe:
    CATEGORY = "mAI / Video"
    FUNCTION = "prepare"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    DESCRIPTION = (
        "MiniMax H3 reference images and timed still guides in one node. "
        "Upload/select local images or connect IMAGE sockets. Non-empty JSON overrides rows. "
        "Each image is either a semantic reference without a timestamp or a timed image guide. "
        "Times are seconds at 24 fps; Picture tags number reference-enabled images in entry order."
    )

    @classmethod
    def INPUT_TYPES(cls):
        files = get_image_loader().INPUT_TYPES()["required"]["image"][0]
        optional = {}
        for index in range(1, MAX_IMAGES + 1):
            optional[f"image_file_{index}"] = ([NO_IMAGE, *files], {
                "default": NO_IMAGE, "image_upload": True,
                "tooltip": f"Image {index}. A connected image_{index} overrides this file.",
            })
            optional[f"time_{index}"] = ("FLOAT", {
                "default": (index - 1) * 1.5, "min": 0.0, "max": 15.0, "step": 0.01,
                "tooltip": "Timed guides only: seconds on H3's 24 fps timeline. Ignored for reference images.",
            })
            optional[f"image_{index}"] = ("IMAGE", {
                "tooltip": f"Optional single-image input overriding image_file_{index}.",
            })
        # Append new fields so the released input order remains intact.
        for index in range(1, MAX_IMAGES + 1):
            optional[f"use_as_reference_{index}"] = (list(REFERENCE_MODES), {
                "default": "default",
                "tooltip": "yes: reference only, no timestamp; no: timed image conditioning; default: follow use_as_references.",
            })
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 15.0, "step": 0.1}),
                "reference_count": ("INT", {"default": 2, "min": 1, "max": MAX_IMAGES}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "use_as_references": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Default reference use. Each image can override this with use_as_reference_N.",
                }),
                "reference_json": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": 'Overrides rows when non-empty. Example: [{"image":"shot.png","time":0}]',
                }),
            },
            "optional": optional,
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Custom validation lets unused/overridden file selectors keep saved values,
        # even when those files are no longer in the input directory's combo list.
        try:
            # ComfyUI supplies None for connected values during queue validation.
            # Validate the known literals here; prepare() validates every actual
            # resolved value before loading images or encoding conditioning.
            settings = {
                "width": 1344, "height": 768, "duration": 5.0,
                "reference_count": 2, "ref_image_size": "match",
            }
            settings.update({name: kwargs[name] for name in settings if kwargs.get(name) is not None})
            length = validate_settings(**settings)
            reference_json = kwargs.get("reference_json", "")
            entries = parse_reference_json(reference_json) if reference_json is not None else None
            reference_default = kwargs.get("use_as_references", True)
            if entries is not None and reference_default is not None:
                _, times = resolve_picture_roles(entries, reference_default)
                if times and kwargs.get("duration", 5.0) is not None:
                    resolve_guide_frames(times, length)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, reference_json="", reference_count=2, **kwargs):
        # Fingerprints are also requested before linked values are available.
        # Without the JSON/count we cannot know which local files to hash yet.
        if reference_json is None or reference_count is None:
            return float("nan")
        loader = get_image_loader()
        hashes = []
        for path, _, image, _ in file_entries(reference_json, reference_count, kwargs):
            if image is None:
                if loader.VALIDATE_INPUTS(path) is not True:
                    return float("nan")
                hashes.append(loader.IS_CHANGED(path))
        return tuple(hashes)

    def prepare(self, clip, vae, prompt, width=1344, height=768, duration=5.0,
                reference_count=2, ref_image_size="match", use_as_references=True,
                reference_json="", **kwargs):
        length = validate_settings(width, height, duration, reference_count, ref_image_size)
        if not isinstance(use_as_references, bool):
            raise ValueError("use_as_references must be true or false")
        entries = file_entries(reference_json, reference_count, kwargs)
        reference_flags, times = resolve_picture_roles(
            [(path, time, flag) for path, time, _, flag in entries], use_as_references,
        )
        frames = iter(resolve_guide_frames(times, length) if times else [])
        loader = get_image_loader()
        # Validate every path before loading images or running any model encoding.
        for index, (path, _, image, _) in enumerate(entries, 1):
            if image is None:
                error = loader.VALIDATE_INPUTS(path)
                if error is not True:
                    raise ValueError(f"Image {index}: {error}")
        images = []
        for index, (path, _, image, _) in enumerate(entries, 1):
            if image is None:
                image = loader().load_image(path)[0]
            shape = getattr(image, "shape", ())
            if len(shape) != 4 or shape[0] != 1 or min(shape[1:3]) < 1 or shape[3] not in (3, 4):
                raise ValueError(f"Image {index} must be one RGB/RGBA still image [1,H,W,3/4]; batches and animated files are unsupported")
            images.append(image[..., :3])
        reference_node, guide_node = get_h3_nodes()
        reference_images = [image for image, enabled in zip(images, reference_flags) if enabled]
        refs = {f"ref_image_{index}": image for index, image in enumerate(reference_images)} or None
        positive, latent = reference_node.execute(
            clip=clip, vae=vae, prompt=prompt, width=width, height=height,
            length=length, ref_image_size=ref_image_size, ref_images=refs,
        ).result
        for image, is_reference in zip(images, reference_flags):
            if is_reference:
                continue
            positive = guide_node.execute(
                positive=positive, latent=latent, vae=vae, image=image, frame_idx=next(frames),
            ).result[0]
        return positive, latent
