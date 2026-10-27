"""One-node MiniMax H3 reference conditioning and timed still-image guides."""

from ..utils.h3_multiframe import (
    MAX_IMAGES, NO_IMAGE, parse_reference_json, resolve_guide_frames, validate_settings,
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
        return [(path, time, None) for path, time in entries]
    entries = []
    for index in range(1, reference_count + 1):
        image = inputs.get(f"image_{index}")
        path = inputs.get(f"image_file_{index}", NO_IMAGE)
        if image is not None or path not in (None, "", NO_IMAGE):
            entries.append((path, inputs.get(f"time_{index}", (index - 1) * 1.5), image))
    return entries


class MAIH3Multiframe:
    CATEGORY = "mAI / Video"
    FUNCTION = "prepare"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    DESCRIPTION = (
        "MiniMax H3 reference images and timed still guides in one node. "
        "Upload/select local images or connect IMAGE sockets. Non-empty JSON overrides rows. "
        "Times are seconds at 24 fps; Picture tags follow entry order."
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
                "tooltip": "Time in seconds on H3's 24 fps timeline. Rounded to the nearest frame.",
            })
            optional[f"image_{index}"] = ("IMAGE", {
                "tooltip": f"Optional single-image input overriding image_file_{index}.",
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
                    "tooltip": "Also expose images as semantic <Picture N> references. Off: timed guides only.",
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
            length = validate_settings(
                kwargs.get("width", 1344), kwargs.get("height", 768),
                kwargs.get("duration", 5.0), kwargs.get("reference_count", 2),
                kwargs.get("ref_image_size", "match"),
            )
            entries = parse_reference_json(kwargs.get("reference_json", ""))
            if entries is not None:
                resolve_guide_frames([time for _, time in entries], length)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, reference_json="", reference_count=2, **kwargs):
        loader = get_image_loader()
        hashes = []
        for path, _, image in file_entries(reference_json, reference_count, kwargs):
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
        frames = resolve_guide_frames([time for _, time, _ in entries], length)
        loader = get_image_loader()
        # Validate every path before loading images or running any model encoding.
        for index, (path, _, image) in enumerate(entries, 1):
            if image is None:
                error = loader.VALIDATE_INPUTS(path)
                if error is not True:
                    raise ValueError(f"Image {index}: {error}")
        images = []
        for index, (path, _, image) in enumerate(entries, 1):
            if image is None:
                image = loader().load_image(path)[0]
            shape = getattr(image, "shape", ())
            if len(shape) != 4 or shape[0] != 1 or min(shape[1:3]) < 1 or shape[3] not in (3, 4):
                raise ValueError(f"Image {index} must be one RGB/RGBA still image [1,H,W,3/4]; batches and animated files are unsupported")
            images.append(image[..., :3])
        reference_node, guide_node = get_h3_nodes()
        refs = {f"ref_image_{index}": image for index, image in enumerate(images)} if use_as_references else None
        positive, latent = reference_node.execute(
            clip=clip, vae=vae, prompt=prompt, width=width, height=height,
            length=length, ref_image_size=ref_image_size, ref_images=refs,
        ).result
        for image, frame in zip(images, frames):
            positive = guide_node.execute(
                positive=positive, latent=latent, vae=vae, image=image, frame_idx=frame,
            ).result[0]
        return positive, latent
