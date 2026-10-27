"""Local still-image loading, including explicitly supplied absolute JSON paths."""

import hashlib
from pathlib import Path


def resolve_image_path(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Image must be a local file path")
    if "://" in name or name.startswith(("\\\\", "//")):
        raise ValueError("Use a local image path, not a URL or network share")
    path = Path(name)
    if not path.is_absolute():
        import folder_paths
        # Relative paths follow ComfyUI's input/annotation and containment rules.
        path = Path(folder_paths.get_annotated_filepath(name))
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {name}")
    return path


class H3ImageLoader:
    """Small adapter; dependencies are already included with ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        from nodes import LoadImage
        return LoadImage.INPUT_TYPES()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        try:
            resolve_image_path(image)
        except (ValueError, OSError) as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, image):
        digest = hashlib.sha256()
        with resolve_image_path(image).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_image(self, image):
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        path = resolve_image_path(image)
        try:
            with Image.open(path) as source:
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError(f"Image must be one still frame, not an animated/multipage file: {image}")
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                pixels = np.array(rgb, dtype=np.float32) / 255.0
        except OSError as exc:
            raise ValueError(f"Cannot read image {image}: {exc}") from exc
        return (torch.from_numpy(pixels).unsqueeze(0),)
