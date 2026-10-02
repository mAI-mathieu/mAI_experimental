from .nodes.example_text_node import MAIExampleTextNode
from .nodes.fal_image import MAIFalImage
from .nodes.fal_ideogram_edit import MAIFalIdeogramEdit
from .nodes.fal_image_edit import MAIFalImageEdit

NODE_CLASS_MAPPINGS = {
    "MAIExampleTextNode": MAIExampleTextNode,
    "MAIFalImage": MAIFalImage,
    "MAIFalIdeogramEdit": MAIFalIdeogramEdit,
    "MAIFalImageEdit": MAIFalImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIExampleTextNode": "mAI Example Text Node",
    "MAIFalImage": "mAI fal image",
    "MAIFalIdeogramEdit": "mAI fal ideogram edit",
    "MAIFalImageEdit": "mAI fal image edit",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
