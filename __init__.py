from .nodes.example_text_node import MAIExampleTextNode
from .nodes.fal_image import MAIFalImage

NODE_CLASS_MAPPINGS = {
    "MAIExampleTextNode": MAIExampleTextNode,
    "MAIFalImage": MAIFalImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIExampleTextNode": "mAI Example Text Node",
    "MAIFalImage": "mAI fal image",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
