from .nodes.example_text_node import MAIExampleTextNode
from .nodes.fal_image import MAIFalImage
from .nodes.fal_ideogram_edit import MAIFalIdeogramEdit
from .nodes.fal_image_edit import MAIFalImageEdit
from .nodes.sol_h3_loader import MAISoLH3Loader
from .nodes.sol_h3_refiner import MAISoLH3Refiner

NODE_CLASS_MAPPINGS = {
    "MAIExampleTextNode": MAIExampleTextNode,
    "MAIFalImage": MAIFalImage,
    "MAIFalIdeogramEdit": MAIFalIdeogramEdit,
    "MAIFalImageEdit": MAIFalImageEdit,
    "MAISoLH3Loader": MAISoLH3Loader,
    "MAISoLH3Refiner": MAISoLH3Refiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIExampleTextNode": "mAI Example Text Node",
    "MAIFalImage": "mAI fal image",
    "MAIFalIdeogramEdit": "mAI fal ideogram edit",
    "MAIFalImageEdit": "mAI fal image edit",
    "MAISoLH3Loader": "mAI SoL H3 Loader",
    "MAISoLH3Refiner": "mAI SoL H3 Refiner",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
