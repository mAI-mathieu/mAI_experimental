from .nodes.example_text_node import MAIExampleTextNode
from .nodes.fal_image import MAIFalImage
from .nodes.fal_ideogram_edit import MAIFalIdeogramEdit
from .nodes.fal_image_edit import MAIFalImageEdit
from .nodes.fal_video import MAIFalVideo
from .nodes.sol_h3_loader import MAISoLH3Loader
from .nodes.sol_h3_refiner import MAISoLH3Refiner
from .nodes.h3_multiframe import MAIH3Multiframe

NODE_CLASS_MAPPINGS = {
    "MAIExampleTextNode": MAIExampleTextNode,
    "MAIFalImage": MAIFalImage,
    "MAIFalIdeogramEdit": MAIFalIdeogramEdit,
    "MAIFalImageEdit": MAIFalImageEdit,
    "MAIFalVideo": MAIFalVideo,
    "MAISoLH3Loader": MAISoLH3Loader,
    "MAISoLH3Refiner": MAISoLH3Refiner,
    "MAIH3Multiframe": MAIH3Multiframe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIExampleTextNode": "mAI Example Text Node",
    "MAIFalImage": "mAI fal image",
    "MAIFalIdeogramEdit": "mAI fal ideogram edit",
    "MAIFalImageEdit": "mAI fal image edit",
    "MAIFalVideo": "mAI fal video",
    "MAISoLH3Loader": "mAI SoL H3 Loader",
    "MAISoLH3Refiner": "mAI SoL H3 Refiner",
    "MAIH3Multiframe": "mAI H3 multiframe",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
