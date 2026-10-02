from ..sol_refiner.utils import DEFAULT_SIGMA


class MAISoLH3Refiner:
    CATEGORY = "mAI / Image"
    FUNCTION = "refine"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "refiner": ("MAI_SOL_H3_REFINER",),
            "images": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
            "width": ("INT", {"default": 1920, "min": 224, "max": 4096, "step": 2}),
            "height": ("INT", {"default": 1080, "min": 224, "max": 4096, "step": 2}),
            "sigma": ("FLOAT", {"default": DEFAULT_SIGMA, "min": 0.70, "max": 1.0, "step": 0.001}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "decoder_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "decoder_mode": (["auto", "untiled", "tiled"], {"default": "auto"}),
            "decoder_tile_size": ([256, 384, 512, 768, 1024], {"default": 768}),
            "decoder_tile_frames": ([16, 32, 64, 128], {"default": 128}),
            "decoder_tile_stride": ("INT", {"default": 512, "min": 8, "max": 1016, "step": 8}),
        }}

    def refine(self, refiner, images, prompt, fps, width, height, sigma, seed, decoder_seed,
               decoder_mode, decoder_tile_size, decoder_tile_frames, decoder_tile_stride):
        return (refiner.refine(images, prompt, fps, width, height, sigma, seed, decoder_seed,
                               decoder_mode, decoder_tile_size, decoder_tile_frames, decoder_tile_stride),)
