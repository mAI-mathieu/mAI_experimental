from ..sol_refiner.paths import model_names, resolve_model


class MAISoLH3Loader:
    CATEGORY = "mAI / Image"
    FUNCTION = "load"
    RETURN_TYPES = ("MAI_SOL_H3_REFINER",)
    RETURN_NAMES = ("refiner",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_name": (model_names(),),
            "precision": (["bf16", "fp8"], {"default": "bf16"}),
            "decoder_backend": (["auto", "flex", "natten"], {"default": "auto"}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "gpu_resident": ("BOOLEAN", {"default": True}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, model_name):
        try:
            resolve_model(model_name)
        except (OSError, ValueError) as exc:
            return str(exc)
        return True

    def load(self, model_name, precision, decoder_backend, keep_model_loaded=True, gpu_resident=True):
        directory = resolve_model(model_name)
        from ..sol_refiner.runtime import get_refiner

        return (get_refiner(directory, precision, decoder_backend,
                            keep_model_loaded, gpu_resident),)
