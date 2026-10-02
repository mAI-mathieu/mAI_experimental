"""Own cached components and request GPU residency through ComfyUI model management."""

import logging
import math
import threading
from weakref import WeakValueDictionary

import torch
from comfy import model_management

from .attention import decoder_backend
from .loader import check_environment, dependency_api, load_component
from .utils import (canvas_geometry, decoder_minimum_frames, read_json, seed_generator,
                    tiling_config, validate_images, validate_sigma)

logger = logging.getLogger(__name__)
_refiners = WeakValueDictionary()
_cache_lock = threading.RLock()


def get_refiner(directory, precision, backend, keep_loaded, resident):
    device = check_environment(precision)
    api = dependency_api()
    selected = decoder_backend(backend, device)
    key = (str(directory.resolve()), precision, selected, bool(keep_loaded), bool(resident), str(device))
    with _cache_lock:
        runtime = _refiners.get(key)
        if runtime is None:
            runtime = RefinerRuntime(directory, precision, selected, keep_loaded, resident, device, api)
            _refiners[key] = runtime
    return runtime


class RefinerRuntime:
    def __init__(self, directory, precision, backend, keep_loaded, resident, device, api):
        self.directory = directory
        self.precision, self.backend = precision, backend
        self.keep_loaded, self.resident = keep_loaded, resident
        self.device, self.api = device, api
        self.components = {}
        self.lock = threading.RLock()
        self.minimum_frames = decoder_minimum_frames(read_json(directory / "diffusion_decoder" / "config.json"))
        self.tokenizer = api["tokenizer"].from_pretrained(directory / "tokenizer", local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("[mAI SoL H3] Precision: %s. Decoder backend: %s.", precision, backend)

    def park(self, tensor):
        return tensor if self.resident else tensor.cpu()

    def offload(self, except_name=None):
        for name, patcher in self.components.items():
            if name != except_name and patcher.loaded_size() > 0:
                model_management.unload_model_and_clones(patcher)

    def activate(self, name, memory):
        model_management.throw_exception_if_processing_interrupted()
        if not self.resident:
            self.offload(except_name=name)
        if name not in self.components:
            logger.info("[mAI SoL H3] Loading %s...", name)
            self.components[name] = load_component(self.directory, name, self.precision, self.backend, self.device, self.api)
        patcher = self.components[name]
        # The other upstream components own ordinary parameters/buffers and
        # cannot safely execute under partial ComfyUI layer offload.
        model_management.load_models_gpu([patcher], memory_required=memory,
                                         force_full_load=name != "transformer")
        return patcher.model.component

    def recover_decode_oom(self):
        self.offload(except_name="diffusion_decoder")
        model_management.soft_empty_cache(force=True)
        model_management.throw_exception_if_processing_interrupted()

    def refine(self, images, prompt, fps, width, height, sigma, seed, decoder_seed, mode, size, frames, stride):
        from .pipeline import run_pipeline

        validate_images(images)
        canvas_geometry(width, height)
        validate_sigma(sigma)
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError("fps must be finite and between 1 and 240.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a prompt describing the visible scene and action in the video.")
        if mode not in ("auto", "untiled", "tiled"):
            raise ValueError(f"Unknown decoder mode: {mode}")
        seed_generator(seed, self.device)
        seed_generator(decoder_seed, self.device)
        tiles = tiling_config(size, frames, stride)
        with self.lock:
            completed = False
            try:
                result = run_pipeline(self, images, prompt, fps, width, height, sigma, seed, decoder_seed, mode, tiles)
                completed = True
                return result
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("SoL H3 ran out of CUDA memory. Use gpu_resident=false, fp8 on supported hardware, tiled decoding, or a shorter/smaller video.") from exc
            finally:
                if not completed or not self.resident or not self.keep_loaded:
                    self.offload()
                if not self.keep_loaded:
                    self.components.clear()
