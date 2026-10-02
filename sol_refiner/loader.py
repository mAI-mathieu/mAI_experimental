"""Independent local component loading and ComfyUI parameter/offload integration."""

import json

import torch
from torch import nn
from safetensors import safe_open

from comfy import model_management, model_patcher, ops, quant_ops, utils as comfy_utils

from .utils import read_json


def check_environment(precision):
    if precision not in ("bf16", "fp8"):
        raise ValueError(f"Unsupported SoL precision: {precision}. Select bf16 or fp8; NVFP4 is not implemented.")
    device = model_management.get_torch_device()
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.cuda is None:
        raise ValueError("SoL H3 requires NVIDIA CUDA. CPU and ROCm execution are unsupported.")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This NVIDIA GPU/PyTorch combination does not support BF16 required by SoL H3.")
    if model_management.vram_state == model_management.VRAMState.NO_VRAM:
        raise ValueError("SoL H3 requires full text/VAE/decoder loads; restart ComfyUI without --novram.")
    if precision == "fp8":
        if not model_management.supports_fp8_compute(device):
            raise ValueError("FP8 selected on unsupported hardware. Select bf16.")
        if (not hasattr(ops, "mixed_precision_ops")
                or quant_ops.get_layout_class("TensorCoreFP8Layout") is None):
            raise ValueError("FP8 requires compatible ComfyUI mixed_precision_ops and comfy-kitchen. Select bf16.")
    return device


def dependency_api():
    try:
        from diffusers import (AutoencoderKLLTX2Video, LTX2ConditionPipeline,
                               LTX2VideoDiffusionDecoderModel, LTX2VideoTransformer3DModel)
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer, AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface
    except (ImportError, AttributeError) as exc:
        raise ImportError("SoL H3 needs Diffusers with LTX-2.5 diffusion-decoder APIs (reference commit e0abab83b5df05de9e7abd788643c1a7c1e42e28) and Transformers with Gemma4Unified support (>=5.12.1). See requirements-sol.txt; PyTorch must not be replaced.") from exc
    if not all(hasattr(LTX2VideoDiffusionDecoderModel, name) for name in ("enable_tiling", "disable_tiling", "tiled_decode")):
        raise ImportError("Installed Diffusers lacks the required LTX-2.5 tiled/untiled decoder API.")
    return dict(vae=AutoencoderKLLTX2Video, connectors=LTX2TextConnectors,
                latent_upsampler=LTX2LatentUpsamplerModel,
                text_encoder=AutoModelForImageTextToText, tokenizer=AutoTokenizer)


def patcher_for(module, device):
    # Diffusers' device is a read-only property. A plain owning module exposes
    # writable device metadata to ModelPatcher without copying any parameters.
    owner = nn.Module()
    owner.add_module("component", module)
    return model_patcher.ModelPatcher(owner, device, torch.device("cpu"))


def prepare_operations(module, precision, loaded=False):
    quantized = set()
    mixed = ops.mixed_precision_ops(compute_dtype=torch.bfloat16) if precision == "fp8" else None
    for path, layer in list(module.named_modules()):
        if not loaded and isinstance(layer, nn.RMSNorm):
            module.set_submodule(path, ops.manual_cast.RMSNorm(
                layer.normalized_shape, eps=layer.eps, elementwise_affine=layer.weight is not None,
                device="meta", dtype=torch.bfloat16))
            continue
        if not loaded and isinstance(layer, nn.LayerNorm):
            module.set_submodule(path, ops.manual_cast.LayerNorm(
                layer.normalized_shape, eps=layer.eps, elementwise_affine=layer.weight is not None,
                bias=layer.bias is not None, device="meta", dtype=torch.bfloat16))
            continue
        if not isinstance(layer, nn.Linear):
            continue
        quantize = (precision == "fp8" and path.startswith("transformer_blocks.")
                    and min(layer.in_features, layer.out_features) >= 128)
        operation = mixed.Linear if quantize else ops.manual_cast.Linear
        replacement = operation(layer.in_features, layer.out_features, bias=layer.bias is not None,
                                device="cpu" if quantize else "meta", dtype=torch.bfloat16)
        if loaded:
            replacement.weight, replacement.bias = layer.weight, layer.bias
        module.set_submodule(path, replacement)
        if quantize:
            quantized.add(path)
    return quantized


def read_weights(module, directory, precision, device):
    # Capture shapes before replacing FP8 linears, which do not own weights yet.
    expected = {name: tuple(value.shape) for name, value in module.state_dict().items()}
    quantized = prepare_operations(module, precision)
    state = {}
    loaded_count = 0
    progress = comfy_utils.ProgressBar(len(expected))
    if quantized:
        model_management.free_memory(2 * 1024**3, device)
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name not in expected:
                    continue  # Encoder-only/video-only views omit unused branches.
                model_management.throw_exception_if_processing_interrupted()
                tensor = weights.get_tensor(name)
                if tuple(tensor.shape) != expected[name] or name in state:
                    raise ValueError(f"Incompatible or duplicate checkpoint tensor {directory.name}/{name}.")
                path = name.removesuffix(".weight")
                if name.endswith(".weight") and path in quantized:
                    values, metadata = quant_ops.TensorCoreFP8Layout.quantize(
                        tensor.to(device=device, dtype=torch.bfloat16), scale="recalculate")
                    state[name] = values.cpu()
                    state[path + ".weight_scale"] = metadata.scale.cpu()
                    state[path + ".comfy_quant"] = torch.tensor(
                        list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)
                    del values, metadata
                else:
                    state[name] = tensor.to(dtype=torch.bfloat16)
                loaded_count += 1
                progress.update_absolute(loaded_count)
    missing = expected.keys() - state.keys()
    if missing:
        raise ValueError(f"Incomplete checkpoint {directory}: missing {sorted(missing)[:8]}.")
    module.load_state_dict(state, strict=True, assign=True)
    unresolved = [name for name, tensor in (*module.named_parameters(), *module.named_buffers()) if tensor.is_meta]
    if unresolved:
        raise ValueError(f"Uninitialized checkpoint tensors in {directory}: {unresolved[:8]}.")
    module.eval()


def load_component(directory, name, precision, backend, device, api):
    from .models import VideoTransformer, use_comfy_attention
    from .decoder import make_decoder

    source = directory / name
    try:
        if name == "text_encoder":
            # Load once; discard the output head and unused modality branches.
            parent = api[name].from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                              dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                              attn_implementation="mai_sol_comfy")
            module = parent.model
            del parent
            prepare_operations(module, "bf16", loaded=True)
            module.eval()
        else:
            config = read_json(source / "config.json")
            with torch.device("meta"):
                if name == "transformer":
                    module = VideoTransformer.from_config(config)
                elif name == "diffusion_decoder":
                    module = make_decoder(config, backend)
                else:
                    module = api[name].from_config(config)
                    if name == "vae":
                        module.decoder = None
            read_weights(module, source, precision if name == "transformer" else "bf16", device)
            use_comfy_attention(module)
        return patcher_for(module, device)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        raise ValueError(f"Cannot load SoL component '{name}' from {source}: {exc}") from exc
