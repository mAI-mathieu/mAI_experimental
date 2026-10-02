import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from ..sol_refiner import paths
from ..sol_refiner.utils import (
    DEFAULT_MODEL, DEFAULT_SIGMA, WEIGHT_COMPONENTS, canvas_geometry,
    compatible_frames, decoded_to_images, decoder_minimum_frames, images_to_video,
    seed_generator, select_backend, tiling_config, validate_images,
    validate_package, validate_sigma,
)


@pytest.mark.parametrize("requested,expected", [((1920, 1080), (1920, 1088)), ((2048, 1152), (2048, 1152))])
def test_canvas_geometry(requested, expected):
    assert canvas_geometry(*requested) == expected


@pytest.mark.parametrize("frames,aligned", [(1, 1), (8, 9), (9, 9), (121, 121), (124, 129), (158, 161)])
def test_temporal_alignment_and_decoder_minimum(frames, aligned):
    config = {"decoder_stage5_kernel": [11, 11, 11], "temporal_compression_ratio": 8}
    minimum = decoder_minimum_frames(config)
    assert minimum == 17
    assert compatible_frames(frames) == aligned
    internal = compatible_frames(frames, minimum)
    assert internal == max(17, aligned)
    images = torch.rand(frames, 2, 3, 3)
    video = images_to_video(images, internal)
    assert video.shape == (1, internal, 3, 2, 3)
    if internal > frames:
        torch.testing.assert_close(video[0, frames:], video[0, frames - 1:frames].expand(internal - frames, -1, -1, -1))
    decoded = video.permute(0, 2, 1, 3, 4) * 2 - 1
    output = decoded_to_images(decoded, frames, 3, 2)
    assert output.shape == images.shape
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, images)


@pytest.mark.parametrize("sigma", [0, -1, 1.0001, float("nan"), float("inf")])
def test_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        validate_sigma(sigma)


def test_reference_and_experimental_sigma():
    for sigma in (DEFAULT_SIGMA, 0.82, 0.86, 0.89, 0.92, 0.94, 1):
        validate_sigma(sigma)


@pytest.mark.parametrize("images", [torch.zeros(2, 4, 3), torch.zeros(2, 4, 4, 4), torch.zeros(0, 4, 4, 3), torch.zeros(1, 0, 4, 3), torch.zeros(1, 2, 2, 3, dtype=torch.uint8), torch.full((1, 2, 2, 3), float("nan")), torch.full((1, 2, 2, 3), 1.1)])
def test_invalid_images(images):
    with pytest.raises(ValueError):
        validate_images(images)


def test_center_crop_and_range():
    decoded = torch.zeros(1, 3, 5, 6, 8, dtype=torch.bfloat16)
    decoded[:, :, :, 1:5, 1:7] = 2
    result = decoded_to_images(decoded, 3, 6, 4)
    assert result.shape == (3, 4, 6, 3)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.equal(result, torch.ones_like(result))


def test_independent_seed_streams():
    refiner = seed_generator(0, "cpu")
    decoder = seed_generator(0, "cpu")
    expected = torch.randn(10, generator=seed_generator(0, "cpu"))
    torch.testing.assert_close(torch.randn(10, generator=refiner), expected)
    torch.randn(100, generator=refiner)
    torch.testing.assert_close(torch.randn(10, generator=decoder), expected)
    assert not torch.equal(expected, torch.randn(10, generator=seed_generator(1, "cpu")))


@pytest.mark.parametrize("choice,natten,flex,expected", [("auto", True, True, "natten"), ("auto", False, True, "flex"), ("flex", True, True, "flex"), ("natten", True, False, "natten")])
def test_backend_selection(choice, natten, flex, expected):
    assert select_backend(choice, natten, flex) == expected


def test_explicit_natten_does_not_fallback():
    with pytest.raises(ValueError, match="NATTEN"):
        select_backend("natten", False, True)
    with pytest.raises(ValueError, match="Flex"):
        select_backend("auto", False, False)


def test_tile_defaults_and_validation():
    tiles = tiling_config(768, 128, 512)
    assert tiles == dict(tile_sample_min_height=768, tile_sample_min_width=768,
                         tile_sample_stride_height=512, tile_sample_stride_width=512,
                         tile_sample_min_num_frames=128, tile_sample_stride_num_frames=80)
    for size in (256, 384, 512, 768, 1024):
        for frames in (16, 32, 64, 128):
            assert tiling_config(size, frames, size // 2)["tile_sample_stride_num_frames"] < frames
    with pytest.raises(ValueError, match="stride"):
        tiling_config(256, 16, 512)


@pytest.fixture
def model_package(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    root = tmp_path / DEFAULT_MODEL.split("/")[-1]
    write(root / "model_index.json", {"_class_name": "SoLRefinerH3Pipeline"})
    for name in WEIGHT_COMPONENTS:
        write(root / name / "config.json", {})
        (root / name / "model.safetensors").touch()
    write(root / "diffusion_decoder" / "config.json",
          {"decoder_num_inference_steps": 1, "decoder_model_output_type": "x0",
           "decoder_stage5_kernel": [11, 11, 11], "temporal_compression_ratio": 8})
    write(root / "scheduler" / "scheduler_config.json",
          {"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 1, "use_dynamic_shifting": False})
    write(root / "tokenizer" / "tokenizer_config.json", {})
    return root


def test_valid_model_package(model_package):
    assert validate_package(model_package) == model_package


@pytest.mark.parametrize("index", [{"_class_name": "OtherPipeline"}, [], {}])
def test_wrong_model_index(model_package, index):
    (model_package / "model_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="SoLRefinerH3Pipeline"):
        validate_package(model_package)


def test_missing_shard(model_package):
    index = model_package / "transformer" / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="shard"):
        validate_package(model_package)


@pytest.mark.parametrize("change", [{"shift": 2}, {"use_dynamic_shifting": True}, {"invert_sigmas": True}, {"stochastic_sampling": True}, {"shift_terminal": 0.1}])
def test_incompatible_scheduler(model_package, change):
    config = {"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 1}
    config.update(change)
    (model_package / "scheduler" / "scheduler_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unshifted"):
        validate_package(model_package)


def test_model_discovery_extra_paths_and_containment(monkeypatch, model_package, tmp_path):
    roots = {"diffusers": ([str(tmp_path)], {"folder"}), "sol_refiner": ([str(tmp_path / "extra")], set())}
    folder_paths = ModuleType("folder_paths")
    folder_paths.models_dir = str(tmp_path / "models")
    folder_paths.folder_names_and_paths = roots
    folder_paths.get_folder_paths = lambda name: roots[name][0].copy()

    def add(name, path):
        if path not in roots[name][0]:
            roots[name][0].append(path)

    folder_paths.add_model_folder_path = add
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    assert paths.resolve_model(DEFAULT_MODEL) == model_package
    assert model_package.name in paths.model_names()
    assert str(tmp_path / "extra") in [str(p) for p in paths.model_roots()]
    for name in ("../model", "..\\model", "/absolute", "C:weights", ".."):
        with pytest.raises(ValueError):
            paths.resolve_model(name)


def test_missing_model_reports_searched_paths(monkeypatch, tmp_path):
    roots = [tmp_path / "sol_refiner", tmp_path / "diffusers"]
    monkeypatch.setattr(paths, "model_roots", lambda: roots)
    with pytest.raises(FileNotFoundError) as error:
        paths.resolve_model(DEFAULT_MODEL)
    for root in roots:
        assert str(root / DEFAULT_MODEL.split("/")[-1] / "model_index.json") in str(error.value)
    assert "generation weights are a separate model" in str(error.value)


def test_selected_incomplete_model_reports_missing_file(monkeypatch, model_package):
    monkeypatch.setattr(paths, "model_roots", lambda: [model_package.parent])
    (model_package / "transformer" / "config.json").unlink()
    with pytest.raises(ValueError, match="transformer.*config.json"):
        paths.resolve_model(DEFAULT_MODEL)


def test_selected_wrong_pipeline_reports_real_error(monkeypatch, model_package):
    monkeypatch.setattr(paths, "model_roots", lambda: [model_package.parent])
    (model_package / "model_index.json").write_text(json.dumps({"_class_name": "OtherPipeline"}))
    with pytest.raises(ValueError, match="SoLRefinerH3Pipeline"):
        paths.resolve_model(DEFAULT_MODEL)


def test_queue_validation_rejects_missing_model_without_runtime_import(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "model_roots", lambda: [tmp_path])
    loader = NODE_CLASS_MAPPINGS["MAISoLH3Loader"]
    runtime_name = loader.__module__.rsplit(".", 2)[0] + ".sol_refiner.runtime"
    assert runtime_name not in sys.modules
    result = loader.VALIDATE_INPUTS(DEFAULT_MODEL)
    assert isinstance(result, str) and "was not found" in result
    assert runtime_name not in sys.modules


def test_queue_validation_accepts_complete_local_model(monkeypatch, model_package):
    monkeypatch.setattr(paths, "model_roots", lambda: [model_package.parent])
    loader = NODE_CLASS_MAPPINGS["MAISoLH3Loader"]
    assert loader.VALIDATE_INPUTS(DEFAULT_MODEL) is True
    assert loader.VALIDATE_INPUTS(model_package.name) is True
    assert isinstance(loader.VALIDATE_INPUTS("../model"), str)


def test_node_registration_and_defaults():
    assert set(NODE_CLASS_MAPPINGS) == {"MAIExampleTextNode", "MAIFalImage", "MAIFalIdeogramEdit", "MAIFalImageEdit", "MAISoLH3Loader", "MAISoLH3Refiner"}
    loader = NODE_CLASS_MAPPINGS["MAISoLH3Loader"]
    refiner = NODE_CLASS_MAPPINGS["MAISoLH3Refiner"]
    assert NODE_DISPLAY_NAME_MAPPINGS["MAISoLH3Loader"] == "mAI SoL H3 Loader"
    assert NODE_DISPLAY_NAME_MAPPINGS["MAISoLH3Refiner"] == "mAI SoL H3 Refiner"
    assert loader.RETURN_TYPES == ("MAI_SOL_H3_REFINER",)
    assert refiner.RETURN_TYPES == ("IMAGE",)
    inputs = refiner.INPUT_TYPES()["required"]
    assert inputs["sigma"][1]["default"] == DEFAULT_SIGMA
    assert inputs["decoder_mode"][1]["default"] == "auto"
    assert not {"cfg", "steps", "negative_prompt"} & inputs.keys()


@pytest.fixture
def pipeline_module(monkeypatch):
    diffusers = ModuleType("diffusers")
    diffusers.LTX2ConditionPipeline = SimpleNamespace(
        _normalize_latents=lambda x, mean, std, scale: (x - mean.view(1, -1, 1, 1, 1).to(x)) * scale / std.view(1, -1, 1, 1, 1).to(x),
        _denormalize_latents=lambda x, mean, std, scale: x * std.view(1, -1, 1, 1, 1).to(x) / scale + mean.view(1, -1, 1, 1, 1).to(x))
    processor = ModuleType("diffusers.video_processor")

    class VideoProcessor:
        def __init__(self, vae_scale_factor):
            assert vae_scale_factor == 32

        def preprocess_video(self, video, height, width):
            assert video.ndim == 5 and video.shape[2] == 3
            pixels = torch.nn.functional.interpolate(video[0], (height, width), mode="bilinear")
            return pixels.permute(1, 0, 2, 3).unsqueeze(0) * 2 - 1

    processor.VideoProcessor = VideoProcessor
    comfy = ModuleType("comfy")
    comfy.model_management = SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None)
    comfy.utils = SimpleNamespace(ProgressBar=lambda total: SimpleNamespace(update_absolute=lambda done: None))
    for name, module in (("diffusers", diffusers), ("diffusers.video_processor", processor), ("comfy", comfy)):
        monkeypatch.setitem(sys.modules, name, module)
    package = __package__.rsplit(".", 1)[0]
    spec = importlib.util.spec_from_file_location(f"{package}.sol_refiner._test_pipeline", Path(__file__).parents[1] / "sol_refiner" / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_transformer_call_euler_and_seed(pipeline_module):
    calls = []
    latent = torch.ones(1, 3, 3, 2, 2, dtype=torch.bfloat16)

    def transformer(noisy, context, mask, sigma, fps, progress):
        calls.append(noisy.clone())
        return torch.full_like(noisy, 2)

    result = pipeline_module.one_step(transformer, latent, None, None, DEFAULT_SIGMA, 24, 7, None)
    assert len(calls) == 1
    noise = torch.randn(latent.shape, dtype=latent.dtype, generator=seed_generator(7, "cpu"))
    expected_noisy = ((1 - DEFAULT_SIGMA) * latent.float() + DEFAULT_SIGMA * noise.float()).bfloat16()
    torch.testing.assert_close(calls[0], expected_noisy)
    torch.testing.assert_close(result, expected_noisy.float() - DEFAULT_SIGMA * 2)


class FakeDecoder:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []
        self.use_tiling = False
        self.tiles = None

    def enable_tiling(self, **tiles):
        self.use_tiling, self.tiles = True, tiles

    def disable_tiling(self):
        self.use_tiling = False

    def decode(self, latent, generator, num_inference_steps):
        assert num_inference_steps == 1
        noise = torch.randn(10, generator=generator)
        self.calls.append((self.use_tiling, noise))
        if len(self.calls) == 1 and self.failure:
            raise self.failure
        return SimpleNamespace(sample=latent)


@pytest.mark.parametrize("mode,tiled", [("auto", False), ("untiled", False), ("tiled", True)])
def test_real_decoder_mode_configuration(pipeline_module, mode, tiled):
    decoder = FakeDecoder()
    decoder.use_tiling = True  # A cached previous tiled run must be reset.
    pipeline_module.decode_video(decoder, torch.zeros(1), mode, tiling_config(768, 128, 512), 4, lambda: None)
    assert decoder.calls[0][0] is tiled
    assert len(decoder.calls) == 1


def test_auto_retries_only_cuda_oom_and_resets_seed(pipeline_module):
    decoder = FakeDecoder(torch.cuda.OutOfMemoryError("test OOM"))
    recovered = []
    pipeline_module.decode_video(decoder, torch.zeros(1), "auto", tiling_config(768, 128, 512), 4, lambda: recovered.append(True))
    assert recovered == [True]
    assert [call[0] for call in decoder.calls] == [False, True]
    torch.testing.assert_close(decoder.calls[0][1], decoder.calls[1][1])
    for error in (RuntimeError("ordinary failure"), ValueError("bad shape")):
        decoder = FakeDecoder(error)
        with pytest.raises(type(error), match=str(error)):
            pipeline_module.decode_video(decoder, torch.zeros(1), "auto", {}, 4, lambda: pytest.fail("Must not retry"))
        assert len(decoder.calls) == 1


def test_explicit_untiled_does_not_retry(pipeline_module):
    decoder = FakeDecoder(torch.cuda.OutOfMemoryError("test OOM"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        pipeline_module.decode_video(decoder, torch.zeros(1), "untiled", {}, 0, lambda: pytest.fail("Must not retry"))
    assert len(decoder.calls) == 1


def test_full_tensor_pipeline_preserves_frames_and_uses_mode(pipeline_module):
    stages = []
    transformer_calls = []
    encoded_pixels = []

    class Tokens:
        input_ids = torch.zeros(1, 1024, dtype=torch.long)
        attention_mask = torch.ones(1, 1024)

        def to(self, device):
            return self

    def encode(pixels):
        encoded_pixels.append(pixels.shape)
        latent = torch.nn.functional.interpolate(pixels, size=(3, 4, 4), mode="trilinear")
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def transformer(latent, context, mask, sigma, fps, progress):
        transformer_calls.append(True)
        progress(1, 1)
        return torch.zeros_like(latent)

    def decode(latent, generator, num_inference_steps):
        assert num_inference_steps == 1
        return SimpleNamespace(sample=torch.nn.functional.interpolate(latent.float(), (17, 256, 256), mode="trilinear"))

    components = {
        "text_encoder": lambda **kwargs: SimpleNamespace(hidden_states=(torch.ones(1, 1024, 2), torch.ones(1, 1024, 2))),
        "connectors": lambda embeddings, mask, padding_side: (embeddings, None, mask),
        "vae": SimpleNamespace(enable_tiling=lambda: None, encode=encode,
                               latents_mean=torch.zeros(3), latents_std=torch.ones(3), config=SimpleNamespace(scaling_factor=1)),
        "latent_upsampler": lambda latent: torch.nn.functional.interpolate(latent, scale_factor=(1, 2, 2), mode="nearest"),
        "transformer": transformer,
        "diffusion_decoder": SimpleNamespace(disable_tiling=lambda: None, decode=decode),
    }

    def activate(name, memory):
        stages.append(name)
        return components[name]

    runtime = SimpleNamespace(device="cpu", minimum_frames=17, tokenizer=lambda *a, **kw: Tokens(),
                              activate=activate, park=lambda x: x, recover_decode_oom=lambda: pytest.fail("No OOM"))
    result = pipeline_module.run_pipeline(runtime, torch.rand(9, 4, 6, 3), "A scene", 24, 224, 224,
                                          DEFAULT_SIGMA, 0, 0, "untiled", tiling_config(768, 128, 512))
    assert stages == list(components)
    assert encoded_pixels == [torch.Size([1, 3, 17, 128, 128])]
    assert transformer_calls == [True]
    assert result.shape == (9, 224, 224, 3)
    assert result.dtype == torch.float32 and 0 <= result.min() <= result.max() <= 1


@pytest.fixture
def runtime_module(monkeypatch, pipeline_module):
    package = __package__.rsplit(".", 1)[0]
    loader = ModuleType(f"{package}.sol_refiner.loader")
    loader.check_environment = lambda precision: torch.device("cpu")
    loader.dependency_api = lambda: {"tokenizer": SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(pad_token=None, eos_token="eos"))}
    loader.load_component = lambda *a: pytest.fail("Replace the component loader in tests")
    attention = ModuleType(f"{package}.sol_refiner.attention")
    attention.decoder_backend = lambda backend, device: "flex" if backend == "auto" else backend
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    monkeypatch.setitem(sys.modules, f"{package}.sol_refiner.pipeline", pipeline_module)
    spec = importlib.util.spec_from_file_location(f"{package}.sol_refiner._test_runtime", Path(__file__).parents[1] / "sol_refiner" / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_cache_keys_and_component_reuse(runtime_module, model_package, monkeypatch):
    runtime = runtime_module.get_refiner(model_package, "bf16", "auto", True, True)
    assert runtime_module.get_refiner(model_package, "bf16", "flex", True, True) is runtime
    assert runtime_module.get_refiner(model_package, "fp8", "flex", True, True) is not runtime
    assert runtime_module.get_refiner(model_package, "bf16", "flex", True, False) is not runtime
    assert runtime.minimum_frames == 17
    assert runtime.tokenizer.pad_token == "eos"
    loads, activations = [], []
    component = object()
    patcher = SimpleNamespace(model=SimpleNamespace(component=component), loaded_size=lambda: 0)
    monkeypatch.setattr(runtime_module, "load_component", lambda *args: loads.append(args[1]) or patcher)
    monkeypatch.setattr(runtime_module.model_management, "load_models_gpu", lambda models, **kw: activations.append(kw), raising=False)
    assert runtime.activate("transformer", 100) is component
    assert runtime.activate("transformer", 200) is component
    assert loads == ["transformer"]
    assert activations == [{"memory_required": 100, "force_full_load": False}, {"memory_required": 200, "force_full_load": False}]


@pytest.mark.parametrize("resident,keep,offloaded,retained", [(True, True, False, True), (False, True, True, True), (True, False, True, False), (False, False, True, False)])
def test_runtime_success_cleanup(runtime_module, pipeline_module, model_package, monkeypatch, resident, keep, offloaded, retained):
    runtime = runtime_module.get_refiner(model_package, "bf16", "flex", keep, resident)
    patcher = SimpleNamespace(loaded_size=lambda: 1)
    runtime.components["vae"] = patcher
    unloads = []
    monkeypatch.setattr(runtime_module.model_management, "unload_model_and_clones", lambda p: unloads.append(p), raising=False)
    output = torch.rand(1, 224, 224, 3)
    monkeypatch.setattr(pipeline_module, "run_pipeline", lambda *args: output)
    result = runtime.refine(torch.rand(1, 2, 2, 3), "scene", 24, 224, 224, DEFAULT_SIGMA, 0, 0, "auto", 768, 128, 512)
    assert result is output
    assert bool(unloads) is offloaded
    assert bool(runtime.components) is retained


@pytest.mark.parametrize("error", [RuntimeError("bad decode"), torch.cuda.OutOfMemoryError("OOM"), KeyboardInterrupt()])
def test_failure_and_interruption_offload(runtime_module, pipeline_module, model_package, monkeypatch, error):
    runtime = runtime_module.get_refiner(model_package, "bf16", "flex", False, True)
    runtime.components["vae"] = SimpleNamespace(loaded_size=lambda: 1)
    unloads = []
    monkeypatch.setattr(runtime_module.model_management, "unload_model_and_clones", lambda p: unloads.append(p), raising=False)

    def fail(*args):
        raise error

    monkeypatch.setattr(pipeline_module, "run_pipeline", fail)
    expected = RuntimeError if isinstance(error, torch.cuda.OutOfMemoryError) else type(error)
    with pytest.raises(expected) as raised:
        runtime.refine(torch.rand(1, 2, 2, 3), "scene", 24, 224, 224, DEFAULT_SIGMA, 0, 0, "auto", 768, 128, 512)
    if isinstance(error, torch.cuda.OutOfMemoryError):
        assert raised.value.__cause__ is error
    assert len(unloads) == 1
    assert runtime.components == {}


def test_decode_recovery_unloads_others_only(runtime_module, model_package, monkeypatch):
    runtime = runtime_module.get_refiner(model_package, "bf16", "flex", True, True)
    vae, decoder = SimpleNamespace(loaded_size=lambda: 1), SimpleNamespace(loaded_size=lambda: 1)
    runtime.components.update(vae=vae, diffusion_decoder=decoder)
    unloaded, cleared = [], []
    monkeypatch.setattr(runtime_module.model_management, "unload_model_and_clones", lambda p: unloaded.append(p), raising=False)
    monkeypatch.setattr(runtime_module.model_management, "soft_empty_cache", lambda **kw: cleared.append(kw), raising=False)
    runtime.recover_decode_oom()
    assert unloaded == [vae]
    assert cleared == [{"force": True}]


@pytest.fixture
def decoder_module(monkeypatch, pipeline_module):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    base = ModuleType("diffusers.models.autoencoders.ltx2_diffusion_decoder")
    base.LTX2VideoVaeNeighborhoodAttention = torch.nn.Module
    base.LTX2VideoVaeNeighborhoodNattenProcessor = type("NattenProcessor", (), {})
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setattr(sys.modules["diffusers"], "LTX2VideoDiffusionDecoderModel", object, raising=False)
    package = __package__.rsplit(".", 1)[0]
    spec = importlib.util.spec_from_file_location(f"{package}.sol_refiner._test_decoder", Path(__file__).parents[1] / "sol_refiner" / "decoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "compiled_functions", lambda: (create_block_mask, flex_attention))
    return module


def test_flex_neighborhood_block_mask(decoder_module):
    frames, height, width = 3, 3, 5
    mask = decoder_module.neighborhood_mask(frames, height, width, (3, 3, 3), "cpu")
    tokens = frames * height * width
    positions = torch.arange(tokens)
    actual = mask.mask_mod(0, 0, positions[:, None], positions[None, :])
    allowed = torch.zeros(tokens, tokens, dtype=torch.bool)
    for q in range(tokens):
        qw = q % width
        start = max(0, min(qw - 1, width - 3))
        for k in range(tokens):
            allowed[q, k] = start <= k % width < start + 3
    assert torch.equal(actual, allowed)
    assert actual.sum(1).unique().tolist() == [27]


def test_flex_neighborhood_without_natten(decoder_module):
    """Actual PyTorch Flex math on CPU; Linux CUDA compilation is a manual test."""
    pytest.importorskip("filelock", reason="PyTorch Flex compiler requires filelock")
    frames, height, width = 3, 3, 5
    mask = decoder_module.neighborhood_mask(frames, height, width, (3, 3, 3), "cpu")
    values = torch.randn(1, frames, height, width, 8)

    def project(hidden):
        return tuple(hidden.unsqueeze(-2) for _ in range(3))

    attention = SimpleNamespace(project_qkv=project, to_out=[torch.nn.Identity()])
    actual = decoder_module.FlexNeighborhood()(attention, values, block_mask=mask)
    tokens = frames * height * width
    allowed = torch.zeros(tokens, tokens, dtype=torch.bool)
    for q in range(tokens):
        qw = q % width
        start = max(0, min(qw - 1, width - 3))
        for k in range(tokens):
            allowed[q, k] = start <= k % width < start + 3
    packed = values.flatten(1, 3).unsqueeze(1)
    expected = torch.nn.functional.scaled_dot_product_attention(packed, packed, packed, attn_mask=allowed, scale=1.0)
    torch.testing.assert_close(actual, expected.reshape(values.shape), rtol=1e-4, atol=1e-5)


@pytest.fixture
def component_loader(monkeypatch, pipeline_module):
    comfy = sys.modules["comfy"]
    comfy.ops = SimpleNamespace(manual_cast=SimpleNamespace(Linear=torch.nn.Linear, RMSNorm=torch.nn.RMSNorm, LayerNorm=torch.nn.LayerNorm))
    comfy.quant_ops = SimpleNamespace()
    comfy.model_patcher = SimpleNamespace(ModelPatcher=lambda owner, device, offload: SimpleNamespace(model=owner))
    package = __package__.rsplit(".", 1)[0]
    spec = importlib.util.spec_from_file_location(f"{package}.sol_refiner._test_loader", Path(__file__).parents[1] / "sol_refiner" / "loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bf16_safetensors_meta_load_and_patcher(component_loader, tmp_path):
    from safetensors.torch import save_file

    expected = {"0.weight": torch.rand(4, 3), "0.bias": torch.rand(4),
                "1.weight": torch.rand(4), "1.bias": torch.rand(4)}
    save_file(expected, str(tmp_path / "model.safetensors"))
    with torch.device("meta"):
        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    component_loader.read_weights(module, tmp_path, "bf16", "cpu")
    assert not module.training
    for name, parameter in module.named_parameters():
        assert not parameter.is_meta and parameter.dtype == torch.bfloat16
        torch.testing.assert_close(parameter, expected[name].bfloat16())
    patcher = component_loader.patcher_for(module, "cpu")
    assert patcher.model.component is module


@pytest.mark.parametrize("weights,message", [({}, "missing"), ({"weight": torch.ones(5, 3), "bias": torch.ones(4)}, "Incompatible")])
def test_missing_or_incompatible_checkpoint_weights(component_loader, tmp_path, weights, message):
    from safetensors.torch import save_file

    save_file(weights, str(tmp_path / "model.safetensors"))
    # A child path mirrors the real components, which have no root Linear.
    with torch.device("meta"):
        module = torch.nn.Sequential(torch.nn.Linear(3, 4))
    if weights:
        save_file({"0." + name: value for name, value in weights.items()}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match=message):
        component_loader.read_weights(module, tmp_path, "bf16", "cpu")
