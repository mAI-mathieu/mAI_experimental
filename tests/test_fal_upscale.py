import base64
import json
import sys
from types import SimpleNamespace

import pytest

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from ..nodes.fal_image import MAIFalImage
from ..utils.fal_image import image_batch_to_png_bytes, resolve_image_input_name
from ..utils.fal_upscale import (
    AURA_SR_ENDPOINT,
    CLARITY_UPSCALE_ENDPOINT,
    CRYSTAL_UPSCALE_ENDPOINT,
    DEFAULT_TOPAZ_MODEL,
    SEEDVR_UPSCALE_ENDPOINT,
    TOPAZ_MODELS,
    TOPAZ_UPSCALE_ENDPOINT,
    UPSCALE_ENDPOINTS,
    build_upscale_arguments,
    validate_upscale_images,
)


EXPECTED_DEFAULT_ARGUMENTS = {
    SEEDVR_UPSCALE_ENDPOINT: {
        "upscale_factor": 4.0, "upscale_mode": "factor",
        "output_format": "png", "noise_scale": 0.1,
    },
    TOPAZ_UPSCALE_ENDPOINT: {
        "upscale_factor": 4.0, "model": "High Fidelity V2",
        "output_format": "png", "crop_to_fill": False, "face_enhancement": False,
    },
    CLARITY_UPSCALE_ENDPOINT: {
        "upscale_factor": 4.0, "creativity": 0.2, "resemblance": 0.8,
    },
    CRYSTAL_UPSCALE_ENDPOINT: {
        "scale_factor": 4.0, "creativity": 0, "output_format": "png",
    },
    AURA_SR_ENDPOINT: {
        "upscale_factor": 4, "checkpoint": "v2", "overlapping_tiles": True,
    },
}


def test_registration_and_workflow_contract_are_preserved():
    assert NODE_CLASS_MAPPINGS["MAIFalImage"] is MAIFalImage
    assert "MAIExampleTextNode" in NODE_CLASS_MAPPINGS
    assert NODE_DISPLAY_NAME_MAPPINGS["MAIFalImage"] == "mAI fal image"
    assert MAIFalImage.CATEGORY == "mAI / Image"
    assert MAIFalImage.RETURN_TYPES == ("IMAGE", "STRING", "INT", "STRING")
    assert MAIFalImage.RETURN_NAMES == ("image", "image_url", "seed", "response_json")
    inputs = MAIFalImage.INPUT_TYPES()
    assert list(inputs["required"]) == [
        "api_key", "model_endpoint", "prompt", "image_size", "seed",
        "output_format", "resolution_mode", "custom_width", "custom_height",
    ]
    assert set(UPSCALE_ENDPOINTS) <= set(inputs["required"]["model_endpoint"][0])
    assert inputs["optional"]["upscale_factor"][1]["default"] == 4.0
    assert inputs["optional"]["topaz_model"][1]["default"] == DEFAULT_TOPAZ_MODEL


@pytest.mark.parametrize("endpoint", UPSCALE_ENDPOINTS)
def test_upscale_defaults_match_endpoint_schema(endpoint):
    assert build_upscale_arguments(endpoint) == EXPECTED_DEFAULT_ARGUMENTS[endpoint]
    assert resolve_image_input_name(endpoint, 1) == "image_url"
    with pytest.raises(ValueError, match="exactly one"):
        resolve_image_input_name(endpoint, 2)


@pytest.mark.parametrize("endpoint", UPSCALE_ENDPOINTS)
def test_prompt_seed_and_format_are_only_sent_when_supported(endpoint):
    args = build_upscale_arguments(endpoint, prompt="a fox", seed=0, output_format="jpeg")
    assert ("seed" in args) == (endpoint in (SEEDVR_UPSCALE_ENDPOINT, CLARITY_UPSCALE_ENDPOINT))
    assert ("prompt" in args) == (endpoint == CLARITY_UPSCALE_ENDPOINT)
    if endpoint in (SEEDVR_UPSCALE_ENDPOINT, CRYSTAL_UPSCALE_ENDPOINT):
        assert args["output_format"] == "jpg"
    elif endpoint == TOPAZ_UPSCALE_ENDPOINT:
        assert args["output_format"] == "jpeg"
    else:
        assert "output_format" not in args


@pytest.mark.parametrize("model", TOPAZ_MODELS)
def test_topaz_model_selection(model):
    args = build_upscale_arguments(TOPAZ_UPSCALE_ENDPOINT, topaz_model=model, prompt="a fox")
    assert args["model"] == model
    assert ("prompt" in args) == (model == "Redefine")
    if model == "Redefine":
        assert args["creativity"] == 1


def test_topaz_rejects_invalid_model_and_long_redefine_prompt():
    with pytest.raises(ValueError, match="topaz_model"):
        build_upscale_arguments(TOPAZ_UPSCALE_ENDPOINT, topaz_model="unknown")
    with pytest.raises(ValueError, match="1024"):
        build_upscale_arguments(TOPAZ_UPSCALE_ENDPOINT, topaz_model="Redefine", prompt="x" * 1025)


@pytest.mark.parametrize("factor", [0, -1, 11, True, "4", float("nan"), float("inf")])
@pytest.mark.parametrize("endpoint", UPSCALE_ENDPOINTS)
def test_invalid_scale_is_rejected(endpoint, factor):
    with pytest.raises(ValueError, match="upscale_factor"):
        build_upscale_arguments(endpoint, factor)


@pytest.mark.parametrize("endpoint", [TOPAZ_UPSCALE_ENDPOINT, CLARITY_UPSCALE_ENDPOINT, AURA_SR_ENDPOINT])
def test_endpoint_scale_limits(endpoint):
    with pytest.raises(ValueError, match="4"):
        build_upscale_arguments(endpoint, 5)


def test_aura_requires_four_and_other_models_allow_larger_scales():
    with pytest.raises(ValueError, match="only upscale_factor 4"):
        build_upscale_arguments(AURA_SR_ENDPOINT, 2)
    assert build_upscale_arguments(SEEDVR_UPSCALE_ENDPOINT, 10)["upscale_factor"] == 10
    assert build_upscale_arguments(CRYSTAL_UPSCALE_ENDPOINT, 8)["scale_factor"] == 8


@pytest.mark.parametrize("shape", [(0, 8, 8, 3), (2, 8, 8, 3), (1, 0, 8, 3), (8, 8, 3), (1, 8, 8, 2)])
def test_invalid_image_shape_is_rejected(shape):
    with pytest.raises(ValueError):
        validate_upscale_images([SimpleNamespace(shape=shape)])


@pytest.mark.parametrize("endpoint", UPSCALE_ENDPOINTS)
def test_node_upscales_with_default_factor_and_preserves_returned_resolution(monkeypatch, endpoint):
    np = pytest.importorskip("numpy")
    source = np.zeros((1, 3, 5, 3), dtype=np.float32)
    output_png = image_batch_to_png_bytes(np.zeros((1, 12, 20, 3), dtype=np.float32))[0]
    uri = "data:image/png;base64," + base64.b64encode(output_png).decode("ascii")
    calls = []

    class FakeClient:
        def __init__(self, key):
            pass

        def upload(self, data, content_type, file_name):
            assert data == image_batch_to_png_bytes(source)[0]
            calls.append("upload")
            return "https://example.com/input.png"

        def subscribe(self, selected_endpoint, arguments):
            assert selected_endpoint == endpoint
            assert arguments == {
                **EXPECTED_DEFAULT_ARGUMENTS[endpoint], "image_url": "https://example.com/input.png",
            }
            calls.append("subscribe")
            # Crystal documents an image list; other upscalers return a single image.
            return {"images": [uri]} if endpoint == CRYSTAL_UPSCALE_ENDPOINT else {"image": {"url": uri}}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))
    image, image_url, seed, response_json = MAIFalImage().generate(
        "test-key", endpoint, "", "square", -1, "png", image_2=source,
        resolution_mode="custom", custom_width=64, custom_height=64,
    )
    assert calls == ["upload", "subscribe"]
    assert tuple(image.shape) == (1, 12, 20, 3)
    assert image_url == uri
    assert seed == -1
    assert json.loads(response_json)


@pytest.mark.parametrize("endpoint", UPSCALE_ENDPOINTS)
@pytest.mark.parametrize("case", ["missing", "multiple", "batch", "empty", "scale"])
def test_invalid_upscale_is_rejected_before_client_creation(monkeypatch, endpoint, case):
    def unexpected_client(**kwargs):
        pytest.fail("Invalid input must not initialize the API client")

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=unexpected_client))
    one = SimpleNamespace(shape=(1, 8, 8, 3))
    kwargs = {
        "missing": {},
        "multiple": {"image_1": one, "image_2": one},
        "batch": {"image_1": SimpleNamespace(shape=(2, 8, 8, 3))},
        "empty": {"image_1": SimpleNamespace(shape=(0, 8, 8, 3))},
        "scale": {"image_1": one, "upscale_factor": 0},
    }[case]
    with pytest.raises(ValueError):
        MAIFalImage().generate("test-key", endpoint, "", "square", -1, "png", **kwargs)
