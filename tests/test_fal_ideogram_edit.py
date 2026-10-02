import base64
import json
import math
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from ..nodes.fal_ideogram_edit import MAIFalIdeogramEdit
from ..utils.fal_ideogram_edit import (
    build_ideogram_edit_arguments,
    prepare_ideogram_edit_inputs,
)


def sample_inputs():
    np = pytest.importorskip("numpy")
    return (
        np.zeros((1, 2, 3, 3), dtype=np.float32),
        np.array([[[0, 0.49, 0.5], [1, 0, 1]]], dtype=np.float32),
    )


def test_registration_and_fresh_request_behavior():
    assert NODE_CLASS_MAPPINGS["MAIFalIdeogramEdit"] is MAIFalIdeogramEdit
    assert NODE_DISPLAY_NAME_MAPPINGS["MAIFalIdeogramEdit"] == "mAI fal ideogram edit"
    assert "MAIFalImage" in NODE_CLASS_MAPPINGS
    assert "MAIExampleTextNode" in NODE_CLASS_MAPPINGS
    assert math.isnan(MAIFalIdeogramEdit.IS_CHANGED())


def test_quality_and_precision_are_selection_dropdowns():
    inputs = MAIFalIdeogramEdit.INPUT_TYPES()["required"]
    assert inputs["quality"][0] == ["very_low", "low", "medium", "high"]
    assert inputs["quality"][1]["default"] == "medium"
    assert set(inputs["edit_precision"][0]) == {"regular", "high"}
    assert inputs["edit_precision"][1]["default"] == "high"


@pytest.mark.parametrize("seed", [-1, 0, 123])
def test_request_uses_source_size_and_omits_random_seed(seed):
    args = build_ideogram_edit_arguments("replace the sign", seed, "high", "medium")
    assert args["image_size"] == "auto"
    assert args["num_images"] == 1
    assert ("seed" in args) == (seed >= 0)
    if seed >= 0:
        assert args["seed"] == seed


@pytest.mark.parametrize("kwargs, message", [
    ({"prompt": " "}, "prompt"),
    ({"edit_precision": "unknown"}, "edit_precision"),
    ({"quality": "unknown"}, "quality"),
    ({"seed": -2}, "seed"),
    ({"seed": True}, "seed"),
    ({"seed": 1.5}, "seed"),
])
def test_invalid_settings_are_rejected(kwargs, message):
    settings = dict(prompt="edit", seed=-1, edit_precision="high", quality="medium")
    settings.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_ideogram_edit_arguments(**settings)


@pytest.mark.parametrize("batched", [True, False])
def test_mask_is_thresholded_inverted_and_encoded_without_resizing(batched):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    image, mask = sample_inputs()
    source_png, mask_png = prepare_ideogram_edit_inputs(image, mask if batched else mask[0])
    with Image.open(BytesIO(source_png)) as source:
        assert source.size == (3, 2)
    with Image.open(BytesIO(mask_png)) as encoded:
        assert encoded.mode == "L"
        np.testing.assert_array_equal(np.asarray(encoded), [[255, 255, 0], [0, 255, 0]])
    # Encoding must not modify the workflow's input tensors/arrays.
    np.testing.assert_array_equal(mask, sample_inputs()[1])


def test_actual_comfy_tensors_are_supported():
    torch = pytest.importorskip("torch")
    image, mask = sample_inputs()
    assert prepare_ideogram_edit_inputs(torch.from_numpy(image), torch.from_numpy(mask)) == (
        prepare_ideogram_edit_inputs(image, mask)
    )


@pytest.mark.parametrize("case", [
    "image_batch", "empty_image", "image_channels", "image_nan", "mask_batch",
    "mask_size", "mask_nan", "mask_inf", "all_black", "all_white", "uniform_gray",
])
def test_invalid_inputs_fail_before_any_upload_or_request(monkeypatch, case):
    np = pytest.importorskip("numpy")
    image, mask = sample_inputs()
    if case == "image_batch":
        image = np.repeat(image, 2, axis=0)
    elif case == "empty_image":
        image = np.zeros((1, 0, 3, 3))
    elif case == "image_channels":
        image = np.zeros((1, 2, 3, 2))
    elif case == "image_nan":
        image[0, 0, 0, 0] = np.nan
    elif case == "mask_batch":
        mask = np.repeat(mask, 2, axis=0)
    elif case == "mask_size":
        mask = np.zeros((1, 3, 3))
    elif case in ("mask_nan", "mask_inf"):
        mask[0, 0, 0] = np.nan if case == "mask_nan" else np.inf
    else:
        mask.fill({"all_black": 0, "all_white": 1, "uniform_gray": 0.5}[case])

    def unexpected_client(**kwargs):
        pytest.fail("Invalid input must not initialize the API client")

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=unexpected_client))
    with pytest.raises(ValueError):
        MAIFalIdeogramEdit().run(image, mask, "edit", api_key="fake-key")


def test_missing_key_has_clear_error(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key or set FAL_KEY"):
        MAIFalIdeogramEdit().run(None, None, "edit")


@pytest.mark.parametrize("use_environment", [True, False])
def test_node_uploads_source_and_mask_and_decodes_result(monkeypatch, use_environment):
    pytest.importorskip("torch")
    image, mask = sample_inputs()
    source_png, mask_png = prepare_ideogram_edit_inputs(image, mask)
    uri = "data:image/png;base64," + base64.b64encode(source_png).decode("ascii")
    calls = []
    monkeypatch.setenv("FAL_KEY", " environment-key ")

    class FakeClient:
        def __init__(self, key):
            assert key == ("environment-key" if use_environment else "widget-key")

        def upload(self, data, content_type, file_name):
            index = len(calls)
            assert data == (source_png, mask_png)[index]
            assert content_type == "image/png"
            calls.append(file_name)
            return f"https://example.com/{file_name}"

        def subscribe(self, endpoint, arguments):
            assert endpoint == "ideogram/v4.5/edit"
            assert arguments == {
                "prompt": "replace the sign", "seed": 0,
                "edit_precision": "high", "quality": "medium",
                "image_size": "auto", "num_images": 1,
                "image_url": "https://example.com/comfy_source.png",
                "mask_url": "https://example.com/comfy_mask.png",
            }
            calls.append("subscribe")
            return {"images": [{"url": uri}], "seed": 0}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))
    result = MAIFalIdeogramEdit().run(
        image, mask, "replace the sign",
        api_key="" if use_environment else " widget-key ", seed=0,
    )
    assert calls == ["comfy_source.png", "comfy_mask.png", "subscribe"]
    assert tuple(result[0].shape) == (1, 2, 3, 3)
    assert result[1:3] == (uri, 0)
    assert json.loads(result[3]) == {"images": [{"url": uri}], "seed": 0}


@pytest.mark.parametrize("failure", ["upload", "subscribe", "download"])
def test_remote_failures_have_context_without_extra_requests(monkeypatch, failure):
    image, mask = sample_inputs()
    submissions = []

    class FakeClient:
        def __init__(self, key):
            pass

        def upload(self, *args, **kwargs):
            if failure == "upload":
                raise RuntimeError("service unavailable")
            return "https://example.com/input.png"

        def subscribe(self, endpoint, arguments):
            submissions.append(endpoint)
            if failure == "subscribe":
                raise RuntimeError("service unavailable")
            return {"images": [{"url": "data:image/png;base64,aW52YWxpZA=="}]}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))
    message = {
        "upload": "upload the image or mask", "subscribe": "request to",
        "download": "download or decode",
    }[failure]
    with pytest.raises(RuntimeError, match=message):
        MAIFalIdeogramEdit().run(image, mask, "edit", api_key="fake-key")
    assert len(submissions) == (0 if failure == "upload" else 1)
