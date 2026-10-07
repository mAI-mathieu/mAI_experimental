import base64
import json
import sys
from types import SimpleNamespace

import pytest

from .. import NODE_CLASS_MAPPINGS
from ..nodes.fal_image_edit import MAIFalImageEdit
from ..utils.fal_ideogram_edit import IDEOGRAM_EDIT_ENDPOINT
from ..utils.fal_image import image_batch_to_png_bytes
from ..utils.fal_image_edit import (
    FLUX_3_EDIT_ENDPOINT,
    FLUX_3_TEXT_ENDPOINT,
    IMAGE_EDIT_ENDPOINTS,
    build_image_edit_arguments,
    prepare_image_edit_uploads,
)


def test_generic_registration_and_optional_image_inputs():
    assert NODE_CLASS_MAPPINGS["MAIFalImageEdit"] is MAIFalImageEdit
    assert "MAIFalIdeogramEdit" in NODE_CLASS_MAPPINGS
    assert "MAIFalImage" in NODE_CLASS_MAPPINGS
    inputs = MAIFalImageEdit.INPUT_TYPES()
    assert inputs["required"]["model_endpoint"][0] == list(IMAGE_EDIT_ENDPOINTS)
    assert inputs["optional"]["image"][0] == "IMAGE"
    assert inputs["optional"]["mask"][0] == "MASK"
    assert set(inputs["required"]["edit_precision"][0]) == {"regular", "high"}
    assert inputs["required"]["quality"][0] == ["very_low", "low", "medium", "high"]


@pytest.mark.parametrize("endpoint", [FLUX_3_TEXT_ENDPOINT, FLUX_3_EDIT_ENDPOINT])
def test_flux_payload_omits_unsupported_ideogram_fields(endpoint):
    args = build_image_edit_arguments(
        endpoint, "a fox", seed=123, edit_precision="high", quality="high",
        resolution="4k", aspect_ratio="16:9", output_format="jpeg",
        enable_prompt_expansion=True, safety_tolerance=3,
    )
    assert args == {
        "prompt": "a fox", "resolution": "4k", "aspect_ratio": "16:9",
        "output_format": "jpeg", "enable_prompt_expansion": True,
        "safety_tolerance": 3,
    }


def test_ideogram_payload_omits_flux_fields():
    args = build_image_edit_arguments(
        IDEOGRAM_EDIT_ENDPOINT, "edit", seed=0, edit_precision="regular", quality="low",
        resolution="4k", aspect_ratio="16:9", output_format="jpeg",
    )
    assert args == {
        "prompt": "edit", "seed": 0, "edit_precision": "regular", "quality": "low",
        "image_size": "auto", "num_images": 1,
    }


@pytest.mark.parametrize("setting, value", [
    ("prompt", " "), ("resolution", "8k"), ("aspect_ratio", "100:1"),
    ("output_format", "webp"), ("enable_prompt_expansion", "yes"),
    ("safety_tolerance", -1), ("safety_tolerance", 5), ("safety_tolerance", True),
])
def test_invalid_flux_settings_are_rejected(setting, value):
    kwargs = {"endpoint": FLUX_3_TEXT_ENDPOINT, "prompt": "a fox", setting: value}
    with pytest.raises(ValueError, match=setting):
        build_image_edit_arguments(**kwargs)


@pytest.mark.parametrize("batch", [1, 2, 10])
def test_flux_reference_batch_is_uploaded_in_order(batch):
    np = pytest.importorskip("numpy")
    pixels = np.zeros((batch, 256, 256, 3), dtype=np.float32)
    pixels[-1] = 1
    uploads = prepare_image_edit_uploads(FLUX_3_EDIT_ENDPOINT, pixels)
    assert len(uploads) == batch
    assert [name for name, _ in uploads] == [f"comfy_reference_{i}.png" for i in range(1, batch + 1)]
    assert [png for _, png in uploads] == image_batch_to_png_bytes(pixels)


@pytest.mark.parametrize("shape", [
    (11, 256, 256, 3), (0, 256, 256, 3), (1, 255, 256, 3),
    (1, 256, 255, 3), (1, 2001, 2000, 3), (256, 256, 3), (1, 256, 256, 2),
])
def test_invalid_flux_image_dimensions_are_rejected(shape):
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        prepare_image_edit_uploads(FLUX_3_EDIT_ENDPOINT, np.zeros(shape, dtype=np.float32))


def test_ideogram_without_mask_uploads_only_source():
    np = pytest.importorskip("numpy")
    pixels = np.zeros((1, 2, 3, 3), dtype=np.float32)
    assert prepare_image_edit_uploads(IDEOGRAM_EDIT_ENDPOINT, pixels) == [
        ("comfy_source.png", image_batch_to_png_bytes(pixels)[0]),
    ]


@pytest.mark.parametrize("endpoint, kwargs, message", [
    (FLUX_3_EDIT_ENDPOINT, {}, "requires a connected image"),
    (IDEOGRAM_EDIT_ENDPOINT, {}, "requires a connected image"),
    (FLUX_3_TEXT_ENDPOINT, {"image": object()}, "Disconnect image and mask"),
    (FLUX_3_TEXT_ENDPOINT, {"mask": object()}, "Disconnect image and mask"),
    (FLUX_3_EDIT_ENDPOINT, {"image": object(), "mask": object()}, "does not support a mask"),
    ("unknown/model", {}, "Unsupported model_endpoint"),
])
def test_invalid_connections_fail_before_client_creation(monkeypatch, endpoint, kwargs, message):
    def unexpected_client(**kwargs):
        pytest.fail("Validation must run before creating a client")

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=unexpected_client))
    with pytest.raises(ValueError, match=message):
        MAIFalImageEdit().run(
            prompt="edit", api_key="fake-key", model_endpoint=endpoint, **kwargs,
        )


@pytest.mark.parametrize("endpoint", [IDEOGRAM_EDIT_ENDPOINT, FLUX_3_TEXT_ENDPOINT, FLUX_3_EDIT_ENDPOINT])
def test_generic_node_submits_endpoint_specific_payload_and_decodes_output(monkeypatch, endpoint):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    output_png = image_batch_to_png_bytes(np.ones((1, 7, 9, 3), dtype=np.float32))[0]
    uri = "data:image/png;base64," + base64.b64encode(output_png).decode("ascii")
    image = mask = None
    if endpoint == FLUX_3_EDIT_ENDPOINT:
        image = np.zeros((2, 256, 256, 3), dtype=np.float32)
        image[1] = 1
    elif endpoint == IDEOGRAM_EDIT_ENDPOINT:
        image = np.zeros((1, 2, 3, 3), dtype=np.float32)
        mask = np.array([[[0, 1, 0], [1, 0, 1]]], dtype=np.float32)
    uploads = prepare_image_edit_uploads(endpoint, image, mask)
    submitted = []
    uploaded = []

    class FakeClient:
        def __init__(self, key):
            assert key == "fake-key"

        def upload(self, data, content_type, file_name):
            assert (file_name, data) == uploads[len(uploaded)]
            assert content_type == "image/png"
            url = f"https://example.com/{file_name}"
            uploaded.append(url)
            return url

        def subscribe(self, selected_endpoint, arguments):
            assert selected_endpoint == endpoint
            if endpoint == IDEOGRAM_EDIT_ENDPOINT:
                assert arguments == {
                    "prompt": "edit", "seed": 42, "edit_precision": "high",
                    "quality": "medium", "num_images": 1, "image_size": "auto",
                    "image_url": uploaded[0], "mask_url": uploaded[1],
                }
            else:
                expected = {
                    "prompt": "edit", "resolution": "2k", "aspect_ratio": "4:3",
                    "output_format": "png", "enable_prompt_expansion": True,
                    "safety_tolerance": 2,
                }
                if endpoint == FLUX_3_EDIT_ENDPOINT:
                    expected["image_urls"] = uploaded
                assert arguments == expected
            submitted.append(selected_endpoint)
            return {"images": [{"url": uri}]}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))
    result = MAIFalImageEdit().run(
        image=image, mask=mask, prompt="edit", api_key="fake-key", model_endpoint=endpoint,
        seed=42, resolution="2k", aspect_ratio="4:3", enable_prompt_expansion=True,
    )
    assert len(uploaded) == len(uploads)
    assert submitted == [endpoint]
    assert tuple(result[0].shape) == (1, 7, 9, 3)
    assert result[1:3] == (uri, -1)
    assert json.loads(result[3]) == {"images": [{"url": uri}]}
