import base64
import json
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

from ..nodes.fal_image_edit import MAIFalImageEdit
from ..utils.fal_image import image_batch_to_png_bytes
from ..utils.fal_image_edit import build_image_edit_arguments, prepare_image_edit_uploads
from ..utils.fal_nano_banana_edit import (
    NANO_BANANA_21_EDIT_ENDPOINT,
    composite_nano_banana_edit,
    prepare_nano_banana_21_uploads,
)


def sample_inputs():
    np = pytest.importorskip("numpy")
    return (
        np.full((1, 2, 3, 3), 0.2, dtype=np.float32),
        np.array([[[0, 0.5, 1], [1, 0, 0.25]]], dtype=np.float32),
    )


def test_model_is_available_with_existing_controls_and_socket_contract():
    inputs = MAIFalImageEdit.INPUT_TYPES()
    assert NANO_BANANA_21_EDIT_ENDPOINT in inputs["required"]["model_endpoint"][0]
    assert inputs["required"]["api_key"][0] == "STRING"
    assert inputs["optional"]["mask"][0] == "MASK"
    assert MAIFalImageEdit.RETURN_NAMES == ("image", "image_url", "seed", "response_json")


@pytest.mark.parametrize("resolution, seed", [("1k", -1), ("2k", 0), ("4k", 42)])
def test_nano_payload_translates_resolution_and_excludes_other_models_settings(resolution, seed):
    args = build_image_edit_arguments(
        NANO_BANANA_21_EDIT_ENDPOINT, "change the shirt", seed=seed,
        resolution=resolution, aspect_ratio="auto", output_format="jpeg",
        quality="high", edit_precision="regular", safety_tolerance=0,
        enable_prompt_expansion=True,
    )
    expected = {
        "prompt": "change the shirt", "num_images": 1, "limit_generations": True,
        "resolution": resolution.upper(), "aspect_ratio": "auto", "output_format": "jpeg",
    }
    if seed >= 0:
        expected["seed"] = seed
    assert args == expected


def test_mask_guidance_describes_white_edits_and_preserves_user_prompt():
    args = build_image_edit_arguments(
        NANO_BANANA_21_EDIT_ENDPOINT, "replace the chair", masked=True,
    )
    assert "Image 1 is the source" in args["prompt"]
    assert "Image 2 is a grayscale editing mask" in args["prompt"]
    assert "white region" in args["prompt"]
    assert args["prompt"].endswith("replace the chair")
    assert "mask_url" not in args


@pytest.mark.parametrize("kwargs, message", [
    ({"prompt": " "}, "prompt"), ({"seed": -2}, "seed"), ({"seed": True}, "seed"),
    ({"resolution": "512sq"}, "resolution"), ({"resolution": "768sq"}, "resolution"),
    ({"aspect_ratio": "2:1"}, "aspect_ratio"), ({"output_format": "webp"}, "output_format"),
    ({"masked": True, "aspect_ratio": "16:9"}, "aspect_ratio=auto"),
])
def test_invalid_settings_fail_before_request(kwargs, message):
    options = dict(endpoint=NANO_BANANA_21_EDIT_ENDPOINT, prompt="edit")
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_image_edit_arguments(**options)


def test_reference_mask_preserves_white_edit_direction_and_soft_edges():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    source, mask = sample_inputs()
    uploads = prepare_nano_banana_21_uploads(source, mask)
    assert uploads[0] == ("comfy_source.png", image_batch_to_png_bytes(source)[0])
    assert uploads[1][0] == "comfy_mask_reference.png"
    with Image.open(BytesIO(uploads[1][1])) as encoded:
        assert encoded.mode == "L"
        np.testing.assert_array_equal(np.asarray(encoded), [[0, 128, 255], [255, 0, 64]])


@pytest.mark.parametrize("case", [
    "missing_image", "batch", "source_nan", "mask_size", "mask_batch",
    "mask_nan", "mask_inf", "mask_negative", "mask_high", "mask_empty",
])
def test_invalid_inputs_fail_before_upload(monkeypatch, case):
    np = pytest.importorskip("numpy")
    source, mask = sample_inputs()
    if case == "missing_image":
        source = None
    elif case == "batch":
        source = np.repeat(source, 2, axis=0)
    elif case == "source_nan":
        source[0, 0, 0, 0] = np.nan
    elif case == "mask_size":
        mask = np.ones((1, 3, 3), dtype=np.float32)
    elif case == "mask_batch":
        mask = np.repeat(mask, 2, axis=0)
    elif case == "mask_empty":
        mask.fill(0)
    else:
        mask[0, 0, 0] = {
            "mask_nan": np.nan, "mask_inf": np.inf,
            "mask_negative": -0.1, "mask_high": 1.1,
        }[case]

    def unexpected_client(**kwargs):
        pytest.fail("Invalid inputs must not initialize the client")

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=unexpected_client))
    with pytest.raises(ValueError):
        MAIFalImageEdit().run(
            image=source, mask=mask, prompt="edit", api_key="test-key",
            model_endpoint=NANO_BANANA_21_EDIT_ENDPOINT,
        )


@pytest.mark.parametrize("shape", [(1, 2, 3, 3), (1, 4, 6, 3)])
def test_composite_preserves_black_pixels_blends_gray_and_restores_source_size(shape):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    source, mask = sample_inputs()
    generated = np.full(shape, 0.8, dtype=np.float32)
    result = composite_nano_banana_edit(torch.from_numpy(source), torch.from_numpy(generated), torch.from_numpy(mask))
    assert tuple(result.shape) == source.shape
    np.testing.assert_allclose(result.numpy(), source * (1 - mask[..., None]) + 0.8 * mask[..., None])
    assert torch.equal(result[0][mask[0] == 0], torch.from_numpy(source[0][mask[0] == 0]))
    np.testing.assert_array_equal(source, sample_inputs()[0])


@pytest.mark.parametrize("masked", [False, True])
def test_nano_node_routes_uploads_and_composites_only_when_mask_connected(monkeypatch, masked):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    source, sample_mask = sample_inputs()
    mask = sample_mask if masked else None
    uploads = prepare_image_edit_uploads(NANO_BANANA_21_EDIT_ENDPOINT, source, mask)
    output_png = image_batch_to_png_bytes(np.ones((1, 4, 6, 3), dtype=np.float32))[0]
    uri = "data:image/png;base64," + base64.b64encode(output_png).decode("ascii")
    uploaded = []
    submissions = []
    monkeypatch.setenv("FAL_KEY", "different-env-key")

    class FakeClient:
        def __init__(self, key):
            assert key == "widget-key"

        def upload(self, data, content_type, file_name):
            assert (file_name, data) == uploads[len(uploaded)]
            assert content_type == "image/png"
            url = f"https://example.com/{file_name}"
            uploaded.append(url)
            return url

        def subscribe(self, endpoint, arguments):
            assert endpoint == NANO_BANANA_21_EDIT_ENDPOINT
            assert arguments["image_urls"] == uploaded
            assert arguments["resolution"] == "2K"
            assert arguments["seed"] == 0
            assert arguments["num_images"] == 1
            assert arguments["limit_generations"] is True
            assert "mask_url" not in arguments
            assert ("Image 2 is a grayscale editing mask" in arguments["prompt"]) == masked
            submissions.append(endpoint)
            return {"images": [{"url": uri}], "description": "edited"}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))
    result = MAIFalImageEdit().run(
        image=source, mask=mask, prompt="replace the chair", api_key=" widget-key ",
        model_endpoint=NANO_BANANA_21_EDIT_ENDPOINT, seed=0, resolution="2k",
    )
    assert len(uploaded) == (2 if masked else 1)
    assert len(submissions) == 1
    if masked:
        assert tuple(result[0].shape) == source.shape
        np.testing.assert_allclose(result[0].numpy(), source * (1 - mask[..., None]) + mask[..., None])
    else:
        assert tuple(result[0].shape) == (1, 4, 6, 3)
    assert result[1:3] == (uri, -1)
    assert json.loads(result[3])["description"] == "edited"
