import base64
import json
import sys
from types import SimpleNamespace

import pytest

from ..nodes.fal_image import MAIFalImage
from ..utils.fal_image import (
    CUSTOM_IMAGE_SIZE_ENDPOINTS,
    DEFAULT_ENDPOINT,
    RESOLUTION_MODES,
    SUPPORTED_ENDPOINTS,
    add_image_urls,
    build_arguments,
    build_nano_banana_2_edit_arguments,
    build_seedream_5_lite_edit_arguments,
    build_seedream_5_pro_edit_arguments,
    download_image_bytes,
    extract_image_url,
    get_result_seed,
    image_batch_to_png_bytes,
    normalize_endpoint,
    resolve_image_size_argument,
    resolve_image_input_name,
)


def test_node_exposes_api_key_input():
    required_inputs = MAIFalImage.INPUT_TYPES()["required"]
    api_key_type = required_inputs["api_key"]

    assert api_key_type[0] == "STRING"
    assert api_key_type[1]["default"] == ""
    assert "extra_arguments_json" not in required_inputs
    assert "image_input_name" not in required_inputs


def test_node_exposes_supported_endpoint_dropdown():
    model_endpoint_type = MAIFalImage.INPUT_TYPES()["required"]["model_endpoint"]

    assert model_endpoint_type[0] == list(SUPPORTED_ENDPOINTS)
    assert model_endpoint_type[1]["default"] == DEFAULT_ENDPOINT
    assert "fal-ai/nano-banana-2/edit" in model_endpoint_type[0]
    assert "bytedance/seedream/v5/lite/edit" in model_endpoint_type[0]
    assert "bytedance/seedream/v5/pro/edit" in model_endpoint_type[0]


def test_node_exposes_custom_resolution_inputs():
    required_inputs = MAIFalImage.INPUT_TYPES()["required"]

    assert required_inputs["resolution_mode"][0] == list(RESOLUTION_MODES)
    assert required_inputs["resolution_mode"][1]["default"] == "preset"
    assert required_inputs["custom_width"][1]["default"] == 1024
    assert required_inputs["custom_height"][1]["default"] == 1024
    assert "Nano Banana 2" in required_inputs["resolution_mode"][1]["tooltip"]


def test_node_rejects_empty_api_key_before_loading_client():
    with pytest.raises(ValueError, match="api_key cannot be empty"):
        MAIFalImage().generate(
            "",
            "fal-ai/flux/schnell",
            "a fox",
            "square",
            -1,
            "png",
        )


def test_normalize_endpoint_accepts_endpoint_id():
    assert normalize_endpoint(" /fal-ai/flux/schnell/ ") == "fal-ai/flux/schnell"


def test_normalize_endpoint_rejects_url():
    with pytest.raises(ValueError, match="not a URL"):
        normalize_endpoint("https://fal.run/fal-ai/flux/schnell")


def test_build_arguments_omits_random_seed_and_applies_overrides():
    arguments = build_arguments(
        "a fox",
        "square",
        -1,
        "png",
        json.dumps({"num_inference_steps": 8, "num_images": 1}),
    )

    assert "seed" not in arguments
    assert arguments["prompt"] == "a fox"
    assert arguments["num_inference_steps"] == 8


def test_build_arguments_rejects_non_object_json():
    with pytest.raises(ValueError, match="JSON object"):
        build_arguments("a fox", "square", 1, "png", "[]")


def test_build_arguments_accepts_custom_image_size():
    arguments = build_arguments(
        "a fox",
        {"width": 1536, "height": 1024},
        7,
        "png",
        "{}",
    )

    assert arguments["image_size"] == {"width": 1536, "height": 1024}
    assert arguments["seed"] == 7


def test_resolve_image_size_argument_uses_preset_by_default():
    assert resolve_image_size_argument(
        "fal-ai/nano-banana-2/edit",
        "portrait_4_3",
        "preset",
        1024,
        1024,
    ) == "portrait_4_3"


def test_resolve_image_size_argument_builds_custom_size_for_supported_endpoint():
    assert "openai/gpt-image-2/edit" in CUSTOM_IMAGE_SIZE_ENDPOINTS
    assert resolve_image_size_argument(
        "openai/gpt-image-2/edit",
        "square",
        "custom",
        1536,
        1024,
    ) == {"width": 1536, "height": 1024}


@pytest.mark.parametrize(
    "endpoint, expected_detail",
    [
        ("fal-ai/flux/dev/image-to-image", "does not expose an image_size"),
        ("fal-ai/nano-banana-2/edit", "aspect_ratio and resolution tiers"),
    ],
)
def test_resolve_image_size_argument_explains_unsupported_models(
    endpoint, expected_detail
):
    with pytest.raises(ValueError, match=expected_detail):
        resolve_image_size_argument(
            endpoint,
            "square",
            "custom",
            1024,
            1024,
        )


def test_resolve_image_size_argument_validates_model_limits():
    with pytest.raises(ValueError, match="between 512 and 2048"):
        resolve_image_size_argument(
            "fal-ai/flux-2/edit",
            "square",
            "custom",
            4096,
            4096,
        )

    with pytest.raises(ValueError, match="fal would otherwise rescale"):
        resolve_image_size_argument(
            "bytedance/seedream/v5/lite/edit",
            "square",
            "custom",
            1024,
            1024,
        )


def test_build_nano_banana_2_edit_arguments_uses_endpoint_schema():
    arguments = build_nano_banana_2_edit_arguments(
        "edit the image",
        "landscape_16_9",
        -1,
        "png",
    )

    assert arguments == {
        "prompt": "edit the image",
        "num_images": 1,
        "aspect_ratio": "16:9",
        "output_format": "png",
        "resolution": "1K",
        "limit_generations": True,
    }


def test_build_seedream_5_pro_edit_arguments_uses_endpoint_schema():
    arguments = build_seedream_5_pro_edit_arguments(
        "edit the image",
        "portrait_4_3",
        "jpeg",
    )

    assert arguments == {
        "prompt": "edit the image",
        "image_size": "portrait_4_3",
        "num_images": 1,
        "output_format": "jpeg",
    }


def test_build_seedream_5_lite_edit_arguments_uses_endpoint_schema():
    arguments = build_seedream_5_lite_edit_arguments(
        "edit the image",
        "landscape_16_9",
    )

    assert arguments == {
        "prompt": "edit the image",
        "image_size": "landscape_16_9",
        "num_images": 1,
        "max_images": 1,
    }


def test_extract_image_url_handles_images_and_image_shapes():
    assert extract_image_url({"images": [{"url": "https://example.com/a.png"}]}) == (
        "https://example.com/a.png"
    )
    assert extract_image_url({"data": {"image": {"url": "data:image/png;base64,AA=="}}}) == (
        "data:image/png;base64,AA=="
    )


def test_extract_image_url_rejects_non_image_response():
    with pytest.raises(ValueError, match="did not contain an image URL"):
        extract_image_url({"video": {"url": "https://example.com/a.mp4"}})


def test_get_result_seed_falls_back_when_missing():
    assert get_result_seed({"seed": 42}) == 42
    assert get_result_seed({}) == -1


def test_download_image_bytes_decodes_data_uri():
    assert download_image_bytes("data:image/png;base64,aGVsbG8=") == b"hello"


def test_add_image_urls_supports_singular_and_plural_fields():
    urls = ["https://example.com/a.png", "https://example.com/b.png"]

    assert add_image_urls({}, "image_url", urls[:1]) == {"image_url": urls[0]}
    assert add_image_urls({}, "reference_image_urls", urls) == {
        "reference_image_urls": urls
    }


def test_add_image_urls_rejects_multiple_images_for_singular_field():
    with pytest.raises(ValueError, match="singular image field"):
        add_image_urls({}, "image_url", ["one", "two"])


def test_resolve_image_input_name_uses_list_for_flux_2_edit_family():
    assert resolve_image_input_name("fal-ai/flux-2-max/edit", 1) == "image_urls"
    assert (
        resolve_image_input_name("fal-ai/flux-2/klein/9b/edit/lora", 1)
        == "image_urls"
    )
    assert resolve_image_input_name("fal-ai/flux-2/lora/edit", 1) == "image_urls"
    assert resolve_image_input_name("fal-ai/flux/dev/image-to-image", 1) == "image_url"
    assert resolve_image_input_name("fal-ai/nano-banana-2/edit", 1) == "image_urls"
    assert resolve_image_input_name("bytedance/seedream/v5/lite/edit", 1) == "image_urls"
    assert resolve_image_input_name("bytedance/seedream/v5/pro/edit", 1) == "image_urls"
    assert resolve_image_input_name("some-owner/multi-reference", 2) == "image_urls"


def test_image_batch_to_png_bytes_encodes_every_batch_item():
    np = pytest.importorskip("numpy")
    images = np.zeros((2, 3, 4, 3), dtype=np.float32)

    encoded = image_batch_to_png_bytes(images)

    assert len(encoded) == 2
    assert all(image.startswith(b"\x89PNG\r\n\x1a\n") for image in encoded)


def test_node_uploads_connected_images_and_sends_their_urls(monkeypatch):
    np = pytest.importorskip("numpy")
    input_images = np.zeros((2, 2, 2, 3), dtype=np.float32)
    output_png = image_batch_to_png_bytes(input_images[:1])[0]
    output_uri = "data:image/png;base64," + base64.b64encode(output_png).decode("ascii")
    captured = {}

    class FakeClient:
        def __init__(self, key):
            captured["key"] = key

        def upload(self, data, content_type, file_name):
            captured.setdefault("uploads", []).append((data, content_type, file_name))
            return f"https://example.com/{file_name}"

        def subscribe(self, endpoint, arguments):
            captured["endpoint"] = endpoint
            captured["arguments"] = arguments
            return {"images": [{"url": output_uri}], "seed": 9}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))

    image, _, seed, _ = MAIFalImage().generate(
        " test-key ",
        "fal-ai/example/image-to-image",
        "a fox",
        "square",
        -1,
        "png",
        image_1=input_images,
    )

    assert captured["key"] == "test-key"
    assert len(captured["uploads"]) == 2
    assert captured["arguments"]["image_urls"] == [
        "https://example.com/comfy_input_1.png",
        "https://example.com/comfy_input_2.png",
    ]
    assert tuple(image.shape) == (1, 2, 2, 3)
    assert seed == 9


def test_node_sends_nano_banana_2_edit_schema(monkeypatch):
    np = pytest.importorskip("numpy")
    input_image = np.zeros((1, 2, 2, 3), dtype=np.float32)
    output_png = image_batch_to_png_bytes(input_image)[0]
    output_uri = "data:image/png;base64," + base64.b64encode(output_png).decode("ascii")
    captured = {}

    class FakeClient:
        def __init__(self, key):
            captured["key"] = key

        def upload(self, data, content_type, file_name):
            return "https://example.com/input.png"

        def subscribe(self, endpoint, arguments):
            captured["endpoint"] = endpoint
            captured["arguments"] = arguments
            return {"images": [{"url": output_uri}]}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))

    MAIFalImage().generate(
        "test-key",
        "fal-ai/nano-banana-2/edit",
        "make the sky purple",
        "landscape_16_9",
        -1,
        "png",
        image_1=input_image,
    )

    assert captured["endpoint"] == "fal-ai/nano-banana-2/edit"
    assert captured["arguments"] == {
        "prompt": "make the sky purple",
        "num_images": 1,
        "aspect_ratio": "16:9",
        "output_format": "png",
        "resolution": "1K",
        "limit_generations": True,
        "image_urls": ["https://example.com/input.png"],
    }


def test_node_rejects_nano_banana_2_edit_without_an_input_image(monkeypatch):
    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace())

    with pytest.raises(ValueError, match="requires at least one connected input image"):
        MAIFalImage().generate(
            "test-key",
            "fal-ai/nano-banana-2/edit",
            "edit the image",
            "square",
            -1,
            "png",
        )


def test_node_sends_seedream_5_pro_edit_schema_without_seed(monkeypatch):
    np = pytest.importorskip("numpy")
    input_image = np.zeros((1, 2, 2, 3), dtype=np.float32)
    output_png = image_batch_to_png_bytes(input_image)[0]
    output_uri = "data:image/png;base64," + base64.b64encode(output_png).decode("ascii")
    captured = {}

    class FakeClient:
        def __init__(self, key):
            captured["key"] = key

        def upload(self, data, content_type, file_name):
            return "https://example.com/input.png"

        def subscribe(self, endpoint, arguments):
            captured["endpoint"] = endpoint
            captured["arguments"] = arguments
            return {"images": [{"url": output_uri}]}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))

    MAIFalImage().generate(
        "test-key",
        "bytedance/seedream/v5/pro/edit",
        "replace the background",
        "portrait_4_3",
        123,
        "jpeg",
        image_1=input_image,
    )

    assert captured["endpoint"] == "bytedance/seedream/v5/pro/edit"
    assert captured["arguments"] == {
        "prompt": "replace the background",
        "image_size": "portrait_4_3",
        "num_images": 1,
        "output_format": "jpeg",
        "image_urls": ["https://example.com/input.png"],
    }


def test_node_rejects_seedream_5_pro_edit_without_an_input_image(monkeypatch):
    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace())

    with pytest.raises(ValueError, match="requires at least one connected input image"):
        MAIFalImage().generate(
            "test-key",
            "bytedance/seedream/v5/pro/edit",
            "edit the image",
            "square",
            -1,
            "png",
        )


def test_node_sends_seedream_5_lite_edit_schema(monkeypatch):
    np = pytest.importorskip("numpy")
    input_image = np.zeros((1, 2, 2, 3), dtype=np.float32)
    output_png = image_batch_to_png_bytes(input_image)[0]
    output_uri = "data:image/png;base64," + base64.b64encode(output_png).decode("ascii")
    captured = {}

    class FakeClient:
        def __init__(self, key):
            captured["key"] = key

        def upload(self, data, content_type, file_name):
            return "https://example.com/input.png"

        def subscribe(self, endpoint, arguments):
            captured["endpoint"] = endpoint
            captured["arguments"] = arguments
            return {"images": [{"url": output_uri}], "seed": 42}

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))

    _, _, result_seed, _ = MAIFalImage().generate(
        "test-key",
        "bytedance/seedream/v5/lite/edit",
        "replace the background",
        "landscape_16_9",
        123,
        "jpeg",
        image_1=input_image,
        resolution_mode="custom",
        custom_width=2560,
        custom_height=1440,
    )

    assert captured["endpoint"] == "bytedance/seedream/v5/lite/edit"
    assert captured["arguments"] == {
        "prompt": "replace the background",
        "image_size": {"width": 2560, "height": 1440},
        "num_images": 1,
        "max_images": 1,
        "image_urls": ["https://example.com/input.png"],
    }
    assert result_seed == 42


def test_node_rejects_seedream_5_lite_edit_without_an_input_image(monkeypatch):
    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace())

    with pytest.raises(ValueError, match="requires at least one connected input image"):
        MAIFalImage().generate(
            "test-key",
            "bytedance/seedream/v5/lite/edit",
            "edit the image",
            "square",
            -1,
            "png",
        )
