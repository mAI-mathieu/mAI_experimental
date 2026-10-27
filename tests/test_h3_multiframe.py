from types import SimpleNamespace

import pytest

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
from ..nodes import h3_multiframe as node_module
from ..nodes.h3_multiframe import MAIH3Multiframe
from ..utils.h3_multiframe import (
    duration_to_length, parse_reference_json, resolve_guide_frames, validate_settings,
)
from ..utils.h3_reference_images import H3ImageLoader, resolve_image_path


@pytest.mark.parametrize("duration,length", [(0.01, 5), (5, 124), (6, 158), (15, 362)])
def test_duration_matches_example_workflow(duration, length):
    assert duration_to_length(duration) == length


@pytest.mark.parametrize("duration", [0, -1, 15.1, float("nan"), float("inf"), "5", True])
def test_invalid_duration(duration):
    with pytest.raises(ValueError, match="duration"):
        duration_to_length(duration)


def test_times_preserve_picture_order_and_use_nearest_frame():
    assert resolve_guide_frames([5, 0, 1.5, 3, 0.05], 124) == [120, 0, 36, 72, 1]
    assert resolve_guide_frames([123 / 24], 124) == [123]


@pytest.mark.parametrize("times,match", [
    ([], "at least one"), ([0, 0.01], "already uses frame"),
    ([124 / 24], "outside"), ([-0.01], "non-negative"),
    ([float("nan")], "finite"), ([float("inf")], "finite"),
    (["1.5"], "seconds"), ([True], "seconds"),
])
def test_invalid_times(times, match):
    with pytest.raises(ValueError, match=match):
        resolve_guide_frames(times, 124)


def test_json_paths_and_blank_fallback():
    assert parse_reference_json(" \n ") is None
    assert parse_reference_json('[{"image":"refs/a.png","time":1.5}]') == [("refs/a.png", 1.5)]
    assert parse_reference_json('[{"image":"C:/references/a.png","time":0}]') == [("C:/references/a.png", 0)]


@pytest.mark.parametrize("value", [
    "[", "{}", "[]", "null", '[{"image":"a.png"}]',
    '[{"image":"","time":0}]', '[{"image":12,"time":0}]',
    '[{"image":"https://example.org/a.png","time":0}]',
    '[{"image":"//server/share/a.png","time":0}]',
    '[{"image":"a.png","time":0,"extra":1}]',
    '[{"image":"a.png","time":0},' * 10 + ']',
])
def test_invalid_json(value):
    with pytest.raises(ValueError):
        parse_reference_json(value)


@pytest.mark.parametrize("settings", [
    (1345, 768, 5, 2, "match"), (1344, 0, 5, 2, "match"),
    (1344, 768, 5, 0, "match"), (1344, 768, 5, 10, "match"),
    (1344, 768, 5, 2, "unknown"),
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        validate_settings(*settings)


class StillImage:
    shape = (1, 64, 96, 3)

    def __getitem__(self, item):
        return self


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    image = StillImage()
    latent = {"samples": object()}

    class Loader:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": (["a.png", "b.png"], {"image_upload": True})}}

        @classmethod
        def VALIDATE_INPUTS(cls, path):
            calls.append(("validate", path))
            return "Missing image file" if path == "missing.png" else True

        @classmethod
        def IS_CHANGED(cls, path):
            return f"hash:{path}"

        def load_image(self, path):
            calls.append(("load", path))
            return image, None

    class Reference:
        @classmethod
        def execute(cls, **kwargs):
            calls.append(("reference", kwargs))
            return SimpleNamespace(result=([], latent))

    class Guide:
        @classmethod
        def execute(cls, **kwargs):
            calls.append(("guide", kwargs))
            return SimpleNamespace(result=(kwargs["positive"] + [kwargs["frame_idx"]],))

    monkeypatch.setattr(node_module, "get_image_loader", lambda: Loader)
    monkeypatch.setattr(node_module, "get_h3_nodes", lambda: (Reference, Guide))
    return SimpleNamespace(calls=calls, image=image, latent=latent)


def test_registration_and_stable_schema(runtime):
    assert NODE_CLASS_MAPPINGS["MAIH3Multiframe"] is MAIH3Multiframe
    assert NODE_DISPLAY_NAME_MAPPINGS["MAIH3Multiframe"] == "mAI H3 multiframe"
    assert MAIH3Multiframe.RETURN_TYPES == ("CONDITIONING", "LATENT")
    assert MAIH3Multiframe.RETURN_NAMES == ("positive", "latent")
    assert MAIH3Multiframe.CATEGORY == "mAI / Video"
    assert WEB_DIRECTORY == "./web"
    inputs = MAIH3Multiframe.INPUT_TYPES()
    for index in range(1, 10):
        assert inputs["optional"][f"image_{index}"][0] == "IMAGE"
        assert inputs["optional"][f"image_file_{index}"][0][0] == "(none)"


def test_upload_and_socket_mix_socket_overrides_file(runtime):
    positive, latent = MAIH3Multiframe().prepare(
        "clip", "vae", "<Picture 1> moves toward <Picture 2>",
        image_1=runtime.image, image_file_1="missing.png", time_1=0,
        image_file_2="b.png", time_2=1.5,
        image_file_3="missing.png", time_3=99,
    )
    assert positive == [0, 36]
    assert latent is runtime.latent
    assert ("load", "b.png") in runtime.calls
    assert ("validate", "missing.png") not in runtime.calls
    ref = next(args for name, args in runtime.calls if name == "reference")
    assert ref["length"] == 124
    assert list(ref["ref_images"]) == ["ref_image_0", "ref_image_1"]
    assert all(args["latent"] is latent for name, args in runtime.calls if name == "guide")


def test_json_overrides_widgets_count_and_connected_images(runtime):
    result = MAIH3Multiframe().prepare(
        "clip", "vae", "prompt", reference_count=1,
        reference_json='[{"image":"a.png","time":3},{"image":"b.png","time":0}]',
        image_1=runtime.image, image_file_1="missing.png", time_1=99,
    )
    assert result[0] == [72, 0]
    assert [args for name, args in runtime.calls if name == "load"] == ["a.png", "b.png"]


def test_empty_optional_rows_are_ignored_and_guide_only_supported(runtime):
    result = MAIH3Multiframe().prepare("clip", "vae", "prompt", image_2=runtime.image, use_as_references=False)
    assert result[0] == [36]
    ref = next(args for name, args in runtime.calls if name == "reference")
    assert ref["ref_images"] is None


@pytest.mark.parametrize("inputs,match", [
    ({}, "at least one"),
    ({"image_file_1": "a.png", "time_1": 10}, "outside"),
    ({"image_file_1": "a.png", "time_1": 0, "image_file_2": "b.png", "time_2": 0.01}, "already uses"),
    ({"image_file_1": "a.png", "image_file_2": "missing.png", "time_2": 1}, "Missing image"),
])
def test_invalid_inputs_fail_before_loading_or_encoding(runtime, inputs, match):
    with pytest.raises(ValueError, match=match):
        MAIH3Multiframe().prepare("clip", "vae", "prompt", **inputs)
    assert not any(name in ("load", "reference", "guide") for name, _ in runtime.calls)


def test_rejects_image_batch_before_encoding(runtime):
    image = SimpleNamespace(shape=(2, 64, 96, 3))
    with pytest.raises(ValueError, match="one RGB/RGBA still"):
        MAIH3Multiframe().prepare("clip", "vae", "prompt", image_1=image)
    assert not any(name in ("reference", "guide") for name, _ in runtime.calls)


def test_validation_and_active_file_cache_keys(runtime):
    assert MAIH3Multiframe.VALIDATE_INPUTS(image_file_9="deleted.png") is True
    assert "outside" in MAIH3Multiframe.VALIDATE_INPUTS(reference_json='[{"image":"a.png","time":99}]')
    assert MAIH3Multiframe.IS_CHANGED(image_file_1="a.png", image_file_9="missing.png") == ("hash:a.png",)
    assert MAIH3Multiframe.IS_CHANGED(image_1=runtime.image, image_file_1="missing.png") == ()
    assert MAIH3Multiframe.IS_CHANGED(reference_count=1, reference_json='[{"image":"b.png","time":0}]') == ("hash:b.png",)


def test_absolute_paths_and_file_content_hashes(tmp_path):
    image = tmp_path / "still.png"
    image.write_bytes(b"first image contents")
    assert resolve_image_path(str(image)) == image
    assert H3ImageLoader.VALIDATE_INPUTS(str(image)) is True
    first = H3ImageLoader.IS_CHANGED(str(image))
    image.write_bytes(b"other image contents")
    assert H3ImageLoader.IS_CHANGED(str(image)) != first
    assert "does not exist" in H3ImageLoader.VALIDATE_INPUTS(str(tmp_path / "missing.png"))


@pytest.mark.parametrize("path", ["", None, "https://example.org/a.png", "//server/share/a.png", "\\\\server\\share\\a.png"])
def test_local_path_validation(path):
    assert isinstance(H3ImageLoader.VALIDATE_INPUTS(path), str)


def test_valid_json_rejects_too_many_entries():
    import json
    with pytest.raises(ValueError, match="1 to 9"):
        parse_reference_json(json.dumps([{"image": "a.png", "time": i} for i in range(10)]))
