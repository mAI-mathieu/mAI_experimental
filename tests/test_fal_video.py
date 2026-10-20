import json
import math
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from ..nodes import fal_video as node_module
from ..nodes.fal_video import MAIFalVideo
from ..utils import fal_video as video_utils
from ..utils.fal_video import (
    MINIMAX_H3_ENDPOINT,
    VIDEO_ENDPOINTS,
    WAN_ENDPOINT,
    add_video_frame_urls,
    build_video_arguments,
    download_video_bytes,
    extract_video_url,
    validate_video_frames,
)


def test_registration_inputs_and_output_contract():
    assert NODE_CLASS_MAPPINGS["MAIFalVideo"] is MAIFalVideo
    assert NODE_DISPLAY_NAME_MAPPINGS["MAIFalVideo"] == "mAI fal video"
    assert MAIFalVideo.CATEGORY == "mAI / Video"
    assert MAIFalVideo.RETURN_TYPES == ("VIDEO", "STRING", "INT", "STRING")
    assert MAIFalVideo.RETURN_NAMES == ("video", "video_url", "seed", "response_json")
    inputs = MAIFalVideo.INPUT_TYPES()
    assert inputs["required"]["api_key"][0] == "STRING"
    assert inputs["required"]["api_key"][1]["default"] == ""
    assert inputs["required"]["model_endpoint"][0] == list(VIDEO_ENDPOINTS)
    assert inputs["required"]["first_frame"] == ("IMAGE",)
    assert inputs["optional"]["last_frame"] == ("IMAGE",)
    assert math.isnan(MAIFalVideo.IS_CHANGED())
    assert {
        "MAIExampleTextNode", "MAIFalImage", "MAIFalImageEdit", "MAIFalIdeogramEdit",
        "MAISoLH3Loader", "MAISoLH3Refiner", "MAIFalVideo",
    } <= set(NODE_CLASS_MAPPINGS)


def test_minimax_defaults_and_model_specific_fields():
    assert build_video_arguments(MINIMAX_H3_ENDPOINT, "A slow pan") == {
        "prompt": "A slow pan", "duration": 5, "resolution": "2K",
        "prompt_expansion_mode": "balanced",
    }
    args = build_video_arguments(MINIMAX_H3_ENDPOINT, "A slow pan", seed=0,
                                 resolution="768P", prompt_expansion_mode="disabled")
    assert args["seed"] == 0
    assert args["resolution"] == "768P"
    assert args["prompt_expansion_mode"] == "disabled"
    assert "start_image_url" not in args
    assert "audio" not in args
    assert "enable_prompt_expansion" not in args


def test_wan_defaults_and_model_specific_fields():
    assert build_video_arguments(WAN_ENDPOINT, "") == {
        "prompt": "", "duration": 5, "resolution": "1080p", "aspect_ratio": "adaptive",
        "audio": True, "enable_prompt_expansion": True, "enable_thinking": False,
    }
    args = build_video_arguments(
        WAN_ENDPOINT, "A slow pan", duration=30, resolution="720p", seed=2147483647,
        aspect_ratio="9:16", audio=False, enable_prompt_expansion=False, enable_thinking=True,
    )
    assert args["seed"] == 2147483647
    assert args["duration"] == 30
    assert args["aspect_ratio"] == "9:16"
    assert args["audio"] is False
    assert args["enable_prompt_expansion"] is False
    assert args["enable_thinking"] is True
    assert "prompt_expansion_mode" not in args


def test_wan_smart_duration_sends_null():
    args = build_video_arguments(WAN_ENDPOINT, "A slow pan", duration=0, smart_duration=True)
    assert args["duration"] is None
    with pytest.raises(ValueError, match="smart_duration"):
        build_video_arguments(MINIMAX_H3_ENDPOINT, "A slow pan", smart_duration=True)


@pytest.mark.parametrize("endpoint,minimum,maximum", [
    (MINIMAX_H3_ENDPOINT, 5, 15), (WAN_ENDPOINT, 2, 30),
])
def test_duration_bounds(endpoint, minimum, maximum):
    for duration in (minimum, maximum):
        assert build_video_arguments(endpoint, "pan", duration=duration)["duration"] == duration
    for duration in (minimum - 1, maximum + 1, 5.5, "5", True):
        with pytest.raises(ValueError, match="duration"):
            build_video_arguments(endpoint, "pan", duration=duration)


@pytest.mark.parametrize("endpoint,resolutions", [
    (MINIMAX_H3_ENDPOINT, ("480P", "768P", "2K", "4K")),
    (WAN_ENDPOINT, ("480p", "720p", "1080p")),
])
def test_resolutions_match_api_case(endpoint, resolutions):
    for resolution in resolutions:
        assert build_video_arguments(endpoint, "pan", resolution=resolution)["resolution"] == resolution
    other_resolution = "1080p" if endpoint == MINIMAX_H3_ENDPOINT else "2K"
    with pytest.raises(ValueError, match="resolution"):
        build_video_arguments(endpoint, "pan", resolution=other_resolution)


@pytest.mark.parametrize("endpoint", VIDEO_ENDPOINTS)
@pytest.mark.parametrize("seed", [-2, 2147483648, 1.5, True, "1"])
def test_invalid_seeds(endpoint, seed):
    with pytest.raises(ValueError, match="seed"):
        build_video_arguments(endpoint, "pan", seed=seed)


@pytest.mark.parametrize("endpoint,length", [(MINIMAX_H3_ENDPOINT, 50000), (WAN_ENDPOINT, 20000)])
def test_prompt_limits(endpoint, length):
    assert build_video_arguments(endpoint, "x" * length)
    with pytest.raises(ValueError, match="prompt"):
        build_video_arguments(endpoint, "x" * (length + 1))
    with pytest.raises(ValueError, match="prompt"):
        build_video_arguments(endpoint, None)


def test_invalid_model_and_model_specific_options():
    with pytest.raises(ValueError, match="model_endpoint"):
        build_video_arguments("unknown/model", "pan")
    with pytest.raises(ValueError, match="prompt"):
        build_video_arguments(MINIMAX_H3_ENDPOINT, "  ")
    with pytest.raises(ValueError, match="prompt_expansion_mode"):
        build_video_arguments(MINIMAX_H3_ENDPOINT, "pan", prompt_expansion_mode="unknown")
    with pytest.raises(ValueError, match="aspect_ratio"):
        build_video_arguments(WAN_ENDPOINT, "pan", aspect_ratio="unknown")
    with pytest.raises(ValueError, match="audio"):
        build_video_arguments(WAN_ENDPOINT, "pan", audio="false")


@pytest.mark.parametrize("shape", [
    (0, 8, 8, 3), (2, 8, 8, 3), (1, 0, 8, 3), (1, 8, 0, 3),
    (8, 8, 3), (1, 8, 8, 2), (),
])
@pytest.mark.parametrize("socket", ["first_frame", "last_frame"])
def test_frame_shape_rejection(shape, socket):
    kwargs = {"first_frame": SimpleNamespace(shape=(1, 8, 8, 3))}
    kwargs[socket] = SimpleNamespace(shape=shape)
    with pytest.raises(ValueError, match=socket):
        validate_video_frames(**kwargs)


def test_missing_first_frame_and_optional_last_frame():
    frame = SimpleNamespace(shape=(1, 8, 8, 3))
    with pytest.raises(ValueError, match="first_frame"):
        validate_video_frames(None, frame)
    for channels in (1, 3, 4):
        validate_video_frames(SimpleNamespace(shape=(1, 8, 8, channels)))
    validate_video_frames(frame, frame)


@pytest.mark.parametrize("endpoint,first_field", [
    (MINIMAX_H3_ENDPOINT, "image_url"), (WAN_ENDPOINT, "start_image_url"),
])
def test_frame_url_mapping_without_mutation(endpoint, first_field):
    original = {"prompt": "pan"}
    assert add_video_frame_urls(endpoint, original, "first", "last") == {
        "prompt": "pan", first_field: "first", "end_image_url": "last",
    }
    assert add_video_frame_urls(endpoint, original, "first") == {"prompt": "pan", first_field: "first"}
    assert original == {"prompt": "pan"}


@pytest.mark.parametrize("wrapped", [False, True])
def test_extract_video_url(wrapped):
    result = {"video": {"url": "https://example.com/video.mp4"}}
    assert extract_video_url({"data": result} if wrapped else result) == "https://example.com/video.mp4"


@pytest.mark.parametrize("result", [None, {}, {"video": {}}, {"video": {"url": ""}},
                                         {"video": {"url": "file:///video.mp4"}}])
def test_invalid_video_response(result):
    with pytest.raises(ValueError, match="video"):
        extract_video_url(result)


def test_download_video_uses_timeout_and_preserves_container_bytes(monkeypatch):
    def fake_open(request, timeout):
        assert request.full_url == "https://example.com/video.mp4"
        assert timeout == 300
        return BytesIO(b"encoded-video-with-audio")

    monkeypatch.setattr(video_utils, "urlopen", fake_open)
    assert download_video_bytes("https://example.com/video.mp4") == b"encoded-video-with-audio"
    with pytest.raises(ValueError, match="HTTP"):
        download_video_bytes("file:///local.mp4")
    monkeypatch.setattr(video_utils, "urlopen", lambda *args, **kwargs: BytesIO())
    with pytest.raises(ValueError, match="empty"):
        download_video_bytes("https://example.com/video.mp4")


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = []
    result = {"video": {"url": "https://example.com/generated.mp4"}, "seed": 123}

    class FakeClient:
        def __init__(self, key):
            assert key == "test-key"
            calls.append(("client",))

        def upload(self, data, content_type, file_name):
            assert content_type == "image/png"
            assert data in (b"first-png", b"last-png")
            calls.append(("upload", file_name))
            return f"https://example.com/{file_name}"

        def subscribe(self, endpoint, arguments):
            calls.append(("subscribe", endpoint, arguments))
            return result

    class FakeVideo:
        def __init__(self, buffer):
            assert buffer.read() == b"encoded-video"

        def get_dimensions(self):
            calls.append(("dimensions",))
            return (1920, 1080)

    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(SyncClient=FakeClient))
    monkeypatch.setitem(sys.modules, "comfy_api.input_impl", SimpleNamespace(VideoFromFile=FakeVideo))
    monkeypatch.setattr(node_module, "image_batch_to_png_bytes", lambda frame: [frame.png])
    monkeypatch.setattr(node_module, "download_video_bytes", lambda url: b"encoded-video")
    return SimpleNamespace(
        calls=calls, result=result, video_class=FakeVideo,
        first=SimpleNamespace(shape=(1, 8, 8, 3), png=b"first-png"),
        last=SimpleNamespace(shape=(1, 8, 8, 3), png=b"last-png"),
    )


@pytest.mark.parametrize("endpoint,first_field", [
    (MINIMAX_H3_ENDPOINT, "image_url"), (WAN_ENDPOINT, "start_image_url"),
])
@pytest.mark.parametrize("use_last", [True, False])
def test_node_uploads_frames_and_returns_native_video(fake_runtime, endpoint, first_field, use_last):
    runtime = fake_runtime
    video, url, seed, response_json = MAIFalVideo().generate(
        " test-key ", endpoint, runtime.first, "pan",
        last_frame=runtime.last if use_last else None,
    )
    assert isinstance(video, runtime.video_class)
    assert url == "https://example.com/generated.mp4"
    assert seed == 123
    assert json.loads(response_json) == runtime.result
    args = next(call[2] for call in runtime.calls if call[0] == "subscribe")
    assert args[first_field] == "https://example.com/first_frame.png"
    assert ("end_image_url" in args) == use_last
    if use_last:
        assert args["end_image_url"] == "https://example.com/last_frame.png"
    assert len([call for call in runtime.calls if call[0] == "upload"]) == (2 if use_last else 1)
    assert len([call for call in runtime.calls if call[0] == "subscribe"]) == 1


@pytest.mark.parametrize("case", ["key", "missing", "batch", "resolution", "duration", "seed"])
def test_invalid_inputs_never_create_client(fake_runtime, case):
    runtime = fake_runtime
    kwargs = {"api_key": "test-key", "model_endpoint": MINIMAX_H3_ENDPOINT,
              "first_frame": runtime.first, "prompt": "pan"}
    kwargs.update({
        "key": {"api_key": ""}, "missing": {"first_frame": None},
        "batch": {"last_frame": SimpleNamespace(shape=(2, 8, 8, 3))},
        "resolution": {"resolution": "1080p"}, "duration": {"duration": 2},
        "seed": {"seed": -2},
    }[case])
    with pytest.raises(ValueError):
        MAIFalVideo().generate(**kwargs)
    assert not runtime.calls


def test_unavailable_native_video_support_fails_before_upload(fake_runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy_api.input_impl", None)
    with pytest.raises(ImportError, match="native VIDEO"):
        MAIFalVideo().generate("test-key", MINIMAX_H3_ENDPOINT, fake_runtime.first, "pan")
    assert not fake_runtime.calls


def test_minimax_missing_result_seed_returns_unknown(fake_runtime):
    fake_runtime.result.pop("seed")
    assert MAIFalVideo().generate("test-key", MINIMAX_H3_ENDPOINT, fake_runtime.first, "pan")[2] == -1


def test_download_failure_includes_recovery_url_without_resubmission(fake_runtime, monkeypatch):
    def fail_download(url):
        raise OSError("connection lost")

    monkeypatch.setattr(node_module, "download_video_bytes", fail_download)
    with pytest.raises(RuntimeError, match="https://example.com/generated.mp4"):
        MAIFalVideo().generate("test-key", MINIMAX_H3_ENDPOINT, fake_runtime.first, "pan")
    assert len([call for call in fake_runtime.calls if call[0] == "subscribe"]) == 1
