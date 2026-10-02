"""Regression coverage for Gemma's wide heads under global CK attention."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


@pytest.fixture
def text_attention_module(monkeypatch):
    # Stub only library wiring; actual tensors, GQA and attention math use torch.
    transformers = ModuleType("transformers")
    transformers.AttentionInterface = SimpleNamespace(register=lambda *args: None)
    masking = ModuleType("transformers.masking_utils")
    masking.AttentionMaskInterface = SimpleNamespace(register=lambda *args: None)
    masking.ALL_MASK_ATTENTION_FUNCTIONS = {"eager": object()}
    diffusers = ModuleType("diffusers")
    diffusers.LTX2VideoTransformer3DModel = object
    ltx = ModuleType("diffusers.models.transformers.transformer_ltx2")
    ltx.LTX2Attention = torch.nn.Module
    ltx.LTX2VideoTransformerBlock = object
    ltx.apply_split_rotary_emb = lambda tensor, rope: tensor
    comfy = ModuleType("comfy")
    comfy.model_management = SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None)
    comfy.ops = SimpleNamespace()
    attention = ModuleType("comfy.ldm.modules.attention")
    calls = []

    def global_attention(*args, **kwargs):
        pytest.fail("Text encoding must not call the global CK INT8 attention kernel")

    def text_attention(query, key, value, heads, mask, skip_reshape, skip_output_reshape, scale, enable_gqa):
        assert skip_reshape and skip_output_reshape
        assert heads == query.shape[1]
        calls.append((query.shape, key.shape, value.shape, scale, enable_gqa))
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale, enable_gqa=enable_gqa)

    def select(device, mask=False, small_input=False):
        assert small_input is True
        calls.append((device, mask, small_input))
        return text_attention

    attention.optimized_attention = global_attention
    attention.optimized_attention_for_device = select
    for name, module in (("transformers", transformers), (masking.__name__, masking),
                         ("diffusers", diffusers), (ltx.__name__, ltx), ("comfy", comfy),
                         (attention.__name__, attention)):
        monkeypatch.setitem(sys.modules, name, module)
    package = __package__.rsplit(".", 1)[0]
    spec = importlib.util.spec_from_file_location(f"{package}.sol_refiner._test_text_models", Path(__file__).parents[1] / "sol_refiner" / "models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


@pytest.mark.parametrize("head_dim,query_heads,key_heads", [(512, 4, 2), (256, 4, 2), (512, 2, 2)])
@pytest.mark.parametrize("scaling", [1.0, None])
@pytest.mark.parametrize("masked", [False, True])
def test_text_attention_preserves_wide_heads_gqa_mask_and_scaling(text_attention_module, head_dim, query_heads, key_heads, scaling, masked):
    module, calls = text_attention_module
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, query_heads, 5, head_dim, generator=generator)
    key = torch.randn(2, key_heads, 5, head_dim, generator=generator)
    value = torch.randn(2, key_heads, 5, head_dim, generator=generator)
    mask = None
    if masked:
        # Eager Transformers masks combine causal and padding biases.
        mask = torch.zeros(2, 1, 5, 5)
        mask.masked_fill_(torch.ones(5, 5, dtype=torch.bool).triu(1), float("-inf"))
        mask[1, :, 1:, 0] = float("-inf")
    actual, weights = module.comfy_text_attention(None, query, key, value, mask, scaling=scaling)
    # An independent float64 reference expands KV and computes the full matrix.
    factor = query_heads // key_heads
    expanded_key = key.repeat_interleave(factor, dim=1).double()
    expanded_value = value.repeat_interleave(factor, dim=1).double()
    scale = head_dim ** -0.5 if scaling is None else scaling
    scores = (query.double() @ expanded_key.transpose(-1, -2)) * scale
    if mask is not None:
        scores += mask.double()
    expected = (scores.softmax(dim=-1) @ expanded_value).float().transpose(1, 2).contiguous()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-5)
    assert actual.shape == (2, 5, query_heads, head_dim)
    assert actual.is_contiguous() and weights is None
    assert calls[0] == (query.device, masked, True)
    assert calls[1] == (query.shape, key.shape, value.shape, scale, query_heads != key_heads)


def test_text_attention_propagates_interruption(text_attention_module, monkeypatch):
    module, calls = text_attention_module

    def interrupted():
        raise KeyboardInterrupt()

    monkeypatch.setattr(module.model_management, "throw_exception_if_processing_interrupted", interrupted)
    tensor = torch.zeros(1, 2, 3, 512)
    with pytest.raises(KeyboardInterrupt):
        module.comfy_text_attention(None, tensor, tensor, tensor, None)
    assert calls == []
