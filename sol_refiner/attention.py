"""Probe decoder backends with small real operations, without installing packages."""

import importlib.util
import logging
from functools import lru_cache

import torch

from .utils import select_backend

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def natten_compatible(device):
    try:
        from natten.functional import na3d

        q = torch.zeros((1, 3, 3, 3, 1, 64), device=device, dtype=torch.bfloat16)
        na3d(q, q, q, kernel_size=(3, 3, 3), scale=1.0)
        return True
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, NotImplementedError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        logger.info("[mAI SoL H3] NATTEN unavailable: %s", exc)
        return False


def decoder_backend(requested, device):
    natten = requested != "flex" and natten_compatible(str(device))
    flex = (importlib.util.find_spec("triton") is not None
            and importlib.util.find_spec("torch.nn.attention.flex_attention") is not None)
    return select_backend(requested, natten, flex)
