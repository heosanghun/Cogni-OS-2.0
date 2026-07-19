"""Bounded latent conditioning at the local model's logits boundary.

The processor stores one fixed vocabulary bias.  Its state is constant with
respect to decode length and it never retains token history, text, or JSON.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MAX_VOCAB_SIZE = 262_144
MAX_LATENT_WIDTH = 8_192
MAX_HEAD_ELEMENTS = 1_000_000_000
MAX_INPUT_ELEMENTS = 131_072
MAX_ABS_BIAS = 1.0

_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


class LatentConditioningError(RuntimeError):
    """Base class for fail-closed latent conditioning failures."""


class BoundedLatentLogitsProcessor:
    """Add one immutable, bounded bias to a single-batch logits tensor."""

    __slots__ = ("_bias", "_vocab_size")

    def __init__(self, bias: Tensor) -> None:
        if not isinstance(bias, Tensor):
            raise TypeError("bias must be a tensor")
        if bias.ndim != 2 or bias.shape[0] != 1:
            raise ValueError("bias must have shape [1, V]")
        if not 1 <= bias.shape[1] <= MAX_VOCAB_SIZE:
            raise ValueError("bias vocabulary exceeds its hard bound")
        if bias.dtype not in _FLOAT_DTYPES:
            raise TypeError("bias must use a supported floating dtype")
        if bias.device.type == "meta" or bias.layout != torch.strided:
            raise ValueError("bias must be a materialized strided tensor")
        if not bool(torch.isfinite(bias).all()):
            raise ValueError("bias must be finite")
        self._bias = bias.detach().clone().contiguous()
        self._vocab_size = int(bias.shape[1])

    @property
    def bias(self) -> Tensor:
        """Return a defensive copy of the fixed vocabulary bias."""

        return self._bias.detach().clone()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        if not isinstance(input_ids, Tensor) or not isinstance(scores, Tensor):
            raise TypeError("input_ids and scores must be tensors")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, S]")
        if not 1 <= input_ids.numel() <= MAX_INPUT_ELEMENTS:
            raise ValueError("input_ids exceeds its bounded sequence storage")
        if input_ids.dtype != torch.int64:
            raise TypeError("input_ids must use torch.int64 token ids")
        if scores.ndim != 2 or tuple(scores.shape) != (1, self._vocab_size):
            raise ValueError("scores must have shape [1, V] matching the fixed bias")
        if scores.numel() > MAX_VOCAB_SIZE:
            raise ValueError("scores exceeds the vocabulary element bound")
        if scores.dtype != self._bias.dtype:
            raise TypeError("scores dtype must match the fixed bias dtype")
        if scores.device != self._bias.device or input_ids.device != scores.device:
            raise ValueError("input_ids, scores, and bias must share one device")
        if scores.layout != torch.strided or input_ids.layout != torch.strided:
            raise ValueError("input_ids and scores must be strided tensors")
        if bool((torch.isnan(scores) | torch.isposinf(scores)).any()):
            raise ValueError("scores cannot contain NaN or positive infinity")
        if not bool(torch.isfinite(scores).any(dim=-1).all()):
            raise ValueError("scores must retain at least one finite token candidate")
        if bool((input_ids < 0).any()) or bool((input_ids >= self._vocab_size).any()):
            raise ValueError("input_ids contains a token outside the vocabulary")

        result = scores + self._bias
        if bool((torch.isnan(result) | torch.isposinf(result)).any()):
            raise LatentConditioningError(
                "bounded bias produced NaN or positive-infinite logits"
            )
        if not bool(torch.isfinite(result).any(dim=-1).all()):
            raise LatentConditioningError("bounded bias left no finite token candidate")
        return result


def _output_head(model: Any) -> tuple[nn.Module, Tensor, Tensor | None]:
    getter = getattr(model, "get_output_embeddings", None)
    if not callable(getter):
        raise TypeError("model must provide get_output_embeddings()")
    head = getter()
    if not isinstance(head, nn.Module):
        raise TypeError("get_output_embeddings() must return a torch module")
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise LatentConditioningError("output head must be frozen")

    weight = getattr(head, "weight", None)
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise TypeError("output head must expose a rank-two weight tensor")
    if weight.requires_grad:
        raise LatentConditioningError("output head weight must be frozen")
    if weight.dtype not in _FLOAT_DTYPES:
        raise TypeError("output head weight must use a supported floating dtype")
    if weight.device.type == "meta" or weight.layout != torch.strided:
        raise ValueError("output head weight must be materialized and strided")
    vocab_size, hidden_size = map(int, weight.shape)
    if not 1 <= vocab_size <= MAX_VOCAB_SIZE:
        raise ValueError("output vocabulary exceeds its hard bound")
    if not 1 <= hidden_size <= MAX_LATENT_WIDTH:
        raise ValueError("output head hidden width exceeds its hard bound")
    if weight.numel() > MAX_HEAD_ELEMENTS:
        raise ValueError("output head exceeds its element bound")

    bias = getattr(head, "bias", None)
    if bias is not None:
        if not isinstance(bias, Tensor) or tuple(bias.shape) != (vocab_size,):
            raise TypeError("output head bias must have shape [V]")
        if bias.requires_grad:
            raise LatentConditioningError("output head bias must be frozen")
        if bias.dtype != weight.dtype or bias.device != weight.device:
            raise TypeError("output head bias must match weight device and dtype")
        if bias.layout != torch.strided:
            raise ValueError("output head bias must be strided")
    return head, weight, bias


def _safe_dtype_limit(value: float, reference: Tensor) -> Tensor:
    limit = torch.tensor(value, dtype=reference.dtype, device=reference.device)
    if float(limit.detach().cpu()) > value:
        limit = torch.nextafter(limit, torch.zeros_like(limit))
    return limit


def _bounded_bias(projected: Tensor, max_abs_bias: float) -> Tensor:
    work_dtype = torch.float64 if projected.dtype == torch.float64 else torch.float32
    work = projected.to(dtype=work_dtype)
    centered = work - work.mean(dim=-1, keepdim=True)
    scale = centered.abs().amax(dim=-1, keepdim=True)
    safe_scale = scale.clamp_min(torch.finfo(work_dtype).tiny)
    scaled = centered / safe_scale
    rms = safe_scale * scaled.square().mean(dim=-1, keepdim=True).sqrt()
    normalized = torch.where(
        scale > 0,
        centered / rms.clamp_min(torch.finfo(work_dtype).tiny),
        torch.zeros_like(centered),
    )
    bounded = torch.tanh(normalized) * max_abs_bias
    if not bool(torch.isfinite(bounded).all()):
        raise LatentConditioningError("latent normalization produced non-finite bias")
    result = bounded.to(dtype=projected.dtype)
    safe_limit = _safe_dtype_limit(max_abs_bias, result)
    return result.clamp(min=-safe_limit, max=safe_limit).contiguous()


def build_latent_logits_processor(
    model: Any,
    latent: Tensor,
    *,
    max_abs_bias: float = 0.05,
) -> BoundedLatentLogitsProcessor:
    """Compile one finite ``[1, H]`` latent into a fixed bounded logits bias."""

    if not isinstance(max_abs_bias, (int, float)) or isinstance(max_abs_bias, bool):
        raise TypeError("max_abs_bias must be a finite real number")
    limit = float(max_abs_bias)
    if not math.isfinite(limit) or not 0.0 < limit <= MAX_ABS_BIAS:
        raise ValueError(f"max_abs_bias must lie in (0, {MAX_ABS_BIAS}]")
    if not isinstance(latent, Tensor):
        raise TypeError("latent must be a tensor")
    if latent.ndim != 2 or latent.shape[0] != 1:
        raise ValueError("latent must have shape [1, H]")
    if not 1 <= latent.shape[1] <= MAX_LATENT_WIDTH:
        raise ValueError("latent hidden width exceeds its hard bound")
    if latent.dtype not in _FLOAT_DTYPES:
        raise TypeError("latent must use a supported floating dtype")
    if latent.device.type == "meta" or latent.layout != torch.strided:
        raise ValueError("latent must be a materialized strided tensor")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("latent must be finite")

    _head, weight, head_bias = _output_head(model)
    if latent.shape[1] != weight.shape[1]:
        raise ValueError("latent hidden width does not match the output head")
    if latent.device != weight.device or latent.dtype != weight.dtype:
        raise TypeError("latent must match the output head device and dtype")

    weight_version = weight._version
    bias_version = None if head_bias is None else head_bias._version
    with torch.no_grad():
        projected = F.linear(
            latent.detach(),
            weight.detach(),
            None if head_bias is None else head_bias.detach(),
        )
    if weight._version != weight_version or (
        head_bias is not None and head_bias._version != bias_version
    ):
        raise LatentConditioningError("output head mutated during latent projection")
    if tuple(projected.shape) != (1, weight.shape[0]):
        raise LatentConditioningError(
            "output head returned an invalid projection shape"
        )
    if not bool(torch.isfinite(projected).all()):
        raise LatentConditioningError("output head produced non-finite latent logits")

    # Hugging Face generation intentionally promotes next-token scores to
    # FP32 before invoking logits processors, even when Gemma weights and its
    # output head are BF16. Store the fixed bias once in that public boundary
    # dtype so decode never performs an implicit per-step vocabulary cast.
    return BoundedLatentLogitsProcessor(
        _bounded_bias(projected, limit).to(dtype=torch.float32)
    )


__all__ = [
    "BoundedLatentLogitsProcessor",
    "LatentConditioningError",
    "MAX_ABS_BIAS",
    "MAX_HEAD_ELEMENTS",
    "MAX_INPUT_ELEMENTS",
    "MAX_LATENT_WIDTH",
    "MAX_VOCAB_SIZE",
    "build_latent_logits_processor",
]
