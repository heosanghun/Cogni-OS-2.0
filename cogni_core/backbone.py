"""Offline Gemma decoder integration for the memory-bounded DEQ core.

The module deliberately has no import-time dependency on ``transformers``.
Synthetic/HF-like decoder blocks can therefore exercise the complete fixed-
point and implicit-gradient path in CPU-only tests.  A real Hugging Face model
is imported lazily and may only be loaded from a verified local directory.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .deq import (
    ContractivityError,
    DEQConfig,
    SolverInfo,
    _broyden_inverse,
    _damped_fixed_point,
    normalized_residual,
)
from .resources import MAX_VRAM_GIB, ResourceBudgetExceeded


class DecoderLayerContractError(TypeError):
    """Raised when a decoder block cannot be used as a DEQ transition."""


class OfflineModelPolicyError(ValueError):
    """Raised before a model load could attempt remote or unverified access."""


@dataclass(frozen=True)
class BackwardSolverInfo:
    """Diagnostics for the most recent matrix-free IFT adjoint solve."""

    converged: bool
    iterations: int
    residual: float


def extract_hidden_states(output: Any) -> Tensor:
    """Extract the hidden-state tensor from an HF-like decoder return value."""

    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    last_hidden = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden, Tensor):
        return last_hidden
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states and isinstance(hidden_states[-1], Tensor):
        return hidden_states[-1]
    raise DecoderLayerContractError(
        "output must expose a Tensor hidden state directly, in tuple[0], as "
        "last_hidden_state, or as hidden_states[-1]; "
        f"received {type(output).__name__}"
    )


def _signature_policy(layer: nn.Module) -> tuple[frozenset[str], bool]:
    """Return accepted keyword names and whether ``forward`` has ``**kwargs``."""

    try:
        parameters = inspect.signature(layer.forward).parameters.values()
    except (TypeError, ValueError):
        # Compiled/decorated modules may not expose a signature. In that case
        # forwarding is safer than silently losing required Gemma arguments.
        return frozenset(), True
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    accepted = frozenset(
        p.name
        for p in parameters
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    return accepted, accepts_any


class GemmaDEQBackboneAdapter(nn.Module):
    """Turn one Gemma-compatible decoder layer into a tied-weight DEQ block.

    The fixed-point map is ``f(z, x) = layer(z + residual_scale * x, ...)``.
    Broyden/fallback iterations always execute under ``torch.no_grad()`` and
    retain at most ``DEQConfig.history`` state differences.  One final layer
    evaluation reconnects autograd, while a bounded matrix-free solve applies
    the Implicit Function Theorem in backward.

    Positional layer arguments (including Gemma 4 per-layer embeddings) and
    immutable attention arguments are replayed unchanged on every iteration.
    KV-cache mutation and attention collection are disabled inside the loop;
    a supplied ``shared_kv_states`` mapping is shallow-copied per iteration so
    a decoder block cannot leak writes from one fixed-point step to the next.
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        config: DEQConfig | None = None,
        *,
        residual_scale: float = 1.0,
        contractive_delta_scale: float | None = None,
        certified_delta_lipschitz_bound: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(decoder_layer, nn.Module):
            raise TypeError("decoder_layer must be a torch.nn.Module")
        self.decoder_layer = decoder_layer
        self.config = config or DEQConfig()
        if self.config.max_iter < 1 or self.config.history < 1:
            raise ValueError("DEQ max_iter and history must be positive")
        if self.config.fallback_steps < 0:
            raise ValueError("DEQ fallback_steps cannot be negative")
        if not 0.0 < self.config.fallback_damping <= 1.0:
            raise ValueError("DEQ fallback_damping must be in (0, 1]")
        self.residual_scale = float(residual_scale)
        if (
            contractive_delta_scale is not None
            and not 0.0 < contractive_delta_scale <= 1.0
        ):
            raise ValueError("contractive_delta_scale must lie in (0, 1]")
        self.contractive_delta_scale = contractive_delta_scale
        if certified_delta_lipschitz_bound is not None:
            if contractive_delta_scale is None:
                raise ValueError(
                    "certified_delta_lipschitz_bound requires contractive_delta_scale"
                )
            if (
                not torch.isfinite(torch.tensor(certified_delta_lipschitz_bound))
                or certified_delta_lipschitz_bound < 0.0
            ):
                raise ValueError(
                    "certified_delta_lipschitz_bound must be finite and non-negative"
                )
        self.certified_delta_lipschitz_bound = certified_delta_lipschitz_bound
        if (
            contractive_delta_scale is not None
            and certified_delta_lipschitz_bound is None
            and self.config.fail_on_noncontractive
        ):
            raise ContractivityError(
                "contractive_delta_scale requires a certified upper bound for "
                "the decoder delta branch in fail-closed mode"
            )
        self.last_info: SolverInfo | None = None
        self.last_backward_info: BackwardSolverInfo | None = None
        self._accepted_kwargs, self._accepts_any_kwarg = _signature_policy(
            decoder_layer
        )
        self._returns_tuple: bool | None = None
        # A stochastic transition is not a well-defined fixed-point map.
        # Parameters remain trainable; only dropout-like training behaviour is
        # disabled on the tied decoder block.
        self.decoder_layer.eval()

    @property
    def effective_lipschitz_upper_bound(self) -> float:
        """Certified upper bound for the actual tied DEQ transition.

        For ``f(z, x) = x + s * (layer(h) - h)``, a certified bound ``L``
        on the delta branch gives ``Lip_z(f) <= s * L``.  No finite claim is
        made for an arbitrary decoder layer without that certificate.
        """

        if (
            self.contractive_delta_scale is None
            or self.certified_delta_lipschitz_bound is None
        ):
            return float("inf")
        return self.contractive_delta_scale * self.certified_delta_lipschitz_bound

    def train(self, mode: bool = True) -> GemmaDEQBackboneAdapter:
        """Keep the DEQ transition deterministic even while its weights train."""

        super().train(mode)
        self.decoder_layer.eval()
        return self

    def _accepts(self, name: str) -> bool:
        return self._accepts_any_kwarg or name in self._accepted_kwargs

    def _iteration_kwargs(self, captured: Mapping[str, Any]) -> dict[str, Any]:
        """Build a fresh, cache-safe kwarg mapping for one solver iteration."""

        kwargs = dict(captured)

        # A DEQ iteration is not an autoregressive time step. Reusing or
        # extending a KV cache here changes f between iterations and defeats
        # both the fixed-point equation and the bounded-memory guarantee.
        for name in ("past_key_value", "past_key_values"):
            if name in kwargs or name in self._accepted_kwargs:
                kwargs[name] = None
        for name in ("use_cache", "output_attentions", "output_hidden_states"):
            if name in kwargs or name in self._accepted_kwargs:
                kwargs[name] = False

        shared = kwargs.get("shared_kv_states")
        if isinstance(shared, Mapping):
            kwargs["shared_kv_states"] = dict(shared)

        if not self._accepts_any_kwarg:
            kwargs = {
                name: value
                for name, value in kwargs.items()
                if name in self._accepted_kwargs
            }
        return kwargs

    def _transition(
        self,
        z: Tensor,
        x: Tensor,
        layer_args: Sequence[Any],
        captured_kwargs: Mapping[str, Any],
    ) -> Tensor:
        layer_input = z + self.residual_scale * x
        output = self.decoder_layer(
            layer_input,
            *layer_args,
            **self._iteration_kwargs(captured_kwargs),
        )
        self._returns_tuple = isinstance(output, tuple)
        hidden = extract_hidden_states(output)
        if hidden.shape != z.shape:
            raise DecoderLayerContractError(
                "decoder layer changed hidden-state shape inside the DEQ loop: "
                f"expected {tuple(z.shape)}, received {tuple(hidden.shape)}"
            )
        if self.contractive_delta_scale is not None:
            # Remove the decoder's identity residual before forming the DEQ map.
            # The remaining update branch is scaled and anchored at the explicit
            # lower-stack state x; this is the safe initialization used before
            # any DEQ-specific fine-tuning.
            hidden = x + self.contractive_delta_scale * (hidden - layer_input)
        return hidden

    def _solve_forward(
        self,
        x: Tensor,
        layer_args: Sequence[Any],
        captured_kwargs: Mapping[str, Any],
    ) -> Tensor:
        lipschitz_bound = self.effective_lipschitz_upper_bound
        if (
            self.contractive_delta_scale is not None
            and lipschitz_bound >= self.config.spectral_margin
            and self.config.fail_on_noncontractive
        ):
            raise ContractivityError(
                "certified Gemma DEQ Lipschitz upper bound exceeds the safety "
                f"margin ({lipschitz_bound:.4f} >= "
                f"{self.config.spectral_margin:.4f})"
            )

        def transition(state: Tensor) -> Tensor:
            return self._transition(state, x, layer_args, captured_kwargs)

        z0 = torch.zeros_like(x)
        z_star, iterations, residual, converged = _broyden_inverse(
            lambda state: transition(state) - state,
            z0,
            tolerance=self.config.tolerance,
            max_iter=self.config.max_iter,
            history=self.config.history,
        )
        used_fallback = False
        if not converged and self.config.fallback_steps:
            used_fallback = True
            if not torch.isfinite(z_star).all():
                z_star = z0
            z_star = _damped_fixed_point(
                transition,
                z_star,
                self.config.fallback_damping,
                self.config.fallback_steps,
            )
            residual = normalized_residual(transition(z_star) - z_star)
            converged = residual <= self.config.tolerance

        self.last_info = SolverInfo(
            converged=converged,
            iterations=iterations,
            residual=residual,
            # This field is an upper bound, not an empirical spectral-norm
            # estimate.  Infinity explicitly denotes an uncertified layer.
            spectral_norm=lipschitz_bound,
            used_fallback=used_fallback,
        )
        if not converged and self.config.fail_on_noncontractive:
            raise ContractivityError(
                "Gemma DEQ transition did not converge within the configured bounded solve "
                f"(residual={residual:.4e}, iterations={iterations}, "
                f"fallback_steps={self.config.fallback_steps})"
            )
        return z_star

    def _ift_hook(
        self,
        grad_output: Tensor,
        z_in: Tensor,
        x: Tensor,
        layer_args: Sequence[Any],
        captured_kwargs: Mapping[str, Any],
    ) -> Tensor:
        """Solve ``v = grad + J_f(z*)^T v`` without materialising a Jacobian."""

        v = grad_output
        residual = float("inf")
        converged = False
        iterations = 0
        for iterations in range(1, self.config.max_iter + 1):
            with torch.enable_grad():
                probe = self._transition(z_in, x, layer_args, captured_kwargs)
            (jtv,) = torch.autograd.grad(
                probe,
                z_in,
                grad_outputs=v,
                retain_graph=False,
                create_graph=False,
            )
            next_v = grad_output + jtv
            residual = normalized_residual((next_v - v).detach())
            if not torch.isfinite(next_v).all():
                break
            v = next_v
            if residual <= self.config.tolerance:
                converged = True
                break
        self.last_backward_info = BackwardSolverInfo(converged, iterations, residual)
        if not converged and self.config.fail_on_noncontractive:
            raise ContractivityError(
                "Gemma DEQ implicit backward solve did not converge within the "
                f"configured bound (residual={residual:.4e}, "
                f"iterations={iterations})"
            )
        return v

    def forward(
        self, hidden_states: Tensor, *layer_args: Any, **layer_kwargs: Any
    ) -> Any:
        if not isinstance(hidden_states, Tensor) or hidden_states.ndim < 2:
            raise DecoderLayerContractError(
                "hidden_states must be a Tensor with at least 2 dimensions"
            )
        captured_args = tuple(layer_args)
        captured_kwargs = dict(layer_kwargs)

        # The entire nonlinear root search is detached. Its activation memory
        # is bounded by solver history rather than iteration count/depth.
        with torch.no_grad():
            z_star = self._solve_forward(hidden_states, captured_args, captured_kwargs)

        requires_ift = torch.is_grad_enabled() and (
            hidden_states.requires_grad
            or any(p.requires_grad for p in self.decoder_layer.parameters())
        )
        if requires_ift:
            z_in = z_star.detach().requires_grad_(True)
            z_out = self._transition(
                z_in, hidden_states, captured_args, captured_kwargs
            )
            z_out.register_hook(
                lambda grad: self._ift_hook(
                    grad, z_in, hidden_states, captured_args, captured_kwargs
                )
            )
            result = z_out
        else:
            result = z_star.detach()

        return (result,) if self._returns_tuple else result


class LocalGemmaFeatureBackbone(nn.Module):
    """Expose one fixed-size, cache-free latent per local Gemma request.

    CTS preallocates one state tensor per arena node. Passing the full
    ``[batch, sequence, hidden]`` activation would therefore multiply its VRAM
    by conversation length. Attention-weighted pooling here keeps the CTS root
    ``[batch, hidden]`` and its arena allocation independent of token count.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

    def train(self, mode: bool = True) -> LocalGemmaFeatureBackbone:
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        attention_mask = kwargs.get("attention_mask")
        input_ids = args[0] if args else kwargs.get("input_ids")
        embedding_getter = getattr(self.model, "get_input_embeddings", None)
        if (
            isinstance(input_ids, Tensor)
            and input_ids.ndim == 2
            and not torch.is_floating_point(input_ids)
            and callable(embedding_getter)
        ):
            embeddings = embedding_getter()(input_ids)
            if not isinstance(embeddings, Tensor) or embeddings.ndim != 3:
                raise DecoderLayerContractError(
                    "local Gemma token embedding must have [batch, sequence, hidden] shape"
                )
            if (
                isinstance(attention_mask, Tensor)
                and tuple(attention_mask.shape) == tuple(embeddings.shape[:2])
            ):
                weights = attention_mask.to(
                    device=embeddings.device, dtype=embeddings.dtype
                ).unsqueeze(-1)
                return (embeddings * weights).sum(dim=1) / weights.sum(
                    dim=1
                ).clamp_min(1)
            return embeddings.mean(dim=1)
        kwargs["use_cache"] = False
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        output = self.model(*args, **kwargs)
        hidden = extract_hidden_states(output)
        if hidden.ndim < 2:
            raise DecoderLayerContractError(
                "local Gemma hidden state must preserve batch and feature axes"
            )
        if hidden.ndim == 2:
            return hidden
        if (
            hidden.ndim == 3
            and isinstance(attention_mask, Tensor)
            and tuple(attention_mask.shape) == tuple(hidden.shape[:2])
        ):
            weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype)
            weights = weights.unsqueeze(-1)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return hidden.flatten(1, -2).mean(dim=1)


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder ``ModuleList`` on common Gemma/HF model layouts."""

    for path in (
        "model.language_model.layers",
        "model.model.layers",
        "model.layers",
        "transformer.h",
    ):
        node: Any = model
        try:
            for component in path.split("."):
                node = getattr(node, component)
        except AttributeError:
            continue
        if isinstance(node, nn.ModuleList) and node:
            return node
    raise AttributeError(
        "could not locate decoder layers on supported Gemma/HF layouts"
    )


def inject_gemma_deq_layer(
    model: nn.Module,
    *,
    layer_index: int = -1,
    config: DEQConfig | None = None,
    residual_scale: float = 1.0,
    contractive_delta_scale: float | None = None,
    certified_delta_lipschitz_bound: float | None = None,
    freeze_other_parameters: bool = True,
) -> GemmaDEQBackboneAdapter:
    """Replace one local model decoder block with the DEQ adapter."""

    layers = find_decoder_layers(model)
    resolved_index = layer_index if layer_index >= 0 else len(layers) + layer_index
    if not 0 <= resolved_index < len(layers):
        raise IndexError(
            f"layer_index {layer_index} is invalid for {len(layers)} decoder layers"
        )
    original = layers[resolved_index]
    if freeze_other_parameters:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in original.parameters():
            parameter.requires_grad_(True)
    adapter = GemmaDEQBackboneAdapter(
        original,
        config=config,
        residual_scale=residual_scale,
        contractive_delta_scale=contractive_delta_scale,
        certified_delta_lipschitz_bound=certified_delta_lipschitz_bound,
    )
    layers[resolved_index] = adapter
    return adapter


_WEIGHT_PATTERNS = ("model*.safetensors", "pytorch_model*.bin")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.model",
)


def _has_any(root: Path, patterns: Sequence[str]) -> bool:
    return any(any(root.glob(pattern)) for pattern in patterns)


def verify_local_gemma_path(
    model_path: str | Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> Path:
    """Resolve and validate a complete local Gemma artifact directory.

    A Hub ID such as ``google/gemma-*`` is only accepted if it actually
    resolves to a directory on disk.  URL-like sources are rejected before
    any optional loading library is imported.
    """

    raw = str(model_path)
    if "://" in raw:
        raise OfflineModelPolicyError(
            f"remote model URI is forbidden in air-gapped mode: {raw!r}"
        )
    try:
        root = Path(model_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise OfflineModelPolicyError(
            f"model source must be an existing local directory, not a Hub ID: {raw!r}"
        ) from exc
    if not root.is_dir():
        raise OfflineModelPolicyError(f"model source is not a local directory: {root}")

    config_path = root / "config.json"
    if not config_path.is_file():
        raise OfflineModelPolicyError(f"local model is missing config.json: {root}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflineModelPolicyError(
            f"local config.json is unreadable or invalid: {config_path}"
        ) from exc
    model_type = str(config.get("model_type", "")).lower()
    if not model_type.startswith("gemma"):
        raise OfflineModelPolicyError(
            f"local config model_type must be Gemma-compatible; received {model_type or '<missing>'!r}"
        )
    if not _has_any(root, _WEIGHT_PATTERNS):
        raise OfflineModelPolicyError(
            f"local Gemma directory contains no model weight artifact: {root}"
        )
    if not any((root / name).is_file() for name in _TOKENIZER_FILES):
        raise OfflineModelPolicyError(
            f"local Gemma directory contains no tokenizer artifact: {root}"
        )

    for relative_name, expected in (expected_sha256 or {}).items():
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise OfflineModelPolicyError(
                f"checksum entry escapes model directory: {relative_name!r}"
            )
        try:
            artifact = (root / relative).resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise OfflineModelPolicyError(
                f"checksum artifact is missing: {relative_name!r}"
            ) from exc
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise OfflineModelPolicyError(
                f"checksum artifact is outside model directory: {relative_name!r}"
            )
        expected_normalized = expected.lower()
        if len(expected_normalized) != 64 or any(
            c not in "0123456789abcdef" for c in expected_normalized
        ):
            raise OfflineModelPolicyError(
                f"invalid SHA-256 value for {relative_name!r}"
            )
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_normalized:
            raise OfflineModelPolicyError(f"SHA-256 mismatch for {relative_name!r}")
    return root


def _strict_loader_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(kwargs or {})
    if result.get("local_files_only") is False:
        raise OfflineModelPolicyError(
            "local_files_only=False violates the offline loader policy"
        )
    if result.get("trust_remote_code") is True:
        raise OfflineModelPolicyError(
            "trust_remote_code=True violates the offline loader policy"
        )
    if result.get("force_download") is True:
        raise OfflineModelPolicyError(
            "force_download=True violates the offline loader policy"
        )
    result["local_files_only"] = True
    result["trust_remote_code"] = False
    result["force_download"] = False
    return result


def load_local_gemma(
    model_path: str | Path,
    *,
    model_class: Any | None = None,
    tokenizer_class: Any | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    vram_limit_gib: float = MAX_VRAM_GIB,
) -> tuple[Any, Any]:
    """Load ``(model, tokenizer)`` with an enforceable local-only policy.

    ``model_class`` and ``tokenizer_class`` need only expose a
    ``from_pretrained`` method, which keeps tests independent of
    ``transformers``. When omitted, the corresponding Auto class is imported
    lazily after local artifacts pass verification.
    """

    if not 0.0 < float(vram_limit_gib) <= MAX_VRAM_GIB:
        raise ValueError(f"vram_limit_gib must be in (0, {MAX_VRAM_GIB}]")
    root = verify_local_gemma_path(model_path, expected_sha256=expected_sha256)
    if model_class is None or tokenizer_class is None:
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required only for loading a real local Gemma model; "
                "install it into the offline environment or supply loader classes"
            ) from exc
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        architectures = set(config.get("architectures") or ())
        multimodal = bool(config.get("vision_config") or config.get("audio_config"))
        conditional = any("ConditionalGeneration" in name for name in architectures)
        model_class = model_class or (
            AutoModelForImageTextToText
            if multimodal or conditional
            else AutoModelForCausalLM
        )
        tokenizer_class = tokenizer_class or AutoTokenizer

    model_options = _strict_loader_kwargs(model_kwargs)
    tokenizer_options = _strict_loader_kwargs(tokenizer_kwargs)
    if "dtype" not in model_options and "torch_dtype" not in model_options:
        model_options["dtype"] = torch.bfloat16
    model_options.setdefault("low_cpu_mem_usage", True)
    if torch.cuda.is_available():
        limit_bytes = int(float(vram_limit_gib) * 1024**3)
        weight_files = {
            artifact.resolve()
            for pattern in _WEIGHT_PATTERNS
            for artifact in root.glob(pattern)
        }
        artifact_bytes = sum(artifact.stat().st_size for artifact in weight_files)
        allocated = torch.cuda.memory_allocated()
        free_bytes, _ = torch.cuda.mem_get_info()
        if allocated + artifact_bytes > limit_bytes or artifact_bytes > free_bytes:
            raise ResourceBudgetExceeded(
                "local Gemma artifact cannot be admitted inside the CUDA budget: "
                f"allocated={allocated}, artifact_bytes={artifact_bytes}, "
                f"free={free_bytes}, limit={limit_bytes}"
            )
        model_options.setdefault("device_map", "cuda")
        model_options.setdefault("max_memory", {0: limit_bytes})
    tokenizer = tokenizer_class.from_pretrained(str(root), **tokenizer_options)
    try:
        model = model_class.from_pretrained(str(root), **model_options)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise ResourceBudgetExceeded(
            "CUDA allocator rejected the local Gemma load within the hard budget"
        ) from exc
    if torch.cuda.is_available():
        observed = torch.cuda.memory_allocated()
        if observed > int(float(vram_limit_gib) * 1024**3):
            del model
            torch.cuda.empty_cache()
            raise ResourceBudgetExceeded(
                f"local Gemma load crossed the VRAM ceiling: observed={observed}"
            )
    return model, tokenizer


__all__ = [
    "BackwardSolverInfo",
    "DecoderLayerContractError",
    "GemmaDEQBackboneAdapter",
    "LocalGemmaFeatureBackbone",
    "OfflineModelPolicyError",
    "extract_hidden_states",
    "find_decoder_layers",
    "inject_gemma_deq_layer",
    "load_local_gemma",
    "verify_local_gemma_path",
]
