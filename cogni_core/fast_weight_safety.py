"""Fail-closed admission certificates for trained session Fast Weights.

This module is deliberately independent from the runtime integration layer.  It
contains only bounded evidence records, a ten-sample residual-quality gate, and
factor-wise EMA state whose low-rank update is projected before it can be
published.  A programmer's self-reported quality is not accepted by any API in
this module; semantic quality must arrive as :class:`ExternalVerifierEvidence`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import BadZipFile, ZipFile, is_zipfile

import torch
from torch import Tensor, nn

from .fast_weights import FastWeightProgrammer, ResidualBottleneckAdapter


_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_EMA_CHECKPOINT_SCHEMA = "cogni-factor-wise-ema-v1"
_FAST_WEIGHT_CHECKPOINT_SCHEMA = "cogni-fast-weight-programmer-v1"
_MAX_FAST_WEIGHT_CHECKPOINT_BYTES = 128 * 1024 * 1024
_VERIFIED_PROGRAMMER_TOKEN = object()
_VERIFIED_ADMISSION_TOKEN = object()
_SUPPORTED_FLOAT_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _finite_hex(value: float | None) -> str | None:
    if value is None:
        return None
    if not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0.0 else "-inf"
    return value.hex()


def _tensor_digest(value: Tensor) -> str:
    tensor = value.detach().to("cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = sha256()
    digest.update(header)
    digest.update(raw)
    return digest.hexdigest()


def _state_digest(state: Mapping[str, Tensor]) -> str:
    """Digest tensor values, shapes, and dtypes independently of serialization."""

    return _canonical_digest(
        {name: _tensor_digest(value) for name, value in sorted(state.items())}
    )


@dataclass(frozen=True, slots=True)
class TrainedProgrammerEvidence:
    """Immutable provenance proving that an FWP came from a training run."""

    checkpoint_sha256: str
    training_run_sha256: str
    training_steps: int
    training_samples: int

    def __post_init__(self) -> None:
        _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _validate_sha256(self.training_run_sha256, "training_run_sha256")
        if (
            not isinstance(self.training_steps, int)
            or isinstance(self.training_steps, bool)
            or self.training_steps <= 0
        ):
            raise ValueError("training_steps must be a positive integer")
        if (
            not isinstance(self.training_samples, int)
            or isinstance(self.training_samples, bool)
            or self.training_samples <= 0
        ):
            raise ValueError("training_samples must be a positive integer")

    @property
    def digest(self) -> str:
        return _canonical_digest(
            {
                "checkpoint_sha256": self.checkpoint_sha256,
                "training_run_sha256": self.training_run_sha256,
                "training_samples": self.training_samples,
                "training_steps": self.training_steps,
            }
        )


@dataclass(frozen=True, slots=True)
class ExternalVerifierEvidence:
    """Held-out evidence produced by a checkpoint other than the programmer.

    There is intentionally no ``programmer_quality`` field.  The only quality
    accepted by the AQ gate is this verifier-owned, held-out measurement.
    """

    verifier_checkpoint_sha256: str
    evaluated_programmer_sha256: str
    heldout_dataset_sha256: str
    verified_quality: float
    sample_count: int
    passed: bool

    def __post_init__(self) -> None:
        _validate_sha256(self.verifier_checkpoint_sha256, "verifier_checkpoint_sha256")
        _validate_sha256(
            self.evaluated_programmer_sha256, "evaluated_programmer_sha256"
        )
        _validate_sha256(self.heldout_dataset_sha256, "heldout_dataset_sha256")
        if self.verifier_checkpoint_sha256 == self.evaluated_programmer_sha256:
            raise ValueError("the programmer cannot verify its own quality")
        quality = float(self.verified_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("verified_quality must be finite and in [0, 1]")
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or self.sample_count <= 0
        ):
            raise ValueError("sample_count must be a positive integer")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")

    @property
    def digest(self) -> str:
        return _canonical_digest(
            {
                "evaluated_programmer_sha256": self.evaluated_programmer_sha256,
                "heldout_dataset_sha256": self.heldout_dataset_sha256,
                "passed": self.passed,
                "sample_count": self.sample_count,
                "verified_quality": float(self.verified_quality).hex(),
                "verifier_checkpoint_sha256": self.verifier_checkpoint_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class VerifiedFastWeightProgrammer:
    """A frozen programmer/adapter pair loaded through the strict local gate.

    Construction is intentionally unavailable to ordinary callers.  The trust
    root is the caller-pinned artifact SHA-256; the derived checkpoint digest
    binds the evidence to the exact tensor values rather than to pickle bytes.
    """

    programmer: FastWeightProgrammer
    adapter: ResidualBottleneckAdapter
    programmer_evidence: TrainedProgrammerEvidence
    verifier_evidence: ExternalVerifierEvidence
    artifact_sha256: str
    provenance_kind: str
    _verified: object

    def __post_init__(self) -> None:
        if self._verified is not _VERIFIED_PROGRAMMER_TOKEN:
            raise RuntimeError(
                "VerifiedFastWeightProgrammer must come from the checkpoint loader"
            )
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if self.provenance_kind not in {"training_run", "test_fixture"}:
            raise ValueError("Fast Weight provenance kind is unsupported")
        if (
            self.verifier_evidence.evaluated_programmer_sha256
            != self.programmer_evidence.checkpoint_sha256
        ):
            raise RuntimeError("verifier evidence is not bound to loaded tensors")


def fast_weight_checkpoint_architecture(
    programmer: FastWeightProgrammer,
    adapter: ResidualBottleneckAdapter,
) -> dict[str, object]:
    return {
        "adapter": {
            "bottleneck_dim": adapter.bottleneck_dim,
            "core_operator_norm_budget": adapter.core_operator_norm_budget.hex(),
            "latent_dim": adapter.latent_dim,
            "residual_scale": adapter.residual_scale.hex(),
            "spectral_margin": adapter.spectral_margin.hex(),
        },
        "programmer": {
            "internal_dim": programmer.internal_dim,
            "max_operator_norm": programmer.max_operator_norm.hex(),
            "rank": programmer.rank,
            "source_dim": programmer.source_dim,
            "target_dim": programmer.target_dim,
        },
    }


def fast_weight_checkpoint_state(
    programmer: FastWeightProgrammer,
    adapter: ResidualBottleneckAdapter,
) -> dict[str, Tensor]:
    """Return the exact CPU state schema used by the verified loader.

    This helper exists so offline training/export code can create the same
    deterministic state contract.  It does not create evidence or authorize a
    runtime by itself.
    """

    state: dict[str, Tensor] = {}
    for prefix, module in (("programmer", programmer), ("adapter", adapter)):
        for name, value in module.state_dict().items():
            state[f"{prefix}.{name}"] = value.detach().to("cpu").contiguous().clone()
    return state


def _validate_checkpoint_tensor(
    name: str,
    value: object,
    expected: Tensor,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"Fast Weight checkpoint entry is not a tensor: {name}")
    if value.device.type != "cpu":
        raise ValueError("Fast Weight checkpoint tensors must reside on CPU")
    if value.shape != expected.shape or value.dtype != expected.dtype:
        raise ValueError(f"Fast Weight checkpoint tensor contract changed: {name}")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"Fast Weight checkpoint tensor is unsafe: {name}")
    return value.detach().contiguous()


def load_verified_fast_weight_checkpoint(
    programmer: FastWeightProgrammer,
    adapter: ResidualBottleneckAdapter,
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    allow_test_fixture: bool = False,
) -> VerifiedFastWeightProgrammer:
    """Load one local, hash-pinned, trained System-1.5 checkpoint atomically.

    There is no network or implicit default path.  A missing artifact leaves
    the product feature disabled; this function never manufactures evidence.
    ``test_fixture`` provenance is rejected unless a test explicitly opts in.
    """

    if not isinstance(programmer, FastWeightProgrammer):
        raise TypeError("programmer must be FastWeightProgrammer")
    if not isinstance(adapter, ResidualBottleneckAdapter):
        raise TypeError("adapter must be ResidualBottleneckAdapter")
    _validate_sha256(expected_sha256, "expected_sha256")
    if not isinstance(allow_test_fixture, bool):
        raise TypeError("allow_test_fixture must be bool")
    source = Path(checkpoint).expanduser()
    if source.is_symlink():
        raise RuntimeError("Fast Weight checkpoint must not be a symbolic link")
    source = source.resolve(strict=True)
    if not source.is_file():
        raise RuntimeError("Fast Weight checkpoint must be one regular local file")
    size = source.stat().st_size
    if not 0 < size <= _MAX_FAST_WEIGHT_CHECKPOINT_BYTES:
        raise RuntimeError("Fast Weight checkpoint exceeds its bounded size")
    artifact = source.read_bytes()
    artifact_sha256 = sha256(artifact).hexdigest()
    if artifact_sha256 != expected_sha256:
        raise RuntimeError("Fast Weight checkpoint SHA-256 verification failed")
    if not is_zipfile(BytesIO(artifact)):
        raise RuntimeError("Fast Weight checkpoint must use bounded ZIP serialization")
    try:
        with ZipFile(BytesIO(artifact)) as archive:
            expanded = sum(info.file_size for info in archive.infolist())
    except BadZipFile as exc:
        raise RuntimeError("Fast Weight checkpoint ZIP is invalid") from exc
    if expanded > _MAX_FAST_WEIGHT_CHECKPOINT_BYTES:
        raise RuntimeError("Fast Weight checkpoint expanded size exceeds its bound")

    # Deserialize the exact bytes that passed the digest check; reopening the
    # path here would create a check/use race even in an offline environment.
    payload = torch.load(BytesIO(artifact), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {
        "architecture",
        "provenance",
        "schema",
        "state",
        "verifier",
    }:
        raise ValueError("Fast Weight checkpoint top-level schema is invalid")
    if payload["schema"] != _FAST_WEIGHT_CHECKPOINT_SCHEMA:
        raise ValueError("Fast Weight checkpoint schema version is unsupported")
    if payload["architecture"] != fast_weight_checkpoint_architecture(
        programmer, adapter
    ):
        raise ValueError("Fast Weight checkpoint architecture does not match runtime")
    state = payload["state"]
    provenance = payload["provenance"]
    verifier = payload["verifier"]
    if not isinstance(state, Mapping):
        raise TypeError("Fast Weight checkpoint state must be a mapping")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "kind",
        "training_run_sha256",
        "training_samples",
        "training_steps",
    }:
        raise ValueError("Fast Weight training provenance schema is invalid")
    if not isinstance(verifier, Mapping) or set(verifier) != {
        "heldout_dataset_sha256",
        "passed",
        "sample_count",
        "verified_quality",
        "verifier_checkpoint_sha256",
    }:
        raise ValueError("Fast Weight verifier evidence schema is invalid")
    provenance_kind = provenance["kind"]
    if provenance_kind not in {"training_run", "test_fixture"}:
        raise ValueError("Fast Weight checkpoint is not a recognized training run")
    if provenance_kind == "test_fixture" and not allow_test_fixture:
        raise RuntimeError("test-only Fast Weight evidence is forbidden in production")

    expected_state = fast_weight_checkpoint_state(programmer, adapter)
    if set(state) != set(expected_state):
        raise ValueError("Fast Weight checkpoint state keys are not exact")
    checked = {
        name: _validate_checkpoint_tensor(name, state[name], expected)
        for name, expected in expected_state.items()
    }
    checkpoint_state_sha256 = _state_digest(checked)
    programmer_evidence = TrainedProgrammerEvidence(
        checkpoint_sha256=checkpoint_state_sha256,
        training_run_sha256=str(provenance["training_run_sha256"]),
        training_steps=provenance["training_steps"],
        training_samples=provenance["training_samples"],
    )
    verifier_evidence = ExternalVerifierEvidence(
        verifier_checkpoint_sha256=str(verifier["verifier_checkpoint_sha256"]),
        evaluated_programmer_sha256=checkpoint_state_sha256,
        heldout_dataset_sha256=str(verifier["heldout_dataset_sha256"]),
        verified_quality=float(verifier["verified_quality"]),
        sample_count=verifier["sample_count"],
        passed=verifier["passed"],
    )

    programmer_state = {
        name.removeprefix("programmer."): value
        for name, value in checked.items()
        if name.startswith("programmer.")
    }
    adapter_state = {
        name.removeprefix("adapter."): value
        for name, value in checked.items()
        if name.startswith("adapter.")
    }
    original_programmer = {
        name: value.detach().clone() for name, value in programmer.state_dict().items()
    }
    original_adapter = {
        name: value.detach().clone() for name, value in adapter.state_dict().items()
    }
    try:
        programmer.load_state_dict(programmer_state, strict=True)
        adapter.load_state_dict(adapter_state, strict=True)
        core_norm = float(
            torch.linalg.matrix_norm(adapter.core.weight.detach().float(), ord=2)
        )
        if (
            not math.isfinite(core_norm)
            or core_norm > adapter.core_operator_norm_budget
        ):
            raise RuntimeError("trained adapter core violates its spectral certificate")
    except BaseException:
        programmer.load_state_dict(original_programmer, strict=True)
        adapter.load_state_dict(original_adapter, strict=True)
        raise
    programmer.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    return VerifiedFastWeightProgrammer(
        programmer=programmer,
        adapter=adapter,
        programmer_evidence=programmer_evidence,
        verifier_evidence=verifier_evidence,
        artifact_sha256=artifact_sha256,
        provenance_kind=str(provenance_kind),
        _verified=_VERIFIED_PROGRAMMER_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class AQCertificate:
    """Bounded, auditable result of one ten-residual AQ decision."""

    accepted: bool
    reason: str
    digest: str
    residuals: tuple[float, ...]
    log_decays: tuple[float, ...]
    observed_count: int
    trace_was_one_dimensional: bool
    mean_log_decay: float | None
    log_decay_variance: float | None
    terminal_residual: float | None
    monotonic_fraction: float | None
    solver_converged: bool
    solver_used_fallback: bool
    trained_programmer: bool
    external_verifier_passed: bool
    programmer_evidence_digest: str | None
    verifier_evidence_digest: str | None

    def __post_init__(self) -> None:
        _validate_sha256(self.digest, "digest")
        if len(self.residuals) > ResidualDecayAQGate.WINDOW_SIZE:
            raise ValueError("AQCertificate residual storage exceeds its hard bound")
        if len(self.log_decays) > ResidualDecayAQGate.WINDOW_SIZE - 1:
            raise ValueError("AQCertificate decay storage exceeds its hard bound")
        for value, label in (
            (self.programmer_evidence_digest, "programmer_evidence_digest"),
            (self.verifier_evidence_digest, "verifier_evidence_digest"),
        ):
            if value is not None:
                _validate_sha256(value, label)


class ResidualDecayAQGate:
    """Certify stable convergence from exactly the latest ten residuals."""

    WINDOW_SIZE = 10

    def __init__(
        self,
        *,
        max_terminal_residual: float = 5.0e-3,
        max_log_decay_variance: float = 5.0e-2,
        min_mean_log_decay: float = 1.0e-3,
        min_monotonic_fraction: float = 8.0 / 9.0,
        minimum_verified_quality: float = 0.8,
        flat_tolerance: float = 1.0e-12,
        log_epsilon: float = 1.0e-12,
    ) -> None:
        positive = {
            "max_terminal_residual": max_terminal_residual,
            "max_log_decay_variance": max_log_decay_variance,
            "min_mean_log_decay": min_mean_log_decay,
            "flat_tolerance": flat_tolerance,
            "log_epsilon": log_epsilon,
        }
        for name, value in positive.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < min_monotonic_fraction <= 1.0:
            raise ValueError("min_monotonic_fraction must lie in (0, 1]")
        if not 0.0 <= minimum_verified_quality <= 1.0:
            raise ValueError("minimum_verified_quality must lie in [0, 1]")
        self.max_terminal_residual = float(max_terminal_residual)
        self.max_log_decay_variance = float(max_log_decay_variance)
        self.min_mean_log_decay = float(min_mean_log_decay)
        self.min_monotonic_fraction = float(min_monotonic_fraction)
        self.minimum_verified_quality = float(minimum_verified_quality)
        self.flat_tolerance = float(flat_tolerance)
        self.log_epsilon = float(log_epsilon)

    def _certificate(
        self,
        *,
        accepted: bool,
        reason: str,
        residuals: tuple[float, ...],
        observed_count: int,
        trace_was_one_dimensional: bool,
        log_decays: tuple[float, ...] = (),
        mean_log_decay: float | None = None,
        log_decay_variance: float | None = None,
        terminal_residual: float | None = None,
        monotonic_fraction: float | None = None,
        solver_converged: bool,
        solver_used_fallback: bool,
        programmer: TrainedProgrammerEvidence | None,
        verifier: ExternalVerifierEvidence | None,
    ) -> AQCertificate:
        bounded_residuals = residuals[-self.WINDOW_SIZE :]
        bounded_decays = log_decays[-(self.WINDOW_SIZE - 1) :]
        payload = {
            "accepted": accepted,
            "external_verifier_digest": None if verifier is None else verifier.digest,
            "log_decay_variance": _finite_hex(log_decay_variance),
            "log_decays": [_finite_hex(value) for value in bounded_decays],
            "mean_log_decay": _finite_hex(mean_log_decay),
            "monotonic_fraction": _finite_hex(monotonic_fraction),
            "observed_count": observed_count,
            "programmer_digest": None if programmer is None else programmer.digest,
            "reason": reason,
            "residuals": [_finite_hex(value) for value in bounded_residuals],
            "solver_converged": solver_converged,
            "solver_used_fallback": solver_used_fallback,
            "terminal_residual": _finite_hex(terminal_residual),
            "trace_was_one_dimensional": trace_was_one_dimensional,
        }
        return AQCertificate(
            accepted=accepted,
            reason=reason,
            digest=_canonical_digest(payload),
            residuals=bounded_residuals,
            log_decays=bounded_decays,
            observed_count=observed_count,
            trace_was_one_dimensional=trace_was_one_dimensional,
            mean_log_decay=mean_log_decay,
            log_decay_variance=log_decay_variance,
            terminal_residual=terminal_residual,
            monotonic_fraction=monotonic_fraction,
            solver_converged=solver_converged,
            solver_used_fallback=solver_used_fallback,
            trained_programmer=programmer is not None,
            external_verifier_passed=bool(verifier is not None and verifier.passed),
            programmer_evidence_digest=(
                None if programmer is None else programmer.digest
            ),
            verifier_evidence_digest=None if verifier is None else verifier.digest,
        )

    @staticmethod
    def _bounded_trace(
        residuals: Tensor | Sequence[float],
    ) -> tuple[tuple[float, ...], int, bool]:
        tensor = (
            residuals.detach()
            if isinstance(residuals, Tensor)
            else torch.as_tensor(residuals, dtype=torch.float64)
        )
        if tensor.ndim != 1:
            return (), int(tensor.numel()), False
        count = int(tensor.numel())
        bounded = tensor[-ResidualDecayAQGate.WINDOW_SIZE :].to(
            device="cpu", dtype=torch.float64
        )
        return tuple(float(value) for value in bounded), count, True

    def evaluate(
        self,
        residuals: Tensor | Sequence[float],
        *,
        solver_converged: bool,
        solver_used_fallback: bool,
        programmer: TrainedProgrammerEvidence | None,
        verifier: ExternalVerifierEvidence | None,
    ) -> AQCertificate:
        """Return a deterministic certificate without retaining trace state."""

        if not isinstance(solver_converged, bool):
            raise TypeError("solver_converged must be bool")
        if not isinstance(solver_used_fallback, bool):
            raise TypeError("solver_used_fallback must be bool")
        trace, observed_count, one_dimensional = self._bounded_trace(residuals)
        common = {
            "residuals": trace,
            "observed_count": observed_count,
            "trace_was_one_dimensional": one_dimensional,
            "solver_converged": solver_converged,
            "solver_used_fallback": solver_used_fallback,
            "programmer": programmer,
            "verifier": verifier,
        }
        if programmer is None:
            return self._certificate(
                accepted=False,
                reason="untrained programmer checkpoint",
                **common,
            )
        if verifier is None:
            return self._certificate(
                accepted=False,
                reason="external verifier evidence is missing",
                **common,
            )
        if verifier.evaluated_programmer_sha256 != programmer.checkpoint_sha256:
            return self._certificate(
                accepted=False,
                reason="verifier evidence targets a different programmer",
                **common,
            )
        if (
            not verifier.passed
            or verifier.verified_quality < self.minimum_verified_quality
        ):
            return self._certificate(
                accepted=False,
                reason="external verifier quality gate failed",
                **common,
            )
        if not solver_converged:
            return self._certificate(
                accepted=False,
                reason="solver did not converge",
                **common,
            )
        if solver_used_fallback:
            return self._certificate(
                accepted=False,
                reason="solver fallback was used",
                **common,
            )
        if not one_dimensional or observed_count != self.WINDOW_SIZE:
            return self._certificate(
                accepted=False,
                reason="residual trace must contain exactly 10 values",
                **common,
            )
        if any(not math.isfinite(value) for value in trace):
            return self._certificate(
                accepted=False,
                reason="residual trace contains non-finite values",
                **common,
            )
        if any(value < 0.0 for value in trace):
            return self._certificate(
                accepted=False,
                reason="residual trace contains negative values",
                **common,
            )

        trace_tensor = torch.tensor(trace, dtype=torch.float64)
        safe = trace_tensor.clamp_min(self.log_epsilon)
        decay_tensor = safe[:-1].log() - safe[1:].log()
        log_decays = tuple(float(value) for value in decay_tensor)
        mean_decay = float(decay_tensor.mean())
        variance = float(decay_tensor.var(unbiased=False))
        terminal = trace[-1]
        monotonic = float((trace_tensor[1:] < trace_tensor[:-1]).float().mean())
        metrics = {
            "log_decays": log_decays,
            "mean_log_decay": mean_decay,
            "log_decay_variance": variance,
            "terminal_residual": terminal,
            "monotonic_fraction": monotonic,
        }
        if max(trace) - min(trace) <= self.flat_tolerance:
            reason = "residual trace is flat"
        elif trace[-1] >= trace[0] or mean_decay <= 0.0:
            reason = "residual trace is increasing"
        elif terminal > self.max_terminal_residual:
            reason = "terminal residual exceeds the AQ limit"
        elif monotonic < self.min_monotonic_fraction:
            reason = "residual monotonic fraction is too low"
        elif mean_decay < self.min_mean_log_decay:
            reason = "mean log-decay is too small"
        elif variance > self.max_log_decay_variance:
            reason = "log-decay variance exceeds the AQ limit"
        else:
            reason = "accepted"
        return self._certificate(
            accepted=reason == "accepted",
            reason=reason,
            **common,
            **metrics,
        )


@dataclass(frozen=True, slots=True)
class VerifiedFastWeightAdmission:
    """Immutable authorization tying one AQ result to the loaded checkpoint."""

    programmer_checkpoint_sha256: str
    programmer_evidence_digest: str
    verifier_evidence_digest: str
    aq_certificate: AQCertificate
    digest: str
    _verified: object

    def __post_init__(self) -> None:
        if self._verified is not _VERIFIED_ADMISSION_TOKEN:
            raise RuntimeError(
                "VerifiedFastWeightAdmission must come from the authorization gate"
            )
        for value, label in (
            (self.programmer_checkpoint_sha256, "programmer_checkpoint_sha256"),
            (self.programmer_evidence_digest, "programmer_evidence_digest"),
            (self.verifier_evidence_digest, "verifier_evidence_digest"),
            (self.digest, "digest"),
        ):
            _validate_sha256(value, label)
        if not self.aq_certificate.accepted:
            raise RuntimeError("a rejected AQ certificate cannot authorize Fast Weight")

    def valid_for(self, programmer_checkpoint_sha256: str) -> bool:
        expected = _canonical_digest(
            {
                "aq_certificate_digest": self.aq_certificate.digest,
                "programmer_checkpoint_sha256": self.programmer_checkpoint_sha256,
                "programmer_evidence_digest": self.programmer_evidence_digest,
                "verifier_evidence_digest": self.verifier_evidence_digest,
            }
        )
        return (
            self.programmer_checkpoint_sha256 == programmer_checkpoint_sha256
            and self.aq_certificate.accepted
            and self.aq_certificate.programmer_evidence_digest
            == self.programmer_evidence_digest
            and self.aq_certificate.verifier_evidence_digest
            == self.verifier_evidence_digest
            and self.digest == expected
        )


def authorize_fast_weight_admission(
    verified: VerifiedFastWeightProgrammer,
    certificate: AQCertificate,
) -> VerifiedFastWeightAdmission:
    """Authorize admission only when AQ evidence matches the loaded artifact."""

    if not isinstance(verified, VerifiedFastWeightProgrammer):
        raise TypeError("verified must be VerifiedFastWeightProgrammer")
    if not isinstance(certificate, AQCertificate):
        raise TypeError("certificate must be AQCertificate")
    programmer_digest = verified.programmer_evidence.digest
    verifier_digest = verified.verifier_evidence.digest
    if not certificate.accepted:
        raise RuntimeError(f"AQ certificate rejected Fast Weight: {certificate.reason}")
    if (
        certificate.programmer_evidence_digest != programmer_digest
        or certificate.verifier_evidence_digest != verifier_digest
    ):
        raise RuntimeError("AQ certificate is not bound to the loaded Fast Weight")
    payload = {
        "aq_certificate_digest": certificate.digest,
        "programmer_checkpoint_sha256": (
            verified.programmer_evidence.checkpoint_sha256
        ),
        "programmer_evidence_digest": programmer_digest,
        "verifier_evidence_digest": verifier_digest,
    }
    return VerifiedFastWeightAdmission(
        programmer_checkpoint_sha256=verified.programmer_evidence.checkpoint_sha256,
        programmer_evidence_digest=programmer_digest,
        verifier_evidence_digest=verifier_digest,
        aq_certificate=certificate,
        digest=_canonical_digest(payload),
        _verified=_VERIFIED_ADMISSION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class FactorWiseEMACertificate:
    """Post-update norm and immutable-base evidence for factor-wise EMA."""

    accepted: bool
    reason: str
    digest: str
    update_count: int
    rank: int
    overlay_operator_norm: float
    base_operator_norm: float
    composed_upper_bound: float
    operator_norm_limit: float
    composed_norm_limit: float
    projected: bool
    base_unchanged: bool

    def __post_init__(self) -> None:
        _validate_sha256(self.digest, "digest")


class FactorWiseEMA(nn.Module):
    """Fixed-shape A/B EMA with projection and reproducible checkpoints."""

    MAX_FEATURE_DIM = 4096
    MAX_RANK = 64
    MAX_FACTOR_ELEMENTS = 1_048_576

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        decay: float = 0.99,
        max_operator_norm: float = 0.1,
        composed_norm_limit: float = 0.95,
        strict_margin: float = 1.0e-5,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        dimensions = (in_features, out_features, rank)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in dimensions
        ):
            raise ValueError("in_features, out_features, and rank must be positive")
        if in_features > self.MAX_FEATURE_DIM or out_features > self.MAX_FEATURE_DIM:
            raise ValueError("factor feature dimension exceeds the hard bound")
        if rank > min(self.MAX_RANK, in_features, out_features):
            raise ValueError("rank exceeds the hard or feature-dimension bound")
        if (in_features + out_features) * rank > self.MAX_FACTOR_ELEMENTS:
            raise ValueError("factor storage exceeds the hard element bound")
        if (
            not isinstance(dtype, torch.dtype)
            or dtype not in _SUPPORTED_FLOAT_DTYPES.values()
        ):
            raise TypeError("dtype must be a supported floating-point torch dtype")
        numeric = (decay, max_operator_norm, composed_norm_limit, strict_margin)
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("EMA limits must be finite")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must lie in [0, 1)")
        if not 0.0 < max_operator_norm < composed_norm_limit:
            raise ValueError("max_operator_norm must be below composed_norm_limit")
        if not 0.0 < composed_norm_limit <= 0.95:
            raise ValueError("composed_norm_limit must lie in (0, 0.95]")
        if not 0.0 < strict_margin < composed_norm_limit:
            raise ValueError("strict_margin must lie inside the composed limit")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.decay = float(decay)
        self.max_operator_norm = float(max_operator_norm)
        self.composed_norm_limit = float(composed_norm_limit)
        self.strict_margin = float(strict_margin)
        target_device = torch.device(device)
        self.register_buffer(
            "ema_a",
            torch.zeros((in_features, rank), dtype=dtype, device=target_device),
        )
        self.register_buffer(
            "ema_b",
            torch.zeros((out_features, rank), dtype=dtype, device=target_device),
        )
        self.register_buffer(
            "update_count_tensor",
            torch.zeros((), dtype=torch.int64, device=target_device),
        )
        self.register_buffer(
            "initialized_tensor",
            torch.zeros((), dtype=torch.bool, device=target_device),
        )

    @property
    def update_count(self) -> int:
        return int(self.update_count_tensor.detach().cpu())

    @property
    def initialized(self) -> bool:
        return bool(self.initialized_tensor.detach().cpu())

    @staticmethod
    def _operator_norm(a: Tensor, b: Tensor) -> Tensor:
        _, r_a = torch.linalg.qr(a.float(), mode="reduced")
        _, r_b = torch.linalg.qr(b.float(), mode="reduced")
        return torch.linalg.matrix_norm(r_b @ r_a.transpose(-1, -2), ord=2)

    def _validate_update(self, a: Tensor, b: Tensor, base_weight: Tensor) -> None:
        expected_a = (self.in_features, self.rank)
        expected_b = (self.out_features, self.rank)
        expected_base = (self.out_features, self.in_features)
        if tuple(a.shape) != expected_a or tuple(b.shape) != expected_b:
            raise ValueError("factor shapes do not match the fixed EMA structure")
        if tuple(base_weight.shape) != expected_base:
            raise ValueError("base_weight shape does not match the target layer")
        if a.dtype != self.ema_a.dtype or b.dtype != self.ema_b.dtype:
            raise TypeError("factor dtype changed across EMA updates")
        if base_weight.dtype != self.ema_a.dtype:
            raise TypeError("base_weight dtype must match the EMA factors")
        if a.device != self.ema_a.device or b.device != self.ema_b.device:
            raise ValueError("factor device changed across EMA updates")
        if base_weight.device != self.ema_a.device:
            raise ValueError("base_weight device must match the EMA factors")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("EMA factors must be finite")
        if not torch.isfinite(base_weight).all():
            raise ValueError("base_weight must be finite")

    def _certificate(
        self,
        *,
        accepted: bool,
        reason: str,
        update_count: int,
        overlay_norm: float,
        base_norm: float,
        composed_bound: float,
        projected: bool,
        base_unchanged: bool,
        a: Tensor,
        b: Tensor,
        base_weight: Tensor,
    ) -> FactorWiseEMACertificate:
        payload = {
            "a_sha256": _tensor_digest(a),
            "accepted": accepted,
            "b_sha256": _tensor_digest(b),
            "base_sha256": _tensor_digest(base_weight),
            "base_unchanged": base_unchanged,
            "base_operator_norm": _finite_hex(base_norm),
            "composed_norm_limit": self.composed_norm_limit.hex(),
            "composed_upper_bound": _finite_hex(composed_bound),
            "operator_norm_limit": self.max_operator_norm.hex(),
            "overlay_operator_norm": _finite_hex(overlay_norm),
            "projected": projected,
            "rank": self.rank,
            "reason": reason,
            "update_count": update_count,
        }
        return FactorWiseEMACertificate(
            accepted=accepted,
            reason=reason,
            digest=_canonical_digest(payload),
            update_count=update_count,
            rank=self.rank,
            overlay_operator_norm=overlay_norm,
            base_operator_norm=base_norm,
            composed_upper_bound=composed_bound,
            operator_norm_limit=self.max_operator_norm,
            composed_norm_limit=self.composed_norm_limit,
            projected=projected,
            base_unchanged=base_unchanged,
        )

    @torch.no_grad()
    def update(
        self, a: Tensor, b: Tensor, base_weight: Tensor
    ) -> FactorWiseEMACertificate:
        """EMA A/B independently, project ``BA^T``, then atomically commit."""

        self._validate_update(a, b, base_weight)
        base_before = base_weight.detach().clone()
        if self.initialized:
            candidate_a = self.decay * self.ema_a + (1.0 - self.decay) * a
            candidate_b = self.decay * self.ema_b + (1.0 - self.decay) * b
        else:
            candidate_a = a.detach().clone()
            candidate_b = b.detach().clone()

        base_norm = float(torch.linalg.matrix_norm(base_weight.float(), ord=2))
        interior_limit = self.composed_norm_limit - self.strict_margin
        available = interior_limit - base_norm
        candidate_norm = float(self._operator_norm(candidate_a, candidate_b))
        prospective_count = self.update_count + 1
        if available <= 0.0:
            unchanged = torch.equal(base_before, base_weight)
            return self._certificate(
                accepted=False,
                reason="base weight leaves no strict composed-norm margin",
                update_count=self.update_count,
                overlay_norm=candidate_norm,
                base_norm=base_norm,
                composed_bound=base_norm + candidate_norm,
                projected=False,
                base_unchanged=unchanged,
                a=candidate_a,
                b=candidate_b,
                base_weight=base_weight,
            )

        target_norm = min(self.max_operator_norm, available) * (1.0 - 1.0e-5)
        scale = min(1.0, target_norm / max(candidate_norm, 1.0e-12))
        projected_b = candidate_b * candidate_b.new_tensor(scale)
        projected_norm = float(self._operator_norm(candidate_a, projected_b))
        composed_bound = base_norm + projected_norm
        projected = scale < 1.0
        unchanged = torch.equal(base_before, base_weight)
        accepted = bool(
            unchanged
            and math.isfinite(projected_norm)
            and projected_norm <= self.max_operator_norm
            and composed_bound < self.composed_norm_limit
        )
        reason = "accepted" if accepted else "post-projection norm certificate failed"
        certificate = self._certificate(
            accepted=accepted,
            reason=reason,
            update_count=prospective_count if accepted else self.update_count,
            overlay_norm=projected_norm,
            base_norm=base_norm,
            composed_bound=composed_bound,
            projected=projected,
            base_unchanged=unchanged,
            a=candidate_a,
            b=projected_b,
            base_weight=base_weight,
        )
        if accepted:
            self.ema_a.copy_(candidate_a)
            self.ema_b.copy_(projected_b)
            self.update_count_tensor.add_(1)
            self.initialized_tensor.fill_(True)
        return certificate

    def factors(self) -> tuple[Tensor, Tensor]:
        if not self.initialized:
            raise RuntimeError("FactorWiseEMA has not accepted an update")
        return self.ema_a.detach().clone(), self.ema_b.detach().clone()

    def _checkpoint_config(self) -> dict[str, object]:
        dtype_name = str(self.ema_a.dtype).removeprefix("torch.")
        return {
            "composed_norm_limit": self.composed_norm_limit.hex(),
            "decay": self.decay.hex(),
            "dtype": dtype_name,
            "in_features": self.in_features,
            "max_operator_norm": self.max_operator_norm.hex(),
            "out_features": self.out_features,
            "rank": self.rank,
            "strict_margin": self.strict_margin.hex(),
        }

    @staticmethod
    def _checkpoint_digest(
        config: Mapping[str, object], state: Mapping[str, Tensor]
    ) -> str:
        return _canonical_digest(
            {
                "config": dict(config),
                "state": {
                    name: _tensor_digest(tensor)
                    for name, tensor in sorted(state.items())
                },
            }
        )

    def checkpoint(self) -> dict[str, object]:
        """Return a CPU-only, hash-protected checkpoint payload."""

        state = {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in self.state_dict().items()
        }
        config = self._checkpoint_config()
        return {
            "schema": _EMA_CHECKPOINT_SCHEMA,
            "config": config,
            "state": state,
            "sha256": self._checkpoint_digest(config, state),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, object],
        *,
        device: torch.device | str = "cpu",
    ) -> FactorWiseEMA:
        """Restore a checkpoint only after exact schema and digest validation."""

        if set(payload) != {"schema", "config", "state", "sha256"}:
            raise ValueError("FactorWiseEMA checkpoint schema is invalid")
        if payload["schema"] != _EMA_CHECKPOINT_SCHEMA:
            raise ValueError("FactorWiseEMA checkpoint version is unsupported")
        config = payload["config"]
        state = payload["state"]
        digest = payload["sha256"]
        if not isinstance(config, Mapping) or not isinstance(state, Mapping):
            raise TypeError("FactorWiseEMA checkpoint config/state must be mappings")
        if not isinstance(digest, str):
            raise TypeError("FactorWiseEMA checkpoint digest must be text")
        _validate_sha256(digest, "checkpoint sha256")
        tensor_state = dict(state)
        if not all(isinstance(value, Tensor) for value in tensor_state.values()):
            raise TypeError("FactorWiseEMA checkpoint state must contain tensors")
        required_config = {
            "composed_norm_limit",
            "decay",
            "dtype",
            "in_features",
            "max_operator_norm",
            "out_features",
            "rank",
            "strict_margin",
        }
        if set(config) != required_config:
            raise ValueError("FactorWiseEMA checkpoint config is invalid")
        dtype_name = config["dtype"]
        if not isinstance(dtype_name, str) or dtype_name not in _SUPPORTED_FLOAT_DTYPES:
            raise ValueError("FactorWiseEMA checkpoint dtype is unsupported")
        integer_keys = ("in_features", "out_features", "rank")
        if any(
            not isinstance(config[name], int) or isinstance(config[name], bool)
            for name in integer_keys
        ):
            raise TypeError("FactorWiseEMA checkpoint dimensions must be integers")
        restored = cls(
            config["in_features"],
            config["out_features"],
            config["rank"],
            decay=float.fromhex(str(config["decay"])),
            max_operator_norm=float.fromhex(str(config["max_operator_norm"])),
            composed_norm_limit=float.fromhex(str(config["composed_norm_limit"])),
            strict_margin=float.fromhex(str(config["strict_margin"])),
            dtype=_SUPPORTED_FLOAT_DTYPES[dtype_name],
            device=device,
        )
        expected_state = {
            "ema_a": ((restored.in_features, restored.rank), restored.ema_a.dtype),
            "ema_b": ((restored.out_features, restored.rank), restored.ema_b.dtype),
            "initialized_tensor": ((), torch.bool),
            "update_count_tensor": ((), torch.int64),
        }
        if set(tensor_state) != set(expected_state):
            raise ValueError("FactorWiseEMA checkpoint state schema is invalid")
        for name, (shape, dtype) in expected_state.items():
            tensor = tensor_state[name]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"FactorWiseEMA checkpoint tensor is invalid: {name}")
            if tensor.device.type != "cpu":
                raise ValueError("FactorWiseEMA checkpoint tensors must reside on CPU")
        if (
            not torch.isfinite(tensor_state["ema_a"]).all()
            or not torch.isfinite(tensor_state["ema_b"]).all()
        ):
            raise ValueError("FactorWiseEMA checkpoint factors must be finite")
        expected = cls._checkpoint_digest(config, tensor_state)
        if digest != expected:
            raise RuntimeError("FactorWiseEMA checkpoint digest verification failed")
        restored.load_state_dict(
            {name: tensor.to(device=device) for name, tensor in tensor_state.items()},
            strict=True,
        )
        if restored.update_count < 0:
            raise RuntimeError("FactorWiseEMA checkpoint update count is invalid")
        if restored.initialized != (restored.update_count > 0):
            raise RuntimeError(
                "FactorWiseEMA checkpoint initialization state is invalid"
            )
        return restored


__all__ = [
    "AQCertificate",
    "ExternalVerifierEvidence",
    "FactorWiseEMA",
    "FactorWiseEMACertificate",
    "ResidualDecayAQGate",
    "TrainedProgrammerEvidence",
    "VerifiedFastWeightAdmission",
    "VerifiedFastWeightProgrammer",
    "authorize_fast_weight_admission",
    "fast_weight_checkpoint_architecture",
    "fast_weight_checkpoint_state",
    "load_verified_fast_weight_checkpoint",
]
