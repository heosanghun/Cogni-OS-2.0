"""Verified, bounded learned control for Cognitive Tree Search.

The controller is intentionally independent from the tree implementation.  It
reduces any bounded latent tensor to twelve fixed summary features, then uses
three separately trained heads for action logits, critic value, and four
meta-controls.  It retains no tree, token, or trajectory history.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


SUMMARY_DIM = 12
ACTION_WIDTH = 3
META_CONTROL_DIM = 4
MAX_LATENT_ELEMENTS = 1_048_576
MAX_CHECKPOINT_BYTES = 128 * 1024
MAX_STATE_ABS = 32.0
ACTION_LOGIT_BOUND = 2.0
CHECKPOINT_SCHEMA = "cogni-cts-control-v1"
CALIBRATION_DATASET_ID = "cts-analytic-latents-v1"
CALIBRATION_ALGORITHM = "independent-ridge-heads-v1"
CALIBRATION_TRAIN_SAMPLES = 48
CALIBRATION_HELDOUT_SAMPLES = 16
CALIBRATION_LATENT_WIDTH = 32
CALIBRATION_RIDGE = 1.0e-3

EXPLORATION_BOUNDS = (0.25, 2.0)
TOLERANCE_BOUNDS = (1.0e-6, 5.0e-3)
TEMPERATURE_BOUNDS = (0.05, 1.5)
ACT_BOUNDS = (4.0, 32.0)

DEFAULT_CHECKPOINT_PATH = Path(__file__).with_name("cts_policy_checkpoint.json")
# This code-anchored digest is the trust root for the bundled text checkpoint.
DEFAULT_CHECKPOINT_SHA256 = (
    "2fb1e2abd586c4c7fc5541bed7d3cdab22edc4b7582a94d2b083398a0ce723ef"
)

_VERIFIED_LOAD_TOKEN = object()
_STATE_SHAPES = {
    "policy_head.weight": (ACTION_WIDTH, SUMMARY_DIM),
    "policy_head.bias": (ACTION_WIDTH,),
    "critic_head.weight": (1, SUMMARY_DIM),
    "critic_head.bias": (1,),
    "meta_head.weight": (META_CONTROL_DIM, SUMMARY_DIM),
    "meta_head.bias": (META_CONTROL_DIM,),
}


class CTSControlError(RuntimeError):
    """Base class for fail-closed learned CTS control errors."""


class CTSCheckpointError(CTSControlError):
    """Raised when checkpoint integrity, provenance, or evidence is invalid."""


@dataclass(frozen=True, slots=True)
class CTSMetaControls:
    exploration: Tensor
    tolerance: Tensor
    temperature: Tensor
    act: Tensor

    @property
    def tensor(self) -> Tensor:
        return torch.stack(
            (self.exploration, self.tolerance, self.temperature, self.act)
        )


@dataclass(frozen=True, slots=True)
class CTSControlOutput:
    summary: Tensor
    action_logits: Tensor
    critic_value: Tensor
    meta_controls: CTSMetaControls


@dataclass(frozen=True, slots=True)
class CTSCalibrationEvidence:
    policy_accuracy: float
    critic_mae: float
    meta_normalized_mae: float
    action_coverage: int
    max_action_fraction: float


@dataclass(frozen=True, slots=True)
class CTSCheckpointProvenance:
    dataset_id: str
    dataset_sha256: str
    algorithm: str
    train_samples: int
    heldout_samples: int
    state_sha256: str
    evidence: CTSCalibrationEvidence


def _mean_or_zero(values: Tensor, reference: Tensor) -> Tensor:
    return values.mean() if values.numel() else reference.new_zeros(())


def summarize_latent(latent: Tensor) -> Tensor:
    """Return twelve finite features in ``[-1, 1]`` from one latent tensor."""

    if not isinstance(latent, Tensor):
        raise TypeError("latent must be a tensor")
    if not latent.is_floating_point():
        raise TypeError("latent must be floating point")
    if latent.device.type == "meta" or latent.layout != torch.strided:
        raise ValueError("latent must be a materialized strided tensor")
    if not 1 <= latent.numel() <= MAX_LATENT_ELEMENTS:
        raise ValueError("latent element count exceeds the fixed safety bound")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("latent must be finite")

    flat = latent.detach().reshape(-1).to(dtype=torch.float32)
    magnitude = flat.abs().amax()
    normalized = torch.where(
        magnitude > 0,
        flat / magnitude.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(flat),
    )
    midpoint = max(1, (normalized.numel() + 1) // 2)
    first = normalized[:midpoint]
    second = normalized[midpoint:]
    even = normalized[0::2]
    odd = normalized[1::2]
    mean = normalized.mean()
    centered = normalized - mean
    features = torch.stack(
        (
            mean,
            normalized.square().mean().sqrt(),
            normalized.abs().mean(),
            normalized.amin(),
            normalized.amax(),
            centered.square().mean().sqrt().clamp_max(1.0),
            normalized.sign().mean(),
            torch.tanh(torch.log1p(magnitude)),
            _mean_or_zero(first, normalized),
            _mean_or_zero(second, normalized),
            _mean_or_zero(even, normalized),
            _mean_or_zero(odd, normalized),
        )
    ).clamp(-1.0, 1.0)
    if features.shape != (SUMMARY_DIM,) or not bool(torch.isfinite(features).all()):
        raise CTSControlError("latent summary violated its fixed finite contract")
    return features


def _map_linear(raw: Tensor, bounds: tuple[float, float]) -> Tensor:
    lower, upper = bounds
    return lower + (upper - lower) * torch.sigmoid(raw)


def _map_log(raw: Tensor, bounds: tuple[float, float]) -> Tensor:
    lower, upper = map(math.log, bounds)
    return torch.exp(lower + (upper - lower) * torch.sigmoid(raw))


class LearnedCTSController(nn.Module):
    """Frozen learned heads over a non-learned, fixed-size latent summary."""

    def __init__(
        self,
        provenance: CTSCheckpointProvenance,
        checkpoint_sha256: str,
        *,
        _verified: object,
    ) -> None:
        if _verified is not _VERIFIED_LOAD_TOKEN:
            raise CTSCheckpointError(
                "LearnedCTSController must be built by the verified checkpoint loader"
            )
        super().__init__()
        # No learned trunk is shared.  In particular, critic parameters are
        # disjoint from both the action policy and meta-control parameters.
        self.policy_head = nn.Linear(SUMMARY_DIM, ACTION_WIDTH)
        self.critic_head = nn.Linear(SUMMARY_DIM, 1)
        self.meta_head = nn.Linear(SUMMARY_DIM, META_CONTROL_DIM)
        self.provenance = provenance
        self.checkpoint_sha256 = checkpoint_sha256

    def train(self, mode: bool = True) -> LearnedCTSController:
        if mode:
            raise CTSControlError("verified CTS control is frozen for inference")
        return super().train(False)

    def _assert_frozen(self) -> None:
        if self.training or any(
            parameter.requires_grad for parameter in self.parameters()
        ):
            raise CTSControlError("CTS controller left frozen inference mode")

    @property
    def device(self) -> torch.device:
        return self.policy_head.weight.device

    @torch.inference_mode()
    def _summary(self, latent: Tensor) -> Tensor:
        self._assert_frozen()
        if latent.device != self.device:
            raise ValueError(
                f"latent and CTS controller must share a device: "
                f"latent={latent.device}, controller={self.device}"
            )
        return summarize_latent(latent)

    @staticmethod
    def _validate_policy_critic(logits: Tensor, value: Tensor) -> None:
        if logits.shape != (ACTION_WIDTH,) or value.numel() != 1:
            raise CTSControlError("CTS policy or critic output shape changed")
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(value)):
            raise CTSControlError("CTS policy or critic output became non-finite")
        if bool((logits.abs() > ACTION_LOGIT_BOUND).any()) or bool(value.abs() > 1.0):
            raise CTSControlError("CTS policy or critic crossed its output bound")

    @torch.inference_mode()
    def forward(self, latent: Tensor) -> CTSControlOutput:
        summary = self._summary(latent)
        policy_raw = self.policy_head(summary)
        critic_raw = self.critic_head(summary).reshape(())
        meta_raw = self.meta_head(summary)
        if (
            policy_raw.shape != (ACTION_WIDTH,)
            or meta_raw.shape != (META_CONTROL_DIM,)
            or not bool(torch.isfinite(policy_raw).all())
            or not bool(torch.isfinite(critic_raw))
            or not bool(torch.isfinite(meta_raw).all())
        ):
            raise CTSControlError("CTS learned heads produced malformed raw controls")
        action_logits = torch.tanh(policy_raw) * ACTION_LOGIT_BOUND
        critic_value = torch.tanh(critic_raw)
        controls = CTSMetaControls(
            _map_linear(meta_raw[0], EXPLORATION_BOUNDS),
            _map_log(meta_raw[1], TOLERANCE_BOUNDS),
            _map_linear(meta_raw[2], TEMPERATURE_BOUNDS),
            _map_linear(meta_raw[3], ACT_BOUNDS),
        )
        self._validate_output(action_logits, critic_value, controls)
        return CTSControlOutput(summary, action_logits, critic_value, controls)

    @staticmethod
    def _validate_output(
        logits: Tensor,
        value: Tensor,
        controls: CTSMetaControls,
    ) -> None:
        LearnedCTSController._validate_policy_critic(logits, value)
        vector = controls.tensor
        lower = vector.new_tensor(
            (
                EXPLORATION_BOUNDS[0],
                TOLERANCE_BOUNDS[0],
                TEMPERATURE_BOUNDS[0],
                ACT_BOUNDS[0],
            )
        )
        upper = vector.new_tensor(
            (
                EXPLORATION_BOUNDS[1],
                TOLERANCE_BOUNDS[1],
                TEMPERATURE_BOUNDS[1],
                ACT_BOUNDS[1],
            )
        )
        if not bool(torch.isfinite(vector).all()) or bool(
            ((vector < lower) | (vector > upper)).any()
        ):
            raise CTSControlError("CTS meta-controls violated their finite bounds")

    @torch.inference_mode()
    def policy_value(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        summary = self._summary(latent)
        policy_raw = self.policy_head(summary)
        critic_raw = self.critic_head(summary).reshape(())
        if not bool(torch.isfinite(policy_raw).all()) or not bool(
            torch.isfinite(critic_raw)
        ):
            raise CTSControlError("CTS policy or critic raw output became non-finite")
        logits = torch.tanh(policy_raw) * ACTION_LOGIT_BOUND
        value = torch.tanh(critic_raw)
        self._validate_policy_critic(logits, value)
        return logits, value

    @torch.inference_mode()
    def policy_logits(self, latent: Tensor) -> Tensor:
        raw = self.policy_head(self._summary(latent))
        if not bool(torch.isfinite(raw).all()):
            raise CTSControlError("CTS policy raw output became non-finite")
        logits = torch.tanh(raw) * ACTION_LOGIT_BOUND
        if logits.shape != (ACTION_WIDTH,) or not bool(torch.isfinite(logits).all()):
            raise CTSControlError("CTS policy output violated its finite shape")
        if bool((logits.abs() > ACTION_LOGIT_BOUND).any()):
            raise CTSControlError("CTS policy crossed its output bound")
        return logits

    @torch.inference_mode()
    def critic(self, latent: Tensor) -> Tensor:
        raw = self.critic_head(self._summary(latent)).reshape(())
        if not bool(torch.isfinite(raw)):
            raise CTSControlError("CTS critic raw output became non-finite")
        value = torch.tanh(raw)
        if not bool(torch.isfinite(value)) or bool(value.abs() > 1.0):
            raise CTSControlError("CTS critic violated its finite value bound")
        return value

    @torch.inference_mode()
    def meta_controls(self, latent: Tensor) -> CTSMetaControls:
        raw = self.meta_head(self._summary(latent))
        if raw.shape != (META_CONTROL_DIM,) or not bool(torch.isfinite(raw).all()):
            raise CTSControlError("CTS meta-control raw output became malformed")
        controls = CTSMetaControls(
            _map_linear(raw[0], EXPLORATION_BOUNDS),
            _map_log(raw[1], TOLERANCE_BOUNDS),
            _map_linear(raw[2], TEMPERATURE_BOUNDS),
            _map_linear(raw[3], ACT_BOUNDS),
        )
        vector = controls.tensor
        if vector.shape != (META_CONTROL_DIM,) or not bool(
            torch.isfinite(vector).all()
        ):
            raise CTSControlError("CTS meta-control output violated its finite shape")
        return controls


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _offline_latents() -> Tensor:
    sample = torch.arange(
        1,
        CALIBRATION_TRAIN_SAMPLES + CALIBRATION_HELDOUT_SAMPLES + 1,
        dtype=torch.float64,
    )[:, None]
    feature = torch.arange(1, CALIBRATION_LATENT_WIDTH + 1, dtype=torch.float64)[
        None, :
    ]
    latents = (
        torch.sin(sample * feature * 0.113)
        + 0.41 * torch.cos((sample + 2.0) * feature * 0.071)
        + 0.17 * torch.sin((sample.square() + feature) * 0.019)
    )
    return latents.to(torch.float32)


def _teacher_parameters() -> tuple[Tensor, Tensor, Tensor]:
    column = torch.arange(1, SUMMARY_DIM + 1, dtype=torch.float64)[None, :]
    policy = torch.zeros(ACTION_WIDTH, SUMMARY_DIM, dtype=torch.float64)
    # The calibration teacher rewards distinct counterfactual regions rather
    # than forcing one action: early-latent evidence, late-latent evidence, and
    # their joint negative residual.  Ridge fitting still learns all weights.
    policy[0, 8] = 2.0
    policy[1, 9] = 2.0
    policy[2, 8:10] = -1.5
    critic = torch.cos(column * 0.43).reshape(1, SUMMARY_DIM)
    meta_row = torch.arange(1, META_CONTROL_DIM + 1, dtype=torch.float64)[:, None]
    meta = 0.75 * torch.sin(meta_row * column * 0.29) + 0.2 * torch.cos(
        (meta_row + column) * 0.17
    )
    return policy, critic, meta


def _offline_dataset() -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
    latents = _offline_latents()
    summaries = torch.stack([summarize_latent(latent) for latent in latents]).to(
        torch.float64
    )
    policy_teacher, critic_teacher, meta_teacher = _teacher_parameters()
    policy_targets = summaries @ policy_teacher.T
    critic_targets = summaries @ critic_teacher.T
    meta_targets = summaries @ meta_teacher.T
    dataset_payload = {
        "dataset_id": CALIBRATION_DATASET_ID,
        "latents": [[round(float(value), 8) for value in row] for row in latents],
        "policy_teacher": policy_teacher.tolist(),
        "critic_teacher": critic_teacher.tolist(),
        "meta_teacher": meta_teacher.tolist(),
    }
    digest = sha256(_canonical_json(dataset_payload)).hexdigest()
    return summaries, policy_targets, critic_targets, meta_targets, digest


def _ridge_fit(features: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    ones = torch.ones(features.shape[0], 1, dtype=torch.float64)
    design = torch.cat((features, ones), dim=1)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * CALIBRATION_RIDGE
    regularizer[-1, -1] = 0.0
    solution = torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ targets,
    )
    return solution[:-1].T, solution[-1]


def _rounded(values: Tensor) -> list[Any]:
    array = values.detach().to(dtype=torch.float64).tolist()

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        return round(float(value), 8)

    result = visit(array)
    assert isinstance(result, list)
    return result


def _state_tensors(state: Mapping[str, Any]) -> dict[str, Tensor]:
    if set(state) != set(_STATE_SHAPES):
        raise CTSCheckpointError("CTS checkpoint state keys are not exact")
    tensors: dict[str, Tensor] = {}
    for name, shape in _STATE_SHAPES.items():
        try:
            tensor = torch.tensor(state[name], dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise CTSCheckpointError(f"invalid tensor payload for {name}") from exc
        if tuple(tensor.shape) != shape:
            raise CTSCheckpointError(
                f"CTS checkpoint tensor {name} must have shape {shape}"
            )
        if not bool(torch.isfinite(tensor).all()) or bool(
            (tensor.abs() > MAX_STATE_ABS).any()
        ):
            raise CTSCheckpointError(f"CTS checkpoint tensor {name} is unsafe")
        tensors[name] = tensor
    if bool((tensors["policy_head.weight"].std(dim=0) < 1.0e-6).all()):
        raise CTSCheckpointError("action policy rows collapsed to one dominant rule")
    return tensors


def _stable_sigmoid(value: float) -> float:
    """Return sigmoid without backend-specific vector approximations."""

    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _deterministic_activation_mae(
    features: Tensor,
    weight: Tensor,
    bias: Tensor,
    targets: Tensor,
    *,
    activation: Any,
) -> float:
    """Reproduce held-out scalar evidence independent of BLAS kernels."""

    feature_rows = features.detach().to("cpu", dtype=torch.float32).tolist()
    weight_rows = weight.detach().to("cpu", dtype=torch.float32).tolist()
    bias_values = bias.detach().to("cpu", dtype=torch.float32).tolist()
    target_rows = targets.detach().to("cpu", dtype=torch.float32).tolist()
    errors: list[float] = []
    for feature_row, target_row in zip(feature_rows, target_rows, strict=True):
        for head_index, weight_row in enumerate(weight_rows):
            raw = float(bias_values[head_index]) + math.fsum(
                float(coefficient) * float(feature)
                for coefficient, feature in zip(
                    weight_row,
                    feature_row,
                    strict=True,
                )
            )
            errors.append(
                abs(activation(raw) - activation(float(target_row[head_index])))
            )
    if not errors:
        raise CTSCheckpointError("held-out evidence cannot be empty")
    return math.fsum(errors) / len(errors)


def _evidence_from_state(
    state: Mapping[str, Any],
    summaries: Tensor,
    policy_targets: Tensor,
    critic_targets: Tensor,
    meta_targets: Tensor,
) -> CTSCalibrationEvidence:
    tensors = _state_tensors(state)
    heldout = slice(CALIBRATION_TRAIN_SAMPLES, None)
    features = summaries[heldout].to(torch.float32)
    policy_raw = torch.nn.functional.linear(
        features,
        tensors["policy_head.weight"],
        tensors["policy_head.bias"],
    )
    expected_actions = policy_targets[heldout].argmax(dim=1)
    predicted_actions = policy_raw.argmax(dim=1)
    policy_accuracy = float((predicted_actions == expected_actions).float().mean())
    critic_mae = _deterministic_activation_mae(
        features,
        tensors["critic_head.weight"],
        tensors["critic_head.bias"],
        critic_targets[heldout],
        activation=math.tanh,
    )
    meta_mae = _deterministic_activation_mae(
        features,
        tensors["meta_head.weight"],
        tensors["meta_head.bias"],
        meta_targets[heldout],
        activation=_stable_sigmoid,
    )
    counts = torch.bincount(predicted_actions, minlength=ACTION_WIDTH)
    return CTSCalibrationEvidence(
        policy_accuracy=policy_accuracy,
        critic_mae=critic_mae,
        meta_normalized_mae=meta_mae,
        action_coverage=int((counts > 0).sum()),
        max_action_fraction=float(counts.max() / counts.sum()),
    )


def rebuild_offline_checkpoint_payload() -> dict[str, Any]:
    """Reproduce the small deterministic offline fit and held-out evidence."""

    summaries, policy_targets, critic_targets, meta_targets, dataset_digest = (
        _offline_dataset()
    )
    train = slice(0, CALIBRATION_TRAIN_SAMPLES)
    policy_weight, policy_bias = _ridge_fit(summaries[train], policy_targets[train])
    critic_weight, critic_bias = _ridge_fit(summaries[train], critic_targets[train])
    meta_weight, meta_bias = _ridge_fit(summaries[train], meta_targets[train])
    state = {
        "policy_head.weight": _rounded(policy_weight),
        "policy_head.bias": _rounded(policy_bias),
        "critic_head.weight": _rounded(critic_weight),
        "critic_head.bias": _rounded(critic_bias),
        "meta_head.weight": _rounded(meta_weight),
        "meta_head.bias": _rounded(meta_bias),
    }
    evidence = _evidence_from_state(
        state,
        summaries,
        policy_targets,
        critic_targets,
        meta_targets,
    )
    state_digest = sha256(_canonical_json(state)).hexdigest()
    return {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": {
            "summary_dim": SUMMARY_DIM,
            "action_width": ACTION_WIDTH,
            "meta_control_dim": META_CONTROL_DIM,
            "action_logit_bound": ACTION_LOGIT_BOUND,
            "controls": {
                "exploration": list(EXPLORATION_BOUNDS),
                "tolerance": list(TOLERANCE_BOUNDS),
                "temperature": list(TEMPERATURE_BOUNDS),
                "act": list(ACT_BOUNDS),
            },
        },
        "provenance": {
            "trained": True,
            "integrity": "offline-calibrated-heldout-v1",
            "dataset_id": CALIBRATION_DATASET_ID,
            "dataset_sha256": dataset_digest,
            "algorithm": CALIBRATION_ALGORITHM,
            "ridge": CALIBRATION_RIDGE,
            "train_samples": CALIBRATION_TRAIN_SAMPLES,
            "heldout_samples": CALIBRATION_HELDOUT_SAMPLES,
            "state_sha256": state_digest,
            "heldout": {
                "policy_accuracy": round(evidence.policy_accuracy, 8),
                "critic_mae": round(evidence.critic_mae, 8),
                "meta_normalized_mae": round(evidence.meta_normalized_mae, 8),
                "action_coverage": evidence.action_coverage,
                "max_action_fraction": round(evidence.max_action_fraction, 8),
            },
        },
        "state": state,
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CTSCheckpointError(f"duplicate checkpoint key: {key}")
        result[key] = value
    return result


def _parse_evidence(value: Any) -> CTSCalibrationEvidence:
    if not isinstance(value, dict) or set(value) != {
        "policy_accuracy",
        "critic_mae",
        "meta_normalized_mae",
        "action_coverage",
        "max_action_fraction",
    }:
        raise CTSCheckpointError("held-out evidence schema is invalid")
    try:
        evidence = CTSCalibrationEvidence(
            float(value["policy_accuracy"]),
            float(value["critic_mae"]),
            float(value["meta_normalized_mae"]),
            int(value["action_coverage"]),
            float(value["max_action_fraction"]),
        )
    except (TypeError, ValueError) as exc:
        raise CTSCheckpointError("held-out evidence is not numeric") from exc
    numeric = (
        evidence.policy_accuracy,
        evidence.critic_mae,
        evidence.meta_normalized_mae,
        evidence.max_action_fraction,
    )
    if not all(math.isfinite(item) for item in numeric):
        raise CTSCheckpointError("held-out evidence must be finite")
    if (
        evidence.policy_accuracy < 0.80
        or evidence.critic_mae > 0.10
        or evidence.meta_normalized_mae > 0.10
        or evidence.action_coverage < 2
        or evidence.max_action_fraction > 0.80
    ):
        raise CTSCheckpointError("held-out evidence did not pass admission thresholds")
    return evidence


def _parse_provenance(value: Any, state: Mapping[str, Any]) -> CTSCheckpointProvenance:
    required = {
        "trained",
        "integrity",
        "dataset_id",
        "dataset_sha256",
        "algorithm",
        "ridge",
        "train_samples",
        "heldout_samples",
        "state_sha256",
        "heldout",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CTSCheckpointError("checkpoint provenance schema is invalid")
    if value["trained"] is not True:
        raise CTSCheckpointError("untrained CTS checkpoint is forbidden")
    if value["integrity"] != "offline-calibrated-heldout-v1":
        raise CTSCheckpointError("checkpoint integrity provenance is unknown")
    summaries, policy_targets, critic_targets, meta_targets, expected_dataset_digest = (
        _offline_dataset()
    )
    if (
        value["dataset_id"] != CALIBRATION_DATASET_ID
        or value["dataset_sha256"] != expected_dataset_digest
        or value["algorithm"] != CALIBRATION_ALGORITHM
        or float(value["ridge"]) != CALIBRATION_RIDGE
        or value["train_samples"] != CALIBRATION_TRAIN_SAMPLES
        or value["heldout_samples"] != CALIBRATION_HELDOUT_SAMPLES
    ):
        raise CTSCheckpointError("checkpoint calibration provenance is not recognized")
    state_digest = sha256(_canonical_json(state)).hexdigest()
    if not hmac.compare_digest(str(value["state_sha256"]), state_digest):
        raise CTSCheckpointError("checkpoint state digest mismatch")
    reported_evidence = _parse_evidence(value["heldout"])
    reproduced_evidence = _evidence_from_state(
        state,
        summaries,
        policy_targets,
        critic_targets,
        meta_targets,
    )
    reproduced_values = (
        reproduced_evidence.policy_accuracy,
        reproduced_evidence.critic_mae,
        reproduced_evidence.meta_normalized_mae,
        reproduced_evidence.max_action_fraction,
    )
    reported_values = (
        reported_evidence.policy_accuracy,
        reported_evidence.critic_mae,
        reported_evidence.meta_normalized_mae,
        reported_evidence.max_action_fraction,
    )
    if reproduced_evidence.action_coverage != reported_evidence.action_coverage or any(
        abs(actual - claimed) > 1.0e-6
        for actual, claimed in zip(reproduced_values, reported_values, strict=True)
    ):
        raise CTSCheckpointError("held-out evidence does not reproduce from state")
    # Apply the same admission thresholds to the reproduced, not merely claimed,
    # metrics before constructing any inference module.
    _parse_evidence(
        {
            "policy_accuracy": reproduced_evidence.policy_accuracy,
            "critic_mae": reproduced_evidence.critic_mae,
            "meta_normalized_mae": reproduced_evidence.meta_normalized_mae,
            "action_coverage": reproduced_evidence.action_coverage,
            "max_action_fraction": reproduced_evidence.max_action_fraction,
        }
    )
    return CTSCheckpointProvenance(
        dataset_id=CALIBRATION_DATASET_ID,
        dataset_sha256=expected_dataset_digest,
        algorithm=CALIBRATION_ALGORITHM,
        train_samples=CALIBRATION_TRAIN_SAMPLES,
        heldout_samples=CALIBRATION_HELDOUT_SAMPLES,
        state_sha256=state_digest,
        evidence=reported_evidence,
    )


def _validate_architecture(value: Any) -> None:
    expected = rebuild_offline_checkpoint_payload()["architecture"]
    if value != expected:
        raise CTSCheckpointError("checkpoint architecture or control bounds changed")


def load_bounded_cts_controller(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> LearnedCTSController:
    """Load a trained controller only after digest and provenance verification."""

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    source = Path(checkpoint).expanduser()
    if source.is_symlink():
        raise CTSCheckpointError("checkpoint must not be a symbolic link")
    path = source.resolve(strict=True)
    if not path.is_file():
        raise CTSCheckpointError("checkpoint must be one regular local file")
    raw = path.read_bytes()
    if not 1 <= len(raw) <= MAX_CHECKPOINT_BYTES:
        raise CTSCheckpointError("checkpoint text exceeds its bounded size")
    actual_digest = sha256(raw).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_sha256):
        raise CTSCheckpointError("checkpoint SHA-256 verification failed")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CTSCheckpointError("checkpoint must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "architecture",
        "provenance",
        "state",
    }:
        raise CTSCheckpointError("checkpoint top-level schema is invalid")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise CTSCheckpointError("checkpoint schema version is unsupported")
    _validate_architecture(payload["architecture"])
    if not isinstance(payload["state"], dict):
        raise CTSCheckpointError("checkpoint state must be an object")
    tensors = _state_tensors(payload["state"])
    provenance = _parse_provenance(payload["provenance"], payload["state"])

    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or (
        target.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError(
            "CTS controller device must be an available CPU or CUDA device"
        )
    controller = LearnedCTSController(
        provenance,
        actual_digest,
        _verified=_VERIFIED_LOAD_TOKEN,
    )
    controller.load_state_dict(tensors, strict=True)
    controller.to(device=target, dtype=torch.float32)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    controller.train(False)
    controller._assert_frozen()
    return controller


def load_default_bounded_cts_controller(
    *, device: str | torch.device = "cpu"
) -> LearnedCTSController:
    return load_bounded_cts_controller(
        DEFAULT_CHECKPOINT_PATH,
        expected_sha256=DEFAULT_CHECKPOINT_SHA256,
        device=device,
    )


__all__ = [
    "ACTION_LOGIT_BOUND",
    "ACTION_WIDTH",
    "ACT_BOUNDS",
    "CTSCalibrationEvidence",
    "CTSCheckpointError",
    "CTSCheckpointProvenance",
    "CTSControlError",
    "CTSControlOutput",
    "CTSMetaControls",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_CHECKPOINT_SHA256",
    "EXPLORATION_BOUNDS",
    "LearnedCTSController",
    "MAX_LATENT_ELEMENTS",
    "META_CONTROL_DIM",
    "SUMMARY_DIM",
    "TEMPERATURE_BOUNDS",
    "TOLERANCE_BOUNDS",
    "load_bounded_cts_controller",
    "load_default_bounded_cts_controller",
    "rebuild_offline_checkpoint_payload",
    "summarize_latent",
]
