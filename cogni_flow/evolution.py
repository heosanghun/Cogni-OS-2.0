"""Transactional System-2.5 evolution with immutable generation checkpoints.

This module is deliberately night-only.  It couples a real FP-EWC penalty to
an optimizer update, applies the C-FIRE scaled-polar certificate before and
after that update, and restores model/optimizer/Fisher state byte-for-byte if
any stage fails.  Stable generations are written behind an atomic ``CURRENT``
pointer; inference never observes a partially written candidate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from cogni_core.c_fire import CFireCertificate, c_fire_scaled_polar_
from cogni_core.fp_ewc import FPEWCRegularizer, FisherSnapshot
from cogni_flow.rhythm import RhythmController, SystemMode


CHECKPOINT_SCHEMA = "cogni-evolution-generation-v1"
POINTER_SCHEMA = "cogni-evolution-current-v1"
MAX_GENERATIONS = 64
MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024


class EvolutionTransactionError(RuntimeError):
    """Raised after an evolution candidate is fully rolled back."""


class GenerationCheckpointError(RuntimeError):
    """Raised when a generation or pointer cannot be authenticated."""


@dataclass(frozen=True, slots=True)
class CFireTarget:
    name: str
    parameter: nn.Parameter
    gamma: float = 0.90
    spectral_margin: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 160:
            raise ValueError("C-FIRE target name must be bounded non-empty text")
        if not isinstance(self.parameter, nn.Parameter):
            raise TypeError("C-FIRE target must reference an nn.Parameter")
        if self.parameter.ndim != 2 or not self.parameter.is_floating_point():
            raise ValueError("C-FIRE targets must be floating matrices")
        if not 0.0 < self.gamma < self.spectral_margin < 1.0:
            raise ValueError("C-FIRE target requires gamma < spectral margin < 1")


@dataclass(frozen=True, slots=True)
class EvolutionStepReport:
    task_loss: float
    ewc_penalty: float
    total_loss: float
    before_digest: str
    after_digest: str
    pre_certificates: Mapping[str, CFireCertificate]
    post_certificates: Mapping[str, CFireCertificate]


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    checkpoint: Path
    checkpoint_sha256: str
    manifest: Path
    manifest_sha256: str
    parent_checkpoint_sha256: str | None


@dataclass(frozen=True, slots=True)
class TransferMetrics:
    backward_transfer: float
    forward_transfer: float


@dataclass(frozen=True, slots=True)
class SeededTransferSummary:
    seeds: int
    bwt_mean: float
    bwt_std: float
    fwt_mean: float
    fwt_std: float


def _clone_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().to("cpu").clone() for name, value in state.items()}


def _clone_fisher(snapshots: Iterable[FisherSnapshot]) -> list[FisherSnapshot]:
    return [
        FisherSnapshot(
            fisher=_clone_state(snapshot.fisher),
            anchor=_clone_state(snapshot.anchor),
            n_samples=snapshot.n_samples,
            quadratic_offset=snapshot.quadratic_offset,
        )
        for snapshot in snapshots
    ]


def _tensor_bytes(value: Tensor) -> bytes:
    flat = value.detach().to("cpu").contiguous().reshape(-1)
    return bytes(flat.view(torch.uint8).tolist())


def _state_digest(state: Mapping[str, Tensor]) -> str:
    digest = sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _validate_state_for(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    current = module.state_dict()
    if current.keys() != state.keys():
        raise GenerationCheckpointError("checkpoint model keys differ from runtime")
    for name, value in state.items():
        expected = current[name]
        if tuple(value.shape) != tuple(expected.shape) or value.dtype != expected.dtype:
            raise GenerationCheckpointError(
                f"checkpoint tensor contract changed for {name!r}"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise GenerationCheckpointError(
                f"checkpoint tensor is non-finite for {name!r}"
            )


def _finite_optimizer_state(state: object) -> bool:
    if isinstance(state, Tensor):
        return not state.is_floating_point() or bool(torch.isfinite(state).all())
    if isinstance(state, Mapping):
        return all(_finite_optimizer_state(value) for value in state.values())
    if isinstance(state, (list, tuple)):
        return all(_finite_optimizer_state(value) for value in state)
    if isinstance(state, float):
        return math.isfinite(state)
    return isinstance(state, (str, int, bool, type(None)))


class EvolutionTransaction:
    """One rollback-capable optimizer commit under an evolution slot."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        regularizer: FPEWCRegularizer,
        rhythm: RhythmController,
        c_fire_targets: Sequence[CFireTarget],
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        if not isinstance(regularizer, FPEWCRegularizer):
            raise TypeError("regularizer must be FPEWCRegularizer")
        if not isinstance(rhythm, RhythmController):
            raise TypeError("rhythm must be RhythmController")
        targets = tuple(c_fire_targets)
        if not targets or len(targets) > 64:
            raise ValueError("an evolution transaction needs 1..64 C-FIRE targets")
        model_parameters = {id(parameter) for parameter in model.parameters()}
        if len({target.name for target in targets}) != len(targets):
            raise ValueError("C-FIRE target names must be unique")
        if any(id(target.parameter) not in model_parameters for target in targets):
            raise ValueError("every C-FIRE target must belong to the model")
        self.model = model
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.rhythm = rhythm
        self.targets = targets

    @property
    def named_trainable(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )

    def _snapshot(self) -> tuple[dict[str, Tensor], dict, list[FisherSnapshot]]:
        return (
            _clone_state(self.model.state_dict()),
            deepcopy(self.optimizer.state_dict()),
            _clone_fisher(self.regularizer.snapshots),
        )

    def _restore(
        self, snapshot: tuple[dict[str, Tensor], dict, list[FisherSnapshot]]
    ) -> None:
        model_state, optimizer_state, fisher = snapshot
        self.model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)
        self.regularizer.snapshots = _clone_fisher(fisher)

    def _certify(self) -> dict[str, CFireCertificate]:
        return {
            target.name: c_fire_scaled_polar_(
                target.parameter,
                gamma=target.gamma,
                spectral_margin=target.spectral_margin,
            )
            for target in self.targets
        }

    def _verify_targets(self) -> None:
        """Verify a restored certificate without rewriting checkpoint bytes."""

        for target in self.targets:
            value = target.parameter.detach().to(dtype=torch.float32)
            singular = torch.linalg.svdvals(value)
            if singular.numel() == 0 or not bool(torch.isfinite(singular).all()):
                raise GenerationCheckpointError(
                    f"restored C-FIRE spectrum is invalid for {target.name!r}"
                )
            maximum = float(singular.max())
            minimum = float(singular.min())
            if (
                maximum >= target.spectral_margin
                or minimum <= 0.0
                or maximum / minimum > 1.05
            ):
                raise GenerationCheckpointError(
                    f"restored C-FIRE certificate failed for {target.name!r}"
                )

    def step(self, loss_closure: Callable[[], Tensor]) -> EvolutionStepReport:
        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError("optimizer evolution is allowed only in evolution mode")
        if not callable(loss_closure):
            raise TypeError("loss_closure must be callable")
        snapshot = self._snapshot()
        before_digest = _state_digest(snapshot[0])
        try:
            with self.rhythm.evolution_slot():
                pre = self._certify()
                self.optimizer.zero_grad(set_to_none=True)
                task_loss = loss_closure()
                if (
                    not isinstance(task_loss, Tensor)
                    or task_loss.numel() != 1
                    or not bool(torch.isfinite(task_loss))
                ):
                    raise EvolutionTransactionError(
                        "task loss must be one finite scalar tensor"
                    )
                named = self.named_trainable
                if not named:
                    raise EvolutionTransactionError("model has no trainable parameters")
                penalty = self.regularizer.penalty(named)
                if penalty.numel() != 1 or not bool(torch.isfinite(penalty)):
                    raise EvolutionTransactionError("FP-EWC penalty became non-finite")
                total = task_loss + penalty
                total.backward()
                for name, parameter in named:
                    gradient = parameter.grad
                    if gradient is not None and not bool(
                        torch.isfinite(gradient).all()
                    ):
                        raise EvolutionTransactionError(
                            f"optimizer gradient became non-finite for {name!r}"
                        )
                self.optimizer.step()
                if not _finite_optimizer_state(self.optimizer.state_dict()):
                    raise EvolutionTransactionError("optimizer state became non-finite")
                post = self._certify()
                current = self.model.state_dict()
                if any(
                    value.is_floating_point() and not bool(torch.isfinite(value).all())
                    for value in current.values()
                ):
                    raise EvolutionTransactionError("candidate model became non-finite")
                after_digest = _state_digest(current)
                return EvolutionStepReport(
                    task_loss=float(task_loss.detach()),
                    ewc_penalty=float(penalty.detach()),
                    total_loss=float(total.detach()),
                    before_digest=before_digest,
                    after_digest=after_digest,
                    pre_certificates=pre,
                    post_certificates=post,
                )
        except BaseException as error:
            self._restore(snapshot)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise EvolutionTransactionError(
                "evolution candidate failed and was rolled back"
            ) from error


class GenerationCheckpointStore:
    """Hash-linked, immutable generations behind one atomic CURRENT pointer."""

    def __init__(self, root: str | Path) -> None:
        source = Path(root).expanduser()
        source.mkdir(parents=True, exist_ok=True)
        self.root = source.resolve(strict=True)
        if self.root.is_symlink():
            raise ValueError("checkpoint root must not be a symbolic link")

    @property
    def current_pointer(self) -> Path:
        return self.root / "CURRENT"

    def _generation_numbers(self) -> list[int]:
        result: list[int] = []
        for path in self.root.glob("generation-*.manifest.json"):
            stem = path.name.removeprefix("generation-").removesuffix(".manifest.json")
            if stem.isdigit():
                result.append(int(stem))
        return sorted(result)

    @staticmethod
    def _metadata(value: Mapping[str, object] | None) -> dict[str, object]:
        metadata = dict(value or {})
        encoded = json.dumps(metadata, ensure_ascii=True, sort_keys=True).encode(
            "utf-8"
        )
        if len(encoded) > MAX_METADATA_BYTES:
            raise ValueError("checkpoint metadata exceeds its fixed byte bound")
        if any(not isinstance(key, str) or not key for key in metadata):
            raise TypeError("checkpoint metadata keys must be non-empty text")
        return metadata

    def write(
        self,
        transaction: EvolutionTransaction,
        certificates: Mapping[str, CFireCertificate],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> GenerationRecord:
        if transaction.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError("generation writes are allowed only in evolution mode")
        generations = self._generation_numbers()
        if len(generations) >= MAX_GENERATIONS:
            raise GenerationCheckpointError("generation retention bound was reached")
        generation = 1 if not generations else generations[-1] + 1
        current = self.read_current(required=False)
        parent = None if current is None else current.checkpoint_sha256
        base = f"generation-{generation:08d}"
        checkpoint = self.root / f"{base}.pt"
        manifest = self.root / f"{base}.manifest.json"
        if checkpoint.exists() or manifest.exists():
            raise GenerationCheckpointError("generation path already exists")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "generation": generation,
            "parent_checkpoint_sha256": parent,
            "model": _clone_state(transaction.model.state_dict()),
            "optimizer": deepcopy(transaction.optimizer.state_dict()),
            "fisher": [
                {
                    "fisher": _clone_state(snapshot.fisher),
                    "anchor": _clone_state(snapshot.anchor),
                    "n_samples": snapshot.n_samples,
                    "quadratic_offset": snapshot.quadratic_offset,
                }
                for snapshot in transaction.regularizer.snapshots
            ],
            "c_fire": {
                name: asdict(certificate)
                for name, certificate in sorted(certificates.items())
            },
            "metadata": self._metadata(metadata),
        }
        checkpoint_tmp = self.root / f".{base}.pt.tmp"
        manifest_tmp = self.root / f".{base}.manifest.json.tmp"
        pointer_tmp = self.root / ".CURRENT.tmp"
        with transaction.rhythm.evolution_slot():
            try:
                torch.save(payload, checkpoint_tmp)
                size = checkpoint_tmp.stat().st_size
                if not 0 < size <= MAX_CHECKPOINT_BYTES:
                    raise GenerationCheckpointError(
                        "generation checkpoint crossed its byte bound"
                    )
                checkpoint_digest = sha256(checkpoint_tmp.read_bytes()).hexdigest()
                manifest_payload = {
                    "schema": CHECKPOINT_SCHEMA,
                    "generation": generation,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": checkpoint_digest,
                    "parent_checkpoint_sha256": parent,
                }
                manifest_bytes = json.dumps(
                    manifest_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                manifest_tmp.write_bytes(manifest_bytes)
                manifest_digest = sha256(manifest_bytes).hexdigest()
                pointer_payload = {
                    "schema": POINTER_SCHEMA,
                    "manifest": manifest.name,
                    "manifest_sha256": manifest_digest,
                }
                pointer_tmp.write_text(
                    json.dumps(
                        pointer_payload,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="ascii",
                )
                os.replace(checkpoint_tmp, checkpoint)
                os.replace(manifest_tmp, manifest)
                os.replace(pointer_tmp, self.current_pointer)
            except BaseException:
                for temporary in (checkpoint_tmp, manifest_tmp, pointer_tmp):
                    temporary.unlink(missing_ok=True)
                raise
        return GenerationRecord(
            generation,
            checkpoint,
            checkpoint_digest,
            manifest,
            manifest_digest,
            parent,
        )

    def read_current(self, *, required: bool = True) -> GenerationRecord | None:
        pointer = self.current_pointer
        if not pointer.exists():
            if required:
                raise GenerationCheckpointError("CURRENT pointer is missing")
            return None
        if pointer.is_symlink() or not pointer.is_file():
            raise GenerationCheckpointError("CURRENT pointer must be a regular file")
        try:
            data = json.loads(pointer.read_text(encoding="ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise GenerationCheckpointError("CURRENT pointer is malformed") from error
        if set(data) != {"schema", "manifest", "manifest_sha256"}:
            raise GenerationCheckpointError("CURRENT pointer schema is not exact")
        if data["schema"] != POINTER_SCHEMA:
            raise GenerationCheckpointError("CURRENT pointer version is unsupported")
        manifest_name = data["manifest"]
        if (
            not isinstance(manifest_name, str)
            or Path(manifest_name).name != manifest_name
            or not manifest_name.endswith(".manifest.json")
        ):
            raise GenerationCheckpointError("CURRENT manifest path escaped its root")
        manifest = (self.root / manifest_name).resolve(strict=True)
        if manifest.parent != self.root or manifest.is_symlink():
            raise GenerationCheckpointError("generation manifest escaped its root")
        manifest_bytes = manifest.read_bytes()
        manifest_digest = sha256(manifest_bytes).hexdigest()
        if manifest_digest != data["manifest_sha256"]:
            raise GenerationCheckpointError("generation manifest digest changed")
        try:
            item = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise GenerationCheckpointError(
                "generation manifest is malformed"
            ) from error
        expected_keys = {
            "schema",
            "generation",
            "checkpoint",
            "checkpoint_sha256",
            "parent_checkpoint_sha256",
        }
        if set(item) != expected_keys or item["schema"] != CHECKPOINT_SCHEMA:
            raise GenerationCheckpointError("generation manifest schema is not exact")
        generation = item["generation"]
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
        ):
            raise GenerationCheckpointError("generation number is invalid")
        checkpoint_name = item["checkpoint"]
        if (
            not isinstance(checkpoint_name, str)
            or Path(checkpoint_name).name != checkpoint_name
        ):
            raise GenerationCheckpointError("checkpoint path escaped its root")
        checkpoint = (self.root / checkpoint_name).resolve(strict=True)
        if checkpoint.parent != self.root or checkpoint.is_symlink():
            raise GenerationCheckpointError("generation checkpoint escaped its root")
        checkpoint_digest = sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint_digest != item["checkpoint_sha256"]:
            raise GenerationCheckpointError("generation checkpoint digest changed")
        return GenerationRecord(
            generation,
            checkpoint,
            checkpoint_digest,
            manifest,
            manifest_digest,
            item["parent_checkpoint_sha256"],
        )

    def restore_current(self, transaction: EvolutionTransaction) -> GenerationRecord:
        if transaction.rhythm.mode not in {
            SystemMode.EVOLUTION,
            SystemMode.ROLLING_BACK,
        }:
            raise RuntimeError("generation restore is allowed only at night")
        record = self.read_current()
        assert record is not None
        snapshot = transaction._snapshot()
        context = (
            transaction.rhythm.evolution_slot()
            if transaction.rhythm.mode == SystemMode.EVOLUTION
            else _NullContext()
        )
        try:
            payload = torch.load(
                record.checkpoint, map_location="cpu", weights_only=True
            )
            required = {
                "schema",
                "generation",
                "parent_checkpoint_sha256",
                "model",
                "optimizer",
                "fisher",
                "c_fire",
                "metadata",
            }
            if set(payload) != required or payload["schema"] != CHECKPOINT_SCHEMA:
                raise GenerationCheckpointError(
                    "generation payload schema is not exact"
                )
            if payload["generation"] != record.generation:
                raise GenerationCheckpointError("generation payload number changed")
            _validate_state_for(transaction.model, payload["model"])
            restored_fisher = [
                FisherSnapshot(
                    fisher=item["fisher"],
                    anchor=item["anchor"],
                    n_samples=item["n_samples"],
                    quadratic_offset=item["quadratic_offset"],
                )
                for item in payload["fisher"]
            ]
            if set(payload["c_fire"]) != {
                target.name for target in transaction.targets
            }:
                raise GenerationCheckpointError(
                    "generation C-FIRE certificate set changed"
                )
            with context:
                transaction.model.load_state_dict(payload["model"], strict=True)
                transaction.optimizer.load_state_dict(payload["optimizer"])
                if not _finite_optimizer_state(transaction.optimizer.state_dict()):
                    raise GenerationCheckpointError(
                        "restored optimizer contains non-finite state"
                    )
                transaction.regularizer.snapshots = _clone_fisher(restored_fisher)
                transaction._verify_targets()
        except BaseException as error:
            transaction._restore(snapshot)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise GenerationCheckpointError(
                "generation restore failed and runtime was rolled back"
            ) from error
        return record


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def continual_transfer_metrics(scores: Tensor, baselines: Tensor) -> TransferMetrics:
    """Compute BWT/FWT from one finite lower-triangular evaluation matrix."""

    if not isinstance(scores, Tensor) or not isinstance(baselines, Tensor):
        raise TypeError("scores and baselines must be tensors")
    if (
        scores.ndim != 2
        or scores.shape[0] != scores.shape[1]
        or scores.shape[0] < 2
        or scores.shape[0] > 64
        or tuple(baselines.shape) != (scores.shape[0],)
    ):
        raise ValueError("transfer inputs have invalid bounded shapes")
    values = scores.detach().to("cpu", dtype=torch.float64)
    reference = baselines.detach().to("cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or not bool(
        torch.isfinite(reference).all()
    ):
        raise ValueError("transfer inputs must be finite")
    tasks = values.shape[0]
    bwt = torch.stack(
        [values[-1, task] - values[task, task] for task in range(tasks - 1)]
    ).mean()
    fwt = torch.stack(
        [values[task - 1, task] - reference[task] for task in range(1, tasks)]
    ).mean()
    return TransferMetrics(float(bwt), float(fwt))


def summarize_seeded_transfer(
    results: Sequence[TransferMetrics],
) -> SeededTransferSummary:
    if not 3 <= len(results) <= 128:
        raise ValueError("BWT/FWT evidence requires 3..128 seeded runs")
    values = torch.tensor(
        [[result.backward_transfer, result.forward_transfer] for result in results],
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("seeded transfer metrics must be finite")
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=True)
    return SeededTransferSummary(
        len(results),
        float(mean[0]),
        float(std[0]),
        float(mean[1]),
        float(std[1]),
    )


__all__ = [
    "CFireTarget",
    "CHECKPOINT_SCHEMA",
    "EvolutionStepReport",
    "EvolutionTransaction",
    "EvolutionTransactionError",
    "GenerationCheckpointError",
    "GenerationCheckpointStore",
    "GenerationRecord",
    "SeededTransferSummary",
    "TransferMetrics",
    "continual_transfer_metrics",
    "summarize_seeded_transfer",
]
