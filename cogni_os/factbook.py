"""Signed local model facts and honest runtime capability disclosure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import prod
from pathlib import Path
from typing import Any

from .artifacts import VerifiedArtifactSet, verify_artifact_manifest
from .capabilities import CapabilityRegistry, baseline_capability_registry


MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
MAX_PROMPT_CONTEXT_CHARS = 8_192
MAX_PROMPT_CONTEXT_CAPABILITIES = 64
_EMBEDDING_TABLE_SUFFIXES = (
    "embed_tokens.weight",
    "embed_tokens_per_layer.weight",
)


class FactBookError(RuntimeError):
    """Raised when artifact facts cannot be proven from bounded local data."""


@dataclass(frozen=True, slots=True)
class TensorInventory:
    tensor_count: int
    stored_parameters: int
    effective_parameters: int | None
    embedding_parameters: int
    dtype_parameters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.tensor_count < 1 or self.stored_parameters < 1:
            raise ValueError("tensor inventory cannot be empty")
        if not 0 <= self.embedding_parameters <= self.stored_parameters:
            raise ValueError("embedding parameter count is invalid")
        if self.effective_parameters is not None and not (
            0 < self.effective_parameters <= self.stored_parameters
        ):
            raise ValueError("effective parameter count is invalid")
        if sum(count for _, count in self.dtype_parameters) != self.stored_parameters:
            raise ValueError("dtype parameter counts must equal stored parameters")


@dataclass(frozen=True, slots=True)
class ModelArtifactFacts:
    label: str
    architecture: str
    hidden_size: int
    layers: int
    dense: bool
    inventory: TensorInventory
    manifest_sha256: str
    config_sha256: str

    def as_payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "dense": self.dense,
            "stored_parameters": self.inventory.stored_parameters,
            "effective_parameters": self.inventory.effective_parameters,
            "embedding_parameters": self.inventory.embedding_parameters,
            "tensor_count": self.inventory.tensor_count,
            "dtype_parameters": dict(self.inventory.dtype_parameters),
            "manifest_sha256": self.manifest_sha256,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeFactBook:
    schema_version: int
    generated_at: str
    build_version: str
    device: str
    target_device: str
    model: ModelArtifactFacts
    capabilities: CapabilityRegistry

    def as_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "build_version": self.build_version,
            "device": self.device,
            "target_device": self.target_device,
            "model": self.model.as_payload(),
            "capabilities": self.capabilities.as_payload(),
        }

    def identity_summary_ko(self) -> str:
        inventory = self.model.inventory
        if inventory.effective_parameters is None:
            parameter_text = f"저장 파라미터 {inventory.stored_parameters:,}개"
        else:
            parameter_text = (
                f"effective 파라미터 {inventory.effective_parameters:,}개와 "
                f"임베딩을 포함한 저장 파라미터 {inventory.stored_parameters:,}개"
            )
        structure = "dense" if self.model.dense else "expert/MoE"
        return (
            f"저는 Cogni-OS 2.0에서 실행되는 Cogni Agent입니다. 로컬 백본은 "
            f"{structure} {self.model.label}이며, {parameter_text}를 가집니다. "
            "모듈의 실제 권한은 Runtime Fact-book의 capability 상태를 기준으로 설명합니다."
        )

    def prompt_context_ko(self) -> str:
        """Return a bounded Korean fact block for the local model prompt.

        Capability enum values remain verbatim so a natural-language
        paraphrase cannot promote an advisory, gated, night-only, or
        proposal-only module into an authoritative runtime feature.
        """

        records = self.capabilities.records
        if len(records) > MAX_PROMPT_CONTEXT_CAPABILITIES:
            raise FactBookError("prompt capability context exceeds its record bound")

        model = self.model
        inventory = model.inventory
        effective = (
            "미확정"
            if inventory.effective_parameters is None
            else _prompt_count(
                inventory.effective_parameters,
                "effective_parameters",
            )
        )
        lines = [
            "[Runtime Fact-book: 아래 사실과 권한 상태를 추측하거나 과장하지 마십시오.]",
            (
                "정체성: Cogni-OS 2.0의 Cogni Agent; "
                f"build={_prompt_scalar(self.build_version, 'build_version')}; "
                f"device={_prompt_scalar(self.device, 'device')}; "
                f"target_device={_prompt_scalar(self.target_device, 'target_device')}"
            ),
            (
                f"검증 모델: label={_prompt_scalar(model.label, 'model.label')}; "
                f"architecture={_prompt_scalar(model.architecture, 'model.architecture')}; "
                f"structure={'dense' if model.dense else 'expert/MoE'}; "
                f"hidden_size={_prompt_count(model.hidden_size, 'hidden_size')}; "
                f"layers={_prompt_count(model.layers, 'layers')}"
            ),
            (
                "파라미터: "
                f"stored_parameters={_prompt_count(inventory.stored_parameters, 'stored_parameters')}; "
                f"effective_parameters={effective}; "
                f"embedding_parameters={_prompt_count(inventory.embedding_parameters, 'embedding_parameters')}; "
                f"tensor_count={_prompt_count(inventory.tensor_count, 'tensor_count')}"
            ),
            "기능 권한(Runtime Fact-book 원문 상태):",
        ]
        for record in records:
            lines.append(
                f"- {record.name}: state={record.state.value}; "
                f"evidence={record.evidence.value}; "
                f"answer_bearing={'true' if record.answer_bearing else 'false'}; "
                "runtime_mutation_allowed="
                f"{'true' if record.runtime_mutation_allowed else 'false'}"
            )
        lines.append(
            "해석 규칙: 현재 답변 영향은 answer_bearing, 자동 변경 권한은 "
            "runtime_mutation_allowed를 따릅니다. evidence나 구현 존재만으로 "
            "advisory/gated/night_only/proposal_only 상태를 승격해 해석하지 마십시오."
        )
        context = "\n".join(lines)
        if len(context) > MAX_PROMPT_CONTEXT_CHARS:
            raise FactBookError("prompt fact context exceeds its character bound")
        return context


def _prompt_scalar(value: object, label: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise FactBookError(f"{label} is unsafe for prompt context")
    if any(ord(character) < 32 and not character.isspace() for character in value):
        raise FactBookError(f"{label} contains a prompt control character")
    collapsed = " ".join(value.split())
    if not collapsed:
        raise FactBookError(f"{label} is empty after normalization")
    return collapsed


def _prompt_count(value: object, label: str) -> str:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value.bit_length() > 256
    ):
        raise FactBookError(f"{label} is unsafe for prompt context")
    return f"{value:,}"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_parameter_count(shape: object, name: str) -> int:
    if not isinstance(shape, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in shape
    ):
        raise FactBookError(f"invalid tensor shape in safetensors header: {name}")
    return prod(shape) if shape else 1


def inspect_safetensors_headers(paths: tuple[Path, ...]) -> TensorInventory:
    """Count parameters from bounded safetensors headers without loading weights."""

    if not paths:
        raise FactBookError("no verified safetensors weights were supplied")
    names: set[str] = set()
    stored = 0
    embedding = 0
    tensor_count = 0
    dtype_counts: dict[str, int] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise FactBookError("safetensors input must be a verified regular file")
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise FactBookError(f"truncated safetensors prefix: {path.name}")
            header_size = int.from_bytes(prefix, "little", signed=False)
            if not 2 <= header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise FactBookError(f"unsafe safetensors header size: {path.name}")
            if 8 + header_size > file_size:
                raise FactBookError(
                    f"safetensors header exceeds file size: {path.name}"
                )
            encoded = stream.read(header_size)
        try:
            header = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FactBookError(
                f"invalid safetensors JSON header: {path.name}"
            ) from exc
        if not isinstance(header, dict):
            raise FactBookError("safetensors header must be an object")
        data_bytes = file_size - 8 - header_size
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not name or name in names:
                raise FactBookError("safetensors tensor names must be unique")
            if not isinstance(entry, dict):
                raise FactBookError(f"invalid safetensors entry: {name}")
            dtype = entry.get("dtype")
            offsets = entry.get("data_offsets")
            if not isinstance(dtype, str) or not dtype or len(dtype) > 16:
                raise FactBookError(f"invalid tensor dtype: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in offsets
                )
                or offsets[0] > offsets[1]
                or offsets[1] > data_bytes
            ):
                raise FactBookError(f"invalid tensor data offsets: {name}")
            count = _tensor_parameter_count(entry.get("shape"), name)
            names.add(name)
            stored += count
            tensor_count += 1
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
            if name.endswith(_EMBEDDING_TABLE_SUFFIXES):
                embedding += count
    effective = stored - embedding if embedding else None
    return TensorInventory(
        tensor_count=tensor_count,
        stored_parameters=stored,
        effective_parameters=effective,
        embedding_parameters=embedding,
        dtype_parameters=tuple(sorted(dtype_counts.items())),
    )


def _required_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FactBookError(f"model config requires positive integer {key}")
    return value


def build_runtime_factbook(
    model_root: str | Path,
    manifest: str | Path,
    *,
    build_version: str,
    device: str,
    target_device: str = "RTX 4090 24GB",
    capabilities: CapabilityRegistry | None = None,
    generated_at: str | None = None,
) -> RuntimeFactBook:
    """Build facts only after verifying the complete local artifact manifest."""

    verified = verify_artifact_manifest(model_root, manifest)
    return build_runtime_factbook_from_verified(
        verified,
        manifest,
        build_version=build_version,
        device=device,
        target_device=target_device,
        capabilities=capabilities,
        generated_at=generated_at,
    )


def build_runtime_factbook_from_verified(
    verified: VerifiedArtifactSet,
    manifest: str | Path,
    *,
    build_version: str,
    device: str,
    target_device: str = "RTX 4090 24GB",
    capabilities: CapabilityRegistry | None = None,
    generated_at: str | None = None,
) -> RuntimeFactBook:
    """Build a Fact-book from an already verified artifact transaction.

    Product startup verifies multi-gigabyte weights once.  Passing that exact
    immutable result here avoids a second full-file hash while still preventing
    callers from constructing facts from unverified paths.
    """

    if not isinstance(verified, VerifiedArtifactSet):
        raise TypeError("verified must be a VerifiedArtifactSet")
    by_name = {path.name: path for path in verified.files}
    config_path = by_name.get("config.json")
    if config_path is None:
        raise FactBookError("verified artifact set does not contain config.json")
    weights = tuple(
        path for path in verified.files if path.name.endswith(".safetensors")
    )
    inventory = inspect_safetensors_headers(weights)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactBookError("model config is not valid UTF-8 JSON") from exc
    if not isinstance(config, dict):
        raise FactBookError("model config must be an object")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise FactBookError("text_config must be an object")
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not isinstance(architectures[0], str)
    ):
        raise FactBookError("model config requires an architecture")
    experts = text_config.get("num_experts")
    moe_enabled = text_config.get("enable_moe_block") is True
    dense = not moe_enabled and experts in {None, 0}
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    for label, value in {
        "build_version": build_version,
        "device": device,
        "target_device": target_device,
        "generated_at": timestamp,
    }.items():
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise ValueError(f"{label} must contain 1-128 characters")
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    model_facts = ModelArtifactFacts(
        label=verified.root.name,
        architecture=architectures[0],
        hidden_size=_required_positive_int(text_config, "hidden_size"),
        layers=_required_positive_int(text_config, "num_hidden_layers"),
        dense=dense,
        inventory=inventory,
        manifest_sha256=_sha256_file(manifest_path),
        config_sha256=_sha256_file(config_path),
    )
    return RuntimeFactBook(
        schema_version=1,
        generated_at=timestamp,
        build_version=build_version,
        device=device,
        target_device=target_device,
        model=model_facts,
        capabilities=capabilities or baseline_capability_registry(),
    )


__all__ = [
    "FactBookError",
    "MAX_PROMPT_CONTEXT_CAPABILITIES",
    "MAX_PROMPT_CONTEXT_CHARS",
    "MAX_SAFETENSORS_HEADER_BYTES",
    "ModelArtifactFacts",
    "RuntimeFactBook",
    "TensorInventory",
    "build_runtime_factbook",
    "build_runtime_factbook_from_verified",
    "inspect_safetensors_headers",
]
