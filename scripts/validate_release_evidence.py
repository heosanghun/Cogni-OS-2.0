"""Fail-closed validation of independently signed release evidence.

The validator intentionally imports only the Python standard library.  The release
builder executes these reviewed bytes from stdin with ``python -I -S -B`` and verifies that their
bytes are identical to the file in the exact clean commit being published.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


GPU5_UUID = "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
IMAGE_DIGEST = (
    "cogni-os-dev@sha256:"
    "20aaf1d7cde8d6a504ba08f158a34a1907eac9413f3578acc4637f0a1b2ec8ba"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
HEX_RE = re.compile(r"^[0-9a-f]+$")
MODEL_FILE_RE = re.compile(r'^"([^"\\/]+)"\s*=\s*"([0-9a-f]{64})"$')
RSA_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


class ReleaseEvidenceError(ValueError):
    """The publication evidence does not satisfy the closed contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ReleaseEvidenceError(f"{label} must be an exact lowercase SHA-256")
    return value


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ReleaseEvidenceError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ReleaseEvidenceError(
            f"{label} keys differ: missing={sorted(keys - actual)}, "
            f"unexpected={sorted(actual - keys)}"
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReleaseEvidenceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ReleaseEvidenceError(f"non-finite JSON number is forbidden: {value}")


def _read_pinned(path: Path, expected_sha256: str, label: str) -> bytes:
    _require_digest(expected_sha256, f"{label} pinned digest")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReleaseEvidenceError(f"cannot read {label}: {exc}") from exc
    if _sha256(data) != expected_sha256:
        raise ReleaseEvidenceError(f"{label} digest does not match the pinned digest")
    return data


def _load_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[dict[str, Any], bytes]:
    data = _read_pinned(path, expected_sha256, label)
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvidenceError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ReleaseEvidenceError(f"{label} root must be an object")
    return value, data


def _validate_verifier_policy(value: dict[str, Any]) -> tuple[str, str]:
    policy = _require_exact_keys(
        value,
        {"schema", "status", "verifier_id", "public_key_sha256"},
        "release verifier policy",
    )
    if policy["schema"] != "cogni.release.verifier-policy.v1":
        raise ReleaseEvidenceError("release verifier policy schema is invalid")
    if policy["status"] == "unconfigured":
        if policy["verifier_id"] is not None or policy["public_key_sha256"] is not None:
            raise ReleaseEvidenceError(
                "unconfigured verifier policy must contain null pins"
            )
        raise ReleaseEvidenceError(
            "release verifier policy has no independently approved key; publication is blocked"
        )
    if (
        policy["status"] != "approved"
        or type(policy["verifier_id"]) is not str
        or IDENTIFIER_RE.fullmatch(policy["verifier_id"]) is None
    ):
        raise ReleaseEvidenceError("release verifier policy approval is invalid")
    return policy["verifier_id"], _require_digest(
        policy["public_key_sha256"], "approved verifier public key"
    )


def _parse_model_manifest_files(data: bytes) -> dict[str, str]:
    try:
        lines = data.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise ReleaseEvidenceError("model manifest is not strict UTF-8") from exc
    section = ""
    files: dict[str, str] = {}
    saw_files = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            if section == "files":
                if saw_files:
                    raise ReleaseEvidenceError("model manifest repeats [files]")
                saw_files = True
            continue
        if section != "files":
            continue
        match = MODEL_FILE_RE.fullmatch(line)
        if match is None:
            raise ReleaseEvidenceError("model manifest [files] entry is invalid")
        name, digest = match.groups()
        if (
            name in files
            or name in {".", ".."}
            or name.endswith(".")
            or name.endswith(" ")
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
        ):
            raise ReleaseEvidenceError("model manifest contains an unsafe file name")
        files[name] = digest
    if not saw_files or not files:
        raise ReleaseEvidenceError("model manifest has no bounded [files] inventory")
    return files


def _canonical_model_tree_digest(files: dict[str, str]) -> str:
    encoded = "".join(f"{files[name]}  {name}\n" for name in sorted(files)).encode(
        "utf-8"
    )
    return _sha256(encoded)


def _safe_source_name(encoded_name: bytes) -> str:
    try:
        name = encoded_name.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReleaseEvidenceError("source path is not canonical UTF-8") from exc
    relative = Path(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or relative.is_absolute()
        or ".." in relative.parts
        or ".git" in relative.parts
        or "__pycache__" in relative.parts
        or relative.suffix.casefold() in {".pyc", ".pyo"}
        or relative.as_posix() != name
    ):
        raise ReleaseEvidenceError("source path is unsafe for canonical evidence")
    return name


def _run_git(
    git_executable: Path,
    git_executable_sha256: str,
    repo: Path,
    arguments: tuple[str, ...],
) -> bytes:
    try:
        executable = git_executable.resolve(strict=True)
    except OSError as exc:
        raise ReleaseEvidenceError("pinned Git application does not exist") from exc
    if (
        not executable.is_file()
        or executable.is_symlink()
        or executable != git_executable
    ):
        raise ReleaseEvidenceError(
            "Git application path is not one canonical regular file"
        )
    _read_pinned(executable, git_executable_sha256, "Git application")
    environment = {
        "PATH": str(executable.parent),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        completed = subprocess.run(
            (
                str(executable),
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                "credential.helper=",
                "-C",
                str(repo),
                *arguments,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseEvidenceError("trusted Git source inventory failed") from exc
    if completed.returncode != 0:
        raise ReleaseEvidenceError("trusted Git source inventory returned failure")
    return completed.stdout


def compute_guard_source_tree_digest(
    *,
    source_repo: Path,
    expanded_source: Path,
    expected_commit: str,
    git_executable: Path,
    git_executable_sha256: str,
) -> str:
    """Reproduce the GPU5 guard content digest from actual archived bytes.

    The canonical records are exactly ``[path, sha256, git_mode, git_blob_oid]``
    in compact ASCII JSON, matching ``gpu5_boundary_guard._canonical_scope_digest``.
    ``SOURCE_TREE_SHA256SUMS.txt`` is a separate human-readable artifact and is
    deliberately not used as the GPU evidence scope digest.
    """

    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise ReleaseEvidenceError("expected source commit must be lowercase 40-hex")
    try:
        repo = source_repo.resolve(strict=True)
        expanded = expanded_source.resolve(strict=True)
    except OSError as exc:
        raise ReleaseEvidenceError("source inventory roots must exist") from exc
    if not repo.is_dir() or not expanded.is_dir():
        raise ReleaseEvidenceError("source inventory roots must be directories")
    head = (
        _run_git(
            git_executable,
            git_executable_sha256,
            repo,
            ("rev-parse", "--verify", "HEAD"),
        )
        .decode("ascii")
        .strip()
    )
    if head != expected_commit:
        raise ReleaseEvidenceError("source repository HEAD differs from release commit")
    if _run_git(
        git_executable,
        git_executable_sha256,
        repo,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    ):
        raise ReleaseEvidenceError("source repository is not a completely clean HEAD")
    object_format = (
        _run_git(
            git_executable,
            git_executable_sha256,
            repo,
            ("rev-parse", "--show-object-format"),
        )
        .decode("ascii")
        .strip()
    )
    if object_format not in {"sha1", "sha256"}:
        raise ReleaseEvidenceError("unsupported Git object format")
    raw_tree = _run_git(
        git_executable,
        git_executable_sha256,
        repo,
        ("ls-tree", "-r", "-z", "--full-tree", expected_commit),
    )
    oid_pattern = re.compile(
        rb"[0-9a-f]{40}" if object_format == "sha1" else rb"[0-9a-f]{64}"
    )
    entries: dict[str, tuple[str, str]] = {}
    for record in (item for item in raw_tree.split(b"\0") if item):
        try:
            metadata, encoded_name = record.split(b"\t", 1)
            mode, kind, oid = metadata.split(b" ", 2)
        except ValueError as exc:
            raise ReleaseEvidenceError("Git commit tree row is malformed") from exc
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            raise ReleaseEvidenceError(
                "release source permits regular 100644/100755 blobs only"
            )
        if oid_pattern.fullmatch(oid) is None:
            raise ReleaseEvidenceError("Git commit tree returned an invalid blob id")
        name = _safe_source_name(encoded_name)
        if name in entries:
            raise ReleaseEvidenceError("Git commit tree contains a duplicate path")
        entries[name] = (mode.decode("ascii"), oid.decode("ascii"))
    if not entries:
        raise ReleaseEvidenceError("release commit contains no regular files")
    tagged = _run_git(
        git_executable,
        git_executable_sha256,
        repo,
        ("ls-files", "-v", "-z", "--cached"),
    )
    observed_index: list[str] = []
    for record in (item for item in tagged.split(b"\0") if item):
        if len(record) < 3 or record[:2] != b"H ":
            raise ReleaseEvidenceError(
                "assume-unchanged, skip-worktree, or non-normal index state is forbidden"
            )
        observed_index.append(_safe_source_name(record[2:]))
    if len(observed_index) != len(entries) or set(observed_index) != set(entries):
        raise ReleaseEvidenceError("working index differs from the release commit tree")

    observed: set[str] = set()
    for candidate in expanded.rglob("*"):
        if candidate.is_symlink():
            raise ReleaseEvidenceError("expanded source contains a symbolic link")
        if candidate.is_file():
            relative = candidate.relative_to(expanded).as_posix()
            _safe_source_name(relative.encode("utf-8"))
            observed.add(relative)
    if observed != set(entries):
        raise ReleaseEvidenceError(
            "expanded archive inventory differs from the commit tree"
        )

    records: list[list[str]] = []
    for name in sorted(entries):
        mode, expected_oid = entries[name]
        path = expanded.joinpath(*name.split("/"))
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ReleaseEvidenceError(
                f"cannot read expanded source file: {name}"
            ) from exc
        blob = hashlib.new(object_format)
        blob.update(f"blob {len(data)}\0".encode("ascii"))
        blob.update(data)
        if blob.hexdigest() != expected_oid:
            raise ReleaseEvidenceError(
                f"expanded source blob differs from commit: {name}"
            )
        records.append([name, _sha256(data), mode, expected_oid])
    encoded = json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    digest = _sha256(encoded)
    _read_pinned(git_executable, git_executable_sha256, "Git application postcheck")
    return digest


def _canonical_device_digest(device: dict[str, Any]) -> str:
    scope = {
        field: device[field]
        for field in (
            "source_commit",
            "source_tree_digest",
            "physical_gpu_index",
            "gpu_uuid",
            "image_digest",
            "identity_pre_sha256",
            "identity_post_sha256",
        )
    }
    encoded = (
        json.dumps(scope, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return _sha256(encoded)


def _validate_cpu(
    value: dict[str, Any], expected_commit: str, expected_source_tree: str
) -> None:
    cpu = _require_exact_keys(
        value,
        {
            "schema",
            "status",
            "source_commit",
            "source_tree_digest",
            "required_gates",
            "pytest_passed",
            "pytest_failed",
            "stdout_sha256",
            "stderr_sha256",
        },
        "CPU evidence",
    )
    if (
        cpu["schema"] != "cogni.cpu.gates.v1"
        or cpu["status"] != "passed"
        or cpu["source_commit"] != expected_commit
        or cpu["source_tree_digest"] != expected_source_tree
    ):
        raise ReleaseEvidenceError("CPU evidence identity, tree, or status is invalid")
    gates = _require_exact_keys(
        cpu["required_gates"],
        {
            "ruff_check",
            "ruff_format_check",
            "pytest",
            "node_syntax_check",
            "clean_same_commit",
        },
        "CPU required_gates",
    )
    if any(type(item) is not bool or item is not True for item in gates.values()):
        raise ReleaseEvidenceError("every CPU gate must be the JSON boolean true")
    if (
        type(cpu["pytest_passed"]) is not int
        or cpu["pytest_passed"] < 1
        or type(cpu["pytest_failed"]) is not int
        or cpu["pytest_failed"] != 0
    ):
        raise ReleaseEvidenceError("CPU pytest counts are invalid")
    for field in ("source_tree_digest", "stdout_sha256", "stderr_sha256"):
        _require_digest(cpu[field], f"CPU {field}")


def _validate_identity(
    value: dict[str, Any], *, phase: str, commit: str, source_tree: str
) -> None:
    identity = _require_exact_keys(
        value,
        {
            "schema",
            "status",
            "phase",
            "source_commit",
            "source_tree_digest",
            "physical_gpu_index",
            "gpu_uuid",
            "image_digest",
        },
        f"GPU5 identity-{phase}",
    )
    if (
        identity["schema"] != "cogni.gpu5.identity.v1"
        or identity["status"] != "passed"
        or identity["phase"] != phase
        or identity["source_commit"] != commit
        or identity["source_tree_digest"] != source_tree
        or type(identity["physical_gpu_index"]) is not int
        or identity["physical_gpu_index"] != 5
        or identity["gpu_uuid"] != GPU5_UUID
        or identity["image_digest"] != IMAGE_DIGEST
    ):
        raise ReleaseEvidenceError(f"GPU5 identity-{phase} scope is invalid")


def _expected_runtime_scope(
    *,
    commit: str,
    source_tree: str,
    manifest_sha: str,
    model_tree: str,
    config_digest: str,
    device_digest: str,
) -> dict[str, object]:
    return {
        "source_commit": commit,
        "source_tree_digest": source_tree,
        "model_manifest_sha256": manifest_sha,
        "model_tree_digest": model_tree,
        "config_digest": config_digest,
        "device_digest": device_digest,
        "physical_gpu_index": 5,
        "gpu_uuid": GPU5_UUID,
        "image_digest": IMAGE_DIGEST,
    }


def _validate_runtime_component(
    value: dict[str, Any], *, schema: str, expected_scope: dict[str, object]
) -> None:
    component = _require_exact_keys(
        value,
        {"schema", "status", *expected_scope},
        schema,
    )
    if component["schema"] != schema or component["status"] != "passed":
        raise ReleaseEvidenceError(f"{schema} identity or status is invalid")
    if any(
        type(component[field]) is not type(expected) or component[field] != expected
        for field, expected in expected_scope.items()
    ):
        raise ReleaseEvidenceError(f"{schema} scope does not match source facts")


def _validate_components(
    *,
    runtime: dict[str, Any],
    completion: dict[str, Any],
    identity_pre: dict[str, Any],
    identity_post: dict[str, Any],
    config_evidence: dict[str, Any],
    device_evidence: dict[str, Any],
    model_inventory: dict[str, Any],
    component_shas: dict[str, str],
    commit: str,
    source_tree: str,
    manifest_sha: str,
    manifest_files: dict[str, str],
    config_digest: str,
) -> tuple[str, str]:
    _validate_identity(
        identity_pre, phase="pre", commit=commit, source_tree=source_tree
    )
    _validate_identity(
        identity_post, phase="post", commit=commit, source_tree=source_tree
    )

    inventory = _require_exact_keys(
        model_inventory,
        {
            "schema",
            "status",
            "source_commit",
            "source_tree_digest",
            "model_manifest_sha256",
            "model_tree_digest",
            "files",
        },
        "GPU5 model inventory",
    )
    model_tree = _canonical_model_tree_digest(manifest_files)
    if (
        inventory["schema"] != "cogni.gpu5.model-inventory.v1"
        or inventory["status"] != "passed"
        or inventory["source_commit"] != commit
        or inventory["source_tree_digest"] != source_tree
        or inventory["model_manifest_sha256"] != manifest_sha
        or inventory["model_tree_digest"] != model_tree
        or type(inventory["files"]) is not dict
        or inventory["files"] != manifest_files
    ):
        raise ReleaseEvidenceError(
            "GPU5 model inventory does not match the source manifest"
        )

    config_component = _require_exact_keys(
        config_evidence,
        {
            "schema",
            "status",
            "source_commit",
            "source_tree_digest",
            "config_sha256",
        },
        "GPU5 config evidence",
    )
    if (
        config_component["schema"] != "cogni.gpu5.config.v1"
        or config_component["status"] != "passed"
        or config_component["source_commit"] != commit
        or config_component["source_tree_digest"] != source_tree
        or config_component["config_sha256"] != config_digest
    ):
        raise ReleaseEvidenceError("GPU5 config evidence does not match source config")

    device = _require_exact_keys(
        device_evidence,
        {
            "schema",
            "status",
            "source_commit",
            "source_tree_digest",
            "physical_gpu_index",
            "gpu_uuid",
            "image_digest",
            "identity_pre_sha256",
            "identity_post_sha256",
            "device_digest",
        },
        "GPU5 device evidence",
    )
    if (
        device["schema"] != "cogni.gpu5.device.v1"
        or device["status"] != "passed"
        or device["source_commit"] != commit
        or device["source_tree_digest"] != source_tree
        or type(device["physical_gpu_index"]) is not int
        or device["physical_gpu_index"] != 5
        or device["gpu_uuid"] != GPU5_UUID
        or device["image_digest"] != IMAGE_DIGEST
        or device["identity_pre_sha256"] != component_shas["identity_pre"]
        or device["identity_post_sha256"] != component_shas["identity_post"]
        or device["device_digest"] != _canonical_device_digest(device)
    ):
        raise ReleaseEvidenceError("GPU5 device evidence is not canonical or in scope")
    device_digest = device["device_digest"]
    scope = _expected_runtime_scope(
        commit=commit,
        source_tree=source_tree,
        manifest_sha=manifest_sha,
        model_tree=model_tree,
        config_digest=config_digest,
        device_digest=device_digest,
    )
    _validate_runtime_component(
        runtime, schema="cogni.gpu5.runtime.v1", expected_scope=scope
    )
    completion_scope = dict(scope)
    completion_scope["runtime_evidence_sha256"] = component_shas["runtime"]
    _validate_runtime_component(
        completion,
        schema="cogni.gpu5.completion.v1",
        expected_scope=completion_scope,
    )
    return model_tree, device_digest


def _validate_gpu(
    value: dict[str, Any],
    *,
    expected_commit: str,
    expected_source_tree: str,
    manifest_sha: str,
    model_tree: str,
    config_digest: str,
    device_digest: str,
    component_shas: dict[str, str],
) -> None:
    component_fields = {
        "runtime_evidence_sha256",
        "completion_evidence_sha256",
        "identity_pre_sha256",
        "identity_post_sha256",
        "config_evidence_sha256",
        "device_evidence_sha256",
        "model_inventory_sha256",
    }
    gpu = _require_exact_keys(
        value,
        {
            "schema",
            "status",
            "source_commit",
            "source_tree_digest",
            "model_manifest_sha256",
            "model_tree_digest",
            "config_digest",
            "device_digest",
            "physical_gpu_index",
            "gpu_uuid",
            "image_digest",
            "required_gates",
            *component_fields,
        },
        "GPU5 evidence",
    )
    if (
        gpu["schema"] != "cogni.gpu5.gates.v2"
        or gpu["status"] != "passed"
        or gpu["source_commit"] != expected_commit
        or gpu["source_tree_digest"] != expected_source_tree
        or gpu["model_manifest_sha256"] != manifest_sha
        or gpu["model_tree_digest"] != model_tree
        or gpu["config_digest"] != config_digest
        or gpu["device_digest"] != device_digest
        or type(gpu["physical_gpu_index"]) is not int
        or gpu["physical_gpu_index"] != 5
        or gpu["gpu_uuid"] != GPU5_UUID
        or gpu["image_digest"] != IMAGE_DIGEST
    ):
        raise ReleaseEvidenceError(
            "GPU5 evidence identity, component scope, or status is invalid"
        )
    gates = _require_exact_keys(
        gpu["required_gates"], {"runtime", "completion"}, "GPU5 required_gates"
    )
    if gates != {"runtime": "passed", "completion": "passed"}:
        raise ReleaseEvidenceError("GPU5 runtime and completion gates must both pass")
    for field in component_fields:
        _require_digest(gpu[field], f"GPU5 {field}")
        key = field.removesuffix("_sha256").removesuffix("_evidence")
        if key == "model_inventory":
            expected = component_shas["model_inventory"]
        elif key == "config":
            expected = component_shas["config"]
        elif key == "device":
            expected = component_shas["device"]
        else:
            expected = component_shas[key]
        if gpu[field] != expected:
            raise ReleaseEvidenceError(
                f"GPU5 {field} does not pin supplied component bytes"
            )


def _validate_summary(value: dict[str, Any], expected_commit: str) -> None:
    summary = _require_exact_keys(
        value,
        {"schema", "status", "source_commit", "cpu", "gpu5", "independent_verifier"},
        "release summary",
    )
    if (
        summary["schema"] != "cogni.release.gates.v2"
        or summary["status"] != "passed"
        or summary["source_commit"] != expected_commit
    ):
        raise ReleaseEvidenceError("release summary identity or status is invalid")
    _require_exact_keys(
        summary["cpu"],
        {
            "status",
            "evidence_sha256",
            "source_tree_digest",
            "required_gates",
            "pytest_passed",
            "pytest_failed",
        },
        "release summary CPU section",
    )
    _require_exact_keys(
        summary["gpu5"],
        {
            "status",
            "evidence_sha256",
            "required_gates",
            "physical_gpu_index",
            "gpu_uuid",
            "image_digest",
            "model_manifest_sha256",
            "source_tree_digest",
            "model_tree_digest",
            "config_digest",
            "device_digest",
            "runtime_evidence_sha256",
            "completion_evidence_sha256",
            "identity_pre_sha256",
            "identity_post_sha256",
            "config_evidence_sha256",
            "device_evidence_sha256",
            "model_inventory_sha256",
        },
        "release summary GPU5 section",
    )
    _require_exact_keys(
        summary["independent_verifier"],
        {"status", "verifier_id"},
        "release summary independent_verifier section",
    )


def _validate_attestation(value: dict[str, Any], expected_commit: str) -> None:
    attestation = _require_exact_keys(
        value,
        {
            "schema",
            "status",
            "verifier_id",
            "source_commit",
            "summary_sha256",
            "cpu_evidence_sha256",
            "gpu5_evidence_sha256",
            "source_tree_digest",
            "model_manifest_sha256",
            "model_tree_digest",
            "config_digest",
            "device_digest",
            "runtime_evidence_sha256",
            "completion_evidence_sha256",
            "identity_pre_sha256",
            "identity_post_sha256",
            "config_evidence_sha256",
            "device_evidence_sha256",
            "model_inventory_sha256",
            "issued_at_utc",
        },
        "release attestation",
    )
    if (
        attestation["schema"] != "cogni.release.attestation.v2"
        or attestation["status"] != "passed"
        or attestation["source_commit"] != expected_commit
        or type(attestation["verifier_id"]) is not str
        or IDENTIFIER_RE.fullmatch(attestation["verifier_id"]) is None
    ):
        raise ReleaseEvidenceError("release attestation identity or status is invalid")
    for field in set(attestation) - {
        "schema",
        "status",
        "verifier_id",
        "source_commit",
        "issued_at_utc",
    }:
        _require_digest(attestation[field], f"attestation {field}")
    if type(attestation["issued_at_utc"]) is not str:
        raise ReleaseEvidenceError("attestation issued_at_utc must be a string")
    try:
        issued = datetime.fromisoformat(
            attestation["issued_at_utc"].replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ReleaseEvidenceError("attestation issued_at_utc is invalid") from exc
    if issued.tzinfo is None or issued.utcoffset() != timezone.utc.utcoffset(issued):
        raise ReleaseEvidenceError("attestation issued_at_utc must identify UTC")


def _validate_public_key(value: dict[str, Any]) -> tuple[str, int, int]:
    key = _require_exact_keys(
        value,
        {"schema", "key_id", "algorithm", "modulus_hex", "exponent"},
        "public key",
    )
    if (
        key["schema"] != "cogni.rsa.public_key.v1"
        or key["algorithm"] != "rsa-pkcs1v15-sha256"
        or type(key["key_id"]) is not str
        or IDENTIFIER_RE.fullmatch(key["key_id"]) is None
        or type(key["modulus_hex"]) is not str
        or len(key["modulus_hex"]) < 512
        or len(key["modulus_hex"]) > 2048
        or len(key["modulus_hex"]) % 2
        or HEX_RE.fullmatch(key["modulus_hex"]) is None
        or type(key["exponent"]) is not int
        or key["exponent"] != 65537
    ):
        raise ReleaseEvidenceError("public key contract is invalid")
    modulus = int(key["modulus_hex"], 16)
    if not 2048 <= modulus.bit_length() <= 8192:
        raise ReleaseEvidenceError("public key modulus size is outside policy")
    return key["key_id"], modulus, key["exponent"]


def _verify_signature(
    data: bytes, signature_data: bytes, modulus: int, exponent: int
) -> None:
    try:
        signature_text = signature_data.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise ReleaseEvidenceError("attestation signature is not ASCII") from exc
    if HEX_RE.fullmatch(signature_text) is None or len(signature_text) % 2:
        raise ReleaseEvidenceError("attestation signature must be exact lowercase hex")
    signature = bytes.fromhex(signature_text)
    width = (modulus.bit_length() + 7) // 8
    if len(signature) != width:
        raise ReleaseEvidenceError(
            "attestation signature width does not match the public key"
        )
    encoded = pow(int.from_bytes(signature, "big"), exponent, modulus).to_bytes(
        width, "big"
    )
    digest_info = RSA_SHA256_DIGEST_INFO + hashlib.sha256(data).digest()
    padding_length = width - len(digest_info) - 3
    if padding_length < 8:
        raise ReleaseEvidenceError("public key is too small for RSA-SHA256")
    expected = b"\x00\x01" + b"\xff" * padding_length + b"\x00" + digest_info
    if not hmac.compare_digest(encoded, expected):
        raise ReleaseEvidenceError(
            "independent attestation signature verification failed"
        )


def validate_release_evidence(
    *,
    summary_path: Path,
    summary_sha256: str,
    cpu_path: Path,
    cpu_sha256: str,
    gpu5_path: Path,
    gpu5_sha256: str,
    attestation_path: Path,
    attestation_sha256: str,
    signature_path: Path,
    signature_sha256: str,
    public_key_path: Path,
    public_key_sha256: str,
    verifier_policy_path: Path,
    verifier_policy_sha256: str,
    runtime_path: Path,
    runtime_sha256: str,
    completion_path: Path,
    completion_sha256: str,
    identity_pre_path: Path,
    identity_pre_sha256: str,
    identity_post_path: Path,
    identity_post_sha256: str,
    config_evidence_path: Path,
    config_evidence_sha256: str,
    device_evidence_path: Path,
    device_evidence_sha256: str,
    model_inventory_path: Path,
    model_inventory_sha256: str,
    model_manifest_path: Path,
    expected_model_manifest_sha256: str,
    config_path: Path,
    expected_config_sha256: str,
    source_repo_path: Path,
    expanded_source_path: Path,
    git_executable_path: Path,
    git_executable_sha256: str,
    expected_source_commit: str,
) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected_source_commit) is None:
        raise ReleaseEvidenceError("expected source commit must be lowercase 40-hex")
    expected_source_tree_digest = compute_guard_source_tree_digest(
        source_repo=source_repo_path,
        expanded_source=expanded_source_path,
        expected_commit=expected_source_commit,
        git_executable=git_executable_path,
        git_executable_sha256=git_executable_sha256,
    )
    manifest_data = _read_pinned(
        model_manifest_path, expected_model_manifest_sha256, "source model manifest"
    )
    config_data = _read_pinned(
        config_path, expected_config_sha256, "source runtime config"
    )
    manifest_files = _parse_model_manifest_files(manifest_data)
    config_digest = _sha256(config_data)

    policy, _ = _load_json(
        verifier_policy_path, verifier_policy_sha256, "release verifier policy"
    )
    approved_id, approved_key_sha = _validate_verifier_policy(policy)
    if public_key_sha256 != approved_key_sha:
        raise ReleaseEvidenceError(
            "verifier public key is not the source-approved immutable key"
        )

    summary, _ = _load_json(summary_path, summary_sha256, "release summary")
    cpu, _ = _load_json(cpu_path, cpu_sha256, "CPU evidence")
    gpu5, _ = _load_json(gpu5_path, gpu5_sha256, "GPU5 evidence")
    attestation, attestation_data = _load_json(
        attestation_path, attestation_sha256, "release attestation"
    )
    public_key, _ = _load_json(
        public_key_path, public_key_sha256, "verifier public key"
    )
    runtime, _ = _load_json(runtime_path, runtime_sha256, "GPU5 runtime evidence")
    completion, _ = _load_json(
        completion_path, completion_sha256, "GPU5 completion evidence"
    )
    identity_pre, _ = _load_json(
        identity_pre_path, identity_pre_sha256, "GPU5 identity-pre evidence"
    )
    identity_post, _ = _load_json(
        identity_post_path, identity_post_sha256, "GPU5 identity-post evidence"
    )
    config_evidence, _ = _load_json(
        config_evidence_path, config_evidence_sha256, "GPU5 config evidence"
    )
    device_evidence, _ = _load_json(
        device_evidence_path, device_evidence_sha256, "GPU5 device evidence"
    )
    model_inventory, _ = _load_json(
        model_inventory_path, model_inventory_sha256, "GPU5 model inventory"
    )
    signature_data = _read_pinned(
        signature_path, signature_sha256, "attestation signature"
    )

    component_shas = {
        "runtime": runtime_sha256,
        "completion": completion_sha256,
        "identity_pre": identity_pre_sha256,
        "identity_post": identity_post_sha256,
        "config": config_evidence_sha256,
        "device": device_evidence_sha256,
        "model_inventory": model_inventory_sha256,
    }
    model_tree, device_digest = _validate_components(
        runtime=runtime,
        completion=completion,
        identity_pre=identity_pre,
        identity_post=identity_post,
        config_evidence=config_evidence,
        device_evidence=device_evidence,
        model_inventory=model_inventory,
        component_shas=component_shas,
        commit=expected_source_commit,
        source_tree=expected_source_tree_digest,
        manifest_sha=expected_model_manifest_sha256,
        manifest_files=manifest_files,
        config_digest=config_digest,
    )
    _validate_cpu(cpu, expected_source_commit, expected_source_tree_digest)
    _validate_gpu(
        gpu5,
        expected_commit=expected_source_commit,
        expected_source_tree=expected_source_tree_digest,
        manifest_sha=expected_model_manifest_sha256,
        model_tree=model_tree,
        config_digest=config_digest,
        device_digest=device_digest,
        component_shas=component_shas,
    )
    _validate_summary(summary, expected_source_commit)
    _validate_attestation(attestation, expected_source_commit)
    key_id, modulus, exponent = _validate_public_key(public_key)
    if key_id != approved_id:
        raise ReleaseEvidenceError(
            "public key id does not match source-approved verifier id"
        )
    _verify_signature(attestation_data, signature_data, modulus, exponent)

    summary_cpu = summary["cpu"]
    summary_gpu = summary["gpu5"]
    verifier = summary["independent_verifier"]
    if (
        summary_cpu["status"] != "passed"
        or summary_cpu["evidence_sha256"] != cpu_sha256
        or summary_cpu["source_tree_digest"] != cpu["source_tree_digest"]
        or summary_cpu["required_gates"] != cpu["required_gates"]
        or type(summary_cpu["pytest_passed"]) is not int
        or summary_cpu["pytest_passed"] != cpu["pytest_passed"]
        or type(summary_cpu["pytest_failed"]) is not int
        or summary_cpu["pytest_failed"] != cpu["pytest_failed"]
    ):
        raise ReleaseEvidenceError(
            "summary CPU section does not match raw CPU evidence"
        )
    gpu_fields = (
        "gpu_uuid",
        "image_digest",
        "model_manifest_sha256",
        "source_tree_digest",
        "model_tree_digest",
        "config_digest",
        "device_digest",
        "runtime_evidence_sha256",
        "completion_evidence_sha256",
        "identity_pre_sha256",
        "identity_post_sha256",
        "config_evidence_sha256",
        "device_evidence_sha256",
        "model_inventory_sha256",
    )
    if (
        summary_gpu["status"] != "passed"
        or summary_gpu["evidence_sha256"] != gpu5_sha256
        or summary_gpu["required_gates"] != gpu5["required_gates"]
        or type(summary_gpu["physical_gpu_index"]) is not int
        or summary_gpu["physical_gpu_index"] != gpu5["physical_gpu_index"]
        or any(summary_gpu[field] != gpu5[field] for field in gpu_fields)
    ):
        raise ReleaseEvidenceError(
            "summary GPU5 section does not match raw GPU5 evidence"
        )
    if cpu["source_tree_digest"] != gpu5["source_tree_digest"]:
        raise ReleaseEvidenceError("CPU and GPU5 evidence source trees differ")
    attestation_scope = {
        "summary_sha256": summary_sha256,
        "cpu_evidence_sha256": cpu_sha256,
        "gpu5_evidence_sha256": gpu5_sha256,
        **{
            field: gpu5[field]
            for field in gpu_fields
            if field not in {"gpu_uuid", "image_digest"}
        },
    }
    if any(
        attestation[field] != expected for field, expected in attestation_scope.items()
    ):
        raise ReleaseEvidenceError(
            "signed attestation scope does not match raw evidence"
        )
    if (
        verifier["status"] != "passed"
        or verifier["verifier_id"] != approved_id
        or verifier["verifier_id"] != attestation["verifier_id"]
    ):
        raise ReleaseEvidenceError(
            "summary verifier section does not match signed attestation"
        )

    return {
        "schema": "cogni.release.validation.v2",
        "status": "passed",
        "source_commit": expected_source_commit,
        "summary_sha256": summary_sha256,
        "cpu_evidence_sha256": cpu_sha256,
        "gpu5_evidence_sha256": gpu5_sha256,
        "source_tree_digest": expected_source_tree_digest,
        "model_manifest_sha256": expected_model_manifest_sha256,
        "model_tree_digest": model_tree,
        "config_digest": config_digest,
        "device_digest": device_digest,
        "runtime_evidence_sha256": runtime_sha256,
        "completion_evidence_sha256": completion_sha256,
        "identity_pre_sha256": identity_pre_sha256,
        "identity_post_sha256": identity_post_sha256,
        "config_evidence_sha256": config_evidence_sha256,
        "device_evidence_sha256": device_evidence_sha256,
        "model_inventory_sha256": model_inventory_sha256,
        "attestation_sha256": attestation_sha256,
        "attestation_signature_sha256": signature_sha256,
        "verifier_policy_sha256": verifier_policy_sha256,
        "verifier_public_key_sha256": public_key_sha256,
        "verifier_id": approved_id,
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or not path.parent.is_dir():
        raise ReleaseEvidenceError(
            "validation output must be a new file in an existing directory"
        )
    temporary = path.with_name(path.name + ".tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "summary",
        "cpu-evidence",
        "gpu5-evidence",
        "attestation",
        "attestation-signature",
        "verifier-public-key",
        "verifier-policy",
        "runtime",
        "completion",
        "identity-pre",
        "identity-post",
        "config-evidence",
        "device-evidence",
        "model-inventory",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--expanded-source", type=Path, required=True)
    parser.add_argument("--git-executable", type=Path, required=True)
    parser.add_argument("--git-executable-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=Path)
    output_group.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    payload = validate_release_evidence(
        summary_path=args.summary,
        summary_sha256=args.summary_sha256,
        cpu_path=args.cpu_evidence,
        cpu_sha256=args.cpu_evidence_sha256,
        gpu5_path=args.gpu5_evidence,
        gpu5_sha256=args.gpu5_evidence_sha256,
        attestation_path=args.attestation,
        attestation_sha256=args.attestation_sha256,
        signature_path=args.attestation_signature,
        signature_sha256=args.attestation_signature_sha256,
        public_key_path=args.verifier_public_key,
        public_key_sha256=args.verifier_public_key_sha256,
        verifier_policy_path=args.verifier_policy,
        verifier_policy_sha256=args.verifier_policy_sha256,
        runtime_path=args.runtime,
        runtime_sha256=args.runtime_sha256,
        completion_path=args.completion,
        completion_sha256=args.completion_sha256,
        identity_pre_path=args.identity_pre,
        identity_pre_sha256=args.identity_pre_sha256,
        identity_post_path=args.identity_post,
        identity_post_sha256=args.identity_post_sha256,
        config_evidence_path=args.config_evidence,
        config_evidence_sha256=args.config_evidence_sha256,
        device_evidence_path=args.device_evidence,
        device_evidence_sha256=args.device_evidence_sha256,
        model_inventory_path=args.model_inventory,
        model_inventory_sha256=args.model_inventory_sha256,
        model_manifest_path=args.model_manifest,
        expected_model_manifest_sha256=args.expected_model_manifest_sha256,
        config_path=args.config,
        expected_config_sha256=args.expected_config_sha256,
        source_repo_path=args.source_repo,
        expanded_source_path=args.expanded_source,
        git_executable_path=args.git_executable,
        git_executable_sha256=args.git_executable_sha256,
        expected_source_commit=args.expected_source_commit,
    )
    if args.stdout:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _write_output(args.output, payload)
        print("release_evidence_validation=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
