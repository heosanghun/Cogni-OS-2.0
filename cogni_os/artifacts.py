from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
import tomllib


class ArtifactVerificationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Manifest-declared provenance; callers must bind it to a trust root."""

    family: str
    variant: str
    role: str
    source: str
    revision: str


@dataclass(frozen=True)
class VerifiedArtifactSet:
    root: Path
    files: tuple[Path, ...]
    identity: ArtifactIdentity | None = None
    digests: tuple[tuple[str, str], ...] = ()


def _top_level_allowlist(names: tuple[str, ...], label: str) -> frozenset[str]:
    accepted: set[str] = set()
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).parts != (name,)
        ):
            raise ValueError(f"{label} must contain safe top-level names")
        accepted.add(name)
    return frozenset(accepted)


def verify_closed_world_artifact_layout(
    verified: VerifiedArtifactSet,
    *,
    allowed_unmanifested_files: tuple[str, ...] = (),
    allowed_unmanifested_directories: tuple[str, ...] = (),
) -> VerifiedArtifactSet:
    """Reject every top-level entry outside a manifest and explicit benign set.

    This inventory check is intentionally separate from digest verification so a
    loader can opt into a closed-world directory policy without changing generic
    manifests used by non-model artifacts. Allowed unmanifested entries must also
    have the declared file-system kind and may never be symlinks.
    """

    if not isinstance(verified, VerifiedArtifactSet):
        raise TypeError("verified must be a VerifiedArtifactSet")
    allowed_files = _top_level_allowlist(
        allowed_unmanifested_files, "allowed_unmanifested_files"
    )
    allowed_directories = _top_level_allowlist(
        allowed_unmanifested_directories, "allowed_unmanifested_directories"
    )
    if allowed_files & allowed_directories:
        raise ValueError("closed-world file and directory allowlists must be disjoint")
    if not verified.digests:
        raise ArtifactVerificationError(
            "closed-world verification requires manifest digest entries"
        )

    manifested_types: dict[str, bool] = {}
    for relative_name, _digest in verified.digests:
        parts = Path(relative_name).parts
        if not parts:
            raise ArtifactVerificationError("manifest contains an empty artifact path")
        top_level = parts[0]
        expects_directory = len(parts) > 1
        prior = manifested_types.get(top_level)
        if prior is not None and prior != expects_directory:
            raise ArtifactVerificationError(
                f"manifest top-level path has conflicting kinds: {top_level}"
            )
        manifested_types[top_level] = expects_directory

    try:
        root = verified.root.expanduser().resolve(strict=True)
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise ArtifactVerificationError(
            "artifact root could not be inventoried for closed-world loading"
        ) from exc
    if not root.is_dir():
        raise ArtifactVerificationError("artifact root must remain a directory")

    observed: set[str] = set()
    for entry in entries:
        name = entry.name
        observed.add(name)
        try:
            mode = entry.lstat().st_mode
        except OSError as exc:
            raise ArtifactVerificationError(
                f"artifact entry changed during inventory: {name}"
            ) from exc
        if S_ISLNK(mode):
            raise ArtifactVerificationError(
                f"closed-world artifact entry cannot be a symlink: {name}"
            )
        if name in manifested_types:
            expects_directory = manifested_types[name]
            valid_kind = S_ISDIR(mode) if expects_directory else S_ISREG(mode)
            if not valid_kind:
                raise ArtifactVerificationError(
                    f"manifested top-level artifact changed kind: {name}"
                )
            continue
        if name in allowed_files and S_ISREG(mode):
            continue
        if name in allowed_directories and S_ISDIR(mode):
            continue
        if name in allowed_files or name in allowed_directories:
            raise ArtifactVerificationError(
                f"benign artifact entry has the wrong kind: {name}"
            )
        raise ArtifactVerificationError(
            f"unmanifested top-level artifact entry is forbidden: {name}"
        )

    missing = sorted(set(manifested_types) - observed)
    if missing:
        raise ArtifactVerificationError(
            f"manifested top-level artifact disappeared: {missing[0]}"
        )
    return verified


def _manifest_identity(data: object) -> ArtifactIdentity | None:
    if not isinstance(data, dict):
        raise ArtifactVerificationError("manifest root must be a table")
    model = data.get("model")
    if model is None:
        return None
    if not isinstance(model, dict):
        raise ArtifactVerificationError("manifest [model] must be a table")
    values: dict[str, str] = {}
    for key in ("family", "variant", "role", "source", "revision"):
        value = model.get(key)
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 128
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ArtifactVerificationError(
                f"manifest model identity requires bounded text: {key}"
            )
        values[key] = value
    if values["role"] not in {"base", "instruction_tuned"}:
        raise ArtifactVerificationError("manifest model role is unsupported")
    return ArtifactIdentity(**values)


def verify_artifact_manifest(
    root: str | Path, manifest: str | Path
) -> VerifiedArtifactSet:
    """Verify a local offline artifact set without following escaping symlinks."""
    root_path = Path(root).expanduser().resolve(strict=True)
    if not root_path.is_dir():
        raise ArtifactVerificationError("artifact root must be a local directory")
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    with manifest_path.open("rb") as stream:
        data = tomllib.load(stream)
    identity = _manifest_identity(data)
    entries = data.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ArtifactVerificationError(
            "manifest must contain a non-empty [files] table"
        )
    verified: list[Path] = []
    verified_digests: list[tuple[str, str]] = []
    for relative_name, expected in sorted(entries.items()):
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactVerificationError(f"unsafe artifact path: {relative_name}")
        candidate = root_path / relative
        if candidate.is_symlink():
            raise ArtifactVerificationError(
                f"symlink artifacts are not accepted: {relative_name}"
            )
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ArtifactVerificationError(
                f"missing artifact: {relative_name}"
            ) from exc
        if not resolved.is_relative_to(root_path) or not resolved.is_file():
            raise ArtifactVerificationError(f"artifact escaped root: {relative_name}")
        expected_hash = str(expected).lower()
        if len(expected_hash) != 64 or any(
            char not in "0123456789abcdef" for char in expected_hash
        ):
            raise ArtifactVerificationError(
                f"invalid SHA-256 in manifest: {relative_name}"
            )
        digest = sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise ArtifactVerificationError(
                f"artifact digest mismatch: {relative_name}"
            )
        verified.append(resolved)
        verified_digests.append((relative.as_posix(), actual_hash))
    return VerifiedArtifactSet(
        root_path,
        tuple(verified),
        identity,
        tuple(verified_digests),
    )


__all__ = [
    "ArtifactIdentity",
    "ArtifactVerificationError",
    "VerifiedArtifactSet",
    "verify_artifact_manifest",
    "verify_closed_world_artifact_layout",
]
