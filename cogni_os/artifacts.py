from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tomllib


class ArtifactVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedArtifactSet:
    root: Path
    files: tuple[Path, ...]


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
    entries = data.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ArtifactVerificationError(
            "manifest must contain a non-empty [files] table"
        )
    verified: list[Path] = []
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
        if digest.hexdigest() != expected_hash:
            raise ArtifactVerificationError(
                f"artifact digest mismatch: {relative_name}"
            )
        verified.append(resolved)
    return VerifiedArtifactSet(root_path, tuple(verified))
