"""Bounded regular-file-only snapshots for isolated candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import stat


FILE_ATTRIBUTE_REPARSE_POINT = 0x400
DEFAULT_EXCLUDED_ROOTS = frozenset(
    {
        ".git",
        ".cogni_state",
        ".cache",
        ".venv",
        "venv",
        "work",
        "outputs",
        "output",
        "build",
        "dist",
        "release",
        "tmp",
        "model",
        "models",
        "checkpoints",
    }
)
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
)
_SECRET_FILENAMES = frozenset(
    {".env", ".env.local", ".env.production", "credentials", "credentials.json"}
)
_SECRET_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx", ".token"})
_MODEL_SUFFIXES = frozenset({".safetensors", ".gguf", ".ckpt", ".pth", ".pt", ".onnx"})


class SnapshotBoundaryError(RuntimeError):
    """Raised before candidate execution when a snapshot boundary is unsafe."""


@dataclass(frozen=True, slots=True)
class SnapshotEvidence:
    files: int
    total_bytes: int
    tree_sha256: str


class SafeProjectSnapshotBuilder:
    """Copy a bounded project view without following any host link or junction."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        excluded_roots: tuple[str, ...] = (),
        max_files: int = 4_096,
        max_total_bytes: int = 128 * 1024**2,
        max_file_bytes: int = 32 * 1024**2,
    ) -> None:
        self.project_root = Path(project_root).resolve(strict=True)
        if not self.project_root.is_dir():
            raise SnapshotBoundaryError("snapshot source is not a directory")
        if not 1 <= max_files <= 100_000:
            raise ValueError("snapshot file bound is invalid")
        if not 1 <= max_file_bytes <= max_total_bytes <= 1024**3:
            raise ValueError("snapshot byte bounds are invalid")
        for name in excluded_roots:
            if (
                not name
                or Path(name).is_absolute()
                or len(Path(name).parts) != 1
                or name in {".", ".."}
            ):
                raise ValueError("snapshot excluded root is invalid")
        self.excluded_roots = DEFAULT_EXCLUDED_ROOTS | frozenset(excluded_roots)
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self.max_file_bytes = max_file_bytes

    def copy_to(self, destination: str | Path) -> SnapshotEvidence:
        target = Path(destination)
        if target.exists() or target.is_symlink():
            raise SnapshotBoundaryError("snapshot destination must not exist")
        target.mkdir(parents=True, exist_ok=False)
        resolved_target = target.resolve(strict=True)
        if (
            resolved_target == self.project_root
            or self.project_root in resolved_target.parents
        ):
            raise SnapshotBoundaryError("snapshot destination cannot be inside source")
        digest = sha256()
        counters = [0, 0]
        try:
            self._copy_directory(
                self.project_root,
                resolved_target,
                Path(),
                digest,
                counters,
            )
        except BaseException:
            # The caller owns its TemporaryDirectory cleanup.  Leaving the
            # partial stage inert is safer than recursively deleting a path
            # after an identity failure.
            raise
        return SnapshotEvidence(counters[0], counters[1], digest.hexdigest())

    def _copy_directory(
        self,
        source: Path,
        destination: Path,
        relative_directory: Path,
        digest,
        counters: list[int],
    ) -> None:
        try:
            entries = sorted(os.scandir(source), key=lambda item: item.name)
        except OSError as exc:
            raise SnapshotBoundaryError("snapshot directory is unreadable") from exc
        for entry in entries:
            name = entry.name
            relative = relative_directory / name
            if not relative_directory.parts and (
                name in self.excluded_roots or name.startswith(".codex")
            ):
                continue
            if name in _EXCLUDED_DIRECTORY_NAMES:
                continue
            try:
                item_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SnapshotBoundaryError(
                    "snapshot entry identity is unstable"
                ) from exc
            if (
                entry.is_symlink()
                or getattr(item_stat, "st_file_attributes", 0)
                & FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise SnapshotBoundaryError(
                    f"snapshot entry is a link/reparse point: {relative.as_posix()}"
                )
            if stat.S_ISDIR(item_stat.st_mode):
                child = destination / relative
                child.mkdir(parents=True, exist_ok=False)
                self._copy_directory(
                    Path(entry.path),
                    destination,
                    relative,
                    digest,
                    counters,
                )
                continue
            if not stat.S_ISREG(item_stat.st_mode):
                raise SnapshotBoundaryError(
                    f"snapshot entry is not a regular file: {relative.as_posix()}"
                )
            if self._secret_or_model_file(name):
                continue
            payload = self._read_regular_file(Path(entry.path), item_stat)
            counters[0] += 1
            counters[1] += len(payload)
            if counters[0] > self.max_files or counters[1] > self.max_total_bytes:
                raise SnapshotBoundaryError("snapshot crossed its file/byte bound")
            relative_bytes = relative.as_posix().encode("utf-8")
            digest.update(len(relative_bytes).to_bytes(4, "big"))
            digest.update(relative_bytes)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(sha256(payload).digest())
            copied = destination / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                copied,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                item_stat.st_mode & 0o777,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)

    def _read_regular_file(self, path: Path, admitted_stat: os.stat_result) -> bytes:
        if admitted_stat.st_size > self.max_file_bytes:
            raise SnapshotBoundaryError("snapshot file crossed its per-file bound")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise SnapshotBoundaryError(
                "snapshot file could not be opened safely"
            ) from exc
        try:
            opened_stat = os.fstat(descriptor)
            path_stat = os.stat(path, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or not stat.S_ISREG(path_stat.st_mode)
                or getattr(path_stat, "st_file_attributes", 0)
                & FILE_ATTRIBUTE_REPARSE_POINT
                or not self._same_identity(admitted_stat, opened_stat)
                or not self._same_identity(admitted_stat, path_stat)
            ):
                raise SnapshotBoundaryError(
                    "snapshot file identity changed during read"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, self.max_file_bytes + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > self.max_file_bytes:
                    raise SnapshotBoundaryError("snapshot file grew beyond its bound")
                chunks.append(chunk)
            after = os.fstat(descriptor)
            after_path = os.stat(path, follow_symlinks=False)
            if (
                getattr(after_path, "st_file_attributes", 0)
                & FILE_ATTRIBUTE_REPARSE_POINT
                or not self._same_identity(opened_stat, after)
                or not self._same_identity(opened_stat, after_path)
            ):
                raise SnapshotBoundaryError("snapshot file changed during read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
        """Compare file identity, with a conservative Windows fallback.

        Some Windows Python builds return zero device/inode values from a
        directory entry while ``fstat`` exposes a synthetic handle identity.
        Native Windows promotion is independently disabled, so the snapshot
        reader admits that platform only when type, size, timestamps, and file
        attributes all remain stable across the opened handle and path.
        """

        if first.st_dev and first.st_ino and second.st_dev and second.st_ino:
            return (
                first.st_dev == second.st_dev
                and first.st_ino == second.st_ino
                and first.st_size == second.st_size
            )
        if os.name != "nt":
            return False
        return (
            stat.S_IFMT(first.st_mode) == stat.S_IFMT(second.st_mode)
            and first.st_size == second.st_size
            and first.st_mtime_ns == second.st_mtime_ns
            and getattr(first, "st_file_attributes", 0)
            == getattr(second, "st_file_attributes", 0)
        )

    @staticmethod
    def _secret_or_model_file(name: str) -> bool:
        lower = name.lower()
        return (
            lower in _SECRET_FILENAMES
            or Path(lower).suffix in _SECRET_SUFFIXES
            or Path(lower).suffix in _MODEL_SUFFIXES
            or lower.endswith(".bin")
        )


__all__ = [
    "SafeProjectSnapshotBuilder",
    "SnapshotBoundaryError",
    "SnapshotEvidence",
]
