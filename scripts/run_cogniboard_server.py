"""Repository-anchored CogniBoard bootstrap compatible with Python ``-I``.

Isolated mode intentionally removes the current working directory and ignores
``PYTHONPATH``. This entry point admits the operating-system-specific launch
profile before importing any product or torch-capable module.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import sys
from typing import NamedTuple, NoReturn


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SERVER_MODULE = _PROJECT_ROOT / "cogni_demo" / "server.py"
_DESKTOP_PROFILE = "desktop-ui-only"
_SERVER_PROFILE = "server-gpu5-native"
_GPU5_UUID = "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SNAPSHOT_NONCE = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NATIVE_PREPARE_STAGE = "prepare"
_NATIVE_SEALED_STAGE = "sealed"
_FIXED_SERVER_ENVIRONMENT = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "COGNI_OS_GPU_UUID": _GPU5_UUID,
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": _GPU5_UUID,
    "NVIDIA_VISIBLE_DEVICES": _GPU5_UUID,
}
_DYNAMIC_SERVER_ENVIRONMENT = frozenset({"HOME", "COGNI_OS_MODEL_DIR"})
_SERVER_ENVIRONMENT_KEYS = frozenset(_FIXED_SERVER_ENVIRONMENT) | (
    _DYNAMIC_SERVER_ENVIRONMENT
)


class _NativeAuthorityArguments(NamedTuple):
    source_commit: str
    physical_index: int
    query_context: str
    gpu_uuid: str
    snapshot_stage: str
    source_snapshot_root: str | None
    source_snapshot_nonce: str | None
    workspace_root: str | None
    source_content_digest: str | None
    source_identity_digest: str | None
    source_file_count: int | None
    source_root_device: int | None
    source_root_inode: int | None
    model_snapshot_root: str | None
    model_manifest_sha256: str | None
    model_content_digest: str | None
    model_identity_digest: str | None
    model_file_count: int | None
    model_root_device: int | None
    model_root_inode: int | None
    model_total_bytes: int | None


_STATIC_HELP = f"""usage: python scripts/run_cogniboard_server.py [options]

Repository-anchored CogniBoard launcher.

options:
  -h, --help                            show this help message and exit
  --model PATH                          verified local Gemma model
  --manifest PATH                       closed-world model manifest
  --assets PATH                         static CogniBoard assets
  --port PORT                           loopback port
  --no-browser                          do not open the graphical shell
  --validation-profile {{{_DESKTOP_PROFILE},{_SERVER_PROFILE}}}
  --validation-physical-gpu-index 5
  --validation-gpu-query-context native-host
  --validation-gpu-uuid UUID
  --expected-source-commit COMMIT
  --native-snapshot-stage {{prepare,sealed}}  internal native snapshot stage
"""

if not _PROJECT_ROOT.is_dir() or not _SERVER_MODULE.is_file():
    raise RuntimeError("CogniBoard repository layout is incomplete")
sys.path.insert(0, str(_PROJECT_ROOT))


def _option_value(argv: tuple[str, ...], name: str) -> str | None:
    """Extract one exact long option without abbreviation or ambiguity."""

    prefix = f"{name}="
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            break
        if token == name:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise RuntimeError(f"{name} requires one explicit value")
            values.append(argv[index + 1])
            index += 2
            continue
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    if len(values) > 1:
        raise RuntimeError(f"{name} must not be repeated")
    return values[0] if values else None


def _bounded_decimal_option(
    argv: tuple[str, ...],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    value = _option_value(argv, name)
    if value is None:
        return None
    if re.fullmatch(r"(?:0|[1-9][0-9]{0,19})", value) is None:
        raise RuntimeError(f"{name} requires one canonical decimal integer")
    parsed = int(value)
    if parsed < minimum or parsed > maximum:
        raise RuntimeError(f"{name} is outside its admitted bound")
    return parsed


def _replace_exact_option(
    argv: tuple[str, ...], name: str, value: str
) -> tuple[str, ...]:
    """Replace one required long option without changing argument ordering."""

    if not value or any(ord(character) < 32 for character in value):
        raise RuntimeError(f"{name} replacement value is invalid")
    prefix = f"{name}="
    result: list[str] = []
    replacements = 0
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise RuntimeError(f"{name} requires one explicit value")
            result.extend((name, value))
            replacements += 1
            index += 2
            continue
        if token.startswith(prefix):
            result.append(f"{name}={value}")
            replacements += 1
        else:
            result.append(token)
        index += 1
    if replacements != 1:
        raise RuntimeError(f"{name} must appear exactly once")
    return tuple(result)


def _append_absent_option(
    argv: tuple[str, ...], name: str, value: str
) -> tuple[str, ...]:
    if _option_value(argv, name) is not None:
        raise RuntimeError(f"{name} is forbidden before snapshot preparation")
    return (*argv, name, value)


def _admitted_profile(argv: tuple[str, ...], *, platform: str | None = None) -> str:
    """Fail closed on a platform/profile mismatch before product import."""

    profile = _option_value(argv, "--validation-profile")
    current_platform = sys.platform if platform is None else platform
    if current_platform.startswith("linux"):
        if profile != _SERVER_PROFILE:
            raise RuntimeError(
                "Linux CogniBoard requires the exact server-gpu5-native profile"
            )
        return profile
    if current_platform == "win32":
        if profile != _DESKTOP_PROFILE:
            raise RuntimeError(
                "Windows CogniBoard requires the exact desktop-ui-only profile"
            )
        return profile
    raise RuntimeError("CogniBoard launch is unsupported on this platform")


def _validate_exact_server_environment() -> None:
    """Require the exact environment emitted by the Linux operator launcher."""

    if frozenset(os.environ) != _SERVER_ENVIRONMENT_KEYS:
        raise RuntimeError("server-gpu5-native requires an exact clean environment")
    if any(
        os.environ.get(name) != value
        for name, value in _FIXED_SERVER_ENVIRONMENT.items()
    ):
        raise RuntimeError("server-gpu5-native environment values are invalid")
    for name in _DYNAMIC_SERVER_ENVIRONMENT:
        value = os.environ.get(name, "")
        if (
            not value.startswith("/")
            or len(value) > 4096
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise RuntimeError(f"server-gpu5-native {name} is invalid")


def _native_authority_arguments(
    argv: tuple[str, ...],
) -> _NativeAuthorityArguments:
    if _option_value(argv, "--validation-profile") != _SERVER_PROFILE:
        raise RuntimeError("native authority requires server-gpu5-native")
    physical_index = _option_value(argv, "--validation-physical-gpu-index")
    query_context = _option_value(argv, "--validation-gpu-query-context")
    gpu_uuid = _option_value(argv, "--validation-gpu-uuid")
    source_commit = _option_value(argv, "--expected-source-commit")
    snapshot_stage = _option_value(argv, "--native-snapshot-stage")
    source_snapshot_root = _option_value(argv, "--native-source-snapshot-root")
    source_snapshot_nonce = _option_value(argv, "--native-source-snapshot-nonce")
    workspace_root = _option_value(argv, "--native-workspace-root")
    source_content_digest = _option_value(argv, "--native-source-content-digest")
    source_identity_digest = _option_value(argv, "--native-source-identity-digest")
    source_file_count = _bounded_decimal_option(
        argv, "--native-source-file-count", minimum=1, maximum=1_000_000
    )
    source_root_device = _bounded_decimal_option(
        argv, "--native-source-root-device", minimum=0, maximum=(2**64) - 1
    )
    source_root_inode = _bounded_decimal_option(
        argv, "--native-source-root-inode", minimum=1, maximum=(2**64) - 1
    )
    model_snapshot_root = _option_value(argv, "--native-model-snapshot-root")
    model_manifest_sha256 = _option_value(argv, "--native-model-manifest-sha256")
    model_content_digest = _option_value(argv, "--native-model-content-digest")
    model_identity_digest = _option_value(argv, "--native-model-identity-digest")
    model_file_count = _bounded_decimal_option(
        argv, "--native-model-file-count", minimum=1, maximum=1_000_000
    )
    model_root_device = _bounded_decimal_option(
        argv, "--native-model-root-device", minimum=0, maximum=(2**64) - 1
    )
    model_root_inode = _bounded_decimal_option(
        argv, "--native-model-root-inode", minimum=1, maximum=(2**64) - 1
    )
    model_total_bytes = _bounded_decimal_option(
        argv,
        "--native-model-total-bytes",
        minimum=1,
        maximum=96 * 1024 * 1024 * 1024,
    )
    model = _option_value(argv, "--model")
    manifest = _option_value(argv, "--manifest")
    assets = _option_value(argv, "--assets")
    if (
        physical_index != "5"
        or query_context != "native-host"
        or gpu_uuid != _GPU5_UUID
        or source_commit is None
        or _SOURCE_COMMIT.fullmatch(source_commit) is None
        or snapshot_stage not in {_NATIVE_PREPARE_STAGE, _NATIVE_SEALED_STAGE}
        or model is None
        or manifest is None
    ):
        raise RuntimeError(
            "server-gpu5-native requires the exact GPU5 boundary and source commit"
        )
    _validate_exact_server_environment()
    if (
        sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.no_user_site != 1
        or sys.flags.safe_path is not True
    ):
        raise RuntimeError("server-gpu5-native must start Python with -I -B")
    snapshot_values = (
        source_snapshot_root,
        source_snapshot_nonce,
        workspace_root,
        source_content_digest,
        source_identity_digest,
        source_file_count,
        source_root_device,
        source_root_inode,
        model_snapshot_root,
        model_manifest_sha256,
        model_content_digest,
        model_identity_digest,
        model_file_count,
        model_root_device,
        model_root_inode,
        model_total_bytes,
    )
    if snapshot_stage == _NATIVE_PREPARE_STAGE:
        if assets is not None or any(value is not None for value in snapshot_values):
            raise RuntimeError(
                "prepare stage rejects caller-supplied snapshot paths or authority"
            )
    else:
        if (
            source_snapshot_root is None
            or source_snapshot_nonce is None
            or workspace_root is None
            or source_content_digest is None
            or source_identity_digest is None
            or source_file_count is None
            or source_root_device is None
            or source_root_inode is None
            or model_snapshot_root is None
            or model_manifest_sha256 is None
            or model_content_digest is None
            or model_identity_digest is None
            or model_file_count is None
            or model_root_device is None
            or model_root_inode is None
            or model_total_bytes is None
            or assets is None
            or _SNAPSHOT_NONCE.fullmatch(source_snapshot_nonce) is None
            or _SHA256.fullmatch(source_content_digest) is None
            or _SHA256.fullmatch(source_identity_digest) is None
            or _SHA256.fullmatch(model_manifest_sha256) is None
            or _SHA256.fullmatch(model_content_digest) is None
            or _SHA256.fullmatch(model_identity_digest) is None
        ):
            raise RuntimeError("sealed stage requires exact source/model snapshots")
        try:
            raw_source = Path(source_snapshot_root)
            raw_workspace = Path(workspace_root)
            raw_model_snapshot = Path(model_snapshot_root)
            raw_manifest = Path(manifest)
            raw_model_argument = Path(model)
            raw_assets = Path(assets)
            admitted_source = raw_source.resolve(strict=True)
            admitted_workspace = raw_workspace.resolve(strict=True)
            admitted_model = raw_model_snapshot.resolve(strict=True)
            admitted_manifest = raw_manifest.resolve(strict=True)
            admitted_model_argument = raw_model_argument.resolve(strict=True)
            admitted_assets = raw_assets.resolve(strict=True)
        except OSError as error:
            raise RuntimeError("sealed snapshot paths are unavailable") from error
        if (
            not all(
                raw.is_absolute()
                for raw in (
                    raw_source,
                    raw_workspace,
                    raw_model_snapshot,
                    raw_manifest,
                    raw_model_argument,
                    raw_assets,
                )
            )
            or raw_source != admitted_source
            or raw_workspace != admitted_workspace
            or raw_model_snapshot != admitted_model
            or raw_manifest != admitted_manifest
            or raw_model_argument != admitted_model_argument
            or raw_assets != admitted_assets
            or admitted_source != _PROJECT_ROOT
            or admitted_workspace == admitted_source
            or admitted_workspace.is_relative_to(admitted_source)
            or admitted_source.is_relative_to(admitted_workspace)
            or admitted_model != admitted_model_argument
            or admitted_workspace == admitted_model
            or admitted_workspace.is_relative_to(admitted_model)
            or admitted_model.is_relative_to(admitted_workspace)
            or admitted_manifest
            != admitted_source / "config" / "gemma4-e4b-it.manifest.toml"
            or admitted_assets != admitted_source / "cogni_demo" / "static"
            or not admitted_assets.is_dir()
        ):
            raise RuntimeError("sealed stage paths escaped their snapshot capability")
    return _NativeAuthorityArguments(
        source_commit,
        5,
        "native-host",
        _GPU5_UUID,
        snapshot_stage,
        source_snapshot_root,
        source_snapshot_nonce,
        workspace_root,
        source_content_digest,
        source_identity_digest,
        source_file_count,
        source_root_device,
        source_root_inode,
        model_snapshot_root,
        model_manifest_sha256,
        model_content_digest,
        model_identity_digest,
        model_file_count,
        model_root_device,
        model_root_inode,
        model_total_bytes,
    )


def _prepare_and_reexec_native_snapshots(
    argv: tuple[str, ...], source_commit: str
) -> NoReturn:
    """Create immutable inputs, then replace this mutable stage-0 process."""

    model_source = _option_value(argv, "--model")
    manifest_source = _option_value(argv, "--manifest")
    if _option_value(argv, "--assets") is not None:
        raise RuntimeError("native snapshot preparation rejects caller assets")
    if model_source is None or manifest_source is None:
        raise RuntimeError("native snapshot preparation requires model and manifest")
    expected_manifest = _PROJECT_ROOT / "config" / "gemma4-e4b-it.manifest.toml"
    try:
        if Path(manifest_source).resolve(strict=True) != expected_manifest.resolve(
            strict=True
        ):
            raise RuntimeError("native snapshot preparation rejected the manifest")
    except OSError as error:
        raise RuntimeError("native snapshot manifest is unavailable") from error

    import scripts.gpu5_boundary_guard as boundary_guard

    expected_guard = _PROJECT_ROOT / "scripts" / "gpu5_boundary_guard.py"
    if Path(boundary_guard.__file__).resolve(strict=True) != expected_guard:
        raise RuntimeError("native snapshot guard escaped the admitted source root")

    snapshots = boundary_guard.prepare_native_execution_snapshot(
        source_commit,
        model_source,
        "config/gemma4-e4b-it.manifest.toml",
    )
    source_root = Path(snapshots.source.root_path).resolve(strict=True)
    model_root = Path(snapshots.model.root_path).resolve(strict=True)
    manifest_path = Path(snapshots.manifest_path).resolve(strict=True)
    bootstrap = (source_root / "scripts" / "run_cogniboard_server.py").resolve(
        strict=True
    )
    if (
        bootstrap != source_root / "scripts" / "run_cogniboard_server.py"
        or not bootstrap.is_file()
        or not manifest_path.is_relative_to(source_root)
        or _SNAPSHOT_NONCE.fullmatch(snapshots.source.launch_nonce) is None
        or _SHA256.fullmatch(snapshots.model.manifest_sha256) is None
    ):
        raise RuntimeError("native snapshot factory returned an invalid capability")

    sealed = _replace_exact_option(
        argv, "--native-snapshot-stage", _NATIVE_SEALED_STAGE
    )
    sealed = _replace_exact_option(sealed, "--model", os.fspath(model_root))
    sealed = _replace_exact_option(sealed, "--manifest", os.fspath(manifest_path))
    sealed = _append_absent_option(
        sealed, "--assets", os.fspath(source_root / "cogni_demo" / "static")
    )
    sealed = _append_absent_option(
        sealed, "--native-source-snapshot-root", os.fspath(source_root)
    )
    sealed = _append_absent_option(
        sealed, "--native-source-snapshot-nonce", snapshots.source.launch_nonce
    )
    sealed = _append_absent_option(
        sealed, "--native-workspace-root", snapshots.workspace_root
    )
    sealed = _append_absent_option(
        sealed, "--native-source-content-digest", snapshots.source.content_digest
    )
    sealed = _append_absent_option(
        sealed, "--native-source-identity-digest", snapshots.source.identity_digest
    )
    sealed = _append_absent_option(
        sealed, "--native-source-file-count", str(snapshots.source.file_count)
    )
    sealed = _append_absent_option(
        sealed, "--native-source-root-device", str(snapshots.source.root_device)
    )
    sealed = _append_absent_option(
        sealed, "--native-source-root-inode", str(snapshots.source.root_inode)
    )
    sealed = _append_absent_option(
        sealed, "--native-model-snapshot-root", os.fspath(model_root)
    )
    sealed = _append_absent_option(
        sealed,
        "--native-model-manifest-sha256",
        snapshots.model.manifest_sha256,
    )
    sealed = _append_absent_option(
        sealed, "--native-model-content-digest", snapshots.model.content_digest
    )
    sealed = _append_absent_option(
        sealed, "--native-model-identity-digest", snapshots.model.identity_digest
    )
    sealed = _append_absent_option(
        sealed, "--native-model-file-count", str(snapshots.model.file_count)
    )
    sealed = _append_absent_option(
        sealed, "--native-model-root-device", str(snapshots.model.root_device)
    )
    sealed = _append_absent_option(
        sealed, "--native-model-root-inode", str(snapshots.model.root_inode)
    )
    sealed = _append_absent_option(
        sealed, "--native-model-total-bytes", str(snapshots.model.total_bytes)
    )
    environment = dict(os.environ)
    environment["COGNI_OS_MODEL_DIR"] = os.fspath(model_root)
    os.chdir(source_root)
    os.execve(
        sys.executable,
        (
            sys.executable,
            "-I",
            "-B",
            os.fspath(bootstrap),
            *sealed,
        ),
        environment,
    )
    raise RuntimeError("native snapshot exec unexpectedly returned")


def _validate_sealed_sys_path(boundary_guard: object, workspace_root: Path) -> None:
    """Reject mutable or untrusted import roots before product import."""

    validate_directory = getattr(
        boundary_guard, "validate_trusted_import_directory", None
    )
    if not callable(validate_directory):
        raise RuntimeError("sealed import-path validator is unavailable")
    expected_runtime_archives: set[Path] = set()
    for raw_prefix in {sys.base_prefix, sys.prefix, sys.exec_prefix}:
        try:
            prefix = Path(raw_prefix).resolve(strict=True)
        except OSError as error:
            raise RuntimeError("Python runtime prefix is unavailable") from error
        expected_runtime_archives.add(
            prefix
            / "lib"
            / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
        )

    if not sys.path:
        raise RuntimeError("sealed Python import path is empty")
    for raw_entry in sys.path:
        if not raw_entry:
            raise RuntimeError("sealed Python import path contains an empty entry")
        lexical = Path(raw_entry)
        if not lexical.is_absolute():
            raise RuntimeError("sealed Python import path must be absolute")
        try:
            imported_root = lexical.resolve(strict=True)
        except OSError as error:
            unresolved = lexical.resolve(strict=False)
            if unresolved not in expected_runtime_archives:
                raise RuntimeError(
                    "sealed Python import path contains an unavailable entry"
                ) from error
            validate_directory(unresolved.parent)
            continue

        if lexical != imported_root:
            raise RuntimeError("sealed Python import path must already be canonical")

        if (
            imported_root == workspace_root
            or imported_root.is_relative_to(workspace_root)
            or workspace_root.is_relative_to(imported_root)
        ):
            raise RuntimeError("mutable workspace remained on the sealed sys.path")
        if imported_root.is_dir():
            validate_directory(imported_root)
            continue
        if imported_root not in expected_runtime_archives:
            raise RuntimeError("sealed Python import path contains an unapproved file")
        try:
            metadata = os.lstat(imported_root)
        except OSError as error:
            raise RuntimeError("sealed runtime archive is unavailable") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid not in {0, os.geteuid()}
            or stat.S_IMODE(metadata.st_mode) & 0o022
            or metadata.st_nlink != 1
        ):
            raise RuntimeError("sealed runtime archive metadata is untrusted")
        validate_directory(imported_root.parent)


def _run(argv: tuple[str, ...] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments in {("-h",), ("--help",)}:
        print(_STATIC_HELP, end="")
        return 0

    profile = _admitted_profile(arguments)
    if profile == _DESKTOP_PROFILE:
        from cogni_demo.server import main

        return main(arguments)

    admitted = _native_authority_arguments(arguments)
    if admitted.snapshot_stage == _NATIVE_PREPARE_STAGE:
        _prepare_and_reexec_native_snapshots(arguments, admitted.source_commit)
    assert admitted.source_snapshot_root is not None
    assert admitted.source_snapshot_nonce is not None
    assert admitted.workspace_root is not None
    assert admitted.source_content_digest is not None
    assert admitted.source_identity_digest is not None
    assert admitted.source_file_count is not None
    assert admitted.source_root_device is not None
    assert admitted.source_root_inode is not None
    assert admitted.model_snapshot_root is not None
    assert admitted.model_manifest_sha256 is not None
    assert admitted.model_content_digest is not None
    assert admitted.model_identity_digest is not None
    assert admitted.model_file_count is not None
    assert admitted.model_root_device is not None
    assert admitted.model_root_inode is not None
    assert admitted.model_total_bytes is not None
    import scripts.gpu5_boundary_guard as boundary_guard

    expected_guard = _PROJECT_ROOT / "scripts" / "gpu5_boundary_guard.py"
    if Path(boundary_guard.__file__).resolve(strict=True) != expected_guard:
        raise RuntimeError("sealed native authority escaped the source snapshot")

    workspace_root = Path(admitted.workspace_root).resolve(strict=True)
    boundary_guard.validate_trusted_import_directory(workspace_root)
    _validate_sealed_sys_path(boundary_guard, workspace_root)

    with boundary_guard.native_gpu5_server_authority(
        admitted.source_commit,
        physical_gpu_index=admitted.physical_index,
        gpu_query_context=admitted.query_context,
        gpu_uuid=admitted.gpu_uuid,
        source_snapshot_root=admitted.source_snapshot_root,
        source_snapshot_nonce=admitted.source_snapshot_nonce,
        workspace_root=os.fspath(workspace_root),
        source_content_digest=admitted.source_content_digest,
        source_identity_digest=admitted.source_identity_digest,
        source_file_count=admitted.source_file_count,
        source_root_device=admitted.source_root_device,
        source_root_inode=admitted.source_root_inode,
        model_snapshot_root=admitted.model_snapshot_root,
        model_manifest_path=_option_value(arguments, "--manifest"),
        model_manifest_sha256=admitted.model_manifest_sha256,
        model_content_digest=admitted.model_content_digest,
        model_identity_digest=admitted.model_identity_digest,
        model_file_count=admitted.model_file_count,
        model_root_device=admitted.model_root_device,
        model_root_inode=admitted.model_root_inode,
        model_total_bytes=admitted.model_total_bytes,
    ) as authority:
        # The product module is imported only from the commit/model snapshots,
        # after the shared flock and exact-index5 idle/no-PID proof above. Its
        # import graph must remain CUDA-probe-free: the first logical CUDA
        # identity probe is inside _native_gpu5_server_lifecycle.
        if authority.execution_snapshot.workspace_root != os.fspath(workspace_root):
            raise RuntimeError("native authority changed the admitted workspace root")

        import cogni_demo.server as product_server

        expected_product = _PROJECT_ROOT / "cogni_demo" / "server.py"
        if Path(product_server.__file__).resolve(strict=True) != expected_product:
            raise RuntimeError("CogniBoard product import escaped the source snapshot")

        return product_server.main(arguments, native_gpu5_authority=authority)


if __name__ == "__main__":
    raise SystemExit(_run())
