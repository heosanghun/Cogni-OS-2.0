from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from secrets import token_hex
import sys
import tempfile

from cogni_flow.kernel_sandbox import (
    LinuxOciSandboxRunner,
    build_kernel_sandbox_evidence_payload,
)


SAFE_COMMAND = ("python", "-I", "/project/check.py")
TIMEOUT_COMMAND = ("python", "-I", "/project/hang.py")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Linux OCI candidate sandbox without GPU access."
    )
    parser.add_argument("--image", required=True, help="Exact local image@sha256 ref")
    parser.add_argument("--engine", default="/usr/bin/docker")
    parser.add_argument("--socket", default="/var/run/docker.sock")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not sys.platform.startswith("linux"):
        raise SystemExit("kernel sandbox validation requires Linux")
    engine = Path(args.engine).resolve(strict=True)
    engine_sha256 = sha256(engine.read_bytes()).hexdigest()
    canary_name = f"cogni-host-canary-{token_hex(12)}"
    host_canary = Path("/tmp") / canary_name
    host_canary.write_bytes(b"host-only")
    try:
        with tempfile.TemporaryDirectory(prefix="cogni-kernel-sandbox-") as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir(mode=0o755)
            (project / "check.py").write_text(
                "import os\n"
                "from pathlib import Path\n"
                "import socket\n"
                "assert os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void'\n"
                "assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''\n"
                "assert os.environ.get('HIP_VISIBLE_DEVICES') == ''\n"
                "assert os.environ.get('ROCR_VISIBLE_DEVICES') == ''\n"
                "for target in (Path('/project/candidate-write'), Path('/root-write')):\n"
                "    try:\n"
                "        target.write_text('forbidden', encoding='ascii')\n"
                "    except OSError:\n"
                "        pass\n"
                "    else:\n"
                "        raise AssertionError(f'writable boundary: {target}')\n"
                f"assert not Path('/tmp/{canary_name}').exists()\n"
                "probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "probe.settimeout(1.0)\n"
                "try:\n"
                "    probe.connect(('192.0.2.1', 9))\n"
                "except OSError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('network namespace allowed egress')\n"
                "finally:\n"
                "    probe.close()\n"
                "print('KERNEL_SANDBOX_BOUNDARIES=PASS')\n",
                encoding="utf-8",
            )
            (project / "hang.py").write_text(
                "import subprocess, sys, time\n"
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            for path in project.iterdir():
                path.chmod(0o644)
            evidence_bytes = build_kernel_sandbox_evidence_payload(
                runner_id="cogni-linux-oci-v1",
                engine_path=str(engine),
                engine_sha256=engine_sha256,
                daemon_socket=args.socket,
                image_reference=args.image,
                commands=(SAFE_COMMAND, TIMEOUT_COMMAND),
                max_memory_bytes=1024 * 1024 * 1024,
                max_pids=64,
                max_cpus=2.0,
                tmpfs_bytes=128 * 1024 * 1024,
            )
            evidence_path = root / "runner-evidence.json"
            evidence_path.write_bytes(evidence_bytes)
            evidence_path.chmod(0o600)
            runner = LinuxOciSandboxRunner(evidence_path)
            safe = runner.run(project, SAFE_COMMAND, 30)
            if not safe.passed or "KERNEL_SANDBOX_BOUNDARIES=PASS" not in safe.output:
                raise RuntimeError(f"sandbox boundary validation failed: {safe}")
            timeout = runner.run(project, TIMEOUT_COMMAND, 2)
            if timeout.passed or timeout.returncode != 124:
                raise RuntimeError(f"sandbox timeout validation failed: {timeout}")
            if not host_canary.is_file() or host_canary.read_bytes() != b"host-only":
                raise RuntimeError("host canary changed during sandbox execution")
            report = {
                "schema": "cogni.kernel-sandbox-validation.v1",
                "status": "PASS",
                "evidence_sha256": sha256(evidence_bytes).hexdigest(),
                "engine_sha256": engine_sha256,
                "image_reference": args.image,
                "network": "none",
                "project_mount": "read_only",
                "rootfs": "read_only",
                "non_root_uid": 65534,
                "timeout_returncode": timeout.returncode,
                "gpu_query_count": 0,
                "gpu_use_count": 0,
            }
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
    finally:
        host_canary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
