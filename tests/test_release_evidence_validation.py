from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.gpu5_boundary_guard import _canonical_scope_digest
from scripts.validate_release_evidence import (
    GPU5_UUID,
    IMAGE_DIGEST,
    RSA_SHA256_DIGEST_INFO,
    ReleaseEvidenceError,
    _canonical_device_digest,
    _canonical_model_tree_digest,
    compute_guard_source_tree_digest,
    validate_release_evidence,
)


TEST_RSA_N = int(
    "c4a2e666727b78f850d050b6c565bf4a0e79ac56940129c384c8ec253d266054"
    "2ecdd14878c0fb0ff16244551fafa960a0d9cec7c4b15a3d2805c53354924dc4"
    "8e22f8513a2846bf36cddbd457b7fb2e56051b8c3e0ac299da493c236a978c3d"
    "125278fb47b3c519e106d59a99bda0a2cf038d50fac39c230c735b12835e7fef"
    "0e040bf88c8d3af9840a5ab859311c740150bd7377744fd48095e5a0c1323232a"
    "4965ef53e0fd1653d9c6acbb858f53130e667b7a22ec9fbdf5cd942ecf72f858"
    "a8d321eb717c3dc1ce0035477147e6557b07def25abd592e3f37421da2564ed695"
    "2e05dc154a8d027e772f98395d87f1292dc90499727c093d14c56ccf9e397",
    16,
)
TEST_RSA_D = int(
    "2bf5bf27fa66f27b4803937592107adc8efb5259ad46154b92f1b3b4d669303b"
    "5896c6378459c4eba262235b8907a59ac25b13d1b0c7757bf69fc15673d653ba3"
    "468019294f2754c412696deb8cfe440e0cc83853ffe1eb106a0867a76c32aa4b3"
    "787c9cb0af8da5892dcbaea47044eb8eae9376f9743893d4302cb9cdade074ccd"
    "6dec624bbb0ca154673de51380947520448003091e881fa2022101fc725bca77e1"
    "a36e66dbba5822eca9d6a5968ce7212aa3ca49ab8391283a48da1dba45c348030"
    "6ac32782d5f9c7ac75c68c502f1eb8f1269f221c470143b928918901fecc7f532"
    "77d914302faf0877955518947d551cf3a85194a5cd38d3796e2306d1",
    16,
)
ALTERNATE_VALID_RSA_N = int(
    "d3261f60f63700768070f0609e24040e3b9b55822dfc2a7ea6de2ffa825c6491"
    "ed8369c86d75881f664429b44e1daa1959078a09e57d1a5bb05ff7abee4fcef21"
    "01699f31e876b225ddba771314336f3f9e289afaf59df17a943811e99749648bd"
    "909addc03edef5b991a1d636aa142c91ee5a5d0f558133ae36998ff4c25dbac16"
    "2cb46ea4d2bb35b1b14d8b95643964db4c24045393058ccda79812a5c7207536c"
    "c36e4134b2f71abe917d8be03cada1b2cfc4a94d255f43ef966586352080a001c"
    "a168c4731e1e0e616df838a30bc22bf92b3698e19f12b58f94b807f57653864a"
    "d3319a6b278d53065fabd7acc95a85aff0baac8b9bd50cefe668ba8e1ef",
    16,
)


def _encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _sign(payload: bytes) -> bytes:
    width = (TEST_RSA_N.bit_length() + 7) // 8
    digest_info = RSA_SHA256_DIGEST_INFO + sha256(payload).digest()
    encoded = (
        b"\x00\x01" + b"\xff" * (width - len(digest_info) - 3) + b"\x00" + digest_info
    )
    signature = pow(int.from_bytes(encoded, "big"), TEST_RSA_D, TEST_RSA_N)
    return signature.to_bytes(width, "big").hex().encode()


class TestReleaseEvidenceValidation(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        reported_model_tree: str | None = None,
        policy_approved: bool = True,
    ) -> dict[str, object]:
        files = {
            "config.json": "1" * 64,
            "model.safetensors": "2" * 64,
            "tokenizer.json": "3" * 64,
        }
        manifest_bytes = (
            '[model]\nfamily = "fixture"\n\n[files]\n'
            + "".join(f'"{name}" = "{files[name]}"\n' for name in sorted(files))
        ).encode()
        config_bytes = b"[project]\noffline = true\n"
        key = {
            "schema": "cogni.rsa.public_key.v1",
            "key_id": "independent.lab1",
            "algorithm": "rsa-pkcs1v15-sha256",
            "modulus_hex": format(TEST_RSA_N, "x"),
            "exponent": 65537,
        }
        key_path = root / "key"
        key_sha = _write(key_path, _encoded(key))
        policy = {
            "schema": "cogni.release.verifier-policy.v1",
            "status": "approved" if policy_approved else "unconfigured",
            "verifier_id": "independent.lab1" if policy_approved else None,
            "public_key_sha256": key_sha if policy_approved else None,
        }
        repo = root / "repo"
        expanded = root / "expanded"
        (repo / "config").mkdir(parents=True)
        (repo / "config" / "gemma4-e4b-it.manifest.toml").write_bytes(manifest_bytes)
        (repo / "config" / "default.toml").write_bytes(config_bytes)
        (repo / "config" / "release-verifier-policy.json").write_bytes(_encoded(policy))
        (repo / "tracked.txt").write_text("guard-compatible source\n", encoding="utf-8")
        for command in (
            ("git", "init", "-q", str(repo)),
            ("git", "-C", str(repo), "config", "core.autocrlf", "false"),
            ("git", "-C", str(repo), "add", "--all"),
            (
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Release Test",
                "-c",
                "user.email=release@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ),
        ):
            subprocess.run(command, check=True, capture_output=True)
        commit = subprocess.run(
            ("git", "-C", str(repo), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        shutil.copytree(repo, expanded, ignore=shutil.ignore_patterns(".git"))
        located_git = shutil.which("git")
        if located_git is None:
            raise AssertionError("Git is required for the release evidence fixture")
        git_executable = Path(located_git).resolve(strict=True)
        git_executable_sha = sha256(git_executable.read_bytes()).hexdigest()
        source_tree = compute_guard_source_tree_digest(
            source_repo=repo,
            expanded_source=expanded,
            expected_commit=commit,
            git_executable=git_executable,
            git_executable_sha256=git_executable_sha,
        )
        paths = {
            name: root / name
            for name in (
                "summary",
                "cpu",
                "gpu",
                "attestation",
                "signature",
                "runtime",
                "completion",
                "identity_pre",
                "identity_post",
                "config_evidence",
                "device_evidence",
                "model_inventory",
            )
        }
        paths.update(
            {
                "key": key_path,
                "policy": expanded / "config" / "release-verifier-policy.json",
                "manifest": expanded / "config" / "gemma4-e4b-it.manifest.toml",
                "config": expanded / "config" / "default.toml",
            }
        )
        manifest_sha = _write(paths["manifest"], manifest_bytes)
        config_sha = _write(paths["config"], config_bytes)
        policy_sha = sha256(paths["policy"].read_bytes()).hexdigest()
        actual_model_tree = _canonical_model_tree_digest(files)
        model_tree = reported_model_tree or actual_model_tree

        identity_base = {
            "schema": "cogni.gpu5.identity.v1",
            "status": "passed",
            "source_commit": commit,
            "source_tree_digest": source_tree,
            "physical_gpu_index": 5,
            "gpu_uuid": GPU5_UUID,
            "image_digest": IMAGE_DIGEST,
        }
        identity_pre = {**identity_base, "phase": "pre"}
        identity_post = {**identity_base, "phase": "post"}
        identity_pre_sha = _write(paths["identity_pre"], _encoded(identity_pre))
        identity_post_sha = _write(paths["identity_post"], _encoded(identity_post))
        device_evidence = {
            "schema": "cogni.gpu5.device.v1",
            "status": "passed",
            "source_commit": commit,
            "source_tree_digest": source_tree,
            "physical_gpu_index": 5,
            "gpu_uuid": GPU5_UUID,
            "image_digest": IMAGE_DIGEST,
            "identity_pre_sha256": identity_pre_sha,
            "identity_post_sha256": identity_post_sha,
        }
        device_evidence["device_digest"] = _canonical_device_digest(device_evidence)
        device_sha = _write(paths["device_evidence"], _encoded(device_evidence))

        config_evidence = {
            "schema": "cogni.gpu5.config.v1",
            "status": "passed",
            "source_commit": commit,
            "source_tree_digest": source_tree,
            "config_sha256": config_sha,
        }
        config_evidence_sha = _write(
            paths["config_evidence"], _encoded(config_evidence)
        )
        model_inventory = {
            "schema": "cogni.gpu5.model-inventory.v1",
            "status": "passed",
            "source_commit": commit,
            "source_tree_digest": source_tree,
            "model_manifest_sha256": manifest_sha,
            "model_tree_digest": model_tree,
            "files": files,
        }
        model_inventory_sha = _write(
            paths["model_inventory"], _encoded(model_inventory)
        )
        runtime_scope = {
            "source_commit": commit,
            "source_tree_digest": source_tree,
            "model_manifest_sha256": manifest_sha,
            "model_tree_digest": model_tree,
            "config_digest": config_sha,
            "device_digest": device_evidence["device_digest"],
            "physical_gpu_index": 5,
            "gpu_uuid": GPU5_UUID,
            "image_digest": IMAGE_DIGEST,
        }
        runtime = {
            "schema": "cogni.gpu5.runtime.v1",
            "status": "passed",
            **runtime_scope,
        }
        runtime_sha = _write(paths["runtime"], _encoded(runtime))
        completion = {
            "schema": "cogni.gpu5.completion.v1",
            "status": "passed",
            **runtime_scope,
            "runtime_evidence_sha256": runtime_sha,
        }
        completion_sha = _write(paths["completion"], _encoded(completion))

        component_shas = {
            "runtime_evidence_sha256": runtime_sha,
            "completion_evidence_sha256": completion_sha,
            "identity_pre_sha256": identity_pre_sha,
            "identity_post_sha256": identity_post_sha,
            "config_evidence_sha256": config_evidence_sha,
            "device_evidence_sha256": device_sha,
            "model_inventory_sha256": model_inventory_sha,
        }
        cpu = {
            "schema": "cogni.cpu.gates.v1",
            "status": "passed",
            "source_commit": commit,
            "source_tree_digest": source_tree,
            "required_gates": {
                "ruff_check": True,
                "ruff_format_check": True,
                "pytest": True,
                "node_syntax_check": True,
                "clean_same_commit": True,
            },
            "pytest_passed": 1079,
            "pytest_failed": 0,
            "stdout_sha256": "4" * 64,
            "stderr_sha256": "5" * 64,
        }
        gpu = {
            "schema": "cogni.gpu5.gates.v2",
            "status": "passed",
            **runtime_scope,
            "required_gates": {"runtime": "passed", "completion": "passed"},
            **component_shas,
        }
        cpu_sha = _write(paths["cpu"], _encoded(cpu))
        gpu_sha = _write(paths["gpu"], _encoded(gpu))
        summary = {
            "schema": "cogni.release.gates.v2",
            "status": "passed",
            "source_commit": commit,
            "cpu": {
                "status": "passed",
                "evidence_sha256": cpu_sha,
                "source_tree_digest": source_tree,
                "required_gates": deepcopy(cpu["required_gates"]),
                "pytest_passed": 1079,
                "pytest_failed": 0,
            },
            "gpu5": {
                "status": "passed",
                "evidence_sha256": gpu_sha,
                "required_gates": deepcopy(gpu["required_gates"]),
                "physical_gpu_index": 5,
                "gpu_uuid": GPU5_UUID,
                "image_digest": IMAGE_DIGEST,
                "model_manifest_sha256": manifest_sha,
                "source_tree_digest": source_tree,
                "model_tree_digest": model_tree,
                "config_digest": config_sha,
                "device_digest": device_evidence["device_digest"],
                **component_shas,
            },
            "independent_verifier": {
                "status": "passed",
                "verifier_id": "independent.lab1",
            },
        }
        summary_sha = _write(paths["summary"], _encoded(summary))
        attestation = {
            "schema": "cogni.release.attestation.v2",
            "status": "passed",
            "verifier_id": "independent.lab1",
            "source_commit": commit,
            "summary_sha256": summary_sha,
            "cpu_evidence_sha256": cpu_sha,
            "gpu5_evidence_sha256": gpu_sha,
            "source_tree_digest": source_tree,
            "model_manifest_sha256": manifest_sha,
            "model_tree_digest": model_tree,
            "config_digest": config_sha,
            "device_digest": device_evidence["device_digest"],
            **component_shas,
            "issued_at_utc": "2026-07-18T00:00:00Z",
        }
        attestation_bytes = _encoded(attestation)
        attestation_sha = _write(paths["attestation"], attestation_bytes)
        signature_sha = _write(paths["signature"], _sign(attestation_bytes))
        arguments = {
            "summary_path": paths["summary"],
            "summary_sha256": summary_sha,
            "cpu_path": paths["cpu"],
            "cpu_sha256": cpu_sha,
            "gpu5_path": paths["gpu"],
            "gpu5_sha256": gpu_sha,
            "attestation_path": paths["attestation"],
            "attestation_sha256": attestation_sha,
            "signature_path": paths["signature"],
            "signature_sha256": signature_sha,
            "public_key_path": paths["key"],
            "public_key_sha256": key_sha,
            "verifier_policy_path": paths["policy"],
            "verifier_policy_sha256": policy_sha,
            "runtime_path": paths["runtime"],
            "runtime_sha256": runtime_sha,
            "completion_path": paths["completion"],
            "completion_sha256": completion_sha,
            "identity_pre_path": paths["identity_pre"],
            "identity_pre_sha256": identity_pre_sha,
            "identity_post_path": paths["identity_post"],
            "identity_post_sha256": identity_post_sha,
            "config_evidence_path": paths["config_evidence"],
            "config_evidence_sha256": config_evidence_sha,
            "device_evidence_path": paths["device_evidence"],
            "device_evidence_sha256": device_sha,
            "model_inventory_path": paths["model_inventory"],
            "model_inventory_sha256": model_inventory_sha,
            "model_manifest_path": paths["manifest"],
            "expected_model_manifest_sha256": manifest_sha,
            "config_path": paths["config"],
            "expected_config_sha256": config_sha,
            "source_repo_path": repo,
            "expanded_source_path": expanded,
            "git_executable_path": git_executable,
            "git_executable_sha256": git_executable_sha,
            "expected_source_commit": commit,
        }
        return {
            "arguments": arguments,
            "cpu": cpu,
            "gpu": gpu,
            "summary": summary,
            "paths": paths,
            "source_tree": source_tree,
            "repo": repo,
            "expanded": expanded,
            "commit": commit,
        }

    def test_release_source_digest_is_byte_identical_to_gpu5_guard_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            repo = fixture["repo"]
            expanded = fixture["expanded"]
            commit = fixture["commit"]
            raw = subprocess.run(
                ("git", "-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", commit),
                check=True,
                capture_output=True,
            ).stdout
            records = []
            for row in (item for item in raw.split(b"\0") if item):
                metadata, encoded_name = row.split(b"\t", 1)
                mode, _kind, oid = metadata.split(b" ", 2)
                name = encoded_name.decode("utf-8")
                payload = (expanded / Path(name)).read_bytes()
                records.append(
                    (name, sha256(payload).hexdigest(), mode.decode(), oid.decode())
                )
            self.assertEqual(fixture["source_tree"], _canonical_scope_digest(records))

            (expanded / "tracked.txt").write_text("archive changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseEvidenceError, "blob differs"):
                compute_guard_source_tree_digest(
                    source_repo=repo,
                    expanded_source=expanded,
                    expected_commit=commit,
                    git_executable=fixture["arguments"]["git_executable_path"],
                    git_executable_sha256=fixture["arguments"]["git_executable_sha256"],
                )

            (expanded / "tracked.txt").write_text(
                "guard-compatible source\n", encoding="utf-8"
            )
            subprocess.run(
                ("git", "-C", str(repo), "update-index", "--chmod=+x", "tracked.txt"),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                (
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Release Test",
                    "-c",
                    "user.email=release@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "mode change",
                ),
                check=True,
                capture_output=True,
            )
            with self.assertRaisesRegex(ReleaseEvidenceError, "HEAD differs"):
                compute_guard_source_tree_digest(
                    source_repo=repo,
                    expanded_source=expanded,
                    expected_commit=commit,
                    git_executable=fixture["arguments"]["git_executable_path"],
                    git_executable_sha256=fixture["arguments"]["git_executable_sha256"],
                )

    def test_valid_signed_raw_and_component_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            payload = validate_release_evidence(**fixture["arguments"])
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["verifier_id"], "independent.lab1")
        self.assertEqual(payload["schema"], "cogni.release.validation.v2")

    def test_rejects_duplicate_keys_string_true_integer_true_and_scope_drift(
        self,
    ) -> None:
        for mutation in ("duplicate", "string_true", "integer_true", "scope"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = self._fixture(Path(temporary))
                arguments = fixture["arguments"]
                paths = fixture["paths"]
                if mutation == "duplicate":
                    payload = paths["summary"].read_bytes()
                    payload = payload[:-2] + b',"status":"passed"}\n'
                    paths["summary"].write_bytes(payload)
                    arguments["summary_sha256"] = sha256(payload).hexdigest()
                elif mutation in {"string_true", "integer_true"}:
                    cpu = deepcopy(fixture["cpu"])
                    cpu["required_gates"]["pytest"] = (
                        "true" if mutation == "string_true" else 1
                    )
                    payload = _encoded(cpu)
                    paths["cpu"].write_bytes(payload)
                    arguments["cpu_sha256"] = sha256(payload).hexdigest()
                else:
                    (arguments["expanded_source_path"] / "tracked.txt").write_text(
                        "forged but self-consistent scope\n", encoding="utf-8"
                    )
                with self.assertRaises(ReleaseEvidenceError):
                    validate_release_evidence(**arguments)

    def test_rejects_unpinned_bytes_and_bad_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            fixture["arguments"]["summary_sha256"] = "0" * 64
            with self.assertRaisesRegex(ReleaseEvidenceError, "pinned digest"):
                validate_release_evidence(**fixture["arguments"])

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            signature = bytearray(fixture["paths"]["signature"].read_bytes())
            signature[0] = ord("0") if signature[0] != ord("0") else ord("1")
            fixture["paths"]["signature"].write_bytes(signature)
            fixture["arguments"]["signature_sha256"] = sha256(signature).hexdigest()
            with self.assertRaisesRegex(
                ReleaseEvidenceError, "signature verification failed"
            ):
                validate_release_evidence(**fixture["arguments"])

    def test_unconfigured_source_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary), policy_approved=False)
            with self.assertRaisesRegex(ReleaseEvidenceError, "publication is blocked"):
                validate_release_evidence(**fixture["arguments"])

    def test_source_policy_rejects_arbitrary_other_valid_rsa_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            alternate_key = {
                "schema": "cogni.rsa.public_key.v1",
                "key_id": "independent.attacker",
                "algorithm": "rsa-pkcs1v15-sha256",
                "modulus_hex": format(ALTERNATE_VALID_RSA_N, "x"),
                "exponent": 65537,
            }
            payload = _encoded(alternate_key)
            fixture["paths"]["key"].write_bytes(payload)
            fixture["arguments"]["public_key_sha256"] = sha256(payload).hexdigest()
            with self.assertRaisesRegex(
                ReleaseEvidenceError, "source-approved immutable key"
            ):
                validate_release_evidence(**fixture["arguments"])

    def test_rejects_self_consistent_signed_forged_component_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary), reported_model_tree="0" * 64)
            with self.assertRaisesRegex(ReleaseEvidenceError, "model inventory"):
                validate_release_evidence(**fixture["arguments"])


if __name__ == "__main__":
    unittest.main()
