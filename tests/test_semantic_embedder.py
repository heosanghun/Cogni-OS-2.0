from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from cogni_demo.semantic_embedder import (
    LocalSemanticEmbedder,
    MAX_ARTIFACT_FILES,
    SEMANTIC_EMBEDDER_MANIFEST,
    SEMANTIC_EMBEDDER_SCHEMA,
    SemanticEmbedderError,
    verify_semantic_embedder_manifest,
)


class _FakeBackend:
    def __init__(self, dimensions: int, *, output=None) -> None:
        self.dimensions = dimensions
        self.output = output
        self.calls: list[tuple[str, ...]] = []
        self.closed = False

    def encode(self, texts: tuple[str, ...]):
        self.calls.append(texts)
        if self.output is not None:
            return self.output
        return [
            [float(index + offset + 1) for index in range(self.dimensions)]
            for offset, _text in enumerate(texts)
        ]

    def close(self) -> None:
        self.closed = True


def _write_manifest(
    root: Path,
    *,
    dimensions: int = 4,
    max_input_chars: int = 128,
    max_batch_size: int = 3,
    config_overrides: dict[str, object] | None = None,
    overrides: dict[str, object] | None = None,
) -> Path:
    config: dict[str, object] = {
        "model_type": "bert",
        "hidden_size": dimensions,
        "intermediate_size": max(4, dimensions * 2),
        "num_hidden_layers": 2,
        "num_attention_heads": 1,
        "vocab_size": 256,
        "max_position_embeddings": 128,
        "type_vocab_size": 2,
    }
    if config_overrides:
        config.update(config_overrides)
    files = {
        "LICENSE": b"Apache License 2.0\n",
        "config.json": (
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        ),
        "model.safetensors": b"bounded semantic weights fixture\n",
        "tokenizer.json": b'{"version":"1.0"}\n',
    }
    for relative, content in files.items():
        (root / relative).write_bytes(content)
    payload: dict[str, object] = {
        "schema": SEMANTIC_EMBEDDER_SCHEMA,
        "model_id": "local/bounded-semantic-test",
        "revision": "a" * 40,
        "backend": "transformers_mean_pool_v1",
        "dimensions": dimensions,
        "max_input_chars": max_input_chars,
        "max_batch_size": max_batch_size,
        "license_spdx": "Apache-2.0",
        "license_file": "LICENSE",
        "files": {
            relative: sha256(content).hexdigest() for relative, content in files.items()
        },
    }
    if overrides:
        payload.update(overrides)
    manifest = root / SEMANTIC_EMBEDDER_MANIFEST
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return manifest


class TestSemanticEmbedderManifest(unittest.TestCase):
    def test_closed_world_manifest_binds_identity_license_and_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_manifest(root)
            verified = verify_semantic_embedder_manifest(root)

            self.assertEqual(verified.path, manifest_path.resolve())
            self.assertEqual(verified.model_id, "local/bounded-semantic-test")
            self.assertEqual(verified.revision, "a" * 40)
            self.assertEqual(verified.dimensions, 4)
            self.assertEqual(verified.license_file, "LICENSE")
            self.assertEqual(len(verified.files), 4)
            self.assertEqual(verified.model_type, "bert")
            self.assertEqual(verified.model_max_tokens, 128)
            self.assertGreater(verified.estimated_parameter_upper_bound, 0)
            self.assertGreater(verified.estimated_peak_ram_bytes, 0)
            self.assertRegex(verified.manifest_sha256, r"[0-9a-f]{64}")
            self.assertRegex(verified.profile, r"local_semantic_[0-9a-f]{16}_v1")

    def test_digest_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(
                SemanticEmbedderError, "digest does not match"
            ) as raised:
                verify_semantic_embedder_manifest(root)
            self.assertEqual(raised.exception.code, "SEMANTIC_ARTIFACT_DIGEST_MISMATCH")

    def test_unmanifested_file_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root)
            self.assertEqual(
                raised.exception.code, "SEMANTIC_ARTIFACT_CLOSED_WORLD_FAILED"
            )

    def test_escaping_file_and_unmanifested_license_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(
                root,
                overrides={
                    "files": {"../escape": "0" * 64},
                    "license_file": "../LICENSE",
                },
            )
            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root)
            self.assertEqual(raised.exception.code, "SEMANTIC_MANIFEST_INVALID")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["files"].pop("LICENSE")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root)
            self.assertEqual(raised.exception.code, "SEMANTIC_LICENSE_UNVERIFIED")

    def test_fixed_root_manifest_and_backend_are_required(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root)
            renamed = root / "other.json"
            manifest.rename(renamed)
            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root, renamed)
            self.assertEqual(raised.exception.code, "SEMANTIC_MANIFEST_INVALID")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root, overrides={"backend": "remote_api"})
            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root)
            self.assertEqual(raised.exception.code, "SEMANTIC_BACKEND_UNSUPPORTED")

    def test_symlinked_root_is_never_promoted_to_verified_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "model"
            root.mkdir()
            _write_manifest(root)
            linked = parent / "linked-model"
            try:
                linked.symlink_to(root, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlink creation is unavailable")
            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(linked)
            self.assertEqual(raised.exception.code, "SEMANTIC_ARTIFACT_UNSAFE")

    def test_network_root_is_rejected_before_filesystem_access(self) -> None:
        with self.assertRaises(SemanticEmbedderError) as raised:
            verify_semantic_embedder_manifest(r"\\semantic-host\model")
        self.assertEqual(raised.exception.code, "SEMANTIC_ARTIFACT_UNSAFE")

    def test_inventory_file_count_is_bounded_while_enumerating(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            for index in range(MAX_ARTIFACT_FILES):
                (root / f"extra-{index:03d}.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root)
            self.assertEqual(raised.exception.code, "SEMANTIC_ARTIFACT_TOO_LARGE")

    def test_unbounded_or_dynamic_model_configs_are_rejected(self) -> None:
        cases = (
            {"model_type": "custom_remote_encoder"},
            {"auto_map": {"AutoModel": "remote.CustomModel"}},
            {"num_hidden_layers": 49},
        )
        for config_overrides in cases:
            with self.subTest(config_overrides=config_overrides):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    _write_manifest(root, config_overrides=config_overrides)
                    with self.assertRaises(SemanticEmbedderError) as raised:
                        verify_semantic_embedder_manifest(root)
                    self.assertEqual(
                        raised.exception.code, "SEMANTIC_MODEL_CONFIG_UNSAFE"
                    )

    def test_windows_case_insensitive_pickle_weight_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root)
            unsafe_name = "pytorch_model.BIN"
            unsafe_content = b"pickle-compatible decoy"
            (root / unsafe_name).write_bytes(unsafe_content)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["files"][unsafe_name] = sha256(unsafe_content).hexdigest()
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root)

            self.assertEqual(raised.exception.code, "SEMANTIC_ARTIFACT_UNSAFE")

    def test_huge_json_integer_has_a_stable_fail_closed_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root)
            raw = manifest.read_text(encoding="utf-8")
            raw = raw.replace('"dimensions": 4', '"dimensions": ' + "9" * 5_000)
            manifest.write_text(raw, encoding="utf-8")

            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root)

            self.assertEqual(raised.exception.code, "SEMANTIC_MANIFEST_INVALID")
            self.assertIn("oversized number", str(raised.exception))

    def test_retained_attention_outputs_and_quadratic_ram_are_rejected(self) -> None:
        for field in ("output_attentions", "output_hidden_states"):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                root = Path(directory)
                _write_manifest(root, config_overrides={field: True})
                with self.assertRaises(SemanticEmbedderError) as raised:
                    verify_semantic_embedder_manifest(root)
                self.assertEqual(raised.exception.code, "SEMANTIC_MODEL_CONFIG_UNSAFE")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(
                root,
                dimensions=128,
                max_batch_size=16,
                config_overrides={
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 16,
                    "max_position_embeddings": 8_192,
                    "attn_implementation": "eager",
                },
            )
            with self.assertRaises(SemanticEmbedderError) as raised:
                verify_semantic_embedder_manifest(root)
            self.assertEqual(raised.exception.code, "SEMANTIC_MODEL_CONFIG_UNSAFE")
            self.assertIn("peak CPU RAM", str(raised.exception))


class TestLocalSemanticEmbedder(unittest.TestCase):
    def test_production_path_loader_is_disabled_before_transformers_can_read(
        self,
    ) -> None:
        calls: list[Path] = []

        class _AutoLoader:
            @classmethod
            def from_pretrained(cls, root, **_kwargs):
                calls.append(Path(root))
                raise AssertionError("unbound path loader must never run")

        module = ModuleType("transformers")
        module.AutoModel = _AutoLoader
        module.AutoTokenizer = _AutoLoader
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            with patch.dict(sys.modules, {"transformers": module}):
                embedder = LocalSemanticEmbedder(root)
                with self.assertRaises(SemanticEmbedderError) as raised:
                    embedder.load()

            self.assertEqual(raised.exception.code, "SEMANTIC_LOADER_BINDING_UNPROVEN")
            self.assertEqual(calls, [])
            status = embedder.status_payload()
            self.assertFalse(status["loaded"])
            self.assertFalse(status["semantic_embedding"])
            self.assertFalse(status["production_backend_enabled"])
            self.assertEqual(status["backend_mode"], "production_path_loader_disabled")

    def test_unbound_backend_override_requires_explicit_test_admission(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            with self.assertRaises(SemanticEmbedderError) as raised:
                LocalSemanticEmbedder(
                    root,
                    backend_factory=lambda manifest: _FakeBackend(manifest.dimensions),
                )
            self.assertEqual(raised.exception.code, "SEMANTIC_TEST_BACKEND_NOT_ENABLED")

    def test_artifact_mutation_after_admission_fails_before_backend_load(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            calls = 0

            def factory(manifest):
                nonlocal calls
                calls += 1
                return _FakeBackend(manifest.dimensions)

            embedder = LocalSemanticEmbedder(
                root,
                backend_factory=factory,
                allow_unbound_test_backend=True,
            )
            (root / "model.safetensors").write_bytes(b"changed after admission")
            with self.assertRaises(SemanticEmbedderError) as raised:
                embedder.load()
            self.assertEqual(raised.exception.code, "SEMANTIC_ARTIFACT_DIGEST_MISMATCH")
            self.assertEqual(calls, 0)
            self.assertFalse(embedder.loaded)

    def test_artifact_mutation_during_backend_load_closes_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            created: list[_FakeBackend] = []

            def factory(manifest):
                backend = _FakeBackend(manifest.dimensions)
                created.append(backend)
                (root / "model.safetensors").write_bytes(b"changed during load")
                return backend

            embedder = LocalSemanticEmbedder(
                root,
                backend_factory=factory,
                allow_unbound_test_backend=True,
            )
            with self.assertRaises(SemanticEmbedderError) as raised:
                embedder.load()
            self.assertEqual(raised.exception.code, "SEMANTIC_ARTIFACT_DIGEST_MISMATCH")
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].closed)
            self.assertFalse(embedder.loaded)

    def test_encode_is_bounded_normalized_cpu_only_and_not_answer_bearing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            created: list[_FakeBackend] = []

            def factory(manifest):
                backend = _FakeBackend(manifest.dimensions)
                created.append(backend)
                return backend

            embedder = LocalSemanticEmbedder(
                root,
                backend_factory=factory,
                allow_unbound_test_backend=True,
            )
            before = embedder.status_payload()
            self.assertFalse(before["loaded"])
            self.assertEqual(before["device"], "cpu")
            self.assertEqual(before["vram_bytes"], 0)
            self.assertFalse(before["network_access"])
            self.assertTrue(before["artifact_verified"])
            self.assertFalse(before["semantic_embedding"])
            self.assertFalse(before["production_backend_enabled"])
            self.assertFalse(before["loader_binding_verified"])
            self.assertTrue(before["test_backend_override"])
            self.assertFalse(before["quality_attested"])
            self.assertFalse(before["answer_bearing"])
            self.assertFalse(before["production_ready"])

            vectors = embedder.encode(("첫 문장", "두 번째 문장"))
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].calls, [("첫 문장", "두 번째 문장")])
            self.assertEqual(len(vectors), 2)
            for vector in vectors:
                self.assertEqual(len(vector), 4)
                self.assertTrue(
                    isclose(sum(value * value for value in vector), 1.0, abs_tol=1e-7)
                )
            self.assertTrue(embedder.status_payload()["loaded"])
            self.assertTrue(embedder.status_payload()["semantic_embedding"])
            embedder.unload()
            self.assertTrue(created[0].closed)
            self.assertFalse(embedder.loaded)

    def test_batch_text_and_control_character_limits_fail_before_backend_load(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root, max_input_chars=32, max_batch_size=2)
            calls = 0

            def factory(manifest):
                nonlocal calls
                calls += 1
                return _FakeBackend(manifest.dimensions)

            embedder = LocalSemanticEmbedder(
                root,
                backend_factory=factory,
                allow_unbound_test_backend=True,
            )
            invalid = (
                (),
                ("a", "b", "c"),
                ("",),
                (" ",),
                ("x" * 33,),
                ("bad\x00text",),
            )
            for texts in invalid:
                with (
                    self.subTest(texts=texts),
                    self.assertRaises(SemanticEmbedderError),
                ):
                    embedder.encode(texts)
            with self.assertRaises(SemanticEmbedderError):
                embedder.encode("not a sequence of documents")
            self.assertEqual(calls, 0)

    def test_batch_bound_is_checked_before_sequence_copy(self) -> None:
        class _ExplosiveSequence(Sequence[str]):
            reads = 0

            def __len__(self) -> int:
                return 4

            def __getitem__(self, _index: int) -> str:
                self.reads += 1
                raise AssertionError("oversized sequence must not be copied")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root, max_batch_size=3)
            embedder = LocalSemanticEmbedder(
                root,
                backend_factory=lambda manifest: _FakeBackend(manifest.dimensions),
                allow_unbound_test_backend=True,
            )
            texts = _ExplosiveSequence()
            with self.assertRaises(SemanticEmbedderError) as raised:
                embedder.encode(texts)
            self.assertEqual(raised.exception.code, "SEMANTIC_BATCH_LIMIT")
            self.assertEqual(texts.reads, 0)
            self.assertFalse(embedder.loaded)

    def test_invalid_output_shape_nan_zero_norm_and_backend_failure_fail_closed(
        self,
    ) -> None:
        cases = (
            ([[1.0, 2.0]], "SEMANTIC_OUTPUT_INVALID"),
            ([[1.0, 2.0, 3.0, float("nan")]], "SEMANTIC_OUTPUT_INVALID"),
            ([[0.0, 0.0, 0.0, 0.0]], "SEMANTIC_OUTPUT_INVALID"),
            ([], "SEMANTIC_OUTPUT_INVALID"),
        )
        for output, code in cases:
            with self.subTest(output=output), TemporaryDirectory() as directory:
                root = Path(directory)
                _write_manifest(root)
                embedder = LocalSemanticEmbedder(
                    root,
                    backend_factory=lambda manifest, output=output: _FakeBackend(
                        manifest.dimensions, output=output
                    ),
                    allow_unbound_test_backend=True,
                )
                with self.assertRaises(SemanticEmbedderError) as raised:
                    embedder.encode(("bounded",))
                self.assertEqual(raised.exception.code, code)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)

            def broken_factory(_manifest):
                raise RuntimeError("sensitive host detail")

            embedder = LocalSemanticEmbedder(
                root,
                backend_factory=broken_factory,
                allow_unbound_test_backend=True,
            )
            with self.assertRaises(SemanticEmbedderError) as raised:
                embedder.encode(("bounded",))
            self.assertEqual(raised.exception.code, "SEMANTIC_MODEL_LOAD_FAILED")
            self.assertNotIn("sensitive", str(raised.exception))

    def test_lying_sequences_are_copied_by_exact_index_without_iteration(self) -> None:
        class _LyingSequence(Sequence[object]):
            def __init__(self, declared: int, item: object) -> None:
                self.declared = declared
                self.item = item
                self.iterated = False

            def __len__(self) -> int:
                return self.declared

            def __getitem__(self, _index: int) -> object:
                return self.item

            def __iter__(self):
                self.iterated = True
                raise AssertionError("untrusted iteration must not be used")

        for malicious_output in (
            [_LyingSequence(4, 1.0)],
            _LyingSequence(1, [1.0, 2.0, 3.0, 4.0]),
        ):
            with self.subTest(output_type=type(malicious_output).__name__):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    _write_manifest(root)
                    backend = _FakeBackend(4, output=malicious_output)
                    embedder = LocalSemanticEmbedder(
                        root,
                        backend_factory=lambda _manifest, backend=backend: backend,
                        allow_unbound_test_backend=True,
                    )
                    with self.assertRaises(SemanticEmbedderError) as raised:
                        embedder.encode(("bounded",))
                    self.assertEqual(raised.exception.code, "SEMANTIC_OUTPUT_INVALID")
                    liar = (
                        malicious_output[0]
                        if isinstance(malicious_output, list)
                        else malicious_output
                    )
                    self.assertFalse(liar.iterated)

        class _IterationLiar(Sequence[float]):
            def __init__(self) -> None:
                self.iterated = False

            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> float:
                if not 0 <= index < 4:
                    raise IndexError(index)
                return float(index + 1)

            def __iter__(self):
                self.iterated = True
                return iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            vector = _IterationLiar()
            backend = _FakeBackend(4, output=[vector])
            embedder = LocalSemanticEmbedder(
                root,
                backend_factory=lambda _manifest: backend,
                allow_unbound_test_backend=True,
            )

            output = embedder.encode(("bounded",))

            self.assertEqual(len(output[0]), 4)
            self.assertFalse(vector.iterated)

    def test_status_drops_artifact_verification_after_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            embedder = LocalSemanticEmbedder(root)
            self.assertTrue(embedder.status_payload()["artifact_verified"])
            (root / "model.safetensors").write_bytes(b"mutated")
            self.assertFalse(embedder.status_payload()["artifact_verified"])


if __name__ == "__main__":
    unittest.main()
