from __future__ import annotations

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cogni_demo.semantic_embedder import (
    LocalSemanticEmbedder,
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
    overrides: dict[str, object] | None = None,
) -> Path:
    files = {
        "LICENSE": b"Apache License 2.0\n",
        "config.json": b'{"model_type":"bounded-test"}\n',
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


class TestLocalSemanticEmbedder(unittest.TestCase):
    def test_artifact_mutation_after_admission_fails_before_backend_load(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)
            calls = 0

            def factory(manifest):
                nonlocal calls
                calls += 1
                return _FakeBackend(manifest.dimensions)

            embedder = LocalSemanticEmbedder(root, backend_factory=factory)
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

            embedder = LocalSemanticEmbedder(root, backend_factory=factory)
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

            embedder = LocalSemanticEmbedder(root, backend_factory=factory)
            before = embedder.status_payload()
            self.assertFalse(before["loaded"])
            self.assertEqual(before["device"], "cpu")
            self.assertEqual(before["vram_bytes"], 0)
            self.assertFalse(before["network_access"])
            self.assertTrue(before["artifact_verified"])
            self.assertTrue(before["semantic_embedding"])
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

            embedder = LocalSemanticEmbedder(root, backend_factory=factory)
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
                )
                with self.assertRaises(SemanticEmbedderError) as raised:
                    embedder.encode(("bounded",))
                self.assertEqual(raised.exception.code, code)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_manifest(root)

            def broken_factory(_manifest):
                raise RuntimeError("sensitive host detail")

            embedder = LocalSemanticEmbedder(root, backend_factory=broken_factory)
            with self.assertRaises(SemanticEmbedderError) as raised:
                embedder.encode(("bounded",))
            self.assertEqual(raised.exception.code, "SEMANTIC_MODEL_LOAD_FAILED")
            self.assertNotIn("sensitive", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
