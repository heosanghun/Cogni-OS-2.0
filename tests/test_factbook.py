import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from cogni_os.capabilities import (
    CapabilityState,
    baseline_capability_registry,
)
from cogni_os.factbook import (
    FactBookError,
    MAX_PROMPT_CONTEXT_CHARS,
    build_runtime_factbook,
    build_runtime_factbook_from_verified,
    inspect_safetensors_headers,
)
from cogni_os.artifacts import verify_artifact_manifest


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for value in shape:
            count *= value
        item_bytes = 2 if dtype == "BF16" else 4
        size = count * item_bytes
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0" * offset)


def _write_manifest(path: Path, root: Path, names: tuple[str, ...]) -> None:
    lines = ["[files]"]
    for name in names:
        digest = sha256((root / name).read_bytes()).hexdigest()
        lines.append(f'"{name}" = "{digest}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCapabilityRegistry(unittest.TestCase):
    def test_baseline_discloses_actual_answer_authority(self):
        registry = baseline_capability_registry()
        self.assertEqual(
            registry.require("gemma4_e4b").state,
            CapabilityState.AUTHORITATIVE,
        )
        self.assertTrue(registry.require("gemma4_e4b").answer_bearing)
        self.assertIn(
            "instruction-tuned",
            registry.require("gemma4_e4b").detail,
        )
        self.assertNotIn("base Gemma", registry.require("gemma4_e4b").detail)
        self.assertEqual(
            registry.require("cts_deq").state,
            CapabilityState.CANARY,
        )
        self.assertTrue(registry.require("cts_deq").answer_bearing)
        self.assertIn("instruction-tuned", registry.require("cts_deq").detail)
        self.assertNotIn("base Gemma", registry.require("cts_deq").detail)
        self.assertFalse(registry.require("system_3").answer_bearing)
        self.assertEqual(
            registry.require("self_harness").state,
            CapabilityState.PROPOSAL_ONLY,
        )
        self.assertFalse(registry.require("self_harness").runtime_mutation_allowed)


class TestRuntimeFactBook(unittest.TestCase):
    def test_model_facts_come_from_verified_config_and_weight_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "gemma4-e4b"
            root.mkdir()
            config = {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {
                    "hidden_size": 2560,
                    "num_hidden_layers": 42,
                    "enable_moe_block": False,
                    "num_experts": None,
                },
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            _write_safetensors(
                root / "model.safetensors",
                {
                    "model.language_model.embed_tokens_per_layer.weight": (
                        "BF16",
                        [10, 4],
                    ),
                    "model.language_model.embed_tokens.weight": ("BF16", [10, 2]),
                    "model.language_model.layers.0.mlp.weight": ("BF16", [2, 3]),
                },
            )
            manifest = Path(temporary) / "manifest.toml"
            _write_manifest(manifest, root, ("config.json", "model.safetensors"))

            facts = build_runtime_factbook(
                root,
                manifest,
                build_version="0.3.0-dev",
                device="test GPU",
                generated_at="2026-07-12T00:00:00+00:00",
            )

            self.assertTrue(facts.model.dense)
            self.assertEqual(facts.model.inventory.stored_parameters, 66)
            self.assertEqual(facts.model.inventory.embedding_parameters, 60)
            self.assertEqual(facts.model.inventory.effective_parameters, 6)
            self.assertEqual(facts.model.inventory.tensor_count, 3)
            self.assertEqual(facts.model.architecture, "Gemma4ForConditionalGeneration")
            payload = facts.as_payload()
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["model"]["stored_parameters"], 66)
            self.assertIn("effective 파라미터 6개", facts.identity_summary_ko())

            prompt_context = facts.prompt_context_ko()
            self.assertEqual(prompt_context, facts.prompt_context_ko())
            self.assertLessEqual(len(prompt_context), MAX_PROMPT_CONTEXT_CHARS)
            self.assertIn("label=gemma4-e4b", prompt_context)
            self.assertIn("stored_parameters=66", prompt_context)
            self.assertIn("effective_parameters=6", prompt_context)
            self.assertIn("embedding_parameters=60", prompt_context)
            self.assertIn("hidden_size=2,560", prompt_context)
            self.assertIn("layers=42", prompt_context)
            self.assertIn(
                "gemma4_e4b: state=authoritative; evidence=verified; "
                "answer_bearing=true; runtime_mutation_allowed=false",
                prompt_context,
            )
            self.assertIn(
                "cts_deq: state=canary; evidence=measured; "
                "answer_bearing=true; runtime_mutation_allowed=false",
                prompt_context,
            )
            self.assertIn("system_1_5: state=gated", prompt_context)
            self.assertIn("system_2_5: state=night_only", prompt_context)
            self.assertIn(
                "system_2_5: state=night_only; evidence=verified; "
                "answer_bearing=false; runtime_mutation_allowed=true",
                prompt_context,
            )
            self.assertIn("self_harness: state=proposal_only", prompt_context)
            for record in facts.capabilities.records:
                expected = (
                    f"{record.name}: state={record.state.value}; "
                    f"evidence={record.evidence.value}; "
                    f"answer_bearing={'true' if record.answer_bearing else 'false'}; "
                    "runtime_mutation_allowed="
                    f"{'true' if record.runtime_mutation_allowed else 'false'}"
                )
                with self.subTest(capability=record.name):
                    self.assertIn(expected, prompt_context)

            verified = verify_artifact_manifest(root, manifest)
            reused = build_runtime_factbook_from_verified(
                verified,
                manifest,
                build_version="0.3.0-dev",
                device="test GPU",
                generated_at="2026-07-12T00:00:00+00:00",
            )
            self.assertEqual(reused.as_payload(), facts.as_payload())

    def test_unsafe_safetensors_offsets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.safetensors"
            header = {
                "weight": {
                    "dtype": "BF16",
                    "shape": [2, 2],
                    "data_offsets": [0, 1_000],
                }
            }
            encoded = json.dumps(header).encode("utf-8")
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)
            with self.assertRaises(FactBookError):
                inspect_safetensors_headers((path,))

    def test_unverified_config_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            root.mkdir()
            _write_safetensors(root / "model.safetensors", {"weight": ("BF16", [2, 2])})
            manifest = Path(temporary) / "manifest.toml"
            _write_manifest(manifest, root, ("model.safetensors",))
            with self.assertRaises(FactBookError):
                build_runtime_factbook(
                    root,
                    manifest,
                    build_version="test",
                    device="test GPU",
                )


if __name__ == "__main__":
    unittest.main()
