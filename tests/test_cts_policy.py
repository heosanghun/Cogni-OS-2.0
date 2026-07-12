from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import torch

from cogni_core.cts_policy import (
    ACTION_LOGIT_BOUND,
    ACTION_WIDTH,
    ACT_BOUNDS,
    CTSCheckpointError,
    CTSControlError,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CHECKPOINT_SHA256,
    EXPLORATION_BOUNDS,
    MAX_LATENT_ELEMENTS,
    META_CONTROL_DIM,
    SUMMARY_DIM,
    TEMPERATURE_BOUNDS,
    TOLERANCE_BOUNDS,
    load_bounded_cts_controller,
    load_default_bounded_cts_controller,
    rebuild_offline_checkpoint_payload,
    summarize_latent,
)


def _canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _write_checkpoint(root: Path, payload: dict) -> tuple[Path, str]:
    raw = _canonical(payload)
    path = root / "candidate.json"
    path.write_bytes(raw)
    return path, sha256(raw).hexdigest()


class TestLearnedCTSControl(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = load_default_bounded_cts_controller()

    def test_summary_is_fixed_bounded_and_rejects_malformed_latents(self) -> None:
        for latent in (
            torch.zeros(1),
            torch.linspace(-2.0, 3.0, 17),
            torch.arange(48, dtype=torch.float32).reshape(2, 3, 8),
        ):
            with self.subTest(shape=tuple(latent.shape)):
                summary = summarize_latent(latent)
                self.assertEqual(summary.shape, (SUMMARY_DIM,))
                self.assertTrue(bool(torch.isfinite(summary).all()))
                self.assertLessEqual(float(summary.abs().max()), 1.0)

        malformed = (
            torch.empty(0),
            torch.ones(3, dtype=torch.int64),
            torch.tensor([0.0, float("nan")]),
        )
        for latent in malformed:
            with self.subTest(dtype=str(latent.dtype), elements=latent.numel()):
                with self.assertRaises((TypeError, ValueError)):
                    summarize_latent(latent)
        with self.assertRaises(ValueError):
            summarize_latent(torch.empty(MAX_LATENT_ELEMENTS + 1))

    def test_outputs_are_deterministic_frozen_finite_and_bounded(self) -> None:
        latent = torch.linspace(-1.0, 1.0, 64)
        first = self.controller(latent)
        second = self.controller(latent.clone())

        self.assertFalse(self.controller.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in self.controller.parameters()
            )
        )
        self.assertFalse(first.action_logits.requires_grad)
        self.assertEqual(first.action_logits.shape, (ACTION_WIDTH,))
        self.assertEqual(first.critic_value.numel(), 1)
        self.assertEqual(first.meta_controls.tensor.shape, (META_CONTROL_DIM,))
        torch.testing.assert_close(
            first.action_logits, second.action_logits, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            first.critic_value, second.critic_value, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            first.meta_controls.tensor,
            second.meta_controls.tensor,
            rtol=0.0,
            atol=0.0,
        )
        self.assertLessEqual(float(first.action_logits.abs().max()), ACTION_LOGIT_BOUND)
        self.assertLessEqual(float(first.critic_value.abs()), 1.0)
        controls = first.meta_controls.tensor
        lower = controls.new_tensor(
            (
                EXPLORATION_BOUNDS[0],
                TOLERANCE_BOUNDS[0],
                TEMPERATURE_BOUNDS[0],
                ACT_BOUNDS[0],
            )
        )
        upper = controls.new_tensor(
            (
                EXPLORATION_BOUNDS[1],
                TOLERANCE_BOUNDS[1],
                TEMPERATURE_BOUNDS[1],
                ACT_BOUNDS[1],
            )
        )
        self.assertTrue(bool(((controls >= lower) & (controls <= upper)).all()))
        with self.assertRaises(CTSControlError):
            self.controller.train(True)

    def test_counterfactual_latents_cover_all_actions_without_dominant_logits(
        self,
    ) -> None:
        latents = (
            torch.cat((torch.ones(16), -torch.ones(16))),
            torch.cat((-torch.ones(16), torch.ones(16))),
            -torch.ones(32),
        )
        outputs = [self.controller(latent) for latent in latents]
        actions = {int(output.action_logits.argmax()) for output in outputs}

        self.assertEqual(actions, {0, 1, 2})
        self.assertEqual(
            self.controller.provenance.evidence.action_coverage,
            ACTION_WIDTH,
        )
        self.assertLess(
            self.controller.provenance.evidence.max_action_fraction,
            0.5,
        )
        self.assertEqual(len({float(output.critic_value) for output in outputs}), 3)
        self.assertFalse(
            torch.equal(outputs[0].action_logits, outputs[1].action_logits)
        )

    def test_critic_and_action_policy_are_computationally_separate(self) -> None:
        policy_parameters = {
            id(item) for item in self.controller.policy_head.parameters()
        }
        critic_parameters = {
            id(item) for item in self.controller.critic_head.parameters()
        }
        self.assertTrue(policy_parameters.isdisjoint(critic_parameters))
        latent = torch.linspace(-0.5, 0.75, 32)

        def forbidden(*_args):
            raise AssertionError("separate head was unexpectedly evaluated")

        policy_hook = self.controller.policy_head.register_forward_hook(forbidden)
        try:
            value = self.controller.critic(latent)
            self.assertTrue(bool(torch.isfinite(value)))
        finally:
            policy_hook.remove()

        critic_hook = self.controller.critic_head.register_forward_hook(forbidden)
        try:
            logits = self.controller.policy_logits(latent)
            self.assertEqual(logits.shape, (ACTION_WIDTH,))
        finally:
            critic_hook.remove()

    def test_search_pair_and_o1_persistent_state_contract(self) -> None:
        state_keys = tuple(self.controller.state_dict())
        parameter_count = sum(item.numel() for item in self.controller.parameters())
        for elements in (1, 8, 128, 4096):
            logits, value = self.controller.policy_value(torch.ones(elements))
            self.assertEqual(logits.shape, (ACTION_WIDTH,))
            self.assertEqual(value.numel(), 1)
        self.assertEqual(tuple(self.controller.state_dict()), state_keys)
        self.assertEqual(
            sum(item.numel() for item in self.controller.parameters()),
            parameter_count,
        )
        self.assertEqual(len(state_keys), 6)
        self.assertEqual(tuple(self.controller.buffers()), ())

    def test_runtime_non_finite_head_tamper_fails_closed(self) -> None:
        with torch.no_grad():
            self.controller.policy_head.weight[0, 0] = float("inf")
        with self.assertRaises(CTSControlError):
            self.controller.policy_logits(torch.ones(32))


class TestCTSCheckpoint(unittest.TestCase):
    def test_bundled_digest_provenance_and_offline_fit_reproduce(self) -> None:
        raw = DEFAULT_CHECKPOINT_PATH.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(), DEFAULT_CHECKPOINT_SHA256)
        artifact = json.loads(raw.decode("utf-8"))
        self.assertEqual(artifact, rebuild_offline_checkpoint_payload())

        controller = load_default_bounded_cts_controller()
        evidence = controller.provenance.evidence
        self.assertGreaterEqual(evidence.policy_accuracy, 0.8)
        self.assertLessEqual(evidence.critic_mae, 0.1)
        self.assertLessEqual(evidence.meta_normalized_mae, 0.1)
        self.assertEqual(evidence.action_coverage, ACTION_WIDTH)
        self.assertEqual(controller.checkpoint_sha256, DEFAULT_CHECKPOINT_SHA256)

    def test_digest_mismatch_is_rejected_before_json_or_model_load(self) -> None:
        with self.assertRaisesRegex(CTSCheckpointError, "SHA-256"):
            load_bounded_cts_controller(
                DEFAULT_CHECKPOINT_PATH,
                expected_sha256="0" * 64,
            )

    def test_untrained_unknown_or_false_evidence_checkpoints_fail_closed(self) -> None:
        mutations = (
            lambda payload: payload["provenance"].__setitem__("trained", False),
            lambda payload: payload["provenance"].__setitem__(
                "integrity", "self-asserted"
            ),
            lambda payload: payload["provenance"]["heldout"].__setitem__(
                "policy_accuracy", 0.1
            ),
        )
        for index, mutate in enumerate(mutations):
            with (
                self.subTest(mutation=index),
                tempfile.TemporaryDirectory() as temporary,
            ):
                payload = deepcopy(rebuild_offline_checkpoint_payload())
                mutate(payload)
                path, digest = _write_checkpoint(Path(temporary), payload)
                with self.assertRaises(CTSCheckpointError):
                    load_bounded_cts_controller(path, expected_sha256=digest)

    def test_state_tamper_cannot_pass_with_recomputed_file_and_state_digests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = deepcopy(rebuild_offline_checkpoint_payload())
            payload["state"]["critic_head.weight"] = [[0.0] * SUMMARY_DIM]
            payload["state"]["critic_head.bias"] = [0.0]
            payload["provenance"]["state_sha256"] = sha256(
                json.dumps(
                    payload["state"],
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest()
            path, digest = _write_checkpoint(Path(temporary), payload)
            with self.assertRaises(CTSCheckpointError):
                load_bounded_cts_controller(path, expected_sha256=digest)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_verified_controller_can_follow_a_cuda_latent(self) -> None:
        controller = load_default_bounded_cts_controller(device="cuda")
        output = controller(torch.linspace(-1.0, 1.0, 32, device="cuda"))
        self.assertEqual(output.action_logits.device.type, "cuda")
        self.assertEqual(output.meta_controls.tensor.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
