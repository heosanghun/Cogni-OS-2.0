from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from cogni_core.fp_ewc import FPEWCRegularizer
from cogni_flow.evolution import (
    CFireTarget,
    EvolutionTransaction,
    EvolutionTransactionError,
    GenerationCheckpointError,
    GenerationCheckpointStore,
    TransferMetrics,
    continual_transfer_metrics,
    summarize_seeded_transfer,
)
from cogni_flow.rhythm import RhythmController, SystemMode


def _assert_nested_equal(
    test: unittest.TestCase, first: object, second: object
) -> None:
    if isinstance(first, torch.Tensor):
        test.assertIsInstance(second, torch.Tensor)
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertIsInstance(second, dict)
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            _assert_nested_equal(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertIsInstance(second, type(first))
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second, strict=True):
            _assert_nested_equal(test, left, right)
    else:
        test.assertEqual(first, second)


class _SmallEvolutionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.core.weight.copy_(torch.tensor([[1.6, 0.2], [0.1, 1.1]]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.core(value)


def _transaction() -> EvolutionTransaction:
    model = _SmallEvolutionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    regularizer = FPEWCRegularizer(strength=0.2)
    regularizer.consolidate_fisher(
        model.named_parameters(),
        {"core.weight": torch.full_like(model.core.weight, 0.5)},
        n_samples=4,
    )
    rhythm = RhythmController(mode=SystemMode.EVOLUTION)
    return EvolutionTransaction(
        model,
        optimizer,
        regularizer,
        rhythm,
        [CFireTarget("core.weight", model.core.weight)],
    )


class TestEvolutionTransaction(unittest.TestCase):
    def test_real_ewc_optimizer_step_is_c_fire_certified_before_and_after(self) -> None:
        transaction = _transaction()
        inputs = torch.tensor([[1.0, -1.0], [0.25, 0.75]])
        target = torch.zeros_like(inputs)

        report = transaction.step(
            lambda: (transaction.model(inputs) - target).square().mean()
        )

        self.assertGreaterEqual(report.ewc_penalty, 0.0)
        self.assertNotEqual(report.before_digest, report.after_digest)
        self.assertEqual(set(report.pre_certificates), {"core.weight"})
        self.assertEqual(set(report.post_certificates), {"core.weight"})
        self.assertLess(report.post_certificates["core.weight"].after_sigma_max, 0.95)
        self.assertEqual(transaction.rhythm.active_evolution_tasks, 0)

    def test_nonfinite_candidate_restores_model_optimizer_and_fisher(self) -> None:
        transaction = _transaction()
        # Prime Adam so rollback covers non-empty moment tensors.
        transaction.step(lambda: transaction.model(torch.ones(1, 2)).square().mean())
        model_before = {
            name: value.detach().clone()
            for name, value in transaction.model.state_dict().items()
        }
        optimizer_before = deepcopy(transaction.optimizer.state_dict())
        fisher_before = deepcopy(transaction.regularizer.snapshots)

        with self.assertRaises(EvolutionTransactionError):
            transaction.step(
                lambda: transaction.model(torch.full((1, 2), float("nan"))).sum()
            )

        for name, value in transaction.model.state_dict().items():
            self.assertTrue(torch.equal(value, model_before[name]))
        _assert_nested_equal(self, transaction.optimizer.state_dict(), optimizer_before)
        self.assertEqual(len(transaction.regularizer.snapshots), len(fisher_before))
        for current, expected in zip(
            transaction.regularizer.snapshots, fisher_before, strict=True
        ):
            _assert_nested_equal(self, current.fisher, expected.fisher)
            _assert_nested_equal(self, current.anchor, expected.anchor)

    def test_inference_mode_cannot_enter_optimizer_transaction(self) -> None:
        transaction = _transaction()
        transaction.rhythm.mode = SystemMode.INFERENCE
        before = transaction.model.core.weight.detach().clone()

        with self.assertRaisesRegex(RuntimeError, "evolution mode"):
            transaction.step(lambda: transaction.model(torch.ones(1, 2)).sum())

        self.assertTrue(torch.equal(transaction.model.core.weight, before))


class TestGenerationCheckpointStore(unittest.TestCase):
    def test_roundtrip_restores_model_optimizer_fisher_and_c_fire(self) -> None:
        transaction = _transaction()
        report = transaction.step(
            lambda: transaction.model(torch.ones(1, 2)).square().mean()
        )
        expected_model = {
            name: value.detach().clone()
            for name, value in transaction.model.state_dict().items()
        }
        expected_optimizer = deepcopy(transaction.optimizer.state_dict())
        expected_penalty = float(
            transaction.regularizer.penalty(transaction.named_trainable).detach()
        )
        with tempfile.TemporaryDirectory() as directory:
            store = GenerationCheckpointStore(directory)
            record = store.write(
                transaction,
                report.post_certificates,
                metadata={"seed": 7, "domain": "alpha"},
            )
            with torch.no_grad():
                transaction.model.core.weight.add_(2.0)
            transaction.optimizer.state.clear()
            transaction.regularizer.snapshots.clear()

            restored = store.restore_current(transaction)

            self.assertEqual(restored, record)
            for name, value in transaction.model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), expected_model[name]))
            _assert_nested_equal(
                self, transaction.optimizer.state_dict(), expected_optimizer
            )
            self.assertAlmostEqual(
                float(
                    transaction.regularizer.penalty(
                        transaction.named_trainable
                    ).detach()
                ),
                expected_penalty,
                places=6,
            )
            self.assertTrue(store.current_pointer.is_file())

    def test_generations_form_parent_chain_and_current_is_atomic_file(self) -> None:
        transaction = _transaction()
        with tempfile.TemporaryDirectory() as directory:
            store = GenerationCheckpointStore(directory)
            first_report = transaction.step(
                lambda: transaction.model(torch.ones(1, 2)).square().mean()
            )
            first = store.write(transaction, first_report.post_certificates)
            second_report = transaction.step(
                lambda: transaction.model(torch.zeros(1, 2)).square().mean()
            )
            second = store.write(transaction, second_report.post_certificates)

            self.assertEqual(second.parent_checkpoint_sha256, first.checkpoint_sha256)
            self.assertEqual(store.read_current(), second)
            self.assertFalse(any(Path(directory).glob("*.tmp")))

    def test_tampered_payload_fails_before_mutation(self) -> None:
        transaction = _transaction()
        with tempfile.TemporaryDirectory() as directory:
            store = GenerationCheckpointStore(directory)
            report = transaction.step(
                lambda: transaction.model(torch.ones(1, 2)).square().mean()
            )
            record = store.write(transaction, report.post_certificates)
            model_before = transaction.model.core.weight.detach().clone()
            record.checkpoint.write_bytes(record.checkpoint.read_bytes() + b"tamper")

            with self.assertRaises(GenerationCheckpointError):
                store.restore_current(transaction)

            self.assertTrue(torch.equal(transaction.model.core.weight, model_before))


class TestTransferEvidence(unittest.TestCase):
    def test_three_seed_bwt_fwt_summary_is_finite_and_reproducible(self) -> None:
        baselines = torch.tensor([0.2, 0.2, 0.2])
        scores = torch.tensor(
            [
                [0.8, 0.3, 0.2],
                [0.75, 0.82, 0.35],
                [0.73, 0.79, 0.85],
            ]
        )
        result = continual_transfer_metrics(scores, baselines)
        self.assertAlmostEqual(result.backward_transfer, -0.05, places=6)
        self.assertAlmostEqual(result.forward_transfer, 0.125, places=6)

        summary = summarize_seeded_transfer(
            [
                result,
                TransferMetrics(-0.04, 0.12),
                TransferMetrics(-0.06, 0.13),
            ]
        )
        self.assertEqual(summary.seeds, 3)
        self.assertTrue(torch.isfinite(torch.tensor(summary.bwt_std)))
        self.assertTrue(torch.isfinite(torch.tensor(summary.fwt_std)))

    def test_fewer_than_three_seeds_are_not_publishable_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "3..128"):
            summarize_seeded_transfer([TransferMetrics(0.0, 0.0)] * 2)


if __name__ == "__main__":
    unittest.main()
