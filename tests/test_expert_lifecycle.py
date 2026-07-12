from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from cogni_core.expert_lifecycle import (
    ExpertCandidateLifecycle,
    ExpertLifecycleError,
    ExternalVerifierAttestation,
    SparseRoutedFPEWC,
    estimate_routed_fisher,
)
from cogni_core.experts import (
    BoundedSparseImplicitExperts,
    EXPERT_ACTIVE,
    EXPERT_QUARANTINED,
    ExpertCalibrationError,
    ExpertConfig,
)
from cogni_core.resources import VRAMGuard
from cogni_core.search import BoundedPUCTSearch, PUCTConfig
from cogni_os.runtime import GenesisRuntime


def _pool() -> BoundedSparseImplicitExperts:
    pool = BoundedSparseImplicitExperts(
        ExpertConfig(
            input_dim=4,
            state_dim=5,
            router_dim=4,
            max_experts=8,
            initial_experts=2,
            min_experts=1,
            top_k=2,
            novelty_threshold=0.8,
            recruit_fraction=0.5,
            routing_temperature=0.2,
            spectral_margin=0.75,
            max_parameter_bytes=2**20,
            max_vram_bytes=4 * 2**20,
        )
    )
    with torch.no_grad():
        pool.router_weight.copy_(torch.eye(4))
        pool.router_bias.zero_()
        pool.prototypes.zero_()
        pool.prototypes[0, 0] = 1
        pool.prototypes[1, 1] = 1
    return pool


def _calibrate(pool: BoundedSparseImplicitExperts) -> None:
    in_domain = torch.cat(
        (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(8, 1),
        )
    )
    out_domain = torch.cat(
        (
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(8, 1),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(8, 1),
        )
    )
    pool.calibrate_novelty_(in_domain, out_domain, max_fpr=0.0, max_fnr=0.0)


def _novel() -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(16, 1)


def _advance_to_held_out(
    lifecycle: ExpertCandidateLifecycle,
) -> tuple[int, str]:
    slot = lifecycle.start_candidate(_novel())
    digest = lifecycle.pool.router_digest()
    lifecycle.certify_c_fire()
    with lifecycle.candidate_gradient_scope():
        lifecycle.pool.zero_grad(set_to_none=True)
        loss = sum(
            getattr(lifecycle.pool, name).sum()
            for name in ("recurrent", "input_weight", "bias")
        )
        loss.backward()
    lifecycle.finish_training(steps=1, before_loss=1.0, after_loss=0.9)
    routing_samples = torch.cat(
        (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(8, 1),
            _novel(),
        )
    )
    lifecycle.accept_held_out(
        baseline_metric=0.50,
        candidate_metric=0.55,
        sample_count=32,
        routing_evidence=lifecycle.pool.route_candidate(routing_samples, slot),
        minimum_improvement=0.01,
    )
    return slot, digest


def _candidate_fisher(lifecycle: ExpertCandidateLifecycle, slot: int, digest: str):
    route = lifecycle.pool.route_candidate(_novel(), slot)
    gradients = {
        name: torch.ones(
            _novel().shape[0],
            *getattr(lifecycle.pool, name).shape,
            dtype=getattr(lifecycle.pool, name).dtype,
        )
        for name in ("recurrent", "input_weight", "bias")
    }
    return estimate_routed_fisher(
        lifecycle.pool, route, gradients, router_digest=digest
    )


class TestPhase8ExpertSafety(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)

    def test_certified_profile_is_fixed_at_eight_slots_top_two(self):
        _pool().assert_phase8_profile()
        other = BoundedSparseImplicitExperts(
            ExpertConfig(input_dim=4, max_experts=4, top_k=2)
        )
        with self.assertRaisesRegex(RuntimeError, "exactly eight"):
            ExpertCandidateLifecycle(other)

    def test_gemma_width_pool_has_a_fixed_sub_16_7_gib_admission_bound(self):
        limit = int(16.7 * 1024**3)
        pool = BoundedSparseImplicitExperts(
            ExpertConfig(
                input_dim=2560,
                state_dim=64,
                router_dim=32,
                max_experts=8,
                initial_experts=1,
                top_k=2,
                spectral_margin=0.90,
                max_parameter_bytes=8 * 1024**3,
                max_vram_bytes=limit,
            )
        )
        required = pool.persistent_bytes + pool.estimated_working_set_bytes(
            1, include_backward=True
        )
        self.assertLess(required, limit)
        self.assertEqual(pool.config.max_experts, 8)

    def test_novelty_is_disabled_as_verified_evidence_until_calibrated(self):
        pool = _pool()
        self.assertFalse(bool(pool.route(_novel()).calibration_verified))
        with self.assertRaisesRegex(ExpertLifecycleError, "calibration"):
            ExpertCandidateLifecycle(pool).start_candidate(_novel())
        _calibrate(pool)
        route = pool.route(_novel())
        self.assertTrue(bool(route.calibration_verified))
        self.assertEqual(float(pool.calibration_fpr), 0.0)
        self.assertEqual(float(pool.calibration_fnr), 0.0)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in pool.router_parameters())
        )

    def test_overlapping_calibration_fails_without_claiming_evidence(self):
        pool = _pool()
        samples = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1)
        with self.assertRaises(ExpertCalibrationError):
            pool.calibrate_novelty_(samples, samples, max_fpr=0.0, max_fnr=0.0)
        self.assertFalse(bool(pool.novelty_calibrated))

    def test_collapsed_held_out_route_is_rejected(self):
        pool = _pool()
        collapsed = pool.route(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(32, 1))
        with self.assertRaisesRegex(RuntimeError, "collapsed"):
            pool.assert_routing_not_collapsed(collapsed)

    def test_router_is_z_independent_and_inactive_expert_gradients_are_zero(self):
        pool = _pool()
        x = torch.randn(7, 4)
        route = pool.route(x)
        z = torch.randn(7, 5, requires_grad=True)
        state = pool.mixture(z, x, route)
        state.square().sum().backward()
        inactive = ~pool.active_mask
        self.assertEqual(int(route.gates[:, inactive].count_nonzero()), 0)
        self.assertEqual(int(pool.recurrent.grad[inactive].count_nonzero()), 0)
        self.assertEqual(int(pool.input_weight.grad[inactive].count_nonzero()), 0)
        self.assertEqual(int(pool.bias.grad[inactive].count_nonzero()), 0)
        # No API accepts z for routing; the gates are exactly repeatable while z changes.
        self.assertTrue(torch.equal(route.gates, pool.route(x).gates))
        z_probe = torch.randn(7, 5, requires_grad=True)
        self.assertEqual(int(pool.assert_z_independent(x, z_probe).count_nonzero()), 0)

    def test_candidate_gradient_scope_freezes_router_and_other_experts(self):
        pool = _pool()
        _calibrate(pool)
        lifecycle = ExpertCandidateLifecycle(pool)
        slot = lifecycle.start_candidate(_novel())
        lifecycle.certify_c_fire()
        with lifecycle.candidate_gradient_scope():
            pool.zero_grad(set_to_none=True)
            (
                pool.recurrent.sum() + pool.input_weight.sum() + pool.bias.sum()
            ).backward()
        for name in ("recurrent", "input_weight", "bias"):
            gradient = getattr(pool, name).grad
            self.assertGreater(int(gradient[slot].count_nonzero()), 0)
            self.assertEqual(int(gradient[torch.arange(8) != slot].count_nonzero()), 0)
        self.assertTrue(
            all(parameter.grad is None for parameter in pool.router_parameters())
        )

    def test_optimizer_weight_decay_on_other_slots_triggers_atomic_quarantine(self):
        pool = _pool()
        _calibrate(pool)
        lifecycle = ExpertCandidateLifecycle(pool)
        slot = lifecycle.start_candidate(_novel())
        lifecycle.certify_c_fire()
        optimizer = torch.optim.AdamW(
            (pool.recurrent, pool.input_weight, pool.bias), lr=1.0e-3, weight_decay=0.1
        )
        with lifecycle.candidate_gradient_scope():
            optimizer.zero_grad(set_to_none=True)
            (
                pool.recurrent.sum() + pool.input_weight.sum() + pool.bias.sum()
            ).backward()
            optimizer.step()
        with self.assertRaisesRegex(ExpertLifecycleError, "post-training"):
            lifecycle.finish_training(steps=1, before_loss=1.0, after_loss=0.9)
        self.assertFalse(bool(pool.active_mask[slot]))
        self.assertEqual(int(pool.slot_state[slot]), EXPERT_QUARANTINED)

    def test_routed_fisher_is_sparse_finite_nonnegative_and_penalizes_slot(self):
        pool = _pool()
        _calibrate(pool)
        lifecycle = ExpertCandidateLifecycle(pool)
        slot, digest = _advance_to_held_out(lifecycle)
        snapshot = _candidate_fisher(lifecycle, slot, digest)
        self.assertGreater(float(snapshot.routing_mass[slot]), 0)
        for value in snapshot.fisher.values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue((value >= 0).all())
            self.assertEqual(
                int(value[~(snapshot.routing_mass > 0)].count_nonzero()), 0
            )
        lifecycle.consolidate_fisher(snapshot)
        before = lifecycle.regularizer.penalty(pool)
        with torch.no_grad():
            pool.bias[slot].add_(0.1)
        after = lifecycle.regularizer.penalty(pool)
        self.assertEqual(float(before.detach()), 0.0)
        self.assertGreater(float(after.detach()), 0.0)

    def test_full_promotion_stays_advisory_until_independent_attestation(self):
        pool = _pool()
        _calibrate(pool)
        lifecycle = ExpertCandidateLifecycle(pool)
        slot, digest = _advance_to_held_out(lifecycle)
        lifecycle.consolidate_fisher(_candidate_fisher(lifecycle, slot, digest))
        lifecycle.begin_canary()
        self.assertFalse(bool(pool.answer_authority_mask[slot]))
        lifecycle.complete_canary(sample_count=100, failures=0)
        self.assertEqual(int(pool.slot_state[slot]), EXPERT_ACTIVE)
        self.assertFalse(bool(pool.answer_authority_mask[slot]))
        self.assertFalse(bool(pool.novelty_calibrated))
        with self.assertRaisesRegex(ExpertLifecycleError, "calibration"):
            lifecycle.start_candidate(_novel())
        artifact = lifecycle.slot_digest(slot)
        with self.assertRaisesRegex(ExpertLifecycleError, "independent"):
            lifecycle.grant_answer_authority(
                slot,
                ExternalVerifierAttestation("self", artifact, True, False),
            )
        lifecycle.grant_answer_authority(
            slot,
            ExternalVerifierAttestation("external-lab", artifact, True, True),
        )
        self.assertTrue(bool(pool.answer_authority_mask[slot]))
        with torch.no_grad():
            pool.recurrent[slot].mul_(2.0)
        pool.project_contractivity_()
        self.assertFalse(bool(pool.answer_authority_mask[slot]))

    def test_failed_held_out_rolls_back_router_and_quarantines_slot(self):
        pool = _pool()
        _calibrate(pool)
        before_router = tuple(
            value.detach().clone() for value in pool.router_parameters()
        )
        before_active = pool.active_mask.clone()
        lifecycle = ExpertCandidateLifecycle(pool)
        slot = lifecycle.start_candidate(_novel())
        lifecycle.certify_c_fire()
        lifecycle.finish_training(steps=1, before_loss=1.0, after_loss=0.9)
        with self.assertRaisesRegex(ExpertLifecycleError, "held-out"):
            lifecycle.accept_held_out(
                baseline_metric=0.8,
                candidate_metric=0.2,
                sample_count=32,
                routing_evidence=pool.route_candidate(_novel(), slot),
            )
        self.assertEqual(int(pool.slot_state[slot]), EXPERT_QUARANTINED)
        self.assertFalse(bool(pool.active_mask[slot]))
        self.assertTrue(torch.equal(pool.active_mask, before_active))
        for before, after in zip(before_router, pool.router_parameters(), strict=True):
            self.assertTrue(torch.equal(before, after))

    def test_canary_failure_and_post_promotion_rollback_are_fail_closed(self):
        for fail_canary in (True, False):
            pool = _pool()
            _calibrate(pool)
            lifecycle = ExpertCandidateLifecycle(pool)
            slot, digest = _advance_to_held_out(lifecycle)
            lifecycle.consolidate_fisher(_candidate_fisher(lifecycle, slot, digest))
            lifecycle.begin_canary()
            if fail_canary:
                with self.assertRaisesRegex(ExpertLifecycleError, "canary"):
                    lifecycle.complete_canary(sample_count=100, failures=50)
            else:
                lifecycle.complete_canary(sample_count=100, failures=0)
                lifecycle.rollback_last_promotion()
            self.assertFalse(bool(pool.active_mask[slot]))
            self.assertEqual(int(pool.slot_state[slot]), EXPERT_QUARANTINED)
            self.assertEqual(lifecycle.regularizer.snapshots, [])

    def test_checkpoint_digest_tamper_and_restore_are_transactional(self):
        pool = _pool()
        _calibrate(pool)
        lifecycle = ExpertCandidateLifecycle(
            pool, regularizer=SparseRoutedFPEWC(max_domains=2)
        )
        with tempfile.TemporaryDirectory() as directory:
            path, digest = lifecycle.write_checkpoint(directory)
            original = {
                name: value.clone() for name, value in pool.state_dict().items()
            }
            with torch.no_grad():
                pool.recurrent[0].zero_()
            lifecycle.restore_checkpoint(path, digest)
            for name, value in pool.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), original[name]))
            Path(path).write_bytes(Path(path).read_bytes() + b"tamper")
            before = {name: value.clone() for name, value in pool.state_dict().items()}
            with self.assertRaisesRegex(ExpertLifecycleError, "digest"):
                lifecycle.restore_checkpoint(path, digest)
            for name, value in pool.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))

    def test_quarantine_cannot_be_bypassed_by_legacy_recruitment_or_checkpoint(self):
        pool = _pool()
        pool.quarantine_mask[2] = True
        pool.slot_state[2] = EXPERT_QUARANTINED
        result = pool.recruit_(_novel())
        self.assertNotEqual(int(result.slot), 2)
        self.assertTrue(bool(pool.quarantine_mask[2]))

        lifecycle = ExpertCandidateLifecycle(pool)
        pool.answer_authority_mask[0] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExpertLifecycleError, "attestations"):
                lifecycle.write_checkpoint(directory)

    def test_ten_thousand_routes_remain_sparse_finite_and_contractive(self):
        pool = _pool().eval()
        generator = torch.Generator().manual_seed(20260712)
        x = torch.randn(10_000, 4, generator=generator)
        z = torch.randn(10_000, 5, generator=generator)
        with torch.no_grad():
            route = pool.route(x)
            state = pool.mixture(z, x, route)
            bound = pool.routing_contractivity_bound(route)
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue((bound < 0.95).all())
        self.assertTrue((route.gates.count_nonzero(-1) <= 2).all())
        self.assertEqual(int(route.gates[:, ~pool.active_mask].count_nonzero()), 0)

    def test_routed_fisher_long_stream_merges_with_hard_domain_bound(self):
        pool = _pool()
        _calibrate(pool)
        lifecycle = ExpertCandidateLifecycle(
            pool,
            regularizer=SparseRoutedFPEWC(max_domains=2, max_total_bytes=2**20),
        )
        slot, digest = _advance_to_held_out(lifecycle)
        snapshot = _candidate_fisher(lifecycle, slot, digest)
        for _ in range(20):
            lifecycle.regularizer.consolidate(pool, snapshot)
            self.assertLessEqual(len(lifecycle.regularizer.snapshots), 2)
            self.assertLessEqual(lifecycle.regularizer.total_bytes, 2**20)
        penalty = lifecycle.regularizer.penalty(pool)
        self.assertTrue(torch.isfinite(penalty))
        self.assertGreaterEqual(float(penalty.detach()), 0.0)


class TestPhase8RuntimeBoundary(unittest.TestCase):
    @staticmethod
    def _runtime() -> GenesisRuntime:
        pool = _pool()
        search = BoundedPUCTSearch(
            PUCTConfig(
                width=2,
                max_depth=1,
                max_nodes=3,
                simulations=1,
                ancestor_k=0,
            )
        )
        return GenesisRuntime(
            nn.Linear(4, 4),
            search,
            experts=pool,
            expert_lifecycle=ExpertCandidateLifecycle(pool),
            vram_guard=VRAMGuard(device="cpu"),
        )

    def test_runtime_forbids_direct_day_recruitment_and_checkpoints_lifecycle(self):
        runtime = self._runtime()
        with self.assertRaisesRegex(RuntimeError, "evolution"):
            runtime.recruit_expert(_novel())

        runtime.rhythm.enter_evolution(lambda: None)
        in_domain = torch.cat(
            (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1),
                torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(8, 1),
            )
        )
        out_of_domain = torch.cat(
            (
                torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(8, 1),
                torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(8, 1),
            )
        )
        runtime.calibrate_expert_novelty(
            in_domain,
            out_of_domain,
            max_fpr=0.0,
            max_fnr=0.0,
        )
        slot = runtime.recruit_expert(_novel())
        self.assertEqual(runtime.expert_lifecycle.candidate_slot, slot)
        runtime.certify_expert_candidate()
        with runtime.expert_candidate_training_scope():
            runtime.experts.zero_grad(set_to_none=True)
            loss = sum(
                getattr(runtime.experts, name).sum()
                for name in ("recurrent", "input_weight", "bias")
            )
            loss.backward()
        runtime.finish_expert_candidate_training(
            steps=1,
            before_loss=1.0,
            after_loss=0.9,
        )
        runtime.rollback_expert_candidate(quarantine=True)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint, digest = runtime.checkpoint(directory)
            self.assertTrue((Path(directory) / "system3-phase8.pt").is_file())
            original = runtime.experts.recurrent.detach().clone()
            with torch.no_grad():
                runtime.experts.recurrent.zero_()
            runtime.restore_checkpoint(checkpoint, digest)
            self.assertTrue(torch.equal(runtime.experts.recurrent, original))


if __name__ == "__main__":
    unittest.main()
