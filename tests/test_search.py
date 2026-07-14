import math
import unittest

import torch

from cogni_core.search import (
    BoundedPUCTSearch,
    BoundedPUCTSearchV2,
    CertifiedBroydenTransitionV2,
    CertifiedPUCTConfigV2,
    CertifiedTransitionBatchV2,
    ContractiveBroydenTransition,
    FixedCapacityTreeRetriever,
    PUCTConfig,
    SearchControlsV2,
    SearchRequestV2,
    SemanticAncestorRetriever,
    TransitionContractError,
    deq_tensor_transition,
    puct_scores,
)
from cogni_core.deq import BroydenWarmStart, ContractivityError


@deq_tensor_transition
def _transition(state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    increments = (actions + 1).to(dtype=state.dtype)
    view = (actions.shape[0],) + (1,) * state.ndim
    return state.unsqueeze(0) + increments.view(view)


def _uniform_policy_value(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(3, device=state.device), state.float().mean()


class _DeterministicCertifiedTransition:
    __cogni_tensor_transition__ = True
    __cogni_deq_transition__ = True
    __cogni_stateless__ = True
    __cogni_uses_kv_cache__ = False
    __cogni_broyden_solver__ = True
    width = 3
    rank = 16
    operator_id = "test-certified-v2"

    def __init__(self, failed_actions: tuple[int, ...] = ()) -> None:
        self.failed_actions = frozenset(failed_actions)
        self.calls = 0
        self.warm_inputs: list[BroydenWarmStart | None] = []

    def __call__(
        self,
        parent: torch.Tensor,
        actions: torch.Tensor,
        *,
        tolerance: float,
        warm_start: BroydenWarmStart | None,
    ) -> CertifiedTransitionBatchV2:
        del tolerance
        self.calls += 1
        self.warm_inputs.append(warm_start)
        view = (3,) + (1,) * parent.ndim
        states = parent.unsqueeze(0) + (actions + 1).to(parent.dtype).view(view)
        success = torch.tensor(
            [action not in self.failed_actions for action in range(3)],
            dtype=torch.bool,
        )
        warm_starts: list[BroydenWarmStart | None] = []
        for action in range(3):
            if not bool(success[action]):
                states[action].copy_(parent)
                warm_starts.append(None)
                continue
            empty = torch.empty(
                (0, *parent.shape), device=parent.device, dtype=parent.dtype
            )
            warm_starts.append(
                BroydenWarmStart(
                    state=states[action].clone(),
                    x_history=empty,
                    f_history=empty.clone(),
                    rank=16,
                    operator_id=self.operator_id,
                )
            )
        return CertifiedTransitionBatchV2(
            states=states,
            success=success,
            residuals=torch.where(
                success,
                torch.zeros(3, dtype=torch.float32),
                torch.full((3,), float("inf"), dtype=torch.float32),
            ),
            iterations=torch.ones(3, dtype=torch.int64),
            linear_solve_fallbacks=torch.zeros(3, dtype=torch.int64),
            warm_used=torch.full((3,), warm_start is not None, dtype=torch.bool),
            warm_rejected=torch.zeros(3, dtype=torch.int64),
            warm_starts=tuple(warm_starts),
        )


def _deep_policy(state: torch.Tensor) -> torch.Tensor:
    return torch.tensor([100.0, -100.0, -100.0], device=state.device)


def _negative_critic(state: torch.Tensor) -> torch.Tensor:
    return torch.tensor(-5.0, device=state.device)


def _controls(simulations: int) -> SearchControlsV2:
    return SearchControlsV2(
        exploration=0.0,
        tolerance=1.0e-4,
        policy_temperature=1.0,
        act_simulations=simulations,
    )


class PUCTMathTests(unittest.TestCase):
    def test_puct_matches_reference_equation(self) -> None:
        priors = torch.tensor([0.3, 0.7])
        visits = torch.tensor([2, 0])
        value_sums = torch.tensor([0.2, 0.0])
        scores = puct_scores(priors, visits, value_sums, parent_visits=10, c_puct=2.0)
        expected_0 = 0.1 + 2.0 * 0.3 * math.sqrt(10) / 3.0
        expected_1 = 2.0 * 0.7 * math.sqrt(10)
        self.assertAlmostEqual(float(scores[0]), expected_0, places=6)
        self.assertAlmostEqual(float(scores[1]), expected_1, places=6)

    def test_puct_rejects_negative_visits(self) -> None:
        with self.assertRaises(ValueError):
            puct_scores(
                torch.tensor([1.0]),
                torch.tensor([-1]),
                torch.tensor([0.0]),
                parent_visits=1,
                c_puct=1.0,
            )


class ContractiveBroydenTransitionTests(unittest.TestCase):
    def test_certified_transition_converges_with_fixed_shape(self) -> None:
        transition = ContractiveBroydenTransition(
            width=3, contraction=0.4, tolerance=1.0e-5, max_iter=30, history=5
        )
        state = torch.randn(2, 4)
        output = transition(state, torch.arange(3, dtype=torch.int64))
        self.assertEqual(output.shape, (3, 2, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(transition.last_info)
        self.assertTrue(transition.last_info.converged)
        self.assertLess(transition.last_info.spectral_norm, 0.95)

    def test_noncontractive_configuration_is_rejected(self) -> None:
        with self.assertRaises(ContractivityError):
            ContractiveBroydenTransition(
                width=2, contraction=0.95, spectral_margin=0.95
            )


class AncestorRetrieverTests(unittest.TestCase):
    def test_ancestor_blend_preserves_low_precision_query_dtype(self) -> None:
        store = SemanticAncestorRetriever(
            feature_dim=2,
            capacity=4,
            retrieval_k=2,
            storage_dtype=torch.bfloat16,
        )
        root = store.add(torch.tensor([1.0, 0.0], dtype=torch.bfloat16))
        child = store.add(torch.tensor([0.0, 1.0], dtype=torch.bfloat16), root)
        query = torch.tensor([0.5, 0.5], dtype=torch.bfloat16)
        result = store.retrieve(query, child).blend(query, 0.5)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_only_true_ancestors_are_semantically_ranked(self) -> None:
        store = SemanticAncestorRetriever(
            feature_dim=2, capacity=8, retrieval_k=3, storage_dtype=torch.float32
        )
        root = store.add(torch.tensor([1.0, 0.0]))
        child = store.add(torch.tensor([0.0, 1.0]), root)
        sibling = store.add(torch.tensor([1.0, 0.0]), root)
        grandchild = store.add(torch.tensor([1.0, 0.1]), child)

        result = store.retrieve(torch.tensor([1.0, 0.1]), grandchild)
        valid_handles = result.handles[result.valid.cpu()].tolist()
        self.assertEqual(valid_handles, [root, child])
        self.assertNotIn(sibling, valid_handles)
        self.assertGreater(float(result.scores[0]), float(result.scores[1]))
        self.assertEqual(result.states.shape, (3, 2))

    def test_ring_capacity_and_allocation_remain_constant(self) -> None:
        store = SemanticAncestorRetriever(feature_dim=4, capacity=3, retrieval_k=2)
        parent = -1
        parent = store.add(torch.ones(4), parent)
        allocated = store.memory_bytes()
        for step in range(20):
            parent = store.add(torch.full((4,), float(step + 2)), parent)
        self.assertEqual(store.size, 3)
        self.assertEqual(store.memory_bytes(), allocated)

        result = store.retrieve(torch.ones(4), parent)
        self.assertEqual(int(result.valid.sum()), 2)
        self.assertTrue(torch.all(result.handles[result.valid.cpu()] >= parent - 2))

    def test_empty_lineage_blend_is_identity(self) -> None:
        store = SemanticAncestorRetriever(feature_dim=4, capacity=4, retrieval_k=2)
        root_state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        root = store.add(root_state)
        result = store.retrieve(root_state, root)
        self.assertTrue(torch.equal(result.blend(root_state, 0.5), root_state))

    def test_parent_must_precede_child(self) -> None:
        store = SemanticAncestorRetriever(feature_dim=2, capacity=4, retrieval_k=2)
        with self.assertRaises(ValueError):
            store.add(torch.ones(2), parent_handle=0)

    def test_shared_state_bank_avoids_duplicate_latent_storage(self) -> None:
        bank = torch.zeros(4, 4, 8, dtype=torch.float32)
        shared = SemanticAncestorRetriever(
            feature_dim=8,
            capacity=4,
            retrieval_k=2,
            storage_dtype=torch.float32,
            shared_state_bank=bank,
        )
        owned = SemanticAncestorRetriever(
            feature_dim=8,
            capacity=4,
            retrieval_k=2,
            storage_dtype=torch.float32,
        )

        parent_shared = -1
        parent_owned = -1
        for index in range(3):
            bank[index].fill_(float(index + 1))
            parent_shared = shared.add(bank[index], parent_shared, state_index=index)
            parent_owned = owned.add(bank[index], parent_owned)

        self.assertTrue(shared.shares_state_storage)
        self.assertFalse(owned.shares_state_storage)
        self.assertLess(shared.memory_bytes(), owned.memory_bytes())
        result = shared.retrieve(bank[2], parent_shared)
        self.assertEqual(result.handles[result.valid.cpu()].tolist(), [1, 0])

    def test_shared_bank_requires_the_exact_referenced_row(self) -> None:
        bank = torch.zeros(4, 2)
        store = SemanticAncestorRetriever(
            feature_dim=2,
            capacity=4,
            retrieval_k=2,
            shared_state_bank=bank,
        )
        with self.assertRaisesRegex(ValueError, "referenced shared_state_bank row"):
            store.add(torch.zeros(2), state_index=0)


class BoundedSearchTests(unittest.TestCase):
    def test_selection_and_backup_prefer_higher_value_branch(self) -> None:
        cfg = PUCTConfig(
            width=2,
            max_depth=1,
            max_nodes=3,
            simulations=40,
            c_puct=1.5,
            ancestor_k=0,
        )

        def policy_value(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(2), state.mean()

        result = BoundedPUCTSearch(cfg).search(
            torch.zeros(1), _transition, policy_value
        )
        self.assertEqual(int(result.root_visit_counts.sum()), cfg.simulations - 1)
        self.assertGreater(
            int(result.root_visit_counts[1]), int(result.root_visit_counts[0])
        )
        self.assertGreater(
            float(result.root_q_values[1]), float(result.root_q_values[0])
        )
        self.assertEqual(int(result.best_action), 1)
        self.assertEqual(result.telemetry.max_depth_reached, 1)

    def test_tree_and_stat_memory_are_bounded_by_node_capacity(self) -> None:
        common = dict(
            width=3,
            max_depth=20,
            max_nodes=13,
            c_puct=2.0,
            ancestor_k=2,
            ancestor_capacity=5,
        )
        short = BoundedPUCTSearch(PUCTConfig(simulations=20, **common)).search(
            torch.zeros(4), _transition, _uniform_policy_value
        )
        long = BoundedPUCTSearch(PUCTConfig(simulations=200, **common)).search(
            torch.zeros(4), _transition, _uniform_policy_value
        )

        self.assertLessEqual(long.telemetry.nodes_used, 13)
        self.assertEqual(long.telemetry.nodes_used, 13)
        self.assertTrue(long.telemetry.capacity_exhausted)
        self.assertLessEqual(long.telemetry.max_depth_reached, 20)
        self.assertEqual(
            short.telemetry.allocated_bytes, long.telemetry.allocated_bytes
        )
        self.assertGreater(long.telemetry.frontier_rollouts, 0)

    def test_requested_depth_does_not_change_any_preallocated_bytes(self) -> None:
        common = dict(
            width=3,
            max_nodes=13,
            simulations=60,
            ancestor_k=2,
            ancestor_capacity=7,
        )
        shallow = BoundedPUCTSearch(PUCTConfig(max_depth=2, **common)).search(
            torch.zeros(4), _transition, _uniform_policy_value
        )
        effectively_unbounded = BoundedPUCTSearch(
            PUCTConfig(max_depth=10_000_000, **common)
        ).search(torch.zeros(4), _transition, _uniform_policy_value)

        self.assertEqual(
            shallow.telemetry.allocated_bytes,
            effectively_unbounded.telemetry.allocated_bytes,
        )
        self.assertLessEqual(
            effectively_unbounded.best_path.numel(), common["max_nodes"]
        )
        self.assertLessEqual(
            effectively_unbounded.telemetry.nodes_used, common["max_nodes"]
        )
        self.assertEqual(
            effectively_unbounded.telemetry.allocated_bytes,
            BoundedPUCTSearch(
                PUCTConfig(max_depth=10_000_000, **common)
            ).estimated_preallocated_bytes(torch.zeros(4)),
        )

    def test_ancestor_capacity_cannot_escape_tree_hard_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed max_nodes"):
            PUCTConfig(
                width=2,
                max_nodes=3,
                ancestor_capacity=4,
                ancestor_k=2,
            )

    def test_fixed_depth_is_never_exceeded(self) -> None:
        cfg = PUCTConfig(
            width=3,
            max_depth=2,
            max_nodes=40,
            simulations=100,
            ancestor_k=0,
        )
        result = BoundedPUCTSearch(cfg).search(
            torch.zeros(2, 4), _transition, _uniform_policy_value
        )
        self.assertEqual(result.telemetry.max_depth_reached, 2)
        self.assertLessEqual(result.best_path.numel(), 3)
        self.assertEqual(result.best_state.shape, (2, 4))

    def test_seeded_gumbel_selection_is_reproducible(self) -> None:
        cfg = PUCTConfig(
            width=3,
            max_depth=2,
            max_nodes=13,
            simulations=50,
            selection_temperature=0.75,
            seed=712,
            ancestor_k=0,
        )

        first = BoundedPUCTSearch(cfg).search(
            torch.zeros(2), _transition, _uniform_policy_value
        )
        second = BoundedPUCTSearch(cfg).search(
            torch.zeros(2), _transition, _uniform_policy_value
        )
        self.assertTrue(torch.equal(first.root_visit_counts, second.root_visit_counts))
        self.assertTrue(torch.equal(first.root_q_values, second.root_q_values))
        self.assertTrue(torch.equal(first.best_path, second.best_path))
        self.assertTrue(torch.equal(first.best_state, second.best_state))

    def test_search_uses_tensor_ancestor_context(self) -> None:
        def policy_value(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(2), state.mean()

        common = dict(
            width=2,
            max_depth=2,
            max_nodes=7,
            simulations=3,
            c_puct=0.0,
            ancestor_k=2,
            ancestor_capacity=7,
        )
        without_context = BoundedPUCTSearch(
            PUCTConfig(ancestor_mix=0.0, **common)
        ).search(torch.zeros(1), _transition, policy_value)
        with_context = BoundedPUCTSearch(PUCTConfig(ancestor_mix=1.0, **common)).search(
            torch.zeros(1), _transition, policy_value
        )

        self.assertGreater(with_context.telemetry.ancestor_queries, 0)
        self.assertEqual(with_context.best_path.numel(), 3)
        self.assertGreater(
            float(without_context.best_state), float(with_context.best_state)
        )

    def test_search_core_contract_rejects_non_tensor_transition(self) -> None:
        cfg = PUCTConfig(
            width=2,
            max_depth=1,
            max_nodes=3,
            simulations=1,
            ancestor_k=0,
        )

        def policy_value(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(2), torch.zeros(())

        @deq_tensor_transition
        def invalid_transition(state, actions):
            return [state, state]

        with self.assertRaises(TypeError):
            BoundedPUCTSearch(cfg).search(
                torch.zeros(1),
                invalid_transition,
                policy_value,  # type: ignore[arg-type,return-value]
            )

    def test_strict_contract_rejects_plain_or_kv_transition_before_execution(
        self,
    ) -> None:
        cfg = PUCTConfig(
            width=2,
            max_depth=1,
            max_nodes=3,
            simulations=1,
            ancestor_k=0,
        )
        calls = 0

        def plain_transition(state, actions):
            nonlocal calls
            calls += 1
            return torch.stack([state, state])

        def policy_value(state):
            return torch.zeros(2), torch.zeros(())

        with self.assertRaises(TransitionContractError):
            BoundedPUCTSearch(cfg).search(
                torch.zeros(1), plain_transition, policy_value
            )
        self.assertEqual(calls, 0)

        plain_transition.__cogni_tensor_transition__ = True
        plain_transition.__cogni_deq_transition__ = True
        plain_transition.__cogni_stateless__ = True
        plain_transition.__cogni_uses_kv_cache__ = True
        with self.assertRaisesRegex(TransitionContractError, "uses_kv_cache"):
            BoundedPUCTSearch(cfg).search(
                torch.zeros(1), plain_transition, policy_value
            )
        self.assertEqual(calls, 0)

    def test_legacy_opt_out_is_explicit_and_input_mutation_is_rejected(self) -> None:
        def policy_value(state):
            return torch.zeros(2), torch.zeros(())

        legacy_cfg = PUCTConfig(
            width=2,
            max_depth=1,
            max_nodes=3,
            simulations=1,
            ancestor_k=0,
            strict_deq_transition=False,
        )
        result = BoundedPUCTSearch(legacy_cfg).search(
            torch.zeros(1),
            lambda state, actions: torch.stack([state, state]),
            policy_value,
        )
        self.assertEqual(result.telemetry.expansions, 1)

        @deq_tensor_transition
        def mutating_transition(state, actions):
            state.add_(1.0)
            return torch.stack([state, state])

        strict_cfg = PUCTConfig(
            width=2,
            max_depth=1,
            max_nodes=3,
            simulations=1,
            ancestor_k=0,
        )
        with self.assertRaisesRegex(RuntimeError, "mutating its input"):
            BoundedPUCTSearch(strict_cfg).search(
                torch.zeros(1), mutating_transition, policy_value
            )


class CertifiedSearchV2Tests(unittest.TestCase):
    def test_certified_broyden_transition_emits_rank16_action_diagnostics(
        self,
    ) -> None:
        transition = CertifiedBroydenTransitionV2(max_iter=24)
        parent = torch.zeros(1, 2)
        actions = torch.arange(3, dtype=torch.int64)
        cold = transition(
            parent,
            actions,
            tolerance=1.0e-3,
            warm_start=None,
        )
        self.assertTrue(bool(cold.success.all()))
        self.assertEqual(cold.states.shape, (3, 1, 2))
        self.assertTrue(bool((cold.residuals <= 1.0e-3).all()))
        self.assertTrue(all(item is not None for item in cold.warm_starts))
        for item in cold.warm_starts:
            assert item is not None
            self.assertEqual(item.rank, 16)
            self.assertLessEqual(item.history_size, 17)

        warm = transition(
            parent,
            actions,
            tolerance=1.0e-3,
            warm_start=cold.warm_starts[0],
        )
        self.assertTrue(bool(warm.warm_used.all()))
        self.assertTrue(bool(warm.success.all()))

    def test_fixed_tree_retrieval_includes_sibling_and_excludes_failed(self) -> None:
        arena = torch.zeros(301, 2)
        arena[0] = torch.tensor([1.0, 0.0])
        arena[1] = torch.tensor([1.0, 0.1])
        arena[2] = torch.tensor([1.0, 0.0])
        arena[3] = torch.tensor([0.8, 0.2])
        arena[4] = torch.tensor([0.0, 1.0])
        store = FixedCapacityTreeRetriever(arena)
        store.add(0, arena[0], success=True)
        store.add(1, arena[1], success=True)  # sibling in whole-tree scope
        store.add(2, arena[2], success=False)  # failed edge state
        store.add(3, arena[3], success=True)
        store.add(4, arena[4], success=True)

        result = store.retrieve(arena[0], current_node=0)
        handles = result.handles[result.valid.cpu()].tolist()
        self.assertIn(1, handles)
        self.assertNotIn(0, handles)
        self.assertNotIn(2, handles)
        self.assertEqual(handles, [1, 3, 4])

    def test_retrieval_starts_at_depth_ten_not_depth_nine(self) -> None:
        def run(simulations: int):
            cfg = CertifiedPUCTConfigV2(
                max_depth=20,
                simulations=simulations,
            )
            return BoundedPUCTSearchV2(cfg).search(
                SearchRequestV2(torch.zeros(1), mac_budget=10_000, seed=7),
                _DeterministicCertifiedTransition(),
                _deep_policy,
                lambda state: torch.tensor(1.0, device=state.device),
                lambda state: _controls(simulations),
            )

        through_depth_nine = run(10)
        at_depth_ten = run(11)
        self.assertEqual(through_depth_nine.telemetry.max_depth_reached, 10)
        self.assertEqual(through_depth_nine.telemetry.retrieval_queries, 0)
        self.assertEqual(at_depth_ten.telemetry.retrieval_queries, 1)
        self.assertEqual(at_depth_ten.telemetry.retrieval_hits, 3)

    def test_failed_edge_gets_q_zero_once_and_is_masked_with_negative_critic(
        self,
    ) -> None:
        cfg = CertifiedPUCTConfigV2(max_depth=5, simulations=8)
        transition = _DeterministicCertifiedTransition(failed_actions=(0,))
        result = BoundedPUCTSearchV2(cfg).search(
            SearchRequestV2(torch.zeros(1), mac_budget=100),
            transition,
            _deep_policy,
            _negative_critic,
            lambda state: _controls(8),
        )
        self.assertEqual(int(result.root_visit_counts[0]), 1)
        self.assertEqual(float(result.root_q_values[0]), 0.0)
        self.assertNotEqual(int(result.best_action), 0)
        self.assertEqual(result.telemetry.failed_edges, transition.calls)
        self.assertEqual(result.telemetry.q_zero_backups, transition.calls)
        self.assertEqual(result.telemetry.nodes_used, 1 + 2 * transition.calls)
        self.assertTrue(result.telemetry.safe_for_decode)

    def test_all_failed_parent_is_terminal_and_never_materializes_child(self) -> None:
        cfg = CertifiedPUCTConfigV2(max_depth=5, simulations=3)
        result = BoundedPUCTSearchV2(cfg).search(
            SearchRequestV2(torch.zeros(1), mac_budget=100),
            _DeterministicCertifiedTransition(failed_actions=(0, 1, 2)),
            _deep_policy,
            _negative_critic,
            lambda state: _controls(3),
        )
        self.assertEqual(result.telemetry.nodes_used, 1)
        self.assertEqual(result.telemetry.all_fail_terminals, 1)
        self.assertEqual(result.telemetry.failed_edges, 3)
        self.assertEqual(result.root_visit_counts.tolist(), [1, 1, 1])
        self.assertEqual(int(result.best_action), -1)
        self.assertFalse(result.telemetry.safe_for_decode)

    def test_mac_denial_prevents_every_later_callback(self) -> None:
        cfg = CertifiedPUCTConfigV2(
            max_depth=3,
            simulations=3,
            meta_policy_macs=5,
            action_policy_macs=7,
            critic_macs=11,
            transition_macs=13,
        )
        calls = {"meta": 0, "policy": 0, "critic": 0}
        transition = _DeterministicCertifiedTransition()

        def meta(state: torch.Tensor) -> SearchControlsV2:
            calls["meta"] += 1
            return _controls(3)

        def policy(state: torch.Tensor) -> torch.Tensor:
            calls["policy"] += 1
            return _deep_policy(state)

        def critic(state: torch.Tensor) -> torch.Tensor:
            calls["critic"] += 1
            return _negative_critic(state)

        result = BoundedPUCTSearchV2(cfg).search(
            # Meta fits, but the atomic policy+critic reservation does not.
            SearchRequestV2(torch.zeros(1), mac_budget=12),
            transition,
            policy,
            critic,
            meta,
        )
        self.assertEqual(calls, {"meta": 1, "policy": 0, "critic": 0})
        self.assertEqual(transition.calls, 0)
        self.assertEqual(result.telemetry.mac_reserved, 5)
        self.assertTrue(result.telemetry.mac_budget_exhausted)
        self.assertEqual(result.telemetry.simulations_completed, 0)

        calls.update(meta=0, policy=0, critic=0)
        transition = _DeterministicCertifiedTransition()
        result = BoundedPUCTSearchV2(cfg).search(
            # Policy and critic fit; the transition is denied before callback.
            SearchRequestV2(torch.zeros(1), mac_budget=23),
            transition,
            policy,
            critic,
            meta,
        )
        self.assertEqual(calls, {"meta": 1, "policy": 1, "critic": 1})
        self.assertEqual(transition.calls, 0)
        self.assertEqual(result.telemetry.mac_reserved, 23)
        self.assertTrue(result.telemetry.mac_budget_exhausted)

    def test_same_seed_replays_trace_and_act_is_hard_clamped(self) -> None:
        cfg = CertifiedPUCTConfigV2(max_depth=4, simulations=5, seed=91)

        def run():
            return BoundedPUCTSearchV2(cfg).search(
                SearchRequestV2(torch.zeros(1), mac_budget=1_000),
                _DeterministicCertifiedTransition(),
                _deep_policy,
                _negative_critic,
                lambda state: SearchControlsV2(0.0, 1.0e-4, 1.0, 10_000),
            )

        first = run()
        second = run()
        self.assertEqual(first.telemetry.trace_digest, second.telemetry.trace_digest)
        self.assertTrue(torch.equal(first.root_visit_counts, second.root_visit_counts))
        self.assertTrue(torch.equal(first.best_path, second.best_path))
        self.assertEqual(first.telemetry.act_requested, 10_000)
        self.assertEqual(first.telemetry.act_applied, 5)
        self.assertTrue(first.telemetry.act_clamped)

    def test_depth_15_35_100_150_share_exact_preallocation(self) -> None:
        root = torch.zeros(1, 2)
        allocated: list[int] = []
        estimates: list[int] = []
        for depth in (15, 35, 100, 150):
            simulations = min(depth + 1, 101)
            search = BoundedPUCTSearchV2(
                CertifiedPUCTConfigV2(
                    max_depth=depth,
                    simulations=simulations,
                )
            )
            result = search.search(
                SearchRequestV2(root, mac_budget=10_000),
                _DeterministicCertifiedTransition(),
                _deep_policy,
                lambda state: torch.tensor(1.0, device=state.device),
                lambda state: _controls(simulations),
            )
            allocated.append(result.telemetry.allocated_bytes)
            estimates.append(search.estimated_preallocated_bytes(root))
            self.assertEqual(
                result.telemetry.max_depth_reached,
                min(depth, 100),
            )
            self.assertEqual(
                result.telemetry.capacity_exhausted,
                depth == 150,
            )
        self.assertEqual(len(set(allocated)), 1)
        self.assertEqual(allocated, estimates)

    def test_saturated_breadth_search_uses_certified_frontier_to_depth_100(
        self,
    ) -> None:
        root = torch.zeros(1, 2)
        search = BoundedPUCTSearchV2(
            CertifiedPUCTConfigV2(max_depth=100, simulations=301)
        )

        result = search.search(
            SearchRequestV2(root, mac_budget=10_000),
            _DeterministicCertifiedTransition(),
            lambda state: torch.zeros(3, device=state.device),
            lambda state: torch.zeros((), device=state.device),
            lambda state: SearchControlsV2(1.0, 1.0e-4, 1.0, 301),
        )

        self.assertEqual(result.telemetry.nodes_used, 301)
        self.assertTrue(result.telemetry.capacity_exhausted)
        self.assertGreater(result.telemetry.frontier_rollouts, 0)
        self.assertEqual(result.telemetry.max_depth_reached, 100)
        self.assertEqual(
            result.telemetry.allocated_bytes,
            search.estimated_preallocated_bytes(root),
        )
        self.assertTrue(result.telemetry.safe_for_decode)
        self.assertTrue(torch.isfinite(result.best_state).all())

    def test_programming_shape_error_fails_closed_globally(self) -> None:
        class InvalidTransition(_DeterministicCertifiedTransition):
            def __call__(self, *args, **kwargs):
                batch = super().__call__(*args, **kwargs)
                return CertifiedTransitionBatchV2(
                    states=batch.states[:2],
                    success=batch.success,
                    residuals=batch.residuals,
                    iterations=batch.iterations,
                    linear_solve_fallbacks=batch.linear_solve_fallbacks,
                    warm_used=batch.warm_used,
                    warm_rejected=batch.warm_rejected,
                    warm_starts=batch.warm_starts,
                )

        with self.assertRaisesRegex(ValueError, "transition states"):
            BoundedPUCTSearchV2(
                CertifiedPUCTConfigV2(max_depth=2, simulations=1)
            ).search(
                SearchRequestV2(torch.zeros(1), mac_budget=100),
                InvalidTransition(),
                _deep_policy,
                _negative_critic,
                lambda state: _controls(1),
            )


if __name__ == "__main__":
    unittest.main()
