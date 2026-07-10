import math
import unittest

import torch

from cogni_core.search import (
    BoundedPUCTSearch,
    ContractiveBroydenTransition,
    PUCTConfig,
    SemanticAncestorRetriever,
    TransitionContractError,
    deq_tensor_transition,
    puct_scores,
)
from cogni_core.deq import ContractivityError


@deq_tensor_transition
def _transition(state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    increments = (actions + 1).to(dtype=state.dtype)
    view = (actions.shape[0],) + (1,) * state.ndim
    return state.unsqueeze(0) + increments.view(view)


def _uniform_policy_value(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(3, device=state.device), state.float().mean()


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


if __name__ == "__main__":
    unittest.main()
