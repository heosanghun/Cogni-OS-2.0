from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from cogni_agent.conditioning import (
    BoundedLatentLogitsProcessor,
    LatentConditioningError,
    MAX_INPUT_ELEMENTS,
    MAX_VOCAB_SIZE,
    build_latent_logits_processor,
)


class _TinyModel(nn.Module):
    def __init__(self, *, frozen: bool = True) -> None:
        super().__init__()
        self.head = nn.Linear(3, 6, bias=True)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, -1.0],
                        [0.5, -0.5, 1.0],
                        [-1.0, 0.25, 0.5],
                        [0.0, 1.0, -0.25],
                        [0.75, 0.5, 0.25],
                        [-0.5, -1.0, 0.75],
                    ]
                )
            )
            self.head.bias.copy_(torch.linspace(-0.3, 0.3, 6))
        for parameter in self.head.parameters():
            parameter.requires_grad_(not frozen)

    def get_output_embeddings(self) -> nn.Module:
        return self.head


class _WideModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, MAX_VOCAB_SIZE + 1, bias=False)
        self.head.weight.requires_grad_(False)

    def get_output_embeddings(self) -> nn.Module:
        return self.head


class TestLatentConditioning(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _TinyModel()
        self.latent = torch.tensor([[0.75, -0.5, 0.25]], dtype=torch.float32)

    def test_counterfactual_latents_change_logits_without_mutating_scores(self) -> None:
        first = build_latent_logits_processor(self.model, self.latent)
        second = build_latent_logits_processor(
            self.model,
            torch.tensor([[-0.5, 0.25, 0.75]], dtype=torch.float32),
        )
        input_ids = torch.tensor([[0, 2, 5]], dtype=torch.int64)
        scores = torch.zeros(1, 6)
        original = scores.clone()

        first_logits = first(input_ids, scores)
        second_logits = second(input_ids, scores)

        self.assertTrue(torch.equal(scores, original))
        self.assertNotEqual(first_logits.data_ptr(), scores.data_ptr())
        self.assertFalse(torch.equal(first_logits, second_logits))
        torch.testing.assert_close(first_logits, scores + first.bias)

    def test_hf_suppression_chain_preserves_negative_infinity(self) -> None:
        try:
            from transformers import LogitsProcessorList
            from transformers.generation.logits_process import (
                SuppressTokensLogitsProcessor,
            )
        except ImportError:  # pragma: no cover - optional Gemma dependency
            self.skipTest("transformers is not installed")

        processor = build_latent_logits_processor(self.model, self.latent)
        input_ids = torch.tensor([[0, 2, 5]], dtype=torch.int64)
        scores = torch.zeros(1, 6)
        chain = LogitsProcessorList([SuppressTokensLogitsProcessor([1, 4]), processor])

        result = chain(input_ids, scores)

        self.assertTrue(torch.isneginf(result[0, 1]))
        self.assertTrue(torch.isneginf(result[0, 4]))
        finite = torch.tensor([0, 2, 3, 5])
        torch.testing.assert_close(
            result[0, finite], processor.bias[0, finite], rtol=0.0, atol=0.0
        )

    def test_projection_normalization_bound_and_determinism(self) -> None:
        processor = build_latent_logits_processor(
            self.model, self.latent, max_abs_bias=0.05
        )
        repeated = build_latent_logits_processor(
            self.model, self.latent.clone(), max_abs_bias=0.05
        )
        bias = processor.bias

        raw = F.linear(
            self.latent,
            self.model.head.weight,
            self.model.head.bias,
        )
        centered = raw - raw.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
        expected = torch.tanh(centered / rms) * 0.05

        self.assertEqual(tuple(bias.shape), (1, 6))
        self.assertTrue(bool(torch.isfinite(bias).all()))
        self.assertLessEqual(float(bias.abs().max()), 0.05)
        torch.testing.assert_close(bias, expected)
        torch.testing.assert_close(bias, repeated.bias, rtol=0.0, atol=0.0)

    def test_fixed_bias_is_input_independent_and_defensively_exposed(self) -> None:
        processor = build_latent_logits_processor(self.model, self.latent)
        scores = torch.arange(6, dtype=torch.float32).unsqueeze(0)
        first = processor(torch.tensor([[0, 1]], dtype=torch.int64), scores)
        second = processor(torch.tensor([[4, 5, 2]], dtype=torch.int64), scores)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

        exposed = processor.bias
        exposed.zero_()
        self.assertFalse(torch.equal(exposed, processor.bias))
        self.assertFalse(hasattr(processor, "__dict__"))

    def test_build_rejects_malformed_latent_head_and_budget(self) -> None:
        malformed = (
            torch.ones(3),
            torch.ones(2, 3),
            torch.ones(1, 2),
            torch.ones(1, 3, dtype=torch.int64),
            torch.tensor([[0.0, float("nan"), 1.0]]),
        )
        for latent in malformed:
            with self.subTest(shape=tuple(latent.shape), dtype=str(latent.dtype)):
                with self.assertRaises((TypeError, ValueError)):
                    build_latent_logits_processor(self.model, latent)

        with self.assertRaises(TypeError):
            build_latent_logits_processor(object(), self.latent)
        with self.assertRaises(LatentConditioningError):
            build_latent_logits_processor(_TinyModel(frozen=False), self.latent)
        with self.assertRaises(ValueError):
            build_latent_logits_processor(_WideModel(), torch.ones(1, 1))
        for invalid in (0.0, -0.1, 1.1, float("inf"), float("nan")):
            with self.subTest(max_abs_bias=invalid):
                with self.assertRaises(ValueError):
                    build_latent_logits_processor(
                        self.model, self.latent, max_abs_bias=invalid
                    )

    def test_processor_rejects_malformed_call_inputs_fail_closed(self) -> None:
        processor = build_latent_logits_processor(self.model, self.latent)
        valid_ids = torch.tensor([[0, 1]], dtype=torch.int64)
        valid_scores = torch.zeros(1, 6)
        cases = (
            (torch.tensor([0, 1]), valid_scores),
            (valid_ids, torch.zeros(6)),
            (valid_ids, torch.zeros(1, 5)),
            (valid_ids, torch.zeros(2, 6)),
            (valid_ids.to(torch.int32), valid_scores),
            (valid_ids, valid_scores.to(torch.float64)),
            (torch.tensor([[6]], dtype=torch.int64), valid_scores),
            (valid_ids, torch.tensor([[0, 0, 0, 0, 0, float("nan")]])),
            (valid_ids, torch.tensor([[0, 0, 0, 0, 0, float("inf")]])),
            (valid_ids, torch.full((1, 6), float("-inf"))),
        )
        for input_ids, scores in cases:
            with self.subTest(ids=tuple(input_ids.shape), scores=tuple(scores.shape)):
                with self.assertRaises((TypeError, ValueError)):
                    processor(input_ids, scores)

        meta_ids = torch.empty((1, 1), dtype=torch.int64, device="meta")
        with self.assertRaises(ValueError):
            processor(meta_ids, valid_scores)
        oversized = torch.zeros((1, MAX_INPUT_ELEMENTS + 1), dtype=torch.int64)
        with self.assertRaises(ValueError):
            processor(oversized, valid_scores)

    def test_base_head_is_immutable_across_build_and_application(self) -> None:
        training = self.model.head.training
        parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.head.named_parameters()
        }
        versions = {
            name: parameter._version
            for name, parameter in self.model.head.named_parameters()
        }

        processor = build_latent_logits_processor(self.model, self.latent)
        processor(torch.tensor([[1, 2]], dtype=torch.int64), torch.zeros(1, 6))

        self.assertEqual(self.model.head.training, training)
        for name, parameter in self.model.head.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertEqual(parameter._version, versions[name])
            torch.testing.assert_close(
                parameter,
                parameters[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_non_finite_head_projection_is_rejected(self) -> None:
        with torch.no_grad():
            self.model.head.weight[0, 0] = float("inf")
        with self.assertRaises(LatentConditioningError):
            build_latent_logits_processor(self.model, self.latent)

    def test_bfloat16_head_compiles_once_to_generation_fp32(self) -> None:
        model = _TinyModel().to(dtype=torch.bfloat16)
        latent = self.latent.to(dtype=torch.bfloat16)

        processor = build_latent_logits_processor(model, latent)
        result = processor(
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.zeros(1, 6, dtype=torch.float32),
        )

        self.assertEqual(processor.bias.dtype, torch.float32)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_direct_processor_constructor_enforces_vocab_and_finiteness(self) -> None:
        with self.assertRaises(ValueError):
            BoundedLatentLogitsProcessor(torch.zeros(1, MAX_VOCAB_SIZE + 1))
        with self.assertRaises(ValueError):
            BoundedLatentLogitsProcessor(torch.tensor([[float("inf")]]))


if __name__ == "__main__":
    unittest.main()
