import unittest

import torch

from cogni_flow.harness import FailureTrace, PatchPolicy, WeaknessCluster
from cogni_flow.local_proposer import (
    HARD_MAX_NEW_TOKENS,
    LocalGemmaPatchProposer,
    ResolvedPatchTarget,
)


class FakeTokenizer:
    def __init__(self, decoded):
        self.decoded = decoded
        self.call = None
        self.decode_call = None

    def __call__(self, prompt, **kwargs):
        self.call = (prompt, kwargs)
        return {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, tokens, **kwargs):
        self.decode_call = (tokens.clone(), kwargs)
        return self.decoded


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.generate_call = None
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_call = kwargs
        suffix = torch.tensor([[99, 100]], device=kwargs["input_ids"].device)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


def cluster():
    trace = FailureTrace("test_x", "AssertionError", "V1", "answer", "wrong")
    return WeaknessCluster(("AssertionError", "V1", "answer"), (trace,))


class TestLocalGemmaPatchProposer(unittest.TestCase):
    def test_injected_model_generates_replacement_only_under_hard_caps(self):
        model = FakeModel()
        tokenizer = FakeTokenizer("```python\nVALUE = 2\n```")
        resolved = ResolvedPatchTarget("pkg/value.py", "A" * 64, "VALUE = 1\n")
        proposer = LocalGemmaPatchProposer(
            model,
            tokenizer,
            lambda weakness: resolved,
            policy=PatchPolicy(allowed_roots=("pkg",)),
            max_new_tokens=100_000,
            max_input_tokens=100_000,
        )

        proposal = proposer(cluster())[0]
        self.assertTrue(model.eval_called)
        self.assertEqual(proposal.relative_path, "pkg/value.py")
        self.assertEqual(proposal.base_sha256, "a" * 64)
        self.assertEqual(proposal.replacement, "VALUE = 2\n")
        self.assertFalse(model.generate_call["use_cache"])
        self.assertFalse(model.generate_call["do_sample"])
        self.assertEqual(model.generate_call["max_new_tokens"], HARD_MAX_NEW_TOKENS)
        self.assertNotIn("pkg/value.py", tokenizer.decoded)
        self.assertEqual(tokenizer.call[1]["max_length"], 8_192)

    def test_candidate_is_not_executed_by_proposer(self):
        model = FakeModel()
        tokenizer = FakeTokenizer("raise RuntimeError('must never execute here')")
        target = ResolvedPatchTarget("pkg/value.py", "0" * 64, "VALUE = 1\n")
        proposer = LocalGemmaPatchProposer(
            model,
            tokenizer,
            lambda weakness: target,
            policy=PatchPolicy(allowed_roots=("pkg",)),
        )
        proposal = proposer(cluster())[0]
        self.assertIn("must never execute", proposal.replacement)

    def test_network_import_is_rejected_by_early_policy_gate(self):
        model = FakeModel()
        tokenizer = FakeTokenizer("import requests\nVALUE = 2")
        target = ResolvedPatchTarget("pkg/value.py", "0" * 64, "VALUE = 1\n")
        proposer = LocalGemmaPatchProposer(
            model,
            tokenizer,
            lambda weakness: target,
            policy=PatchPolicy(allowed_roots=("pkg",)),
        )
        with self.assertRaisesRegex(ValueError, "network"):
            proposer(cluster())

    def test_malformed_fence_and_path_based_model_loading_are_rejected(self):
        target = ResolvedPatchTarget("pkg/value.py", "0" * 64, "VALUE = 1\n")
        proposer = LocalGemmaPatchProposer(
            FakeModel(),
            FakeTokenizer("explanation\n```python\nVALUE = 2\n```"),
            lambda weakness: target,
            policy=PatchPolicy(allowed_roots=("pkg",)),
        )
        with self.assertRaisesRegex(ValueError, "markdown"):
            proposer(cluster())
        with self.assertRaisesRegex(TypeError, "objects"):
            LocalGemmaPatchProposer(
                "C:/models/gemma",
                FakeTokenizer("VALUE = 2"),
                lambda weakness: target,
            )


if __name__ == "__main__":
    unittest.main()
