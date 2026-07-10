from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Callable

import torch

from .harness import PatchPolicy, PatchProposal, WeaknessCluster


HARD_MAX_NEW_TOKENS = 2_048
HARD_MAX_INPUT_TOKENS = 8_192


@dataclass(frozen=True)
class ResolvedPatchTarget:
    """Trusted host resolution; the language model cannot select this target."""

    relative_path: str
    base_sha256: str
    current_source: str


TargetResolver = Callable[[WeaknessCluster], ResolvedPatchTarget | None]


class LocalGemmaPatchProposer:
    """Generate one replacement using already-injected, local-only objects.

    This component cannot load a model, resolve a filesystem target, write a
    file, or execute candidate code. It performs an early ``PatchPolicy`` AST
    check; ``SafeHarnessPatcher`` must independently repeat that check and run
    the candidate in a kernel-isolated sandbox before promotion.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        target_resolver: TargetResolver,
        *,
        policy: PatchPolicy | None = None,
        max_new_tokens: int = 1_024,
        max_input_tokens: int = 4_096,
        prompt_char_limit: int = 32_000,
    ) -> None:
        if isinstance(model, (str, bytes, os.PathLike)) or isinstance(
            tokenizer, (str, bytes, os.PathLike)
        ):
            raise TypeError("inject loaded local model/tokenizer objects, not paths")
        if not callable(getattr(model, "generate", None)):
            raise TypeError("model must provide generate()")
        if not callable(tokenizer) or not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("tokenizer must be callable and provide decode()")
        if max_new_tokens < 1 or max_input_tokens < 1:
            raise ValueError("token limits must be positive")
        if prompt_char_limit < 1:
            raise ValueError("prompt_char_limit must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.target_resolver = target_resolver
        self.policy = policy or PatchPolicy()
        self.max_new_tokens = min(max_new_tokens, HARD_MAX_NEW_TOKENS)
        self.max_input_tokens = min(max_input_tokens, HARD_MAX_INPUT_TOKENS)
        self.prompt_char_limit = prompt_char_limit
        eval_method = getattr(self.model, "eval", None)
        if callable(eval_method):
            eval_method()

    def __call__(self, cluster: WeaknessCluster) -> tuple[PatchProposal, ...]:
        target = self.target_resolver(cluster)
        if target is None:
            return ()
        self._validate_resolution(target)
        prompt = self._build_prompt(cluster, target)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        if "input_ids" not in inputs:
            raise ValueError("tokenizer did not return input_ids")
        prepared = self._move_inputs(dict(inputs))
        prompt_width = int(prepared["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = self.model.generate(
                **prepared,
                use_cache=False,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        sequences = getattr(generated, "sequences", generated)
        if getattr(sequences, "ndim", 0) != 2 or int(sequences.shape[0]) != 1:
            raise ValueError("model must return one decoder-only token sequence")
        if int(sequences.shape[-1]) <= prompt_width:
            raise ValueError("model returned no replacement tokens")
        replacement_tokens = sequences[0, prompt_width:]
        decoded = self.tokenizer.decode(
            replacement_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        replacement = _clean_replacement(decoded)
        proposal = PatchProposal(
            relative_path=target.relative_path,
            base_sha256=target.base_sha256.lower(),
            replacement=replacement,
            rationale=("local-model repair for " + "/".join(cluster.signature)),
        )

        # This is a static parse/policy gate only. Candidate execution belongs
        # exclusively to SafeHarnessPatcher's kernel-isolated sandbox stage.
        self.policy.validate(proposal)
        return (proposal,)

    def _build_prompt(
        self, cluster: WeaknessCluster, target: ResolvedPatchTarget
    ) -> str:
        excerpts = "\n".join(
            f"- {trace.test_id}: {trace.excerpt}" for trace in cluster.traces[:8]
        )
        prompt = (
            "You are an offline Python repair compiler.\n"
            "Return only the complete replacement source for the supplied file.\n"
            "Do not return a path, hash, diff, explanation, shell command, or markdown.\n"
            f"Failure signature: {' / '.join(cluster.signature)}\n"
            f"Failure evidence:\n{excerpts}\n"
            "Current source begins below.\n"
            "<CURRENT_SOURCE>\n"
            f"{target.current_source}\n"
            "</CURRENT_SOURCE>\n"
            "Complete replacement source:\n"
        )
        return prompt[-self.prompt_char_limit :]

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        device = getattr(self.model, "device", None)
        if device is None:
            parameters = getattr(self.model, "parameters", None)
            if callable(parameters):
                try:
                    device = next(parameters()).device
                except (StopIteration, TypeError):
                    device = None
        if device is None:
            return inputs
        return {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }

    @staticmethod
    def _validate_resolution(target: ResolvedPatchTarget) -> None:
        digest = target.base_sha256
        if len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise ValueError("target resolver returned an invalid SHA-256 digest")
        if not target.relative_path:
            raise ValueError("target resolver returned an empty path")


_FENCED = re.compile(
    r"\A```[A-Za-z0-9_+.-]*[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    re.DOTALL,
)


def _clean_replacement(decoded: str) -> str:
    text = decoded.replace("\ufeff", "").strip()
    match = _FENCED.fullmatch(text)
    if match is not None:
        text = match.group("body").strip()
    elif "```" in text:
        raise ValueError("malformed or explanatory markdown fence rejected")
    if not text:
        raise ValueError("model returned an empty replacement")
    return text + "\n"
