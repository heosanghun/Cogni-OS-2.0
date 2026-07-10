from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from .deq import EquilibriumLayer


@dataclass
class CTSResult:
    state: Tensor
    score: Tensor
    depth: int
    expanded: int


class CognitiveTreeSearch(nn.Module):
    """Fixed-width latent search without retaining an autograd tree or KV cache."""

    def __init__(self, transition: EquilibriumLayer, width: int = 3):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        self.transition = transition
        self.width = width

    @torch.no_grad()
    def search(
        self,
        root: Tensor,
        action_encoder: Callable[[Tensor, int], Tensor],
        critic: Callable[[Tensor], Tensor],
        depth: int,
    ) -> CTSResult:
        state = root
        expanded = 0
        for _ in range(depth):
            candidates = torch.stack(
                [self.transition(action_encoder(state, a)) for a in range(self.width)]
            )
            scores = torch.stack(
                [critic(candidate).reshape(()) for candidate in candidates]
            )
            state = candidates[int(scores.argmax())]
            expanded += self.width
        return CTSResult(state, critic(state), depth, expanded)
