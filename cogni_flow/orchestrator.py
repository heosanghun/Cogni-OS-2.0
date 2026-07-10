from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .aflow import AFlowOptimizer, SearchResult, WorkflowSpec
from .logdb import LogDB
from .rhythm import RhythmController, SystemMode


@dataclass(frozen=True)
class WorkflowEvolutionReport:
    search: SearchResult
    resumed_inference: bool


class WorkflowEvolutionCoordinator:
    """Runs AFlow strictly inside a checkpointed night/evolution window."""

    def __init__(
        self,
        rhythm: RhythmController,
        optimizer: AFlowOptimizer,
        checkpoint: Callable[[], None],
        logdb: LogDB | None = None,
    ) -> None:
        self.rhythm = rhythm
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.logdb = logdb

    def run(self, initial: WorkflowSpec) -> WorkflowEvolutionReport:
        self.rhythm.enter_evolution(self.checkpoint)
        try:
            with self.rhythm.evolution_slot():
                result = self.optimizer.search(initial)
                if self.logdb is not None:
                    self.logdb.audit(
                        "workflow_search",
                        result.best.workflow.workflow_id,
                        f"score={result.best.score};evaluations={result.evaluations}",
                    )
            self.rhythm.resume_inference("workflow search completed")
            return WorkflowEvolutionReport(result, True)
        except Exception:
            if self.rhythm.mode == SystemMode.EVOLUTION:
                self.rhythm.resume_inference("workflow search failed")
            raise
