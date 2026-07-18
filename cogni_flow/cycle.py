from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .harness import PatchProposal, SafeHarnessPatcher, WeaknessCluster, mine_weaknesses
from .logdb import LogDB
from .rhythm import RhythmController, SystemMode


@dataclass(frozen=True)
class EvolutionReport:
    clusters: int
    proposals: int
    promoted: bool
    target: str | None
    proposal_only: bool = False
    blocked_reason: str | None = None
    awaiting_approval: bool = False
    evaluation_id: str | None = None


class SelfHarness:
    """One bounded propose-evaluate-accept cycle over declared harness surfaces."""

    def __init__(
        self,
        rhythm: RhythmController,
        logdb: LogDB,
        patcher: SafeHarnessPatcher,
        proposer: Callable[[WeaknessCluster], Iterable[PatchProposal]],
        checkpoint: Callable[[], None],
        *,
        proposal_only_reason: str | None = None,
    ) -> None:
        self.rhythm = rhythm
        self.logdb = logdb
        self.patcher = patcher
        self.proposer = proposer
        self.checkpoint = checkpoint
        self.proposal_only_reason = proposal_only_reason

    def run_night_cycle(self, since: float = 0.0) -> EvolutionReport:
        self.rhythm.enter_evolution(self.checkpoint)
        clusters = mine_weaknesses(self.logdb.failures_since(since))
        proposal_count = 0
        try:
            promoted_target: str | None = None
            awaiting_target: str | None = None
            evaluation_id: str | None = None
            # Proposal generation is evolution work too. Keep the lifetime
            # counter active across generation, nested sandbox validation, and
            # promotion so another thread cannot resume daytime inference in
            # the gaps between candidates.
            with self.rhythm.evolution_slot():
                for cluster in clusters:
                    for proposal in self.proposer(cluster):
                        proposal_count += 1
                        if self.proposal_only_reason is not None:
                            self.patcher.validate_proposal(proposal)
                            self.logdb.audit(
                                "candidate",
                                proposal.relative_path,
                                "proposal_only",
                            )
                            continue
                        if getattr(self.patcher, "requires_manual_approval", False):
                            identity = getattr(
                                self.proposer, "proposal_id_for_patch", None
                            )
                            if not callable(identity):
                                raise RuntimeError(
                                    "candidate evaluation requires proposal evidence identity"
                                )
                            result = self.patcher.evaluate_candidate(
                                proposal,
                                proposal_id=identity(proposal),
                            )
                            self.logdb.audit(
                                "candidate",
                                proposal.relative_path,
                                "awaiting_approval"
                                if result.passed
                                else f"failed:{result.sandbox.returncode}",
                            )
                            if result.passed:
                                if result.evaluation is None:
                                    raise RuntimeError(
                                        "passing candidate has no evaluation evidence"
                                    )
                                awaiting_target = str(result.target)
                                evaluation_id = result.evaluation.evaluation_id
                                break
                            continue
                        result = self.patcher.validate_and_promote(proposal)
                        self.logdb.audit(
                            "candidate",
                            proposal.relative_path,
                            "passed"
                            if result.promoted
                            else f"failed:{result.sandbox.returncode}",
                        )
                        if result.promoted:
                            promoted_target = str(result.target)
                            break
                    if promoted_target is not None:
                        break
                    if awaiting_target is not None:
                        break
            if promoted_target is not None:
                self.rhythm.resume_inference("validated patch promoted")
                return EvolutionReport(
                    len(clusters),
                    proposal_count,
                    True,
                    promoted_target,
                    False,
                    None,
                )
            if awaiting_target is not None:
                self.rhythm.resume_inference(
                    "candidate evaluation awaits external approval"
                )
                return EvolutionReport(
                    len(clusters),
                    proposal_count,
                    False,
                    awaiting_target,
                    False,
                    "external human approval required",
                    True,
                    evaluation_id,
                )
            self.rhythm.resume_inference("no candidate passed validation")
            return EvolutionReport(
                len(clusters),
                proposal_count,
                False,
                None,
                self.proposal_only_reason is not None,
                self.proposal_only_reason,
            )
        except Exception as exc:
            self.logdb.audit("cycle_error", "self_harness", type(exc).__name__)
            if self.rhythm.mode in {SystemMode.VALIDATING, SystemMode.PROMOTING}:
                self.rhythm.transition(SystemMode.ROLLING_BACK, "validation exception")
                self.rhythm.transition(SystemMode.INFERENCE, "rollback complete")
            elif self.rhythm.mode == SystemMode.EVOLUTION:
                self.rhythm.resume_inference("evolution exception")
            elif self.rhythm.mode == SystemMode.ROLLING_BACK:
                self.rhythm.transition(SystemMode.INFERENCE, "rollback complete")
            raise
