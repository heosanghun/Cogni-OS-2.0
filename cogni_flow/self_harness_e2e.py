"""Operator-only append-only evidence for a full Self-Harness transaction.

The product UI remains proposal-only.  This API does not sign approvals,
invent runner trust, or auto-promote.  It only sequences already attested
runtime primitives and records a bounded, restart-readable hash chain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from time import time_ns
from typing import Any, Mapping

from .approval import (
    CandidateEvaluationV1,
    Ed25519ApprovalVerifier,
    HumanApprovalV1,
    RollbackAuthorizationV1,
    canonical_json_bytes,
)
from .production import (
    BackupRecord,
    JournaledHarnessPatcher,
    ProductionSelfHarness,
    PromotionMode,
    command_sha256,
)


E2E_EVENT_SCHEMA = "cogni.self_harness.operator_e2e_event.v1"
E2E_VALIDATION_SCHEMA = "cogni.self_harness.operator_e2e_validation.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_NONCE = re.compile(r"[A-Za-z0-9_-]{32,128}\Z")
_MAX_EVENT_BYTES = 64 * 1024
_MAX_RUNS = 64
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_ZERO_DIGEST = "0" * 64


class SelfHarnessE2EError(RuntimeError):
    """Raised when an operator E2E run or its evidence fails closed."""


class SelfHarnessE2EReplayError(SelfHarnessE2EError):
    """Raised when an operator run nonce was already used."""


@dataclass(frozen=True, slots=True)
class SelfHarnessE2EEventV1:
    schema: str
    run_id: str
    sequence: int
    stage: str
    previous_event_sha256: str
    created_ns: int
    payload: Mapping[str, Any]
    event_sha256: str

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        sequence: int,
        stage: str,
        previous_event_sha256: str,
        created_ns: int,
        payload: Mapping[str, Any],
    ) -> SelfHarnessE2EEventV1:
        base = {
            "schema": E2E_EVENT_SCHEMA,
            "run_id": run_id,
            "sequence": sequence,
            "stage": stage,
            "previous_event_sha256": previous_event_sha256,
            "created_ns": created_ns,
            "payload": dict(payload),
        }
        event_sha256 = sha256(canonical_json_bytes(base)).hexdigest()
        event = cls(event_sha256=event_sha256, **base)
        event._validate_shape()
        return event

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SelfHarnessE2EEventV1:
        if set(payload) != set(cls.__dataclass_fields__):
            raise SelfHarnessE2EError("operator E2E event fields are invalid")
        try:
            event = cls(**dict(payload))
        except TypeError as exc:
            raise SelfHarnessE2EError("operator E2E event is malformed") from exc
        event._validate_shape()
        expected = cls.create(
            run_id=event.run_id,
            sequence=event.sequence,
            stage=event.stage,
            previous_event_sha256=event.previous_event_sha256,
            created_ns=event.created_ns,
            payload=event.payload,
        )
        if event.event_sha256 != expected.event_sha256:
            raise SelfHarnessE2EError("operator E2E event digest is invalid")
        return event

    def _validate_shape(self) -> None:
        if self.schema != E2E_EVENT_SCHEMA:
            raise SelfHarnessE2EError("operator E2E event schema is unsupported")
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise SelfHarnessE2EError("operator E2E run id is invalid")
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or not 1 <= self.sequence <= 3
        ):
            raise SelfHarnessE2EError("operator E2E event sequence is invalid")
        allowed_stages = {
            1: {"evaluation_ready"},
            2: {"promotion_committed", "promotion_health_restore"},
            3: {"rollback_completed", "rollback_health_restore"},
        }
        if self.stage not in allowed_stages[self.sequence]:
            raise SelfHarnessE2EError("operator E2E stage/sequence is invalid")
        _digest(self.previous_event_sha256, "previous event")
        if self.sequence == 1 and self.previous_event_sha256 != _ZERO_DIGEST:
            raise SelfHarnessE2EError("first E2E event has a previous digest")
        if self.sequence > 1 and self.previous_event_sha256 == _ZERO_DIGEST:
            raise SelfHarnessE2EError("chained E2E event lacks a previous digest")
        if (
            not isinstance(self.created_ns, int)
            or isinstance(self.created_ns, bool)
            or self.created_ns < 1
        ):
            raise SelfHarnessE2EError("operator E2E event time is invalid")
        if not isinstance(self.payload, Mapping):
            raise SelfHarnessE2EError("operator E2E payload must be an object")
        _digest(self.event_sha256, "event")
        _validate_stage_payload(self.stage, self.payload)


@dataclass(frozen=True, slots=True)
class SelfHarnessE2EValidationResult:
    schema: str
    run_id: str
    event_count: int
    terminal_stage: str
    chain_sha256: str
    full_e2e_complete: bool
    runner_attestation_digest_bound: bool
    production_attestation_reverified: bool
    assurance: str


class SelfHarnessE2ELedger:
    """Bounded immutable event store with deterministic restart hydration."""

    def __init__(self, directory: str | Path, *, max_runs: int = 64) -> None:
        if not 1 <= max_runs <= _MAX_RUNS:
            raise ValueError("operator E2E run bound is invalid")
        self.directory = Path(directory)
        self.max_runs = max_runs
        self._lock = RLock()
        self.directory.mkdir(parents=True, exist_ok=True)
        _require_directory(self.directory)

    def events(self, run_id: str | None = None) -> tuple[SelfHarnessE2EEventV1, ...]:
        with self._lock:
            _require_directory(self.directory)
            events: list[SelfHarnessE2EEventV1] = []
            observed_run_ids: set[str] = set()
            entries = sorted(os.scandir(self.directory), key=lambda item: item.name)
            if len(entries) > self.max_runs * 3:
                raise SelfHarnessE2EError("operator E2E ledger crossed its event bound")
            for entry in entries:
                item_stat = entry.stat(follow_symlinks=False)
                attributes = getattr(item_stat, "st_file_attributes", 0)
                if (
                    entry.is_symlink()
                    or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
                    or not stat.S_ISREG(item_stat.st_mode)
                    or not entry.name.endswith(".json")
                ):
                    raise SelfHarnessE2EError(
                        "operator E2E ledger contains an invalid entry"
                    )
                raw = _read_bounded(Path(entry.path))
                data = _strict_json(raw)
                event = SelfHarnessE2EEventV1.from_mapping(data)
                expected_name = f"{event.run_id}-{event.sequence:02d}.json"
                if (
                    entry.name != expected_name
                    or canonical_json_bytes(asdict(event)) != raw
                ):
                    raise SelfHarnessE2EError(
                        "operator E2E event filename/canonical form is invalid"
                    )
                observed_run_ids.add(event.run_id)
                if run_id is None or event.run_id == run_id:
                    events.append(event)
            if len(observed_run_ids) > self.max_runs:
                raise SelfHarnessE2EError("operator E2E ledger crossed its hard bound")
            selected = tuple(
                sorted(events, key=lambda item: (item.run_id, item.sequence))
            )
            if run_id is None:
                for observed in sorted(observed_run_ids):
                    _validate_chain(
                        tuple(item for item in selected if item.run_id == observed)
                    )
            elif selected:
                if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
                    raise SelfHarnessE2EError("operator E2E run id is invalid")
                _validate_chain(selected)
            return selected

    def consume_run_nonce(self, nonce: str, evaluation_id: str) -> str:
        if not isinstance(nonce, str) or _NONCE.fullmatch(nonce) is None:
            raise SelfHarnessE2EError("operator E2E nonce is invalid")
        _digest(evaluation_id, "evaluation")
        for event in self.events():
            if event.sequence == 1 and event.payload["run_nonce"] == nonce:
                raise SelfHarnessE2EReplayError(
                    "operator E2E run nonce was already consumed"
                )
        run_id = sha256(nonce.encode("ascii")).hexdigest()[:32]
        if self.events(run_id):
            raise SelfHarnessE2EReplayError("operator E2E run already exists")
        return run_id

    def append(self, event: SelfHarnessE2EEventV1) -> None:
        if not isinstance(event, SelfHarnessE2EEventV1):
            raise TypeError("event must be a SelfHarnessE2EEventV1")
        with self._lock:
            all_events = self.events()
            run_ids = {item.run_id for item in all_events}
            existing = tuple(item for item in all_events if item.run_id == event.run_id)
            if existing:
                _validate_chain(existing)
            elif len(run_ids) >= self.max_runs:
                raise SelfHarnessE2EError(
                    "operator E2E ledger reached its hard run bound"
                )
            expected_sequence = len(existing) + 1
            if event.sequence != expected_sequence:
                raise SelfHarnessE2EError("operator E2E append sequence is invalid")
            expected_previous = (
                _ZERO_DIGEST if not existing else existing[-1].event_sha256
            )
            if event.previous_event_sha256 != expected_previous:
                raise SelfHarnessE2EError("operator E2E append chain is invalid")
            target = self.directory / f"{event.run_id}-{event.sequence:02d}.json"
            _write_exclusive(target, canonical_json_bytes(asdict(event)))


class OperatorSelfHarnessE2E:
    """Explicit operator sequencer over existing signed transaction APIs."""

    def __init__(
        self,
        service: ProductionSelfHarness,
        ledger: SelfHarnessE2ELedger,
    ) -> None:
        if not isinstance(service, ProductionSelfHarness):
            raise TypeError("service must be ProductionSelfHarness")
        if service.config.promotion_mode != PromotionMode.ATTESTED:
            raise SelfHarnessE2EError(
                "operator E2E is unavailable in proposal-only mode"
            )
        if not isinstance(ledger, SelfHarnessE2ELedger):
            raise TypeError("ledger must be SelfHarnessE2ELedger")
        if not isinstance(service.harness.patcher, JournaledHarnessPatcher):
            raise SelfHarnessE2EError("operator E2E patcher is not attested")
        self.service = service
        self.ledger = ledger

    def prepare(
        self,
        evaluation_id: str,
        *,
        run_nonce: str,
    ) -> SelfHarnessE2EEventV1:
        if not self.service.failure_daemon.running:
            raise SelfHarnessE2EError("operator E2E service is not running")
        if self.service.evaluation_ledger is None:
            raise SelfHarnessE2EError("candidate evaluation ledger is unavailable")
        evaluation = self.service.evaluation_ledger.load(evaluation_id)
        current = time_ns()
        if current >= evaluation.expires_ns:
            raise SelfHarnessE2EError("candidate evaluation expired before E2E prepare")
        if evaluation.returncode != 0:
            raise SelfHarnessE2EError("candidate regression was not a zero-exit pass")
        proposals = self.service.proposal_ledger.proposals
        selected = next(
            (item for item in proposals if item.proposal_id == evaluation.proposal_id),
            None,
        )
        if selected is None:
            raise SelfHarnessE2EError("candidate evaluation has no proposal evidence")
        batch = tuple(
            item for item in proposals if item.signature == selected.signature
        )
        replacements = tuple(sorted({item.replacement_sha256 for item in batch}))
        if len(batch) < 3 or len(replacements) < 3:
            raise SelfHarnessE2EError(
                "operator E2E requires at least three distinct candidate records"
            )
        if (
            self.service.active_source_surface_sha256
            != evaluation.source_surface_sha256
        ):
            raise SelfHarnessE2EError(
                "active source changed after immutable candidate evaluation"
            )
        target = self.service.harness.patcher.project_root / evaluation.relative_path
        before_bytes = target.read_bytes()
        if sha256(before_bytes).hexdigest() != evaluation.base_sha256:
            raise SelfHarnessE2EError("candidate base bytes changed before E2E prepare")
        run_id = self.ledger.consume_run_nonce(run_nonce, evaluation_id)
        failures = tuple(
            sorted({event_id for proposal in batch for event_id in proposal.event_ids})
        )
        failure_ids = set(failures)
        failure_records = tuple(
            item
            for item in self.service.proposal_ledger.failures
            if item.event_id in failure_ids
        )
        if {item.event_id for item in failure_records} != failure_ids:
            raise SelfHarnessE2EError(
                "operator E2E proposal references missing failure evidence"
            )
        payload = {
            "run_nonce": run_nonce,
            "source_before_sha256": evaluation.source_surface_sha256,
            "target_before_sha256": sha256(before_bytes).hexdigest(),
            "failure_event_ids": list(failures),
            "candidate_proposal_ids": sorted(item.proposal_id for item in batch),
            "candidate_replacement_sha256": list(replacements),
            "candidate_count": len(batch),
            "failure_evidence_root_sha256": _record_root(failure_records),
            "candidate_evidence_root_sha256": _record_root(batch),
            "evaluation": asdict(evaluation),
            "runner_attestation_sha256": evaluation.runner_evidence_sha256,
            "regression_result_sha256": evaluation.result_sha256,
        }
        event = SelfHarnessE2EEventV1.create(
            run_id=run_id,
            sequence=1,
            stage="evaluation_ready",
            previous_event_sha256=_ZERO_DIGEST,
            created_ns=current,
            payload=payload,
        )
        self.ledger.append(event)
        return event

    def promote(
        self,
        run_id: str,
        approval: HumanApprovalV1,
    ) -> SelfHarnessE2EEventV1:
        chain = self.ledger.events(run_id)
        if len(chain) != 1 or chain[0].stage != "evaluation_ready":
            raise SelfHarnessE2EError("operator E2E run is not awaiting promotion")
        evaluation = CandidateEvaluationV1.from_mapping(chain[0].payload["evaluation"])
        operation_started = time_ns()
        before_ids = {record.record_id for record in self.service.journal.records()}
        result = self.service.promote_approved_once(evaluation.evaluation_id, approval)
        records = tuple(
            record
            for record in self.service.journal.records()
            if record.record_id not in before_ids
        )
        if len(records) != 1:
            raise SelfHarnessE2EError(
                "operator promotion did not create exactly one journal record"
            )
        record = records[0]
        expected_status = "committed" if result.promoted else "rolled_back"
        if record.status != expected_status:
            raise SelfHarnessE2EError("operator promotion journal status is invalid")
        live = result.target.read_bytes()
        expected_live = record.after_sha256 if result.promoted else record.before_sha256
        if sha256(live).hexdigest() != expected_live:
            raise SelfHarnessE2EError("operator promotion live bytes are invalid")
        stage = "promotion_committed" if result.promoted else "promotion_health_restore"
        payload = {
            "operation_started_ns": operation_started,
            "approval": asdict(approval),
            "approval_id": approval.approval_id,
            "journal_record": asdict(record),
            "health_passed": result.sandbox.passed,
            "health_returncode": result.sandbox.returncode,
            "health_output_sha256": sha256(
                result.sandbox.output.encode("utf-8")
            ).hexdigest(),
            "health_command_sha256": command_sha256(
                self.service.config.health_check_command
            ),
            "source_after_sha256": self.service.active_source_surface_sha256,
            "target_live_sha256": sha256(live).hexdigest(),
            "byte_restore_verified": (not result.promoted),
        }
        event = SelfHarnessE2EEventV1.create(
            run_id=run_id,
            sequence=2,
            stage=stage,
            previous_event_sha256=chain[-1].event_sha256,
            created_ns=time_ns(),
            payload=payload,
        )
        self.ledger.append(event)
        return event

    def rollback(
        self,
        run_id: str,
        authorization: RollbackAuthorizationV1,
    ) -> SelfHarnessE2EEventV1:
        chain = self.ledger.events(run_id)
        if len(chain) != 2 or chain[-1].stage != "promotion_committed":
            raise SelfHarnessE2EError("operator E2E run is not awaiting rollback")
        record = BackupRecord(**dict(chain[-1].payload["journal_record"]))
        material = self.service.journal.committed_rollback_material(record.record_id)
        original_before = material.before
        operation_started = time_ns()
        result = self.service.rollback_committed_once(record.record_id, authorization)
        live = result.target.read_bytes()
        expected_live = (
            record.before_sha256 if result.rolled_back else record.after_sha256
        )
        if sha256(live).hexdigest() != expected_live:
            raise SelfHarnessE2EError("operator rollback live bytes are invalid")
        byte_identical = (
            live == original_before if result.rolled_back else live == material.after
        )
        if not byte_identical:
            raise SelfHarnessE2EError("operator rollback did not restore exact bytes")
        stage = (
            "rollback_completed" if result.rolled_back else "rollback_health_restore"
        )
        payload = {
            "operation_started_ns": operation_started,
            "authorization": asdict(authorization),
            "authorization_id": authorization.authorization_id,
            "journal_record": asdict(result.journal_record),
            "health_passed": result.health.passed,
            "health_returncode": result.health.returncode,
            "health_output_sha256": sha256(
                result.health.output.encode("utf-8")
            ).hexdigest(),
            "health_command_sha256": command_sha256(
                self.service.config.health_check_command
            ),
            "source_final_sha256": self.service.active_source_surface_sha256,
            "target_live_sha256": sha256(live).hexdigest(),
            "byte_identical_restore": byte_identical,
        }
        event = SelfHarnessE2EEventV1.create(
            run_id=run_id,
            sequence=3,
            stage=stage,
            previous_event_sha256=chain[-1].event_sha256,
            created_ns=time_ns(),
            payload=payload,
        )
        self.ledger.append(event)
        return event


def validate_self_harness_e2e(
    ledger: SelfHarnessE2ELedger,
    run_id: str,
    approval_verifier: Ed25519ApprovalVerifier,
) -> SelfHarnessE2EValidationResult:
    """Validate one persisted chain, including both external signatures."""

    if not isinstance(approval_verifier, Ed25519ApprovalVerifier):
        raise TypeError("approval_verifier must be Ed25519ApprovalVerifier")
    chain = ledger.events(run_id)
    _validate_chain(chain)
    first = chain[0]
    evaluation = CandidateEvaluationV1.from_mapping(first.payload["evaluation"])
    expected_run_id = sha256(first.payload["run_nonce"].encode("ascii")).hexdigest()[
        :32
    ]
    if first.run_id != expected_run_id:
        raise SelfHarnessE2EError("E2E run id is not nonce-bound")
    first_bindings = {
        "source_before_sha256": evaluation.source_surface_sha256,
        "target_before_sha256": evaluation.base_sha256,
        "runner_attestation_sha256": evaluation.runner_evidence_sha256,
        "regression_result_sha256": evaluation.result_sha256,
    }
    for field, expected in first_bindings.items():
        if first.payload[field] != expected:
            raise SelfHarnessE2EError(f"E2E {field} binding is invalid")
    if evaluation.returncode != 0:
        raise SelfHarnessE2EError("E2E evaluation is not a zero-exit pass")
    if not evaluation.completed_ns <= first.created_ns < evaluation.expires_ns:
        raise SelfHarnessE2EError("E2E evaluation timeline is invalid")
    if evaluation.proposal_id not in first.payload["candidate_proposal_ids"]:
        raise SelfHarnessE2EError("E2E selected proposal is absent from candidates")
    if (
        evaluation.replacement_sha256
        not in first.payload["candidate_replacement_sha256"]
    ):
        raise SelfHarnessE2EError("E2E selected replacement is absent from candidates")
    terminal = chain[-1].stage
    if len(chain) >= 2:
        promotion = chain[1]
        if not (
            first.created_ns
            <= promotion.payload["operation_started_ns"]
            <= promotion.created_ns
        ):
            raise SelfHarnessE2EError("E2E promotion timeline is invalid")
        approval = HumanApprovalV1.from_mapping(promotion.payload["approval"])
        approval_id = approval_verifier.verify(
            approval,
            evaluation,
            now_ns=promotion.payload["operation_started_ns"],
        )
        if approval_id != promotion.payload["approval_id"]:
            raise SelfHarnessE2EError("E2E approval identity is invalid")
        promotion_record = BackupRecord(**dict(promotion.payload["journal_record"]))
        _validate_backup_record(promotion_record)
        if (
            promotion_record.relative_path != evaluation.relative_path
            or promotion_record.created_ns < promotion.payload["operation_started_ns"]
            or promotion_record.created_ns > promotion.created_ns
            or promotion_record.before_sha256 != evaluation.base_sha256
            or promotion_record.after_sha256 != evaluation.replacement_sha256
        ):
            raise SelfHarnessE2EError("E2E promotion journal is not evaluation-bound")
        if promotion.payload["health_command_sha256"] == _ZERO_DIGEST:
            raise SelfHarnessE2EError("E2E health command binding is invalid")
        if promotion.stage == "promotion_committed" and (
            promotion.payload["health_passed"] is not True
            or promotion.payload["health_returncode"] != 0
            or promotion.payload["byte_restore_verified"] is not False
            or promotion.payload["target_live_sha256"] != promotion_record.after_sha256
            or promotion_record.status != "committed"
        ):
            raise SelfHarnessE2EError("E2E committed promotion evidence is invalid")
        if promotion.stage == "promotion_health_restore":
            if (
                promotion.payload["health_passed"] is not False
                or promotion.payload["health_returncode"] == 0
                or promotion.payload["byte_restore_verified"] is not True
                or promotion.payload["target_live_sha256"]
                != promotion_record.before_sha256
                or promotion.payload["source_after_sha256"]
                != first.payload["source_before_sha256"]
                or promotion_record.status != "rolled_back"
            ):
                raise SelfHarnessE2EError(
                    "E2E promotion health-failure restore is invalid"
                )
    if len(chain) == 3:
        rollback = chain[2]
        if not (
            chain[1].created_ns
            <= rollback.payload["operation_started_ns"]
            <= rollback.created_ns
        ):
            raise SelfHarnessE2EError("E2E rollback timeline is invalid")
        authorization = RollbackAuthorizationV1.from_mapping(
            rollback.payload["authorization"]
        )
        promotion_record = BackupRecord(**dict(chain[1].payload["journal_record"]))
        rollback_record = BackupRecord(**dict(rollback.payload["journal_record"]))
        _validate_backup_record(rollback_record)
        for field in (
            "record_id",
            "relative_path",
            "before_sha256",
            "after_sha256",
            "backup_file",
            "file_mode",
            "created_ns",
        ):
            if getattr(rollback_record, field) != getattr(promotion_record, field):
                raise SelfHarnessE2EError(
                    f"E2E rollback journal changed immutable {field}"
                )
        authorization_id = approval_verifier.verify_rollback(
            authorization,
            journal_record_id=promotion_record.record_id,
            relative_path=promotion_record.relative_path,
            before_sha256=promotion_record.before_sha256,
            after_sha256=promotion_record.after_sha256,
            source_surface_sha256=chain[1].payload["source_after_sha256"],
            runner_id=evaluation.runner_id,
            runner_evidence_sha256=evaluation.runner_evidence_sha256,
            health_command_sha256=chain[1].payload["health_command_sha256"],
            record_created_ns=promotion_record.created_ns,
            now_ns=rollback.payload["operation_started_ns"],
        )
        if authorization_id != rollback.payload["authorization_id"]:
            raise SelfHarnessE2EError("E2E rollback authority identity is invalid")
        if (
            rollback.payload["health_command_sha256"]
            != chain[1].payload["health_command_sha256"]
        ):
            raise SelfHarnessE2EError("E2E rollback health command changed")
        expected = (
            promotion_record.before_sha256
            if rollback.stage == "rollback_completed"
            else promotion_record.after_sha256
        )
        if (
            rollback.payload["target_live_sha256"] != expected
            or rollback.payload["byte_identical_restore"] is not True
        ):
            raise SelfHarnessE2EError("E2E rollback byte restoration is invalid")
        if rollback.stage == "rollback_completed" and (
            rollback.payload["health_passed"] is not True
            or rollback.payload["health_returncode"] != 0
            or rollback.payload["source_final_sha256"]
            != first.payload["source_before_sha256"]
            or rollback_record.status != "operator_rolled_back"
        ):
            raise SelfHarnessE2EError("E2E completed rollback is not source-identical")
        if rollback.stage == "rollback_health_restore" and (
            rollback.payload["health_passed"] is not False
            or rollback.payload["health_returncode"] == 0
            or rollback.payload["source_final_sha256"]
            != chain[1].payload["source_after_sha256"]
            or rollback_record.status != "committed"
        ):
            raise SelfHarnessE2EError("E2E rejected rollback restore is invalid")
    return SelfHarnessE2EValidationResult(
        schema=E2E_VALIDATION_SCHEMA,
        run_id=run_id,
        event_count=len(chain),
        terminal_stage=terminal,
        chain_sha256=chain[-1].event_sha256,
        full_e2e_complete=(terminal == "rollback_completed"),
        runner_attestation_digest_bound=True,
        production_attestation_reverified=False,
        assurance="signed_operator_evidence_chain_validation_only",
    )


def _validate_chain(events: tuple[SelfHarnessE2EEventV1, ...]) -> None:
    if not events:
        raise SelfHarnessE2EError("operator E2E run has no evidence")
    if len(events) > 3 or len({item.run_id for item in events}) != 1:
        raise SelfHarnessE2EError("operator E2E chain shape is invalid")
    for index, event in enumerate(events, start=1):
        if event.sequence != index:
            raise SelfHarnessE2EError("operator E2E chain sequence is invalid")
        expected = _ZERO_DIGEST if index == 1 else events[index - 2].event_sha256
        if event.previous_event_sha256 != expected:
            raise SelfHarnessE2EError("operator E2E previous digest is invalid")
        if index > 1 and event.created_ns < events[index - 2].created_ns:
            raise SelfHarnessE2EError("operator E2E event time moved backwards")
    if len(events) == 3 and events[1].stage != "promotion_committed":
        raise SelfHarnessE2EError("terminal rollback follows a non-commit stage")


def _validate_stage_payload(stage: str, payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "evaluation_ready": {
            "run_nonce",
            "source_before_sha256",
            "target_before_sha256",
            "failure_event_ids",
            "candidate_proposal_ids",
            "candidate_replacement_sha256",
            "candidate_count",
            "failure_evidence_root_sha256",
            "candidate_evidence_root_sha256",
            "evaluation",
            "runner_attestation_sha256",
            "regression_result_sha256",
        },
        "promotion_committed": {
            "operation_started_ns",
            "approval",
            "approval_id",
            "journal_record",
            "health_passed",
            "health_returncode",
            "health_output_sha256",
            "health_command_sha256",
            "source_after_sha256",
            "target_live_sha256",
            "byte_restore_verified",
        },
        "promotion_health_restore": {
            "operation_started_ns",
            "approval",
            "approval_id",
            "journal_record",
            "health_passed",
            "health_returncode",
            "health_output_sha256",
            "health_command_sha256",
            "source_after_sha256",
            "target_live_sha256",
            "byte_restore_verified",
        },
        "rollback_completed": {
            "operation_started_ns",
            "authorization",
            "authorization_id",
            "journal_record",
            "health_passed",
            "health_returncode",
            "health_output_sha256",
            "health_command_sha256",
            "source_final_sha256",
            "target_live_sha256",
            "byte_identical_restore",
        },
        "rollback_health_restore": {
            "operation_started_ns",
            "authorization",
            "authorization_id",
            "journal_record",
            "health_passed",
            "health_returncode",
            "health_output_sha256",
            "health_command_sha256",
            "source_final_sha256",
            "target_live_sha256",
            "byte_identical_restore",
        },
    }[stage]
    if set(payload) != expected_keys:
        raise SelfHarnessE2EError(f"{stage} payload fields are invalid")
    digest_fields = (
        ("source_before_sha256",),
        ("target_before_sha256",),
        ("runner_attestation_sha256",),
        ("regression_result_sha256",),
        ("failure_evidence_root_sha256",),
        ("candidate_evidence_root_sha256",),
        (
            "approval_id",
            "health_output_sha256",
            "health_command_sha256",
            "source_after_sha256",
            "target_live_sha256",
        ),
        (
            "authorization_id",
            "health_output_sha256",
            "health_command_sha256",
            "source_final_sha256",
            "target_live_sha256",
        ),
    )
    keys = set(payload)
    for group in digest_fields:
        for key in group:
            if key in keys:
                _digest(payload[key], key)
    if stage == "evaluation_ready":
        if (
            not isinstance(payload["run_nonce"], str)
            or _NONCE.fullmatch(payload["run_nonce"]) is None
        ):
            raise SelfHarnessE2EError("operator E2E run nonce is invalid")
        for key, minimum in (
            ("failure_event_ids", 1),
            ("candidate_proposal_ids", 3),
            ("candidate_replacement_sha256", 3),
        ):
            values = payload[key]
            if (
                not isinstance(values, list)
                or len(values) < minimum
                or values != sorted(set(values))
            ):
                raise SelfHarnessE2EError(f"operator E2E {key} is invalid")
            for value in values:
                _digest(value, key)
        if (
            not isinstance(payload["candidate_count"], int)
            or isinstance(payload["candidate_count"], bool)
            or payload["candidate_count"] < 3
            or payload["candidate_count"] != len(payload["candidate_proposal_ids"])
        ):
            raise SelfHarnessE2EError("operator E2E candidate count is invalid")
        CandidateEvaluationV1.from_mapping(payload["evaluation"])
    else:
        for key in ("operation_started_ns", "health_returncode"):
            if not isinstance(payload[key], int) or isinstance(payload[key], bool):
                raise SelfHarnessE2EError(f"operator E2E {key} is invalid")
        if not isinstance(payload["health_passed"], bool):
            raise SelfHarnessE2EError("operator E2E health result is invalid")
        BackupRecord(**dict(payload["journal_record"]))
        if stage.startswith("promotion_"):
            HumanApprovalV1.from_mapping(payload["approval"])
            if not isinstance(payload["byte_restore_verified"], bool):
                raise SelfHarnessE2EError("operator promotion restore flag is invalid")
        else:
            RollbackAuthorizationV1.from_mapping(payload["authorization"])
            if payload["byte_identical_restore"] is not True:
                raise SelfHarnessE2EError("operator rollback byte flag is invalid")


def _strict_json(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelfHarnessE2EError("operator E2E event repeats a JSON key")
            result[key] = value
        return result

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelfHarnessE2EError("operator E2E event is not strict JSON") from exc
    if not isinstance(data, dict):
        raise SelfHarnessE2EError("operator E2E event must be a JSON object")
    return data


def _read_bounded(path: Path) -> bytes:
    try:
        item_stat = os.lstat(path)
        attributes = getattr(item_stat, "st_file_attributes", 0)
        if (
            path.is_symlink()
            or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
            or not stat.S_ISREG(item_stat.st_mode)
            or item_stat.st_size > _MAX_EVENT_BYTES
            or path.resolve(strict=True) != path.absolute()
        ):
            raise SelfHarnessE2EError(
                "operator E2E event must be bounded regular non-link data"
            )
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(_MAX_EVENT_BYTES + 1)
            final = os.fstat(stream.fileno())
    except SelfHarnessE2EError:
        raise
    except OSError as exc:
        raise SelfHarnessE2EError("operator E2E event is unavailable") from exc
    identity_changed = bool(item_stat.st_ino and opened.st_ino) and (
        (item_stat.st_dev, item_stat.st_ino) != (opened.st_dev, opened.st_ino)
    )
    if (
        identity_changed
        or not stat.S_ISREG(opened.st_mode)
        or len(raw) > _MAX_EVENT_BYTES
        or len(raw) != item_stat.st_size
        or (opened.st_size, opened.st_mtime_ns) != (final.st_size, final.st_mtime_ns)
    ):
        raise SelfHarnessE2EError("operator E2E event changed during read")
    return raw


def _require_directory(path: Path) -> None:
    try:
        item_stat = os.lstat(path)
    except OSError as exc:
        raise SelfHarnessE2EError("operator E2E ledger is unavailable") from exc
    attributes = getattr(item_stat, "st_file_attributes", 0)
    if (
        path.is_symlink()
        or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
        or not stat.S_ISDIR(item_stat.st_mode)
        or path.resolve(strict=True) != path.absolute()
    ):
        raise SelfHarnessE2EError(
            "operator E2E ledger must be a regular non-link directory"
        )


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except FileExistsError as exc:
        raise SelfHarnessE2EReplayError("operator E2E event already exists") from exc
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise SelfHarnessE2EError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_backup_record(record: BackupRecord) -> None:
    if (
        not isinstance(record.record_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", record.record_id) is None
        or record.backup_file != f"{record.record_id}.bak"
    ):
        raise SelfHarnessE2EError("E2E journal record identity is invalid")
    _digest(record.before_sha256, "journal before")
    _digest(record.after_sha256, "journal after")
    if (
        not isinstance(record.file_mode, int)
        or isinstance(record.file_mode, bool)
        or not 0 <= record.file_mode <= 0o777
    ):
        raise SelfHarnessE2EError("E2E journal file mode is invalid")
    if (
        not isinstance(record.created_ns, int)
        or isinstance(record.created_ns, bool)
        or record.created_ns < 1
    ):
        raise SelfHarnessE2EError("E2E journal creation time is invalid")
    if record.status not in {"committed", "rolled_back", "operator_rolled_back"}:
        raise SelfHarnessE2EError("E2E journal status is invalid")


def _record_root(records: tuple[object, ...]) -> str:
    digest = sha256()
    encoded = sorted(canonical_json_bytes(asdict(item)) for item in records)
    for payload in encoded:
        digest.update(len(payload).to_bytes(4, "big"))
        digest.update(sha256(payload).digest())
    return digest.hexdigest()


__all__ = [
    "E2E_EVENT_SCHEMA",
    "E2E_VALIDATION_SCHEMA",
    "OperatorSelfHarnessE2E",
    "SelfHarnessE2EError",
    "SelfHarnessE2EEventV1",
    "SelfHarnessE2ELedger",
    "SelfHarnessE2EReplayError",
    "SelfHarnessE2EValidationResult",
    "validate_self_harness_e2e",
]
