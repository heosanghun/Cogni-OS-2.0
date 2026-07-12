"""Deterministic, bounded answers for Cogni-OS product facts.

Natural-language generation is useful for open-ended conversation, but it is
the wrong authority for the runtime's own identity and capability boundary.
This module recognizes only a small, explicit set of product-fact questions
and builds their answers from :class:`~cogni_os.factbook.RuntimeFactBook`.
Questions outside that set return ``None`` and continue through the normal
local Gemma conversation path.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from cogni_os.capabilities import CapabilityRecord, CapabilityState
from cogni_os.factbook import RuntimeFactBook


MAX_FACT_QUESTION_CHARS = 4_096
MAX_FACT_ANSWER_CHARS = 8_192


class FactGroundingError(ValueError):
    """Raised when a product-fact request exceeds a declared bound."""


@dataclass(frozen=True, slots=True)
class FactGroundingLimits:
    """Hard-bounded input and output sizes for deterministic fact answers."""

    question_chars: int = MAX_FACT_QUESTION_CHARS
    answer_chars: int = MAX_FACT_ANSWER_CHARS

    def __post_init__(self) -> None:
        for name, value, hard_limit in (
            ("question_chars", self.question_chars, MAX_FACT_QUESTION_CHARS),
            ("answer_chars", self.answer_chars, MAX_FACT_ANSWER_CHARS),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if not 1 <= value <= hard_limit:
                raise ValueError(f"{name} must be between 1 and {hard_limit}")


_STATE_KO = {
    CapabilityState.DISABLED: "비활성",
    CapabilityState.RESEARCH: "연구",
    CapabilityState.ADVISORY: "자문 전용",
    CapabilityState.CANARY: "제한 실험",
    CapabilityState.AUTHORITATIVE: "답변 권한 보유",
    CapabilityState.GATED: "게이트 대기",
    CapabilityState.NIGHT_ONLY: "야간 전용",
    CapabilityState.PROPOSAL_ONLY: "제안 전용",
}

_EVIDENCE_KO = {
    "measured": "실측",
    "verified": "구성 검증",
    "target": "목표",
    "plan": "계획",
}

_TOPIC_ORDER = (
    "identity",
    "parameters",
    "cts_deq",
    "system_1_5",
    "system_2_5",
    "system_3",
    "system_4",
    "self_harness",
)

_ALIASES = {
    "parameters": (
        "파라미터",
        "매개변수",
        "parameter count",
        "parameters",
        "model size",
        "모델 크기",
    ),
    "cts_deq": (
        "cts",
        "cognitive tree search",
        "인지 트리 탐색",
        "인지 탐색",
        "deq",
        "deep equilibrium",
        "broyden",
        "브로이든",
        "평형 탐색",
    ),
    "system_1_5": (
        "system 1.5",
        "system1.5",
        "sys 1.5",
        "sys1.5",
        "시스템 1.5",
        "시스템1.5",
        "fast weight",
        "fast-weight",
        "fastweight",
        "패스트 웨이트",
        "직관 컴파일러",
    ),
    "system_2_5": (
        "system 2.5",
        "system2.5",
        "sys 2.5",
        "sys2.5",
        "시스템 2.5",
        "시스템2.5",
        "fp-ewc",
        "fp ewc",
        "fpewc",
        "c-fire",
        "c fire",
        "cfire",
        "spectral safety",
        "스펙트럼 안전",
    ),
    "system_3": (
        "system 3",
        "system3",
        "sys 3",
        "sys3",
        "시스템 3",
        "시스템3",
        "sparse expert",
        "sparse moe",
        "희소 전문가",
    ),
    "system_4": (
        "system 4",
        "system4",
        "sys 4",
        "sys4",
        "시스템 4",
        "시스템4",
        "tensor swarm",
        "텐서 스웜",
        "텐서 swarm",
    ),
    "self_harness": (
        "self-harness",
        "self harness",
        "selfharness",
        "셀프 하니스",
        "자가 거울치료",
        "자기 거울치료",
        "자가 코드 수정",
        "스스로 코드 수정",
        "자동 패치",
        "autonomous patch",
    ),
}

_IDENTITY_ALIASES = (
    "너는 누구",
    "당신은 누구",
    "정체성이",
    "정체성은",
    "자기소개",
    "무슨 모델",
    "어떤 모델",
    "who are you",
    "what model are you",
    "cogni agent가 뭐",
    "cogni agent는 뭐",
)

_OVERVIEW_ALIASES = (
    "모든 기능",
    "전체 기능",
    "기능을 모두",
    "기능 모두",
    "어떤 기능",
    "모듈을 모두",
    "모든 모듈",
    "전체 모듈",
    "뭘 할 수",
    "무엇을 할 수",
    "all capabilities",
    "all features",
)

_FOLLOWUP_REFERENCE_ALIASES = (
    "방금 답변",
    "앞선 답변",
    "이전 답변",
    "위 답변",
)
_EVIDENCE_BOUNDARY_ALIASES = (
    "실제 검증",
    "검증된 사실",
    "향후 목표",
    "설계 목표",
    "검증과 목표",
)

_EXECUTION_RESULT_ALIASES = (
    "도구 실행 결과",
    "명령 실행 결과",
    "작업 실행 결과",
    "tool result",
    "execution result",
)
_UNVERIFIED_ALIASES = (
    "확인하지 못",
    "확인할 수 없",
    "검증하지 못",
    "검증할 수 없",
    "unverified",
    "not verified",
)
_SUCCESS_ASSERTION_ALIASES = (
    "성공",
    "완료",
    "succeed",
    "success",
    "complete",
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _contains_any(text: str, aliases: tuple[str, ...]) -> bool:
    return any(_normalize(alias) in text for alias in aliases)


def _capability_status(record: CapabilityRecord) -> str:
    state = _STATE_KO[record.state]
    evidence = _EVIDENCE_KO[record.evidence.value]
    answer_authority = "있음" if record.answer_bearing else "없음"
    mutation = "허용" if record.runtime_mutation_allowed else "차단"
    return (
        f"Fact-book: state={record.state.value}({state}), "
        f"evidence={record.evidence.value}({evidence}), "
        f"answer_bearing={answer_authority}, runtime_mutation_allowed={mutation}"
    )


class RuntimeFactGrounder:
    """Answer only recognized runtime-fact questions from a signed Fact-book."""

    def __init__(
        self,
        factbook: RuntimeFactBook,
        limits: FactGroundingLimits | None = None,
    ) -> None:
        if not isinstance(factbook, RuntimeFactBook):
            raise TypeError("factbook must be a RuntimeFactBook")
        if limits is not None and not isinstance(limits, FactGroundingLimits):
            raise TypeError("limits must be FactGroundingLimits or None")
        self._factbook = factbook
        self._limits = limits or FactGroundingLimits()

    @property
    def factbook(self) -> RuntimeFactBook:
        return self._factbook

    @property
    def limits(self) -> FactGroundingLimits:
        return self._limits

    def answer(self, question: str) -> str | None:
        """Return one grounded answer, or ``None`` for ordinary conversation."""

        if not isinstance(question, str):
            raise TypeError("question must be text")
        if len(question) > self._limits.question_chars:
            raise FactGroundingError("fact question exceeds its character bound")
        text = _normalize(question)
        if len(text) > self._limits.question_chars:
            raise FactGroundingError("normalized fact question exceeds its bound")
        if not text:
            return None

        if (
            _contains_any(text, _EXECUTION_RESULT_ALIASES)
            and _contains_any(text, _UNVERIFIED_ALIASES)
            and _contains_any(text, _SUCCESS_ASSERTION_ALIASES)
        ):
            answer = (
                "실행 결과를 확인하지 못하면 성공을 뒷받침할 증거가 없으므로 "
                "완료로 단정해서는 안 됩니다. 미확인 성공 표시는 사용자의 후속 "
                "판단과 복구를 잘못 이끌 수 있으므로 검증 상태를 그대로 밝혀야 합니다."
            )
            if len(answer) > self._limits.answer_chars:
                raise FactGroundingError(
                    "grounded execution policy exceeds its character bound"
                )
            return answer

        topics = self._classify(text)
        if not topics:
            return None
        answer = "\n\n".join(
            self._identity_answer(include_parameters=False)
            if topic == "identity" and "parameters" in topics
            else self._answer_topic(topic)
            for topic in topics
        )
        if "이상입니다" in text and not answer.rstrip().endswith("이상입니다."):
            answer += "\n\n이상입니다."
        if len(answer) > self._limits.answer_chars:
            raise FactGroundingError("grounded fact answer exceeds its character bound")
        return answer

    def answer_followup(self, question: str, previous_answer: str) -> str | None:
        """Ground one narrow evidence-vs-target follow-up from prior Fact-book text."""

        if not isinstance(question, str) or not isinstance(previous_answer, str):
            raise TypeError("question and previous_answer must be text")
        if len(question) > self._limits.question_chars:
            raise FactGroundingError("fact follow-up exceeds its character bound")
        if len(previous_answer) > self._limits.answer_chars:
            return None
        text = _normalize(question)
        if not _contains_any(text, _FOLLOWUP_REFERENCE_ALIASES):
            return None
        if not _contains_any(text, _EVIDENCE_BOUNDARY_ALIASES):
            return None
        if "Fact-book:" not in previous_answer:
            return None
        return (
            "실제 검증은 Runtime Fact-book에서 measured 또는 verified로 표시되고 "
            "현재 실행에서 확인된 범위만 완료된 사실로 답합니다. "
            "향후 목표는 target·plan·research·canary 같은 상태로 분리해 표시하며 "
            "검증이 끝나기 전에는 현재 성과로 단정하지 않습니다."
        )

    def _classify(self, text: str) -> tuple[str, ...]:
        if _contains_any(text, _OVERVIEW_ALIASES):
            return _TOPIC_ORDER

        topics = {
            topic for topic, aliases in _ALIASES.items() if _contains_any(text, aliases)
        }
        if _contains_any(text, _IDENTITY_ALIASES):
            topics.add("identity")

        # An explicit model name plus a size/effective-count question is both
        # an identity and parameter request. The parameter aliases above keep
        # unrelated numerical questions on the normal conversation path.
        if "parameters" in topics and any(
            token in text for token in ("gemma", "e4b", "백본", "모델", "너", "당신")
        ):
            topics.add("identity")
        return tuple(topic for topic in _TOPIC_ORDER if topic in topics)

    def _answer_topic(self, topic: str) -> str:
        if topic == "identity":
            return self._identity_answer()
        if topic == "parameters":
            return self._parameter_answer()
        if topic == "cts_deq":
            return self._capability_answer(
                "CTS · DEQ",
                "cts_deq",
                "Gemma의 causal hidden을 고정 용량 CTS가 탐색하고 DEQ 평형점을 계산한 뒤, 검증된 상한의 logits bias로 생성에 제한적으로 반영하며 기본 Gemma 가중치는 변경하지 않습니다.",
            )
        if topic == "system_1_5":
            return self._capability_answer(
                "System 1.5 · Fast Weight",
                "system_1_5",
                "수렴한 상태를 세션 한정 저랭크 Fast Weight 후보로 압축하는 직관 컴파일 계층입니다.",
            )
        if topic == "system_2_5":
            return self._capability_answer(
                "System 2.5 · FP-EWC/C-FIRE",
                "system_2_5",
                "FP-EWC와 스펙트럼 안전 투영으로 진화 단계의 가중치 변경을 안정화하는 계층입니다.",
            )
        if topic == "system_3":
            return self._capability_answer(
                "System 3 · Sparse Experts",
                "system_3",
                "고정된 자원 경계 안에서 희소 전문가 상태를 선택하고 관측하는 계층입니다.",
            )
        if topic == "system_4":
            return self._capability_answer(
                "System 4 · Tensor Swarm",
                "system_4",
                "모듈의 잠재 상태를 자연어 재파싱 없이 텐서 결합으로 모으는 관측 계층입니다.",
            )
        if topic == "self_harness":
            return self._self_harness_answer()
        raise AssertionError(f"unhandled fact topic: {topic}")

    def _identity_answer(self, *, include_parameters: bool = True) -> str:
        facts = self._factbook
        if include_parameters:
            identity = facts.identity_summary_ko()
        else:
            structure = "dense" if facts.model.dense else "expert/MoE"
            identity = (
                "저는 Cogni-OS 2.0에서 실행되는 Cogni Agent입니다. "
                f"로컬 백본은 {structure} {facts.model.label}입니다."
            )
        return (
            f"{identity} 현재 빌드는 {facts.build_version}, "
            f"실행 장치는 {facts.device}, 목표 장치는 {facts.target_device}입니다."
        )

    def _parameter_answer(self) -> str:
        model = self._factbook.model
        inventory = model.inventory
        structure = "dense" if model.dense else "expert/MoE"
        counts = f"저장 파라미터 {inventory.stored_parameters:,}개"
        if inventory.effective_parameters is not None:
            counts += f", effective 파라미터 {inventory.effective_parameters:,}개"
        if inventory.embedding_parameters:
            counts += f", 임베딩 파라미터 {inventory.embedding_parameters:,}개"
        return (
            f"모델 파라미터 — {model.label}은 {structure} 구조의 "
            f"{model.architecture}입니다. {counts}이며, hidden size는 "
            f"{model.hidden_size:,}, 레이어는 {model.layers:,}개입니다. "
            "수치는 검증된 로컬 가중치 헤더와 config에서 계산된 Runtime Fact-book 값입니다."
        )

    def _capability_answer(
        self,
        title: str,
        capability_name: str,
        description: str,
    ) -> str:
        record = self._factbook.capabilities.require(capability_name)
        return f"{title} — {description.rstrip('.')} ({_capability_status(record)})."

    def _self_harness_answer(self) -> str:
        record = self._factbook.capabilities.require("self_harness")
        if record.runtime_mutation_allowed:
            mutation = "이 Fact-book에서는 실행 중 변경 권한이 허용되어 있습니다"
        else:
            mutation = "현재 활성 소스를 스스로 덮어쓰는 권한은 없습니다"
        if record.state is CapabilityState.PROPOSAL_ONLY:
            boundary = "실패를 수집해 패치 후보를 만들 수 있지만, 격리 검증과 운영자 신뢰 경계 밖에서는 제안으로만 남습니다."
        else:
            boundary = "실제 동작 범위는 아래 Fact-book 상태와 권한 플래그를 따릅니다."
        return (
            "Self-Harness · 자가 거울치료 — Cogni-OS 문맥의 ‘자가 거울치료’는 "
            "의학적 심리치료가 아니라 실패를 되짚어 코드 패치 후보를 만들고 검증·승격·롤백을 통제하는 기능입니다. "
            f"{boundary.rstrip('.')}; {mutation} ({_capability_status(record)})."
        )


def ground_runtime_fact(
    question: str,
    factbook: RuntimeFactBook,
    *,
    limits: FactGroundingLimits | None = None,
) -> str | None:
    """Convenience API for one deterministic runtime-fact lookup."""

    return RuntimeFactGrounder(factbook, limits).answer(question)


__all__ = [
    "FactGroundingError",
    "FactGroundingLimits",
    "MAX_FACT_ANSWER_CHARS",
    "MAX_FACT_QUESTION_CHARS",
    "RuntimeFactGrounder",
    "ground_runtime_fact",
]
