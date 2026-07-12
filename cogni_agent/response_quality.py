"""Bounded, deterministic quality checks for locally generated responses.

The token guard in :mod:`cogni_agent.model_service` intentionally catches only
exact token cycles.  This module operates at the decoded-text boundary and
covers failure modes that cannot be detected safely from token equality alone:

* generated chat roles and reserved model-control tokens;
* sentence templates that change only ordinals, numbers, or modifiers;
* short, low-information phrase cycles; and
* conservative Korean clause-fragment endings on a final response.

The implementation is stateless, performs no model or network calls, and has
hard work bounds.  Long inputs are inspected through fixed-size prefix and
suffix regions so both an initial role header and a trailing degeneration loop
remain visible without allowing caller-controlled memory growth.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
import re
import unicodedata


MAX_INSPECT_CHARS = 8_192
MAX_ANALYSIS_UNITS = 64
MAX_UNIT_TOKENS = 64
MAX_ANALYSIS_TOKENS = 1_024
MAX_TEMPLATE_BLOCK_UNITS = 4
MAX_LOW_INFORMATION_PERIOD = 4


class QualityCode(str, Enum):
    """Stable machine-readable response-quality finding codes."""

    ROLE_MARKER = "role_marker"
    CONTROL_TOKEN = "control_token"
    TEMPLATE_REPETITION = "template_repetition"
    LOW_INFORMATION_REPETITION = "low_information_repetition"
    INCOMPLETE_KOREAN_CLAUSE = "incomplete_korean_clause"


class QualityAction(str, Enum):
    """Recommended action at the generation boundary."""

    ACCEPT = "accept"
    TRIM_AND_STOP = "trim_and_stop"
    CONTINUE = "continue"


class ResponseQualityError(RuntimeError):
    """Raised when no complete, non-degenerate answer can be published."""


_STOP_CODES = frozenset(
    {
        QualityCode.ROLE_MARKER,
        QualityCode.CONTROL_TOKEN,
        QualityCode.TEMPLATE_REPETITION,
        QualityCode.LOW_INFORMATION_REPETITION,
    }
)


@dataclass(frozen=True, slots=True)
class QualityFinding:
    """One bounded finding using offsets into the caller's original text."""

    code: QualityCode
    start: int
    end: int
    occurrences: int = 1

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("quality finding offsets must be ordered and non-negative")
        if self.occurrences < 1:
            raise ValueError("quality finding occurrences must be positive")


@dataclass(frozen=True, slots=True)
class ResponseQualityReport:
    """Immutable result of one response inspection."""

    findings: tuple[QualityFinding, ...]
    input_characters: int
    inspected_characters: int
    input_truncated: bool

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def should_stop_generation(self) -> bool:
        return any(finding.code in _STOP_CODES for finding in self.findings)

    @property
    def needs_continuation(self) -> bool:
        return not self.should_stop_generation and any(
            finding.code is QualityCode.INCOMPLETE_KOREAN_CLAUSE
            for finding in self.findings
        )

    @property
    def recommended_action(self) -> QualityAction:
        if self.should_stop_generation:
            return QualityAction.TRIM_AND_STOP
        if self.needs_continuation:
            return QualityAction.CONTINUE
        return QualityAction.ACCEPT

    @property
    def recommended_cut_index(self) -> int | None:
        """Return the earliest unsafe/repeated boundary, preserving one copy."""

        boundaries = [
            finding.start for finding in self.findings if finding.code in _STOP_CODES
        ]
        return min(boundaries) if boundaries else None

    def has(self, code: QualityCode) -> bool:
        return any(finding.code is code for finding in self.findings)


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    raw_key: tuple[str, ...]
    template_key: tuple[str, ...]
    has_variable: bool
    visible_characters: int


@dataclass(frozen=True, slots=True)
class _Token:
    value: str
    start: int
    end: int


_ROLE_MARKER_RE = re.compile(
    r"(?i)(?<![\w가-힣])(?:user|assistant|model|system|tool|사용자|"
    r"어시스턴트|시스템)\s*:\s*"
)
_CONTROL_TOKEN_RE = re.compile(
    r"(?i)<\|[^|<>\r\n]{1,64}\|>|"
    r"</?(?:start_of_turn|end_of_turn|turn|bos|eos|pad|unk)>|"
    r"<unused\d{1,5}>|\[(?:/?inst|system|/?sys|multimodal)\]|"
    r"<<\s*/?sys\s*>>|\[Runtime Fact-book:|\[턴\s*종료\]|"
    r"</?(?:시스템|컨펌|종료|summary)(?:\s+[^<>\r\n]{0,96})?>|\[##\]"
)
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)")
_UNIT_RE = re.compile(r".+?(?:[.!?。！？]+(?=\s|$)|\n+|$)", re.DOTALL)
_LEXEME_RE = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d{1,4}[.)]\s+)")
_LIST_MARKER_ONLY_RE = re.compile(r"^\s*(?:[-*+]|\d{1,4}[.)])\s*$")
_REQUEST_COUNT_RE = re.compile(
    r"(?P<count>[1-9]|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
    r"(?P<unit>문장|단계|항목)"
)
_MAX_ITEM_REQUEST_RE = re.compile(
    r"(?P<count>[1-9]|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
    r"(?:개|가지|항목)\s*이내"
)
_EXACT_ITEM_COUNT_RE = re.compile(
    r"(?P<count>[1-9]|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
    r"(?P<unit>단계|항목|가지|개)"
)
_EXACT_ITEM_OUTPUT_CUE_RE = re.compile(
    r"^\s*(?:(?:만|을|를|로|으로|에|은|는)\s*)?"
    r"(?:(?:각각|간결하게|간단히|자세히|명확하게|짧게|자연스럽게|"
    r"순서대로|구체적으로)\s*)*"
    r"(?:답|설명|정리|제시|작성|나열|열거|구분|요약|알려|말해|꼽|선정|"
    r"적어|기술|제안|무엇|인가|입니까|일까)"
)
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？]+(?=\s|$)")
_COMPLETE_SENTENCE_RE = re.compile(r".+?[.!?。！？]+(?=\s|$)", re.DOTALL)
_ALL_LIST_PREFIX_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d{1,4}[.)]\s+)")
_INLINE_NUMBERED_CLAUSE_RE = re.compile(
    r"(?<!\S)(?P<number>\d{1,2})(?:[.)]\s+|"
    r"단계(?:는|로|:)?\s*|항목(?:은|으로|:)?\s*)"
)
_NEGATIVE_SECTION_RE = re.compile(
    r"(?im)(?:한계|단점|제약|위험)(?:(?:로)?는\s*|[:：]\s*)"
)
_CATEGORY_COUNT_RE = re.compile(
    r"(?P<category>장점|이점|강점|한계|단점|제약|위험)\s*"
    r"(?P<count>[1-9]|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*가지"
)
_NEGATIVE_CATEGORY_CUES = (
    "한계",
    "단점",
    "제약",
    "제한",
    "부족",
    "위험",
    "어렵",
    "떨어",
    "소모",
    "의존",
)
_META_SENTENCE_ENDINGS = (
    "설명하겠습니다.",
    "답변하겠습니다.",
    "정리하겠습니다.",
    "알려드리겠습니다.",
    "살펴보겠습니다.",
    "다음과 같습니다.",
    "다음과 같아요.",
)
_CLOSING_SENTENCES = frozenset({"이상입니다.", "답변을 마칩니다.", "설명을 마칩니다."})
_KOREAN_COORDINATE_SPLIT_RE = re.compile(
    r"^(?P<head>.{8,}?)(?P<ending>하고|하며|이고|이며),?\s+"
    r"(?P<tail>[가-힣A-Za-z].+[.!?。！？])$",
    re.DOTALL,
)
_ARABIC_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)*(?:st|nd|rd|th)?$", re.I)
_ENGLISH_ORDINAL_RE = re.compile(
    r"^(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)$",
    re.I,
)
_KOREAN_ORDINAL_RE = re.compile(
    r"^(?:(?:첫|두|세|네|다섯|여섯|일곱|여덟|아홉|열)(?:\s*번째|째)?|"
    r"(?:하나|둘|셋|넷)|번째|째)(?:은|는|이|가|을|를|의|로|으로)?$"
)
_KOREAN_INCOMPLETE_WORD_RE = re.compile(
    r"(?:그리고|하지만|그러나|따라서|또는|및|즉|반면에|왜냐하면|"
    r"예를\s*들어)\s*$"
)
_KOREAN_CONNECTIVE_END_RE = re.compile(
    r"(?:하고|하며|하면서|했지만|하지만|지만|는데|으나|거나|이거나|"
    r"므로|으므로|라면|이라면|다면|하면|면서|다가|도록|려고|려면|"
    r"위해|때문에|따라|대해|관해|통해|로서|으로서)\s*$"
)
_KOREAN_PARTICLE_END_RE = re.compile(
    r"[A-Za-z0-9가-힣._-]{2,}(?:은|는|이|가|을|를|에|에서|에게|께|와|과|로|으로)\s*$"
)
_KOREAN_PRONOUN_FRAGMENT_RE = re.compile(
    r"(?:이는|이것은|그것은|나는|내가|제가|우리는|우리가|사용자는|사용자가)\s*$"
)
_KOREAN_COMPLETE_PREDICATE_RE = re.compile(
    r"(?:다|요|죠|니다|습니다|합니다|됩니다|했습니다|됐습니다|"
    r"있습니다|없습니다|입니다|임|함|됨|음)\s*$"
)

_MODIFIERS = frozenset(
    {
        "매우",
        "정말",
        "아주",
        "상당히",
        "대단히",
        "굉장히",
        "훨씬",
        "조금",
        "다소",
        "특히",
        "극히",
        "완전히",
        "부분적으로",
        "비교적",
        "remarkably",
        "very",
        "really",
        "highly",
        "extremely",
        "slightly",
    }
)
_ENGLISH_ADJECTIVES = frozenset(
    {
        "fast",
        "slow",
        "safe",
        "unsafe",
        "accurate",
        "inaccurate",
        "strong",
        "weak",
        "important",
        "critical",
        "effective",
        "efficient",
        "inefficient",
        "simple",
        "complex",
        "new",
        "different",
        "excellent",
        "good",
        "bad",
    }
)
_KOREAN_ADJECTIVE_ROOTS = (
    "빠르",
    "빠른",
    "느리",
    "느린",
    "높",
    "낮",
    "크",
    "작",
    "좋",
    "나쁘",
    "강하",
    "강한",
    "약하",
    "약한",
    "강력하",
    "강력한",
    "안전하",
    "안전한",
    "위험하",
    "위험한",
    "중요하",
    "중요한",
    "정확하",
    "정확한",
    "부정확하",
    "완전하",
    "불완전하",
    "새롭",
    "새로운",
    "다양하",
    "다양한",
    "단순하",
    "단순한",
    "복잡하",
    "복잡한",
    "우수하",
    "탁월하",
    "안정적",
    "불안정",
    "효율적",
    "비효율적",
    "핵심적",
    "적절하",
    "부적절하",
    "적합하",
    "가능하",
    "불가능하",
    "필요하",
    "불필요하",
    "유용하",
    "신속하",
    "명확하",
    "모호하",
    "자연스럽",
    "부자연스럽",
    "훌륭하",
    "빨간",
    "파란",
    "초록",
    "최고",
    "최상",
    "최악",
)
_ADJECTIVAL_SUFFIX_RE = re.compile(
    r"(?:적(?:인|이고|이며|으로|이다|입니다)?|"
    r"스럽(?:다|게|고|지만|습니다|러운)|"
    r"롭(?:다|게|고|지만|습니다|로운)|[가-힣]{2,}한)$"
)
_TOPIC_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]{1,31}|[가-힣]{2,24}")
_TOPIC_STOP_TERMS = frozenset(
    {
        "그럼",
        "그리고",
        "대해서",
        "대한",
        "답하세요",
        "관련",
        "전에",
        "먼저",
        "무엇",
        "문장",
        "설명",
        "설명해",
        "알려",
        "어떤",
        "이번",
        "정리",
        "주세요",
        "질문",
        "하나",
        "항목",
        "현재",
    }
)
_TOPIC_SUFFIXES = (
    "해주세요",
    "입니다",
    "하세요",
    "합니다",
    "하는",
    "하여",
    "해서",
    "하며",
    "하고",
    "되는",
    "에서",
    "으로",
    "에게",
    "까지",
    "부터",
    "처럼",
    "보다",
    "이랑",
    "랑",
    "과",
    "와",
    "의",
    "이",
    "가",
    "은",
    "는",
    "을",
    "를",
    "도",
    "만",
)
_UNSOLICITED_SUBJECT_GROUPS = (
    frozenset({"뉴스", "언론사", "기자", "취재", "기사"}),
    frozenset({"이메일", "메일", "비밀번호", "gmail", "email", "password"}),
    frozenset({"chatgpt", "openai", "gemini", "claude"}),
    frozenset({"여행", "호텔", "항공권", "관광"}),
    frozenset({"짱구", "만화", "애니메이션"}),
)
_GENERIC_TOPIC_TERMS = frozenset(
    {
        "답변",
        "모델",
        "방법",
        "설명",
        "원칙",
        "응답",
        "절차",
        "질문",
        "확인",
        "검증",
    }
)


def _topic_terms(text: str) -> frozenset[str]:
    terms: set[str] = set()
    normalized = unicodedata.normalize("NFKC", text[:MAX_INSPECT_CHARS]).casefold()
    for match in _TOPIC_TERM_RE.finditer(normalized):
        term = match.group(0)
        if re.fullmatch(r"[가-힣]+", term):
            for suffix in _TOPIC_SUFFIXES:
                if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                    term = term[: -len(suffix)]
                    break
        if len(term) >= 2 and term not in _TOPIC_STOP_TERMS:
            terms.add(term)
        if len(terms) >= 128:
            break
    return frozenset(terms)


def response_topically_anchored(request: str, response: str) -> bool:
    """Reject obvious subject drift without pretending to judge semantics.

    The check is intentionally inactive for short/social requests.  For a
    substantive request with at least four bounded content terms, a response
    must preserve at least three (or 40 percent for smaller sets).  This catches
    failures such as answering a model-quality question with an unrelated news
    workflow while allowing ordinary paraphrase and Korean particles.
    """

    if not isinstance(request, str) or not isinstance(response, str):
        raise TypeError("request and response must be strings")
    requested = _topic_terms(request)
    if len(requested) < 4:
        return True
    observed = _topic_terms(response)
    required = 4 if len(requested) >= 6 else max(2, (len(requested) + 1) // 2)
    return len(requested & observed) >= required


def request_topic_terms(request: str, *, limit: int = 8) -> tuple[str, ...]:
    """Return a stable bounded topic vocabulary for one repair directive."""

    if not isinstance(request, str):
        raise TypeError("request must be a string")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 16:
        raise ValueError("limit must be an integer in [1, 16]")
    return tuple(sorted(_topic_terms(request)))[:limit]


def response_preserves_distinctive_topic(request: str, response: str) -> bool:
    """Require one non-generic request concept in a substantive response."""

    if not isinstance(request, str) or not isinstance(response, str):
        raise TypeError("request and response must be strings")
    distinctive = _topic_terms(request) - _GENERIC_TOPIC_TERMS
    if not distinctive:
        return True
    return bool(distinctive & _topic_terms(response))


def response_avoids_unsolicited_subjects(request: str, response: str) -> bool:
    """Reject a small set of observed, high-confidence subject intrusions.

    Lexical overlap alone cannot judge paraphrases.  This guard is therefore
    narrower: it blocks only recurrent unrelated corpora seen in the shipped
    checkpoint (news, credentials/mail, other assistant brands, travel, and
    animation) when the user did not mention that subject at all.
    """

    if not isinstance(request, str) or not isinstance(response, str):
        raise TypeError("request and response must be strings")
    requested = unicodedata.normalize("NFKC", request).casefold()
    observed = unicodedata.normalize("NFKC", response).casefold()
    for group in _UNSOLICITED_SUBJECT_GROUPS:
        if any(term in observed for term in group) and not any(
            term in requested for term in group
        ):
            return False
    return True


def has_near_duplicate_sentences(text: str) -> bool:
    """Detect two almost-identical long sentences within fixed work bounds."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    sentences: list[str] = []
    masked = _mask_code(text[:MAX_INSPECT_CHARS])
    for match in _COMPLETE_SENTENCE_RE.finditer(masked):
        sentence = re.sub(
            r"^(?:[-*+]\s+|\d{1,4}(?:[.)]|단계(?:는|로|:)?)\s*)",
            "",
            " ".join(match.group(0).split()).strip(),
        ).casefold()
        if len(sentence) >= 20:
            sentences.append(sentence[:256])
        if len(sentences) >= 64:
            break
    for index, first in enumerate(sentences):
        for second in sentences[index + 1 :]:
            if (
                SequenceMatcher(
                    None,
                    first,
                    second,
                    autojunk=False,
                ).ratio()
                >= 0.90
            ):
                return True
    return False


def inspect_response(text: str, *, final: bool = False) -> ResponseQualityReport:
    """Inspect decoded response text within fixed CPU work bounds.

    ``final=False`` is appropriate while tokens are still arriving.  It can
    stop role/control leakage and degeneration, but does not label an ordinary
    in-progress clause as incomplete.  Call once with ``final=True`` after the
    backend emits its terminal frame to decide whether a bounded continuation
    is warranted.
    """

    if not isinstance(text, str):
        raise TypeError("response text must be a string")

    regions = _bounded_regions(text)
    findings: list[QualityFinding] = []
    remaining_units = MAX_ANALYSIS_UNITS
    remaining_tokens = MAX_ANALYSIS_TOKENS

    for offset, region in regions:
        masked = _mask_code(region)
        findings.extend(_boundary_findings(masked, offset))
        if remaining_units <= 0 or remaining_tokens <= 0:
            continue
        units, used_tokens = _units(
            masked,
            offset,
            max_units=remaining_units,
            max_tokens=remaining_tokens,
        )
        remaining_units -= len(units)
        remaining_tokens -= used_tokens
        exact_repetition = _exact_unit_repetition(units)
        if exact_repetition is not None:
            findings.append(exact_repetition)
        template = _template_repetition(units)
        if template is not None:
            findings.append(template)
        low_information = _low_information_repetition(masked, offset)
        if low_information is not None:
            findings.append(low_information)
        numbering_restart = _numbering_restart(masked, offset)
        if numbering_restart is not None:
            findings.append(numbering_restart)

    if final and text:
        incomplete = _incomplete_korean_clause(text)
        if incomplete is not None:
            findings.append(incomplete)

    # One earliest result per code keeps the report small and stable even when
    # a malformed response contains many reserved tokens.
    earliest: dict[QualityCode, QualityFinding] = {}
    for finding in findings:
        previous = earliest.get(finding.code)
        if previous is None or (finding.start, finding.end) < (
            previous.start,
            previous.end,
        ):
            earliest[finding.code] = finding
    ordered = tuple(
        sorted(earliest.values(), key=lambda item: (item.start, item.code.value))
    )
    inspected = sum(len(region) for _offset, region in regions)
    return ResponseQualityReport(
        findings=ordered,
        input_characters=len(text),
        inspected_characters=inspected,
        input_truncated=len(text) > MAX_INSPECT_CHARS,
    )


def salvage_complete_prefix(text: str, *, cutoff: int | None = None) -> str:
    """Return the longest complete, quality-clean prefix without inventing text.

    A locally generated answer can contain useful complete sentences before an
    incomplete tail, a generated role marker, or a repetition cycle.  Rejecting
    the whole candidate in that situation produces a less helpful answer than
    publishing the already complete prefix.  This helper is deliberately
    conservative: it only returns text that independently passes the final
    quality inspection, and it considers at most the last 64 observed sentence
    boundaries.

    ``cutoff`` is an optional trusted upper bound supplied by a streaming guard.
    When omitted, hard-stop findings automatically bound the search.  An
    incomplete-clause finding at offset zero does *not* erase earlier complete
    sentences on the same line; the boundary scan still has a chance to retain
    them.
    """

    if not isinstance(text, str):
        raise TypeError("response text must be a string")
    if cutoff is not None and (not isinstance(cutoff, int) or isinstance(cutoff, bool)):
        raise TypeError("cutoff must be an integer or None")
    if not text:
        return ""

    report = inspect_response(text, final=True)
    limit = len(text)
    hard_stops = [
        finding.start for finding in report.findings if finding.code in _STOP_CODES
    ]
    if hard_stops:
        limit = min(limit, min(hard_stops))
    if cutoff is not None:
        limit = min(limit, max(0, cutoff))

    candidate = text[:limit].rstrip()
    if (
        candidate
        and inspect_response(candidate, final=True).recommended_action
        is QualityAction.ACCEPT
    ):
        return candidate

    # If the incomplete clause started after a clean prefix, avoid scanning
    # past it.  Offset zero means the detector conservatively marked the whole
    # line, so sentence boundaries in that line must still be considered.
    incomplete_starts = [
        finding.start
        for finding in report.findings
        if finding.code is QualityCode.INCOMPLETE_KOREAN_CLAUSE
        and 0 < finding.start < limit
    ]
    if incomplete_starts:
        limit = min(limit, min(incomplete_starts))

    boundaries = _content_sentence_boundaries(text[:limit])[-64:]
    for boundary in reversed(boundaries):
        candidate = text[: boundary.end()].rstrip()
        if (
            candidate
            and inspect_response(candidate, final=True).recommended_action
            is QualityAction.ACCEPT
        ):
            return candidate
    return ""


def _bounded_regions(text: str) -> tuple[tuple[int, str], ...]:
    if len(text) <= MAX_INSPECT_CHARS:
        return ((0, text),)
    half = MAX_INSPECT_CHARS // 2
    return ((0, text[:half]), (len(text) - half, text[-half:]))


def _boundary_findings(text: str, offset: int) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    role = _ROLE_MARKER_RE.search(text)
    if role is not None:
        findings.append(
            QualityFinding(
                QualityCode.ROLE_MARKER,
                offset + role.start(),
                offset + role.end(),
            )
        )
    control = _CONTROL_TOKEN_RE.search(text)
    if control is not None:
        findings.append(
            QualityFinding(
                QualityCode.CONTROL_TOKEN,
                offset + control.start(),
                offset + control.end(),
            )
        )
    return findings


def _mask_code(text: str) -> str:
    if "```" not in text:
        return text
    characters = list(text)
    for match in _CODE_FENCE_RE.finditer(text):
        for index in range(match.start(), match.end()):
            if characters[index] not in "\r\n":
                characters[index] = " "
    return "".join(characters)


def _units(
    text: str,
    offset: int,
    *,
    max_units: int,
    max_tokens: int,
) -> tuple[list[_Unit], int]:
    result: list[_Unit] = []
    used_tokens = 0
    for match in _UNIT_RE.finditer(text):
        if len(result) >= max_units or used_tokens >= max_tokens:
            break
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        value = raw.strip()
        if not value:
            continue
        if _LIST_MARKER_ONLY_RE.fullmatch(value) is not None:
            continue
        value = _LIST_PREFIX_RE.sub("", value, count=1)
        available = min(
            MAX_UNIT_TOKENS,
            max_tokens - used_tokens,
        )
        tokens = _token_values(value, available)
        if not tokens:
            continue
        raw_key = tuple(tokens)
        template_key = tuple(_canonical_token(token) for token in tokens)
        template_key = _collapse_placeholders(template_key)
        used_tokens += len(tokens)
        result.append(
            _Unit(
                start=offset + match.start() + leading,
                end=offset + match.end() - trailing,
                raw_key=raw_key,
                template_key=template_key,
                has_variable=raw_key != template_key,
                visible_characters=sum(character.isalnum() for character in value),
            )
        )
    return result, used_tokens


def _token_values(text: str, limit: int) -> list[str]:
    values: list[str] = []
    for match in _LEXEME_RE.finditer(text):
        values.append(unicodedata.normalize("NFKC", match.group(0)).casefold())
        if len(values) >= limit:
            break
    return values


def _tokens_with_positions(text: str, offset: int) -> list[_Token]:
    tokens: list[_Token] = []
    for match in _LEXEME_RE.finditer(text):
        tokens.append(
            _Token(
                unicodedata.normalize("NFKC", match.group(0)).casefold(),
                offset + match.start(),
                offset + match.end(),
            )
        )
        if len(tokens) >= MAX_ANALYSIS_TOKENS:
            break
    return tokens


def _canonical_token(token: str) -> str:
    if (
        _ARABIC_NUMBER_RE.fullmatch(token)
        or _ENGLISH_ORDINAL_RE.fullmatch(token)
        or _KOREAN_ORDINAL_RE.fullmatch(token)
    ):
        return "<number>"
    if token in _MODIFIERS:
        return "<modifier>"
    if _is_adjective(token):
        return "<adjective>"
    return token


def _is_adjective(token: str) -> bool:
    if token in _ENGLISH_ADJECTIVES:
        return True
    for root in _KOREAN_ADJECTIVE_ROOTS:
        if token.startswith(root):
            return True
        # ``정확하다`` and similar ㅎ-irregular surface forms become
        # ``정확합니다/정확한/정확해``.  Keep the conversion lexical and
        # bounded instead of treating every ``-합니다`` predicate as an
        # adjective (which would collapse substantive action lists).
        if root.endswith("하") and token.startswith(
            (root[:-1] + "합", root[:-1] + "한", root[:-1] + "해")
        ):
            return True
        if root.endswith("르") and token.startswith(
            (root[:-1] + "릅", root[:-1] + "른")
        ):
            return True
    return _ADJECTIVAL_SUFFIX_RE.fullmatch(token) is not None


def _collapse_placeholders(tokens: tuple[str, ...]) -> tuple[str, ...]:
    collapsed: list[str] = []
    for token in tokens:
        if collapsed and token == collapsed[-1] and token.startswith("<"):
            continue
        collapsed.append(token)
    return tuple(collapsed)


def _template_repetition(units: list[_Unit]) -> QualityFinding | None:
    count = len(units)
    for start in range(count):
        maximum = min(MAX_TEMPLATE_BLOCK_UNITS, (count - start) // 2)
        for block_size in range(1, maximum + 1):
            first = units[start : start + block_size]
            second = units[start + block_size : start + block_size * 2]
            if tuple(unit.raw_key for unit in first) == tuple(
                unit.raw_key for unit in second
            ):
                visible = sum(unit.visible_characters for unit in first)
                # Two identical Korean list bodies can carry substantial
                # semantics in fewer visible characters than an English
                # sentence. Twelve alphanumerics still excludes ordinary
                # short emphasis while catching duplicated policy items.
                if visible >= 12:
                    return QualityFinding(
                        QualityCode.TEMPLATE_REPETITION,
                        second[0].start,
                        second[-1].end,
                        occurrences=2,
                    )

            if start + block_size * 3 > count:
                continue
            third = units[start + block_size * 2 : start + block_size * 3]
            template = tuple(unit.template_key for unit in first)
            if (
                not template
                or template != tuple(unit.template_key for unit in second)
                or template != tuple(unit.template_key for unit in third)
            ):
                continue
            all_units = first + second + third
            has_variable = any(unit.has_variable for unit in all_units)
            stable_tokens = sum(
                token not in {"<number>", "<modifier>", "<adjective>"}
                for unit in first
                for token in unit.template_key
            )
            if has_variable and stable_tokens >= 1:
                return QualityFinding(
                    QualityCode.TEMPLATE_REPETITION,
                    second[0].start,
                    third[-1].end,
                    occurrences=3,
                )
    return None


def _exact_unit_repetition(units: list[_Unit]) -> QualityFinding | None:
    """Detect a repeated substantive sentence even when copies are separated."""

    seen: dict[tuple[str, ...], _Unit] = {}
    for unit in units:
        if unit.visible_characters < 12 or len(unit.raw_key) < 3:
            continue
        if unit.raw_key in seen:
            return QualityFinding(
                QualityCode.TEMPLATE_REPETITION,
                unit.start,
                unit.end,
                occurrences=2,
            )
        seen[unit.raw_key] = unit
    return None


def _numbering_restart(text: str, offset: int) -> QualityFinding | None:
    """Stop when a generated numbered answer restarts at item one."""

    highest = 0
    for match in _INLINE_NUMBERED_CLAUSE_RE.finditer(text):
        number = int(match.group("number"))
        if number == 1 and highest >= 2:
            return QualityFinding(
                QualityCode.TEMPLATE_REPETITION,
                offset + match.start(),
                offset + match.end(),
                occurrences=2,
            )
        highest = max(highest, number)
    return None


def _low_information_repetition(text: str, offset: int) -> QualityFinding | None:
    tokens = _tokens_with_positions(text, offset)
    values = tuple(token.value for token in tokens)
    count = len(tokens)
    for start in range(count):
        maximum = min(MAX_LOW_INFORMATION_PERIOD, (count - start) // 3)
        for period in range(1, maximum + 1):
            pattern = values[start : start + period]
            if values[start + period : start + period * 2] != pattern:
                continue
            if values[start + period * 2 : start + period * 3] != pattern:
                continue
            if len(set(pattern)) > 3:
                continue
            return QualityFinding(
                QualityCode.LOW_INFORMATION_REPETITION,
                tokens[start + period].start,
                tokens[start + period * 3 - 1].end,
                occurrences=3,
            )

    if count >= 12:
        frequencies = Counter(values)
        most_common, frequency = frequencies.most_common(1)[0]
        if len(frequencies) * 4 <= count and frequency >= 4:
            positions = [token for token in tokens if token.value == most_common]
            return QualityFinding(
                QualityCode.LOW_INFORMATION_REPETITION,
                positions[1].start,
                positions[-1].end,
                occurrences=frequency,
            )
    return None


def _incomplete_korean_clause(text: str) -> QualityFinding | None:
    stripped = text.rstrip()
    if not stripped or not re.search(r"[가-힣]", stripped):
        return None
    if stripped.endswith("```"):
        return None
    visible = stripped.rstrip("*_~'\"”’)]}")
    if not visible:
        return None
    last_line = visible.rsplit("\n", 1)[-1]
    if _LIST_MARKER_ONLY_RE.fullmatch(last_line) is not None:
        return QualityFinding(
            QualityCode.INCOMPLETE_KOREAN_CLAUSE,
            len(visible) - len(last_line),
            len(visible),
        )
    terminal_punctuation = visible.endswith((".", "!", "?", "。", "！", "？"))
    analyzable = visible.rstrip(".!?。！？").rstrip()
    if not analyzable:
        return None
    tail = analyzable[-128:]
    pronoun_fragment = _KOREAN_PRONOUN_FRAGMENT_RE.search(tail)
    if terminal_punctuation and pronoun_fragment is None:
        return None
    if (
        not terminal_punctuation
        and len(visible) >= 48
        and _KOREAN_COMPLETE_PREDICATE_RE.search(tail) is None
    ):
        line_start = visible.rfind("\n") + 1
        return QualityFinding(
            QualityCode.INCOMPLETE_KOREAN_CLAUSE,
            line_start,
            len(visible),
        )
    match = (
        pronoun_fragment
        or _KOREAN_INCOMPLETE_WORD_RE.search(tail)
        or _KOREAN_CONNECTIVE_END_RE.search(tail)
        or _KOREAN_PARTICLE_END_RE.search(tail)
    )
    if match is None and not analyzable.endswith((",", ";", ":", "-", "–", "—")):
        return None
    start = (
        len(analyzable)
        - len(tail)
        + (match.start() if match is not None else len(tail) - 1)
    )
    return QualityFinding(
        QualityCode.INCOMPLETE_KOREAN_CLAUSE,
        start,
        len(visible),
    )


_KOREAN_COUNTS = {
    "한": 1,
    "두": 2,
    "세": 3,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}


def requested_minimum_units(request: str) -> int | None:
    """Return an explicit Korean sentence/step/item minimum when unambiguous.

    ``이내`` and ``이하`` express a maximum, so they are deliberately not
    converted into a minimum.  Sentence contracts take precedence over step
    and item wording elsewhere in the same request.
    """

    if not isinstance(request, str):
        raise TypeError("request must be a string")
    candidates: list[tuple[int, int, int]] = []
    priority = {"문장": 0, "단계": 1, "항목": 2}
    for match in _REQUEST_COUNT_RE.finditer(request[:MAX_INSPECT_CHARS]):
        suffix = request[match.end() : match.end() + 4]
        if re.match(r"\s*(?:이내|이하)", suffix):
            continue
        raw = match.group("count")
        count = int(raw) if raw.isdigit() else _KOREAN_COUNTS[raw]
        candidates.append((priority[match.group("unit")], match.start(), count))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def requested_exact_sentence_count(request: str) -> int | None:
    """Return an exact sentence count when the wording is not a range.

    Minimum/maximum expressions intentionally return ``None`` so an exact-count
    decode directive cannot silently narrow the user's requested range.
    """

    if not isinstance(request, str):
        raise TypeError("request must be a string")
    bounded = request[:MAX_INSPECT_CHARS]
    for match in _REQUEST_COUNT_RE.finditer(bounded):
        if match.group("unit") != "문장":
            continue
        prefix = bounded[max(0, match.start() - 6) : match.start()]
        suffix = bounded[match.end() : match.end() + 8]
        if re.search(r"(?:최소|최대|각각)\s*$", prefix):
            continue
        if re.match(r"\s*(?:씩|이상|이하|이내|미만|초과)", suffix):
            continue
        raw = match.group("count")
        return int(raw) if raw.isdigit() else _KOREAN_COUNTS[raw]
    return None


def requested_maximum_items(request: str) -> int | None:
    """Return an explicit Korean maximum item count such as ``네 가지 이내``."""

    if not isinstance(request, str):
        raise TypeError("request must be a string")
    match = _MAX_ITEM_REQUEST_RE.search(request[:MAX_INSPECT_CHARS])
    if match is None:
        return None
    raw = match.group("count")
    return int(raw) if raw.isdigit() else _KOREAN_COUNTS[raw]


def requested_exact_item_count(request: str) -> int | None:
    """Return one exact step/item count, excluding ranges and category pairs."""

    if not isinstance(request, str):
        raise TypeError("request must be a string")
    bounded = request[:MAX_INSPECT_CHARS]
    matches = list(_EXACT_ITEM_COUNT_RE.finditer(bounded))
    if len(matches) != 1:
        return None
    match = matches[0]
    prefix = bounded[max(0, match.start() - 6) : match.start()]
    suffix = bounded[match.end() : match.end() + 8]
    if re.search(r"(?:최소|최대|각각)\s*$", prefix):
        return None
    if re.match(r"\s*(?:씩|이상|이하|이내|미만|초과)", suffix):
        return None
    output_tail = bounded[match.end() : match.end() + 40]
    if _EXACT_ITEM_OUTPUT_CUE_RE.search(output_tail) is None:
        return None
    raw = match.group("count")
    return int(raw) if raw.isdigit() else _KOREAN_COUNTS[raw]


def response_contract_satisfied(request: str, response: str) -> bool:
    """Check the bounded minimum for an explicitly requested response shape."""

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    exact = requested_exact_sentence_count(request)
    exact_items = requested_exact_item_count(request)
    minimum = requested_minimum_units(request)
    categories = _requested_category_counts(request)
    if minimum is None and exact_items is None and categories is None:
        return True
    completed = _contract_sentence_count(response[:MAX_INSPECT_CHARS])
    if categories is not None and not _category_contract_satisfied(
        response,
        categories,
    ):
        return False
    if exact is not None:
        return completed == exact
    if exact_items is not None:
        return completed == exact_items
    if minimum is None:
        return True
    return completed >= minimum


def _content_sentence_boundaries(text: str) -> list[re.Match[str]]:
    """Return punctuation boundaries while ignoring inline list ordinals."""

    boundaries: list[re.Match[str]] = []
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        if match.group(0).startswith(".") and match.start() >= 1:
            number_index = match.start() - 1
            before_number = text[number_index - 1] if number_index >= 1 else ""
            following = text[match.end() :]
            if (
                text[number_index].isdigit()
                and (number_index == 0 or before_number.isspace())
                and bool(following.lstrip())
            ):
                continue
        boundaries.append(match)
    return boundaries


def _contract_sentence_count(text: str) -> int:
    """Count substantive completed sentences, not headings or list numbers."""

    count = 0
    cursor = 0
    for boundary in _content_sentence_boundaries(text):
        sentence = " ".join(text[cursor : boundary.end()].split()).strip()
        cursor = boundary.end()
        sentence = re.sub(r"^\d{1,4}[.)]\s*", "", sentence)
        if (
            not sentence
            or sentence in _CLOSING_SENTENCES
            or sentence.endswith(_META_SENTENCE_ENDINGS)
        ):
            continue
        count += 1
    return count


def _completed_content_sentences(text: str) -> list[str]:
    cleaned = _ALL_LIST_PREFIX_RE.sub("", text[:MAX_INSPECT_CHARS])
    sentences: list[str] = []
    seen: set[str] = set()
    for match in _COMPLETE_SENTENCE_RE.finditer(cleaned):
        sentence = " ".join(match.group(0).split()).strip()
        sentence = re.sub(
            r"^(?:장점|이점|강점|한계|단점|제약|위험)\s*[:：]\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        if (
            not sentence
            or sentence in _CLOSING_SENTENCES
            or _LIST_MARKER_ONLY_RE.fullmatch(sentence) is not None
        ):
            continue
        if sentence.endswith(_META_SENTENCE_ENDINGS):
            continue
        key = unicodedata.normalize("NFKC", sentence).casefold()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(sentence)
        if len(sentences) >= MAX_ANALYSIS_UNITS:
            break
    return sentences


def _requested_category_counts(request: str) -> tuple[int, int] | None:
    positive = 0
    negative = 0
    for match in _CATEGORY_COUNT_RE.finditer(request[:MAX_INSPECT_CHARS]):
        raw = match.group("count")
        count = int(raw) if raw.isdigit() else _KOREAN_COUNTS[raw]
        if match.group("category") in {"장점", "이점", "강점"}:
            positive += count
        else:
            negative += count
    if positive and negative:
        return positive, negative
    return None


def _category_contract_satisfied(
    response: str,
    categories: tuple[int, int],
) -> bool:
    """Require requested positive and negative categories to remain visible."""

    positive_count, negative_count = categories
    sentences = _completed_content_sentences(response)
    negatives = sum(
        any(cue in sentence for cue in _NEGATIVE_CATEGORY_CUES)
        for sentence in sentences
    )
    positives = len(sentences) - negatives
    return negatives == negative_count and positives == positive_count


def compose_observed_contract_response(
    request: str,
    observed: list[str],
) -> str:
    """Compose a response only from distinct complete sentences already seen.

    This is a bounded last-resort salvage path for local generations whose
    individual attempts degenerate after different useful sentences. It never
    invents content and re-runs the full request contract before publication.
    """

    if not isinstance(request, str) or not isinstance(observed, list):
        raise TypeError("request must be a string and observed must be a list")
    sentences: list[str] = []
    for item in observed[:MAX_ANALYSIS_UNITS]:
        if not isinstance(item, str):
            raise TypeError("observed items must be strings")
        sentences.extend(_completed_content_sentences(item))
        if len(sentences) >= MAX_ANALYSIS_UNITS:
            break

    categories = _requested_category_counts(request)
    if categories is not None:
        positive_count, negative_count = categories
        positives = [
            sentence
            for sentence in sentences
            if not any(cue in sentence for cue in _NEGATIVE_CATEGORY_CUES)
        ]
        negatives = [
            sentence
            for sentence in sentences
            if any(cue in sentence for cue in _NEGATIVE_CATEGORY_CUES)
        ]
        if len(positives) < positive_count or len(negatives) < negative_count:
            return ""
        selected = positives[:positive_count] + negatives[:negative_count]
    else:
        expected = requested_exact_sentence_count(request)
        if expected is None:
            expected = requested_exact_item_count(request)
        if expected is None or len(sentences) < expected:
            return ""
        selected = sentences[:expected]

    normalized = " ".join(selected).strip()
    if (
        not response_contract_satisfied(request, normalized)
        or inspect_response(normalized, final=True).recommended_action
        is not QualityAction.ACCEPT
        or has_near_duplicate_sentences(normalized)
    ):
        return ""
    return normalized


def _split_complete_korean_coordinate(sentence: str) -> list[str] | None:
    match = _KOREAN_COORDINATE_SPLIT_RE.fullmatch(sentence.strip())
    if match is None:
        return None
    predicate = "합니다." if match.group("ending") in {"하고", "하며"} else "입니다."
    first = match.group("head").rstrip(" ,") + predicate
    second = match.group("tail").strip()
    if len(first) < 12 or len(second) < 12:
        return None
    return [first, second]


def _split_inline_numbered_sentences(
    text: str,
    expected: int,
) -> list[str] | None:
    """Turn a model's numbered coordinate clauses into complete sentences.

    Small instruction-tuned models often satisfy a requested item count but
    join ``1) ...하고, 2) ...하며, 3) ...합니다.`` into one grammatical
    sentence.  This bounded normalizer keeps the generated meaning and only
    completes the two common Korean coordinate predicates.  Any ambiguous
    fragment fails closed and is retried by the caller.
    """

    bounded = text[:MAX_INSPECT_CHARS]
    observed = list(_INLINE_NUMBERED_CLAUSE_RE.finditer(bounded))[: expected + 1]
    if len(observed) < expected:
        return None
    matches = observed[:expected]
    if [int(match.group("number")) for match in matches] != list(
        range(1, expected + 1)
    ):
        return None

    result: list[str] = []
    for index, match in enumerate(matches):
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        elif len(observed) > expected:
            end = observed[expected].start()
        else:
            end = len(bounded)
        raw_clause = bounded[match.end() : end].strip()
        boundaries = _content_sentence_boundaries(raw_clause)
        if boundaries:
            raw_clause = raw_clause[: boundaries[0].end()]
        clause = raw_clause.strip().rstrip(" ,;")
        if not clause or len(clause) < 8:
            return None
        if clause.endswith((".", "!", "?", "。", "！", "？")):
            sentence = clause
        elif clause.endswith(("하고", "하며")):
            sentence = clause[:-2].rstrip() + "합니다."
        elif clause.endswith(("이고", "이며")):
            sentence = clause[:-2].rstrip() + "입니다."
        elif clause.endswith(("되고", "되며")):
            sentence = clause[:-2].rstrip() + "됩니다."
        elif clause.endswith(("있고", "있으며")):
            suffix = "있고" if clause.endswith("있고") else "있으며"
            sentence = clause[: -len(suffix)].rstrip() + "있습니다."
        elif clause.endswith(("없고", "없으며")):
            suffix = "없고" if clause.endswith("없고") else "없으며"
            sentence = clause[: -len(suffix)].rstrip() + "없습니다."
        elif clause.endswith(("다", "요", "니다")):
            sentence = clause + "."
        else:
            return None
        if len(sentence) < 8:
            return None
        result.append(sentence)
    return result


def normalize_exact_sentence_response(request: str, response: str) -> str | None:
    """Salvage only complete content from an overlong exact-count candidate.

    No new text is synthesized. The normalizer removes list markers, obvious
    meta-introductions/closings, and an incomplete tail. Paired category requests
    are accepted only when both requested sections remain represented.
    """

    if not isinstance(request, str) or not isinstance(response, str):
        raise TypeError("request and response must be strings")
    exact = requested_exact_sentence_count(request)
    if exact is None or not response.strip():
        return None

    categories = _requested_category_counts(request)
    selected: list[str]
    if categories is not None and sum(categories) == exact:
        positive_count, negative_count = categories
        boundary = _NEGATIVE_SECTION_RE.search(response[:MAX_INSPECT_CHARS])
        if boundary is None:
            return None
        positive = _completed_content_sentences(response[: boundary.start()])
        negative = _completed_content_sentences(response[boundary.end() :])
        if len(positive) < positive_count or len(negative) < negative_count:
            return None
        selected = positive[:positive_count] + negative[:negative_count]
    else:
        inline = _split_inline_numbered_sentences(response, exact)
        candidates = inline or _completed_content_sentences(response)
        if (
            inline is None
            and len(candidates) < exact
            and exact == 2
            and len(candidates) == 1
        ):
            split = _split_complete_korean_coordinate(candidates[0])
            if split is not None:
                candidates = split
        if len(candidates) < exact:
            return None
        selected = candidates[:exact]

    normalized = " ".join(selected).strip()
    if not response_contract_satisfied(request, normalized):
        return None
    if (
        inspect_response(normalized, final=True).recommended_action
        is not QualityAction.ACCEPT
    ):
        return None
    return normalized


def normalize_maximum_item_response(request: str, response: str) -> str | None:
    """Publish the first bounded sequential item block for an ``이내`` request."""

    if not isinstance(request, str) or not isinstance(response, str):
        raise TypeError("request and response must be strings")
    maximum = requested_maximum_items(request)
    if maximum is None or not response.strip():
        return None
    minimum = 2 if maximum >= 2 else 1
    for count in range(maximum, minimum - 1, -1):
        sentences = _split_inline_numbered_sentences(response, count)
        if sentences is None:
            continue
        normalized = " ".join(sentences).strip()
        if (
            normalized
            and inspect_response(normalized, final=True).recommended_action
            is QualityAction.ACCEPT
        ):
            return normalized
    return None


def normalize_exact_item_response(request: str, response: str) -> str | None:
    """Normalize the first sequential block for an exact step/item request."""

    if not isinstance(request, str) or not isinstance(response, str):
        raise TypeError("request and response must be strings")
    expected = requested_exact_item_count(request)
    if expected is None or not response.strip():
        return None
    sentences = _split_inline_numbered_sentences(response, expected)
    if sentences is None:
        return None
    normalized = " ".join(sentences).strip()
    if (
        _contract_sentence_count(normalized) != expected
        or inspect_response(normalized, final=True).recommended_action
        is not QualityAction.ACCEPT
    ):
        return None
    return normalized


__all__ = [
    "MAX_INSPECT_CHARS",
    "QualityAction",
    "QualityCode",
    "QualityFinding",
    "ResponseQualityError",
    "ResponseQualityReport",
    "compose_observed_contract_response",
    "inspect_response",
    "has_near_duplicate_sentences",
    "normalize_exact_sentence_response",
    "normalize_exact_item_response",
    "normalize_maximum_item_response",
    "request_topic_terms",
    "requested_exact_sentence_count",
    "requested_exact_item_count",
    "requested_maximum_items",
    "requested_minimum_units",
    "response_avoids_unsolicited_subjects",
    "response_contract_satisfied",
    "response_preserves_distinctive_topic",
    "response_topically_anchored",
    "salvage_complete_prefix",
]
