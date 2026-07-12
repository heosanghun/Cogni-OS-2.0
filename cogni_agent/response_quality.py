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
    r"<<\s*/?sys\s*>>|\[Runtime Fact-book:|\[턴\s*종료\]"
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
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？]+(?=\s|$)")
_COMPLETE_SENTENCE_RE = re.compile(r".+?[.!?。！？]+(?=\s|$)", re.DOTALL)
_ALL_LIST_PREFIX_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d{1,4}[.)]\s+)")
_NEGATIVE_SECTION_RE = re.compile(r"(?im)^\s*(?:한계|단점|제약|위험)\s*[:：]\s*")
_CATEGORY_COUNT_RE = re.compile(
    r"(?P<category>장점|이점|강점|한계|단점|제약|위험)\s*"
    r"(?P<count>[1-9]|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*가지"
)
_META_SENTENCE_ENDINGS = (
    "설명하겠습니다.",
    "답변하겠습니다.",
    "정리하겠습니다.",
    "알려드리겠습니다.",
    "살펴보겠습니다.",
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
        findings.extend(_boundary_findings(region, offset))
        if remaining_units <= 0 or remaining_tokens <= 0:
            continue
        masked = _mask_code(region)
        units, used_tokens = _units(
            masked,
            offset,
            max_units=remaining_units,
            max_tokens=remaining_tokens,
        )
        remaining_units -= len(units)
        remaining_tokens -= used_tokens
        template = _template_repetition(units)
        if template is not None:
            findings.append(template)
        low_information = _low_information_repetition(masked, offset)
        if low_information is not None:
            findings.append(low_information)

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


def response_contract_satisfied(request: str, response: str) -> bool:
    """Check the bounded minimum for an explicitly requested response shape."""

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    exact = requested_exact_sentence_count(request)
    minimum = requested_minimum_units(request)
    if minimum is None:
        return True
    completed = len(_SENTENCE_BOUNDARY_RE.findall(response[:MAX_INSPECT_CHARS]))
    if exact is not None:
        return completed == exact
    return completed >= minimum


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
        if not sentence or sentence in _CLOSING_SENTENCES:
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
        candidates = _completed_content_sentences(response)
        if len(candidates) < exact and exact == 2 and len(candidates) == 1:
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


__all__ = [
    "MAX_INSPECT_CHARS",
    "QualityAction",
    "QualityCode",
    "QualityFinding",
    "ResponseQualityError",
    "ResponseQualityReport",
    "inspect_response",
    "normalize_exact_sentence_response",
    "requested_exact_sentence_count",
    "requested_minimum_units",
    "response_contract_satisfied",
]
