"""Narrow deterministic replies for small Korean social conversation turns.

This module is deliberately not a general intent router.  It recognizes only
four bounded social exchanges that should never need an expensive model load:
a greeting, an invitation to collaborate on a project/demo, a question about
what work the user can ask for, and one specific first-step follow-up.
Everything else is returned to the normal Fact-book/model pipeline.
"""

from __future__ import annotations

import re
import unicodedata


MAX_FAST_PATH_INPUT_CHARS = 256

_SPACE_RE = re.compile(r"\s+")
_EDGE_PUNCTUATION_RE = re.compile(
    r"^[\s,.!?~。！？…·'\"“”‘’()\[\]{}]+|[\s,.!?~。！？…·'\"“”‘’()\[\]{}]+$"
)
_COMPACT_PUNCTUATION_RE = re.compile(r"[\s,.!?~。！？…·'\"“”‘’()\[\]{}:;]+")
_UNSUPPORTED_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_GREETING_FORMS = frozenset(
    {
        "안녕",
        "안녕하세요",
        "안녕하세여",
        "안녕하셔요",
        "반가워",
        "반가워요",
        "반갑습니다",
        "좋은아침",
        "좋은아침이에요",
    }
)
_GREETING_SOCIAL_SUFFIXES = (
    "오늘은편하게이야기해볼까요",
    "편하게이야기해볼까요",
    "오늘이야기해볼까요",
)
_PROJECT_TERMS = ("프로젝트", "프러젝트", "프로잭트", "프러잭트")
_DEMO_TERMS = ("데모", "대모")
_COLLABORATION_TERMS = ("함께", "같이", "나랑", "저랑", "우리")
_INVITATION_TERMS = (
    "합시다",
    "하자",
    "해요",
    "해보자",
    "해볼래",
    "해볼까요",
    "만들자",
    "만들어요",
    "만들어보자",
    "만들어볼래",
    "만들어볼래요",
    "만들어볼까요",
    "만드러볼래",
    "만드러볼래요",
    "만드러볼까요",
    "만들고싶어",
    "만들고싶어요",
    "시작하자",
    "시작해요",
)
_NON_SOCIAL_TASK_TERMS = (
    "코드작성",
    "코드를작성",
    "구현해줘",
    "구현해주세요",
    "분석해줘",
    "분석해주세요",
    "설명해줘",
    "설명해주세요",
    "계획을작성",
    "파일을",
    "명령",
)
_CAPABILITY_SHAPES = (
    re.compile(
        r"^(?:나와|나랑|저와|저랑|우리와|우리랑)?(?:어떤|무슨)일을"
        r"(?:함께|같이)?(?:할수있(?:나요|어요|어)|도와줄수있(?:나요|어요|어))$"
    ),
    re.compile(
        r"^(?:내가|제가)?(?:당신에게|너에게|여기서)?(?:어떤|무슨)일을"
        r"부탁할수있(?:나요|어요|어)$"
    ),
    re.compile(
        r"^(?:내가|제가)?부탁할수있는일을"
        r"(?:어렵지않게)?(?:다른말로)?알려(?:주세요|줘)$"
    ),
    re.compile(
        r"^(?:내가|제가)?(?:당신에게|너에게)?부탁할수있는일(?:은|이)?"
        r"(?:무엇인가요|뭐예요|뭔가요)$"
    ),
    re.compile(
        r"^(?:나와|나랑|저와|저랑)?(?:함께|같이)할수있는일(?:은|이)?"
        r"(?:무엇인가요|뭐예요|뭔가요)$"
    ),
)
_FIRST_STEP_SHAPES = (
    re.compile(
        r"^(?:좋아요|좋아|그럼|그러면)?(?:그럼|그러면)?"
        r"첫(?:번째)?단계(?:에서|는)?(?:제가정할것)?하나만"
        r"(?:물어봐|물어봐줘|물어봐주세요|물어보세요|질문해줘|질문해주세요)$"
    ),
)

_GREETING_REPLY = "안녕하세요! 오늘은 어떤 이야기를 함께 나눠 볼까요?"
_PROJECT_REPLY = (
    "좋아요. 재미있는 프로젝트를 함께 시작해 봅시다. "
    "먼저 만들고 싶은 결과를 한 문장으로 알려 주세요."
)
_TYPO_PROJECT_REPLY = (
    "좋아요. 재미있는 프로젝트를 같이 만들어 봐요. "
    "먼저 해보고 싶은 주제를 한 문장으로 알려 주세요."
)
_DEMO_REPLY = (
    "좋아요. 오프라인 AI 데모를 함께 만들어 봅시다. "
    "먼저 데모로 보여줄 핵심 기능 하나를 알려 주세요."
)
_COLLABORATION_REPLIES = frozenset({_PROJECT_REPLY, _TYPO_PROJECT_REPLY, _DEMO_REPLY})


def _normalized(text: str) -> tuple[str, str] | None:
    if not isinstance(text, str) or not text or len(text) > MAX_FAST_PATH_INPUT_CHARS:
        return None
    value = unicodedata.normalize("NFKC", text)
    if _UNSUPPORTED_CONTROL_RE.search(value):
        return None
    spaced = _SPACE_RE.sub(" ", value).strip()
    if not spaced:
        return None
    bounded = _EDGE_PUNCTUATION_RE.sub("", spaced).casefold()
    compact = _COMPACT_PUNCTUATION_RE.sub("", bounded)
    if not compact or len(compact) > MAX_FAST_PATH_INPUT_CHARS:
        return None
    return bounded, compact


class ConversationFastPath:
    """Return one safe social reply, or ``None`` for the normal product path."""

    def answer(
        self,
        text: str,
        *,
        previous_user: str | None = None,
        previous_assistant: str | None = None,
    ) -> str | None:
        normalized = _normalized(text)
        if normalized is None:
            return None
        _spaced, compact = normalized

        if self._is_greeting(compact):
            return _GREETING_REPLY
        if self._is_project_invitation(compact):
            if any(term in compact for term in _DEMO_TERMS):
                return _DEMO_REPLY
            if any(term in compact for term in ("프로잭트", "프러잭트", "만드러")):
                return _TYPO_PROJECT_REPLY
            return _PROJECT_REPLY
        if self._is_capability_question(compact):
            if "다른말로" in compact:
                return (
                    "쉽게 말하면 아이디어를 다듬고, 코드나 문서를 검토하고, 로컬에서 "
                    "테스트하는 작업을 함께할 수 있습니다. 원하시는 일을 말씀해 주시면 "
                    "제가 도와드릴 범위부터 정리하겠습니다."
                )
            return (
                "아이디어 정리, 구현 계획, 코드·문서 검토, 로컬 테스트 같은 일을 "
                "함께할 수 있습니다. 원하는 결과를 알려 주시면 가능한 범위와 첫 단계를 "
                "분명히 나눠서 도와드리겠습니다."
            )
        if any(
            pattern.fullmatch(compact) for pattern in _FIRST_STEP_SHAPES
        ) and self._has_collaboration_context(previous_user, previous_assistant):
            return "좋아요. 이 데모를 가장 먼저 보여주고 싶은 사용자는 누구인가요?"
        return None

    @staticmethod
    def _is_greeting(compact: str) -> bool:
        if compact in _GREETING_FORMS:
            return True
        return any(
            compact == greeting + suffix
            for greeting in _GREETING_FORMS
            for suffix in _GREETING_SOCIAL_SUFFIXES
        )

    @staticmethod
    def _is_project_invitation(compact: str) -> bool:
        if len(compact) > 96 or any(term in compact for term in _NON_SOCIAL_TASK_TERMS):
            return False
        has_subject = any(term in compact for term in (*_PROJECT_TERMS, *_DEMO_TERMS))
        has_partner = any(term in compact for term in _COLLABORATION_TERMS)
        has_invitation = any(compact.endswith(term) for term in _INVITATION_TERMS)
        return has_subject and has_partner and has_invitation

    @staticmethod
    def _is_capability_question(compact: str) -> bool:
        if len(compact) > 96 or any(term in compact for term in _NON_SOCIAL_TASK_TERMS):
            return False
        return any(pattern.fullmatch(compact) for pattern in _CAPABILITY_SHAPES)

    @classmethod
    def _has_collaboration_context(
        cls,
        previous_user: str | None,
        previous_assistant: str | None,
    ) -> bool:
        if (
            not isinstance(previous_assistant, str)
            or len(previous_assistant) > MAX_FAST_PATH_INPUT_CHARS * 2
        ):
            return False
        normalized = (
            _normalized(previous_user) if isinstance(previous_user, str) else None
        )
        return (
            normalized is not None
            and cls._is_project_invitation(normalized[1])
            and previous_assistant.strip() in _COLLABORATION_REPLIES
        )


__all__ = ["ConversationFastPath", "MAX_FAST_PATH_INPUT_CHARS"]
