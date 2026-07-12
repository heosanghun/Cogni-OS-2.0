from __future__ import annotations

import unittest

from cogni_agent.conversation_fastpath import (
    MAX_FAST_PATH_INPUT_CHARS,
    ConversationFastPath,
)


class TestConversationFastPath(unittest.TestCase):
    def setUp(self) -> None:
        self.fast_path = ConversationFastPath()

    def test_greetings_are_nfkc_normalized_and_bounded(self) -> None:
        self.assertIn("안녕하세요", self.fast_path.answer("  안녕하세여!!  ") or "")
        self.assertIn(
            "이야기",
            self.fast_path.answer("안녕하세요！ 오늘은 편하게 이야기해 볼까요？") or "",
        )
        self.assertIsNone(
            self.fast_path.answer("안녕" + "요" * MAX_FAST_PATH_INPUT_CHARS)
        )
        self.assertIsNone(self.fast_path.answer("안녕\x00"))

    def test_project_and_demo_invitations_include_common_typos(self) -> None:
        project = self.fast_path.answer("그럼, 나와함께 재미있는 프로젝트를 합시다.")
        typo = self.fast_path.answer("나랑 재미잇는 프로잭트 같이 만드러볼래요?")
        demo = self.fast_path.answer("오프라인 AI 대모를 함께 만들고 싶어요.")

        self.assertIn("프로젝트", project or "")
        self.assertIn("프로젝트", typo or "")
        self.assertIn("오프라인 AI 데모", demo or "")

    def test_capability_and_first_step_answers_are_short_and_direct(self) -> None:
        prompts = (
            "나와 어떤 일을 함께 할 수 있나요?",
            "당신에게 무슨 일을 부탁할 수 있어?",
            "제가 부탁할 수 있는 일을 어렵지 않게 다른 말로 알려 주세요.",
            "부탁할 수 있는 일은 무엇인가요?",
            "나와 함께 할 수 있는 일은 뭐예요?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                answer = self.fast_path.answer(prompt)
                self.assertIsNotNone(answer)
                self.assertIn("함께", answer or "")
                self.assertLessEqual(
                    sum((answer or "").count(mark) for mark in ".!?"), 3
                )

        followup_prompt = "좋아요. 그럼 첫 단계에서 제가 정할 것 하나만 물어봐 주세요."
        self.assertIsNone(self.fast_path.answer(followup_prompt))

        previous_user = "오프라인 AI 데모를 함께 만들고 싶어요."
        previous_assistant = self.fast_path.answer(previous_user)
        followup = self.fast_path.answer(
            followup_prompt,
            previous_user=previous_user,
            previous_assistant=previous_assistant,
        )
        self.assertEqual(
            followup,
            "좋아요. 이 데모를 가장 먼저 보여주고 싶은 사용자는 누구인가요?",
        )

    def test_general_knowledge_code_and_formal_requests_are_not_intercepted(
        self,
    ) -> None:
        prompts = (
            "파이썬 리스트와 튜플의 차이를 알려 주세요.",
            "프로젝트 코드를 작성해 주세요.",
            "오프라인 AI 데모 구현 계획을 작성해 주세요.",
            "모델 응답 품질을 세 문장으로 설명해 주세요.",
            "대한민국의 수도는 어디인가요?",
            "당신은 정확히 어떤 모델인가요?",
            "안녕하세요. 파이썬 코드를 작성해 주세요.",
            "첫 단계에서 하나만 물어봐 주세요.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIsNone(self.fast_path.answer(prompt))


if __name__ == "__main__":
    unittest.main()
