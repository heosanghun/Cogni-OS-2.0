import unittest

from cogni_agent.conversation import BoundedConversationStore, ConversationError


class TestBoundedConversationStore(unittest.TestCase):
    def test_transactional_multi_turn_history(self) -> None:
        store = BoundedConversationStore(
            max_sessions=2,
            max_turns=4,
            max_chars=40,
            max_message_chars=10,
        )
        first = store.begin_user_turn("alpha", "question-1")
        reply = store.commit_assistant_turn("alpha", first, "answer-1")
        second = store.begin_user_turn("alpha", "question-2")
        store.commit_assistant_turn("alpha", second, "answer-2")

        snapshot = store.snapshot("alpha")
        self.assertEqual(reply.sequence, 2)
        self.assertEqual(
            [(turn.role, turn.text) for turn in snapshot.turns],
            [
                ("user", "question-1"),
                ("assistant", "answer-1"),
                ("user", "question-2"),
                ("assistant", "answer-2"),
            ],
        )
        self.assertEqual(snapshot.as_messages()[0]["role"], "user")

    def test_turn_and_session_lru_bounds_preserve_complete_exchanges(self) -> None:
        store = BoundedConversationStore(
            max_sessions=2,
            max_turns=3,
            max_chars=24,
            max_message_chars=6,
        )
        for index in range(3):
            sequence = store.begin_user_turn("alpha", f"u{index}")
            store.commit_assistant_turn("alpha", sequence, f"a{index}")
        snapshot = store.snapshot("alpha")
        self.assertEqual(
            [(turn.role, turn.text) for turn in snapshot.turns],
            [("user", "u2"), ("assistant", "a2")],
        )

        sequence = store.begin_user_turn("beta", "b")
        store.commit_assistant_turn("beta", sequence, "B")
        # Touch alpha, then create gamma; beta is the bounded LRU victim.
        store.snapshot("alpha")
        store.begin_user_turn("gamma", "g")
        self.assertEqual(store.session_ids, ("alpha", "gamma"))
        with self.assertRaises(ConversationError):
            store.snapshot("beta")

    def test_cancellation_removes_only_owned_pending_turn(self) -> None:
        store = BoundedConversationStore(
            max_chars=40,
            max_message_chars=10,
        )
        sequence = store.begin_user_turn("session", "pending")
        self.assertFalse(store.abort_user_turn("session", sequence + 1))
        self.assertTrue(store.abort_user_turn("session", sequence))
        self.assertEqual(store.snapshot("session").turns, ())

    def test_order_identity_and_message_bounds_fail_closed(self) -> None:
        store = BoundedConversationStore(
            max_chars=20,
            max_message_chars=10,
        )
        sequence = store.begin_user_turn("valid-1", "hello")
        with self.assertRaises(ConversationError):
            store.begin_user_turn("valid-1", "overlap")
        with self.assertRaises(ConversationError):
            store.commit_assistant_turn("valid-1", sequence + 1, "wrong")
        with self.assertRaises(ConversationError):
            store.commit_assistant_turn("valid-1", sequence, "x" * 11)
        with self.assertRaises(ConversationError):
            store.begin_user_turn("../escape", "bad")
        with self.assertRaises(ConversationError):
            store.begin_user_turn("other", "nul\x00byte")


if __name__ == "__main__":
    unittest.main()
