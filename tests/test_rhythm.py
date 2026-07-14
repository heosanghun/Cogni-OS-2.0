import unittest

from cogni_flow.rhythm import MAX_RHYTHM_HISTORY, RhythmController, SystemMode


class TestRhythmBounds(unittest.TestCase):
    def test_transition_history_is_o1_bounded(self) -> None:
        rhythm = RhythmController()

        for _ in range(MAX_RHYTHM_HISTORY + 10):
            rhythm.enter_evolution(lambda: None)
            rhythm.resume_inference("bounded cycle complete")

        self.assertEqual(rhythm.mode, SystemMode.INFERENCE)
        self.assertEqual(len(rhythm.history), MAX_RHYTHM_HISTORY)
        self.assertEqual(rhythm.history.maxlen, MAX_RHYTHM_HISTORY)
        self.assertEqual(rhythm.history[-1].target, SystemMode.INFERENCE)


if __name__ == "__main__":
    unittest.main()
