from __future__ import annotations

from io import BytesIO
import math
import struct
import unittest
import wave

from scripts.validate_gemma4_local_voice import _normalized, _resample_to_voice_contract


def _wav(sample_rate: int = 22_050) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(2_000 * math.sin(2 * math.pi * 220 * i / sample_rate)))
        for i in range(sample_rate // 10)
    )
    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(frames)
    return output.getvalue()


class TestVoiceRuntimeValidator(unittest.TestCase):
    def test_resampling_produces_the_exact_voice_input_contract(self) -> None:
        converted = _resample_to_voice_contract(_wav())
        with wave.open(BytesIO(converted), "rb") as stream:
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getsampwidth(), 2)
            self.assertEqual(stream.getframerate(), 16_000)
            self.assertEqual(stream.getnframes(), 1_600)

    def test_normalized_similarity_ignores_spacing_and_punctuation(self) -> None:
        self.assertEqual(
            _normalized("안녕하세요. Cogni Board!"), "안녕하세요cogniboard"
        )


if __name__ == "__main__":
    unittest.main()
