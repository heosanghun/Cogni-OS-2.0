import unittest

import torch

from cogni_agent.protocol import (
    DIGEST_BYTES,
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_LKG_LENGTH,
    FINISH_LKG_STOP,
    FINISH_NONE,
    FINISH_STOP,
    HARD_MAX_INPUT_TOKENS,
    STATUS_CANCELLED,
    STATUS_OK,
    TensorProtocolError,
    make_generate_request,
    make_response,
    make_stop_request,
    parse_request,
    parse_response,
)


class TestAgentTensorProtocol(unittest.TestCase):
    def test_generation_request_is_exactly_four_bounded_cpu_tensors(self) -> None:
        artifact = torch.arange(DIGEST_BYTES, dtype=torch.int64)
        session = torch.arange(DIGEST_BYTES - 1, -1, -1, dtype=torch.int64)
        message = make_generate_request(
            7,
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
            torch.ones((1, 3), dtype=torch.int64),
            max_new_tokens=12,
            stop_token_ids=torch.tensor([2], dtype=torch.int64),
            job_id=7001,
            lease_epoch=9,
            request_deadline_ns=1_000_000,
            lease_deadline_ns=2_000_000,
            artifact_digest=artifact,
            session_digest=session,
        )
        self.assertEqual(len(message), 4)
        self.assertTrue(all(item.device.type == "cpu" for item in message))
        self.assertTrue(all(item.dtype == torch.int64 for item in message))
        request = parse_request(message)
        self.assertIsNotNone(request)
        self.assertEqual(request.request_id, 7)
        self.assertEqual(request.job_id, 7001)
        self.assertEqual(request.lease_epoch, 9)
        self.assertEqual(request.request_deadline_ns, 1_000_000)
        self.assertTrue(torch.equal(request.artifact_digest, artifact))
        self.assertTrue(torch.equal(request.session_digest, session))
        self.assertEqual(request.max_new_tokens, 12)
        self.assertTrue(torch.equal(request.input_ids, torch.tensor([[1, 2, 3]])))
        self.assertIsNone(parse_request(make_stop_request()))

    def test_request_schema_rejects_wrong_dtype_shape_mask_and_size(self) -> None:
        with self.assertRaises(TensorProtocolError):
            make_generate_request(
                1,
                torch.tensor([[1.0]]),
                None,
                max_new_tokens=1,
            )
        with self.assertRaises(TensorProtocolError):
            make_generate_request(
                1,
                torch.ones((2, 1), dtype=torch.int64),
                None,
                max_new_tokens=1,
            )
        with self.assertRaises(TensorProtocolError):
            make_generate_request(
                1,
                torch.ones((1, 2), dtype=torch.int64),
                torch.tensor([[1, 2]], dtype=torch.int64),
                max_new_tokens=1,
            )
        with self.assertRaises(TensorProtocolError):
            make_generate_request(
                1,
                torch.ones((1, HARD_MAX_INPUT_TOKENS + 1), dtype=torch.int64),
                None,
                max_new_tokens=1,
            )

    def test_streaming_and_terminal_response_frames_are_closed(self) -> None:
        chunk = parse_response(
            make_response(
                9,
                STATUS_OK,
                torch.tensor([4, 5], dtype=torch.int64),
                generated_total=2,
                final=False,
            )
        )
        self.assertFalse(chunk.final)
        self.assertEqual(chunk.token_ids.tolist(), [4, 5])
        self.assertEqual(chunk.finish_reason, FINISH_NONE)
        terminal = parse_response(
            make_response(
                9,
                STATUS_CANCELLED,
                generated_total=2,
                final=True,
            )
        )
        self.assertTrue(terminal.final)
        self.assertEqual(terminal.status, STATUS_CANCELLED)
        self.assertEqual(terminal.finish_reason, FINISH_CANCELLED)
        with self.assertRaises(TensorProtocolError):
            make_response(9, STATUS_CANCELLED, final=False)

    def test_terminal_stop_length_and_cancel_reasons_round_trip(self) -> None:
        cases = (
            (STATUS_OK, FINISH_STOP),
            (STATUS_OK, FINISH_LENGTH),
            (STATUS_OK, FINISH_LKG_STOP),
            (STATUS_OK, FINISH_LKG_LENGTH),
            (STATUS_CANCELLED, FINISH_CANCELLED),
        )
        for status, reason in cases:
            with self.subTest(status=status, reason=reason):
                frame = parse_response(
                    make_response(
                        11,
                        status,
                        torch.tensor([7], dtype=torch.int64),
                        generated_total=1,
                        final=True,
                        finish_reason=reason,
                    )
                )
                self.assertTrue(frame.final)
                self.assertEqual(frame.status, status)
                self.assertEqual(frame.finish_reason, reason)

    def test_response_echoes_complete_request_authority(self) -> None:
        artifact = torch.full((DIGEST_BYTES,), 17, dtype=torch.int64)
        session = torch.full((DIGEST_BYTES,), 23, dtype=torch.int64)
        frame = parse_response(
            make_response(
                19,
                STATUS_OK,
                torch.tensor([3]),
                generated_total=1,
                final=True,
                finish_reason=FINISH_STOP,
                job_id=91,
                lease_epoch=4,
                request_deadline_ns=8_000,
                lease_deadline_ns=9_000,
                artifact_digest=artifact,
                session_digest=session,
            )
        )
        self.assertEqual(frame.job_id, 91)
        self.assertEqual(frame.lease_epoch, 4)
        self.assertEqual(frame.request_deadline_ns, 8_000)
        self.assertEqual(frame.lease_deadline_ns, 9_000)
        self.assertTrue(torch.equal(frame.artifact_digest, artifact))
        self.assertTrue(torch.equal(frame.session_digest, session))

    def test_finish_reason_status_pairs_are_fail_closed(self) -> None:
        invalid = (
            (STATUS_OK, True, FINISH_CANCELLED),
            (STATUS_CANCELLED, True, FINISH_STOP),
            (STATUS_OK, False, FINISH_LENGTH),
        )
        for status, final, reason in invalid:
            with self.subTest(status=status, final=final, reason=reason):
                with self.assertRaises(TensorProtocolError):
                    make_response(
                        3,
                        status,
                        generated_total=0,
                        final=final,
                        finish_reason=reason,
                    )


if __name__ == "__main__":
    unittest.main()
