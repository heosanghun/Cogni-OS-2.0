from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import math
import unittest

from cogni_demo.protocol import (
    EVENT_SENTINEL,
    EventEmitter,
    MAX_TRANSITION_RESIDUAL,
    ProtocolError,
    parse_event_line,
    validate_terminal_metrics,
)


def valid_metrics() -> dict:
    return {
        "verified_files": 6,
        "model_class": "Gemma4ForConditionalGeneration",
        "hidden_size": 2560,
        "load_seconds": 19.789,
        "inference_seconds": 3.729,
        "requested_depth": 100,
        "reached_depth": 100,
        "nodes_used": 301,
        "node_capacity": 301,
        "search_allocated_bytes": 14994009,
        "transition_converged": True,
        "transition_residual": 0.00390625,
        "transition_used_fallback": False,
        "cts_protocol_version": "SearchRequestV2",
        "safe_for_decode": True,
        "unsafe_silent_fallbacks": 0,
        "linear_solve_fallbacks": 0,
        "solver_rank": 16,
        "solver_history_peak": 16,
        "solver_failures": 0,
        "failed_edges": 0,
        "q_zero_backups": 0,
        "mac_budget": 1000,
        "mac_reserved": 900,
        "act_applied": 301,
        "trace_digest": "a" * 64,
        "causal_bridge_answer_bearing": True,
        "causal_bridge_bias_nonzero": True,
        "causal_bridge_bias_max": 0.04980469,
        "conditioned_generated_tokens": 1,
        "peak_vram_gib": 14.856,
        "vram_limit_gib": 16.7,
        "finite": True,
        "device": "RTX 5090 Laptop GPU",
    }


class TestDemoWorkerProtocol(unittest.TestCase):
    def test_ordinary_legacy_output_is_ignored(self) -> None:
        self.assertIsNone(parse_event_line("peak_vram_gib=14.8560\n"))

    def test_emitter_is_opt_in_and_round_trips_typed_result(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            EventEmitter(False).phase("verifying", 5)
            emitter = EventEmitter(True)
            emitter.phase("verifying", 5)
            emitter.result(valid_metrics())
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith(EVENT_SENTINEL))
        phase = parse_event_line(lines[0])
        result = parse_event_line(lines[1])
        self.assertEqual(phase.stage, "verifying")
        self.assertEqual(result.metrics["requested_depth"], 100)

    def test_terminal_gate_rejects_bad_types_nan_depth_and_vram(self) -> None:
        cases = []
        bad = valid_metrics()
        bad["requested_depth"] = True
        cases.append(bad)
        bad = valid_metrics()
        bad["transition_residual"] = math.nan
        cases.append(bad)
        bad = valid_metrics()
        bad["reached_depth"] = 99
        cases.append(bad)
        bad = valid_metrics()
        bad["peak_vram_gib"] = 16.8
        cases.append(bad)
        bad = valid_metrics()
        bad["causal_bridge_answer_bearing"] = False
        cases.append(bad)
        bad = valid_metrics()
        bad["causal_bridge_bias_max"] = 0.1001
        cases.append(bad)
        for metrics in cases:
            with self.subTest(metrics=metrics):
                with self.assertRaises(ProtocolError):
                    validate_terminal_metrics(metrics)

    def test_transition_residual_certification_boundary(self) -> None:
        for residual in (0, MAX_TRANSITION_RESIDUAL):
            with self.subTest(residual=residual):
                metrics = valid_metrics()
                metrics["transition_residual"] = residual
                self.assertEqual(
                    validate_terminal_metrics(metrics)["transition_residual"],
                    float(residual),
                )

        rejected = (
            math.nextafter(MAX_TRANSITION_RESIDUAL, math.inf),
            math.nan,
            math.inf,
            -math.inf,
            -math.ulp(0.0),
        )
        for residual in rejected:
            with self.subTest(residual=residual):
                metrics = valid_metrics()
                metrics["transition_residual"] = residual
                with self.assertRaises(ProtocolError):
                    validate_terminal_metrics(metrics)

    def test_unknown_fields_and_malformed_sentinel_fail_closed(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_event_line(EVENT_SENTINEL + "{")
        with self.assertRaises(ProtocolError):
            parse_event_line(
                EVENT_SENTINEL + '{"v":2,"seq":1,"kind":"phase","stage":"verifying",'
                '"progress":5,"extra":true}'
            )
        with self.assertRaises(ProtocolError):
            parse_event_line(
                EVENT_SENTINEL + '{"v":2,"v":2,"seq":1,"kind":"phase",'
                '"stage":"verifying","progress":5}'
            )


if __name__ == "__main__":
    unittest.main()
