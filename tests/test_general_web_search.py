import json
import unittest
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import tempfile
from urllib.parse import parse_qs

from cogni_demo.general_web_search import (
    BRAVE_SEARCH_API_HOST,
    BRAVE_SEARCH_API_PATH,
    MAX_WEB_RESPONSE_BYTES,
    GeneralWebHttpResponse,
    GeneralWebSearchClient,
    GeneralWebSearchConfig,
    GeneralWebSearchError,
    GeneralWebTransportError,
    StandardLibraryGeneralWebTransport,
)
from cogni_demo.workspace_capabilities import (
    VerifiedModelMetadata,
    WorkspaceCapabilityError,
    WorkspaceCapabilityService,
)


TOKEN = "brave-secret-token-that-must-never-be-published"
NOW = datetime(2026, 7, 19, 1, 2, 3, tzinfo=timezone.utc)


def _config(
    *,
    online: bool = True,
    provider: str = "brave",
    allowlist: tuple[str, ...] = (BRAVE_SEARCH_API_HOST,),
    token: str = TOKEN,
    terms_accepted: bool = True,
) -> GeneralWebSearchConfig:
    return GeneralWebSearchConfig(
        online_mode=online,
        provider_id=provider,
        allowlist=allowlist,
        token=token,
        terms_accepted=terms_accepted,
    )


def _json_response(
    payload: object, *, status: int = 200, headers: dict[str, str] | None = None
) -> GeneralWebHttpResponse:
    return GeneralWebHttpResponse(
        status=status,
        headers=headers or {"content-type": "application/json; charset=utf-8"},
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


class FakeTransport:
    def __init__(self, *responses: GeneralWebHttpResponse | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.before_return = None

    def get(self, **kwargs: object) -> GeneralWebHttpResponse:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected general web transport call")
        selected = self.responses.pop(0)
        if isinstance(selected, BaseException):
            raise selected
        if self.before_return is not None:
            self.before_return()
        return selected


def _result_payload() -> dict[str, object]:
    return {
        "type": "search",
        "query": {"original": "deep equilibrium"},
        "web": {
            "results": [
                {
                    "title": "Deep Equilibrium Models",
                    "url": "https://Example.COM:443/paper?id=123",
                    "description": "A fixed-point neural architecture.",
                    "age": "2019-10-01T00:00:00.000Z",
                }
            ]
        },
    }


class TestGeneralWebConfiguration(unittest.TestCase):
    def test_every_operator_gate_is_independent_and_default_is_off(self) -> None:
        self.assertEqual(
            GeneralWebSearchConfig.from_environment({}).state, "disabled_air_gap"
        )
        self.assertEqual(_config(online=False).state, "disabled_air_gap")
        self.assertEqual(_config(provider="").state, "provider_configuration_required")
        self.assertEqual(_config(allowlist=()).state, "allowlist_required")
        self.assertEqual(_config(token="").state, "credentials_required")
        self.assertEqual(
            _config(terms_accepted=False).state, "terms_acceptance_required"
        )
        self.assertEqual(_config().state, "ready_for_session_opt_in")

    def test_environment_contract_and_public_payload_are_secret_free(self) -> None:
        config = GeneralWebSearchConfig.from_environment(
            {
                "COGNI_OS_ONLINE_MODE": "1",
                "COGNI_OS_GENERAL_WEB_PROVIDER": "BRAVE",
                "COGNI_OS_WEB_ALLOWLIST": "example.com,API.SEARCH.BRAVE.COM",
                "COGNI_OS_BRAVE_SEARCH_API_TOKEN": TOKEN,
                "COGNI_OS_BRAVE_SEARCH_TERMS_ACCEPTED": "1",
            }
        )
        payload = config.public_payload()
        serialized = json.dumps(payload, sort_keys=True)
        self.assertTrue(config.enabled)
        self.assertNotIn(TOKEN, serialized)
        self.assertNotIn(TOKEN, repr(config))
        self.assertTrue(payload["official_json_api"])
        self.assertFalse(payload["html_scraping"])
        self.assertFalse(payload["redirects_followed"])
        self.assertTrue(payload["session_online_opt_in_required"])
        self.assertFalse(payload["session_online_opt_in_active"])

    def test_tokens_and_allowlist_entries_cannot_inject_protocol_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "token is malformed"):
            _config(token="secret\r\nX-Injected: yes")
        with self.assertRaisesRegex(ValueError, "invalid hostname"):
            _config(allowlist=("https://api.search.brave.com",))

    def test_disabled_configurations_never_reach_transport(self) -> None:
        cases = (
            (_config(online=False), "WEB_SEARCH_AIR_GAP_BLOCKED"),
            (_config(provider=""), "WEB_SEARCH_PROVIDER_REQUIRED"),
            (_config(allowlist=()), "WEB_SEARCH_HOST_NOT_ALLOWLISTED"),
            (_config(token=""), "WEB_SEARCH_TOKEN_REQUIRED"),
            (_config(terms_accepted=False), "WEB_SEARCH_TERMS_REQUIRED"),
        )
        for config, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                transport = FakeTransport()
                client = GeneralWebSearchClient(config, transport=transport)
                session = client.open_session(online_opt_in=True)
                with self.assertRaises(GeneralWebSearchError) as raised:
                    client.search(session, "deep equilibrium")
                self.assertEqual(raised.exception.code, expected_code)
                self.assertEqual(transport.calls, [])
                self.assertNotIn(TOKEN, str(raised.exception))


class TestGeneralWebSearch(unittest.TestCase):
    def test_normalized_result_has_canonical_citation_and_digests(self) -> None:
        upstream = _result_payload()
        transport = FakeTransport(_json_response(upstream))
        client = GeneralWebSearchClient(
            _config(),
            transport=transport,
            clock=lambda: NOW,
            maximum_retries=0,
        )
        session = client.open_session(online_opt_in=True)

        response = client.search(session, "  deep   equilibrium  ", limit=1)

        self.assertEqual(response.query, "deep equilibrium")
        self.assertEqual(response.retrieved_at, NOW.isoformat())
        self.assertEqual(len(response.results), 1)
        result = response.results[0]
        self.assertEqual(
            result.provenance.canonical_url, "https://example.com/paper?id=123"
        )
        self.assertEqual(
            result.provenance.query_sha256,
            sha256(b"deep equilibrium").hexdigest(),
        )
        raw_record = upstream["web"]["results"][0]
        self.assertEqual(
            result.provenance.source_record_sha256,
            sha256(
                json.dumps(
                    raw_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(
            result.as_payload()["citation_links"][0]["url"],
            "https://example.com/paper?id=123",
        )

        call = transport.calls[0]
        self.assertEqual(call["host"], BRAVE_SEARCH_API_HOST)
        self.assertEqual(call["path"], BRAVE_SEARCH_API_PATH)
        self.assertEqual(call["headers"]["X-Subscription-Token"], TOKEN)
        query = parse_qs(call["query_string"])
        self.assertEqual(query["q"], ["deep equilibrium"])
        self.assertEqual(query["count"], ["1"])
        self.assertEqual(query["safesearch"], ["strict"])

    def test_session_opt_in_is_required_even_when_provider_is_ready(self) -> None:
        transport = FakeTransport()
        client = GeneralWebSearchClient(_config(), transport=transport)
        session = client.open_session(online_opt_in=False)

        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(session, "bounded search")

        self.assertEqual(raised.exception.code, "WEB_SEARCH_SESSION_OPT_IN_REQUIRED")
        self.assertEqual(transport.calls, [])

    def test_sessions_are_bound_to_the_issuing_client(self) -> None:
        first = GeneralWebSearchClient(_config(), transport=FakeTransport())
        second_transport = FakeTransport()
        second = GeneralWebSearchClient(_config(), transport=second_transport)

        with self.assertRaises(GeneralWebSearchError) as raised:
            second.search(first.open_session(online_opt_in=True), "bounded search")

        self.assertEqual(raised.exception.code, "WEB_SEARCH_SESSION_INVALID")
        self.assertEqual(second_transport.calls, [])

    def test_cancel_and_revoke_prevent_calls_and_late_publication(self) -> None:
        for terminal, expected_code in (
            ("cancel", "WEB_SEARCH_CANCELLED"),
            ("revoke", "WEB_SEARCH_REVOKED"),
        ):
            with self.subTest(terminal=terminal):
                transport = FakeTransport(_json_response(_result_payload()))
                client = GeneralWebSearchClient(
                    _config(), transport=transport, maximum_retries=0
                )
                session = client.open_session(online_opt_in=True)
                getattr(session, terminal)()
                with self.assertRaises(GeneralWebSearchError) as raised:
                    client.search(session, "bounded search")
                self.assertEqual(raised.exception.code, expected_code)
                self.assertEqual(transport.calls, [])

        transport = FakeTransport(_json_response(_result_payload()))
        client = GeneralWebSearchClient(
            _config(), transport=transport, maximum_retries=0
        )
        session = client.open_session(online_opt_in=True)
        transport.before_return = session.cancel
        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(session, "bounded search")
        self.assertEqual(raised.exception.code, "WEB_SEARCH_CANCELLED")
        self.assertEqual(len(transport.calls), 1)

        transport = FakeTransport(_json_response(_result_payload()))
        client = GeneralWebSearchClient(
            _config(), transport=transport, maximum_retries=0
        )
        session = client.open_session(online_opt_in=True)
        transport.before_return = client.revoke
        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(session, "bounded search")
        self.assertEqual(raised.exception.code, "WEB_SEARCH_REVOKED")
        self.assertEqual(client.capability_payload()["state"], "revoked")

    def test_query_result_and_response_bounds_fail_before_publication(self) -> None:
        for query, limit, code in (
            ("", 1, "WEB_SEARCH_INVALID_QUERY"),
            ("x" * 513, 1, "WEB_SEARCH_INVALID_QUERY"),
            ("safe", 0, "WEB_SEARCH_INVALID_LIMIT"),
            ("safe", 11, "WEB_SEARCH_INVALID_LIMIT"),
        ):
            transport = FakeTransport()
            client = GeneralWebSearchClient(_config(), transport=transport)
            with self.assertRaises(GeneralWebSearchError) as raised:
                client.search(
                    client.open_session(online_opt_in=True), query, limit=limit
                )
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(transport.calls, [])

        invalid_responses = (
            GeneralWebHttpResponse(
                200, {"content-type": "text/html"}, b"<html>not json</html>"
            ),
            GeneralWebHttpResponse(
                200,
                {"content-type": "application/json"},
                b"x" * (MAX_WEB_RESPONSE_BYTES + 1),
            ),
            _json_response(
                {
                    "web": {
                        "results": [
                            {
                                "title": "unsafe",
                                "url": "http://example.com/not-https",
                            }
                        ]
                    }
                }
            ),
            _json_response(
                {
                    "web": {
                        "results": [{"title": "private", "url": "https://127.0.0.1/"}]
                    }
                }
            ),
        )
        for response in invalid_responses:
            client = GeneralWebSearchClient(
                _config(),
                transport=FakeTransport(response),
                maximum_retries=0,
            )
            with self.assertRaises(GeneralWebSearchError) as raised:
                client.search(
                    client.open_session(online_opt_in=True), "bounded search", limit=1
                )
            self.assertIn(
                raised.exception.code,
                {"WEB_SEARCH_RESPONSE_TOO_LARGE", "WEB_SEARCH_RESPONSE_INVALID"},
            )

    def test_retries_are_bounded_and_abort_aware(self) -> None:
        transport = FakeTransport(
            _json_response(
                {"error": "rate limited"},
                status=429,
                headers={
                    "content-type": "application/json",
                    "retry-after": "9999",
                },
            ),
            _json_response({"web": {"results": []}}),
        )
        delays: list[float] = []

        def waiter(delay: float, cancelled) -> bool:
            delays.append(delay)
            return not cancelled()

        client = GeneralWebSearchClient(
            _config(),
            transport=transport,
            retry_waiter=waiter,
            maximum_retries=1,
        )
        response = client.search(
            client.open_session(online_opt_in=True), "bounded search"
        )
        self.assertEqual(response.external_calls, 2)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(delays, [0.5])

        session = client.open_session(online_opt_in=True)
        transport.responses = [
            GeneralWebTransportError(f"private {TOKEN}"),
            _json_response({"web": {"results": []}}),
        ]

        def cancelling_waiter(delay: float, cancelled) -> bool:
            session.cancel()
            return not cancelled()

        client.retry_waiter = cancelling_waiter
        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(session, "second bounded search")
        self.assertEqual(raised.exception.code, "WEB_SEARCH_CANCELLED")
        self.assertNotIn(TOKEN, str(raised.exception))

    def test_redirect_and_upstream_secrets_are_never_returned(self) -> None:
        for response, expected_code in (
            (
                GeneralWebHttpResponse(
                    302,
                    {"location": "https://evil.example/collect"},
                    b"redirect",
                ),
                "WEB_SEARCH_REDIRECT_BLOCKED",
            ),
            (
                GeneralWebHttpResponse(
                    403,
                    {"content-type": "application/json"},
                    f'{{"token":"{TOKEN}","private":"path"}}'.encode(),
                ),
                "WEB_SEARCH_AUTH_FAILED",
            ),
        ):
            client = GeneralWebSearchClient(
                _config(), transport=FakeTransport(response), maximum_retries=0
            )
            with self.assertRaises(GeneralWebSearchError) as raised:
                client.search(client.open_session(online_opt_in=True), "bounded search")
            self.assertEqual(raised.exception.code, expected_code)
            self.assertNotIn(TOKEN, str(raised.exception))
            self.assertNotIn("private", str(raised.exception))

        reflected = _json_response(
            {
                "web": {
                    "results": [
                        {
                            "title": f"reflected {TOKEN}",
                            "url": "https://example.com/",
                        }
                    ]
                }
            }
        )
        client = GeneralWebSearchClient(
            _config(), transport=FakeTransport(reflected), maximum_retries=0
        )
        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(client.open_session(online_opt_in=True), "bounded search")
        self.assertEqual(raised.exception.code, "WEB_SEARCH_RESPONSE_INVALID")
        self.assertNotIn(TOKEN, str(raised.exception))

    def test_nonstandard_json_and_unexpected_transport_fail_redacted(self) -> None:
        nonstandard = GeneralWebHttpResponse(
            200,
            {"content-type": "application/json"},
            b'{"web":{"results":[]},"score":NaN}',
        )
        for injected in (
            nonstandard,
            RuntimeError(f"transport leaked {TOKEN}"),
        ):
            client = GeneralWebSearchClient(
                _config(), transport=FakeTransport(injected), maximum_retries=0
            )
            with self.assertRaises(GeneralWebSearchError) as raised:
                client.search(client.open_session(online_opt_in=True), "bounded search")
            self.assertIn(
                raised.exception.code,
                {"WEB_SEARCH_RESPONSE_INVALID", "WEB_SEARCH_NETWORK_ERROR"},
            )
            self.assertNotIn(TOKEN, str(raised.exception))

        client = GeneralWebSearchClient(_config(), transport=FakeTransport())
        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(client.open_session(online_opt_in=True), f"query {TOKEN}")
        self.assertEqual(raised.exception.code, "WEB_SEARCH_INVALID_QUERY")
        self.assertNotIn(TOKEN, str(raised.exception))

    def test_standard_transport_rejects_nonfixed_destinations_preflight(self) -> None:
        transport = StandardLibraryGeneralWebTransport()
        valid_query = "q=bounded&count=1&safesearch=strict"
        for host, path, query in (
            ("example.com", BRAVE_SEARCH_API_PATH, valid_query),
            (BRAVE_SEARCH_API_HOST, "/other", valid_query),
            (BRAVE_SEARCH_API_HOST, BRAVE_SEARCH_API_PATH, "q=bounded&count=1"),
            (
                BRAVE_SEARCH_API_HOST,
                BRAVE_SEARCH_API_PATH,
                "q=bounded&count=1&safesearch=off",
            ),
        ):
            with self.assertRaises(GeneralWebTransportError):
                transport.get(
                    host=host,
                    path=path,
                    query_string=query,
                    headers={},
                    timeout_seconds=1.0,
                    maximum_response_bytes=1024,
                    cancelled=lambda: False,
                )


class TestGeneralWebWorkspaceIntegration(unittest.TestCase):
    @staticmethod
    def _model() -> VerifiedModelMetadata:
        return VerifiedModelMetadata(
            model_id="verified",
            label="Gemma 4 e4b",
            architecture="Gemma4ForConditionalGeneration",
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            checkpoint_modalities=("text",),
            runtime_input_modalities=("text",),
        )

    def test_capability_and_one_shot_explicit_opt_in_are_wired(self) -> None:
        transport = FakeTransport(_json_response(_result_payload()))
        client = GeneralWebSearchClient(
            _config(), transport=transport, maximum_retries=0, clock=lambda: NOW
        )
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(
                Path(temporary), self._model(), general_web_client=client
            )
            capability = service.capability_payload()["web_search"][
                "general_web_connector"
            ]
            self.assertEqual(capability["state"], "ready_for_session_opt_in")
            self.assertTrue(capability["executor_implemented"])
            self.assertFalse(capability["answer_integration"])
            self.assertFalse(capability["rag_index_integration"])

            with self.assertRaises(WorkspaceCapabilityError) as raised:
                service.search_web(
                    "must remain offline",
                    limit=1,
                    session_online_opt_in=False,
                )
            self.assertEqual(
                raised.exception.code, "WEB_SEARCH_SESSION_OPT_IN_REQUIRED"
            )
            self.assertEqual(transport.calls, [])

            response = service.search_web(
                "deep equilibrium", limit=1, session_online_opt_in=True
            )
            self.assertEqual(response["count"], 1)
            self.assertEqual(len(transport.calls), 1)

    def test_service_defaults_never_advertise_general_network_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(Path(temporary), self._model())
            connector = service.capability_payload()["web_search"][
                "general_web_connector"
            ]
            self.assertEqual(connector["state"], "disabled_air_gap")
            self.assertEqual(connector["external_calls"], 0)
            with self.assertRaises(WorkspaceCapabilityError) as raised:
                service.search_web("must remain offline", session_online_opt_in=True)
            self.assertEqual(raised.exception.code, "WEB_SEARCH_AIR_GAP_BLOCKED")
            self.assertEqual(
                service.general_web_client.external_calls,
                0,
            )


if __name__ == "__main__":
    unittest.main()
