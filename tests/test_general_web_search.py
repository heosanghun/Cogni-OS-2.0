import json
import ssl
import unittest
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import tempfile
from threading import Event, Thread
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
    _ResolverGate,
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
        self.before_dispatch = None
        self.before_return = None

    def get(self, **kwargs: object) -> GeneralWebHttpResponse:
        if self.before_dispatch is not None:
            self.before_dispatch()
        authorize = kwargs.get("authorize_dispatch")
        if not callable(authorize) or not authorize(
            lambda: self.calls.append(dict(kwargs))
        ):
            raise GeneralWebTransportError("dispatch rejected")
        if not self.responses:
            raise AssertionError("unexpected general web transport call")
        selected = self.responses.pop(0)
        if isinstance(selected, BaseException):
            raise selected
        if self.before_return is not None:
            self.before_return()
        return selected


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _FakeProviderResponse:
    status = 200

    def getheader(self, name: str):
        return "0" if name == "Content-Length" else None

    def getheaders(self):
        return (("Content-Type", "application/json"),)

    def read(self, _size: int) -> bytes:
        return b""


class _FakePinnedConnection:
    def __init__(self) -> None:
        self.sock = _FakeSocket()
        self.timeout = 0.0
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False

    def request(self, method: str, target: str, *, headers: dict[str, str]) -> None:
        self.requests.append((method, target, headers))

    def getresponse(self) -> _FakeProviderResponse:
        return _FakeProviderResponse()

    def close(self) -> None:
        self.closed = True


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
        with self.assertRaisesRegex(ValueError, "token is malformed"):
            _config(token="too-short")
        with self.assertRaisesRegex(ValueError, "token is malformed"):
            _config(token="x" * 20 + "\t")
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

    def test_cancel_dispatch_and_publication_share_one_epoch_fence(self) -> None:
        transport = FakeTransport(_json_response(_result_payload()))
        client = GeneralWebSearchClient(
            _config(), transport=transport, maximum_retries=0
        )
        session = client.open_session(online_opt_in=True)
        transport.before_dispatch = session.cancel
        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(session, "cancel wins before wire")
        self.assertEqual(raised.exception.code, "WEB_SEARCH_CANCELLED")
        self.assertEqual(transport.calls, [])
        self.assertEqual(client.external_calls, 0)

        transport = FakeTransport(_json_response(_result_payload()))
        client = GeneralWebSearchClient(
            _config(), transport=transport, maximum_retries=0
        )
        session = client.open_session(online_opt_in=True)
        response = client.search(session, "publication wins", limit=1)
        self.assertEqual(len(response.results), 1)
        self.assertFalse(session.cancel())
        self.assertEqual(session.state, "published")
        with self.assertRaises(GeneralWebSearchError) as consumed:
            client.search(session, "cannot reuse")
        self.assertEqual(consumed.exception.code, "WEB_SEARCH_SESSION_CONSUMED")

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

        escaped_token = "".join(f"\\u{ord(character):04x}" for character in TOKEN)
        escaped_body = (
            '{"web":{"results":[{"title":"'
            + escaped_token
            + '","url":"https://example.com/"}]}}'
        ).encode("ascii")
        client = GeneralWebSearchClient(
            _config(),
            transport=FakeTransport(
                GeneralWebHttpResponse(
                    200, {"content-type": "application/json"}, escaped_body
                )
            ),
            maximum_retries=0,
        )
        with self.assertRaises(GeneralWebSearchError) as escaped:
            client.search(client.open_session(online_opt_in=True), "bounded search")
        self.assertEqual(escaped.exception.code, "WEB_SEARCH_RESPONSE_INVALID")
        self.assertNotIn(TOKEN, escaped_body.decode("ascii"))

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
            self.assertIsNone(raised.exception.__cause__)

        client = GeneralWebSearchClient(_config(), transport=FakeTransport())
        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(client.open_session(online_opt_in=True), f"query {TOKEN}")
        self.assertEqual(raised.exception.code, "WEB_SEARCH_INVALID_QUERY")
        self.assertNotIn(TOKEN, str(raised.exception))

    def test_total_wall_deadline_covers_retries_and_publication(self) -> None:
        now = [10.0]
        transport = FakeTransport(_json_response({"web": {"results": []}}))
        transport.before_return = lambda: now.__setitem__(0, 11.1)
        client = GeneralWebSearchClient(
            _config(),
            transport=transport,
            timeout_seconds=1.0,
            maximum_retries=1,
            monotonic_clock=lambda: now[0],
        )
        with self.assertRaises(GeneralWebSearchError) as raised:
            client.search(client.open_session(online_opt_in=True), "bounded search")
        self.assertEqual(raised.exception.code, "WEB_SEARCH_TIMEOUT")
        self.assertEqual(len(transport.calls), 1)

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

    def test_standard_transport_rejects_nonpublic_dns_and_pins_public_tls_peer(self) -> None:
        valid_query = "q=bounded&count=1&safesearch=strict"
        for addresses in (
            ("127.0.0.1",),
            ("169.254.10.20",),
            ("10.0.0.7",),
            ("fc00::1",),
            ("93.184.216.34", "127.0.0.1"),
        ):
            connections: list[object] = []

            def forbidden_factory(*args):
                connections.append(args)
                raise AssertionError("non-public DNS must fail before connect")

            transport = StandardLibraryGeneralWebTransport(
                resolver=lambda _host, _port, answer=addresses: answer,
                connection_factory=forbidden_factory,
            )
            with self.subTest(addresses=addresses), self.assertRaises(
                GeneralWebTransportError
            ):
                transport.get(
                    host=BRAVE_SEARCH_API_HOST,
                    path=BRAVE_SEARCH_API_PATH,
                    query_string=valid_query,
                    headers={"Accept": "application/json"},
                    timeout_seconds=1.0,
                    maximum_response_bytes=1024,
                    cancelled=lambda: False,
                    authorize_dispatch=lambda dispatch: (dispatch() or True),
                )
            self.assertEqual(connections, [])

        captured: list[tuple[str, str, float, ssl.SSLContext]] = []
        connection = _FakePinnedConnection()

        def factory(host, pinned_ip, timeout, context):
            captured.append((host, pinned_ip, timeout, context))
            return connection

        transport = StandardLibraryGeneralWebTransport(
            resolver=lambda _host, _port: ("93.184.216.34",),
            connection_factory=factory,
        )
        response = transport.get(
            host=BRAVE_SEARCH_API_HOST,
            path=BRAVE_SEARCH_API_PATH,
            query_string=valid_query,
            headers={"Accept": "application/json"},
            timeout_seconds=1.0,
            maximum_response_bytes=1024,
            cancelled=lambda: False,
            authorize_dispatch=lambda dispatch: (dispatch() or True),
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(captured[0][0], BRAVE_SEARCH_API_HOST)
        self.assertEqual(captured[0][1], "93.184.216.34")
        self.assertTrue(captured[0][3].check_hostname)
        self.assertEqual(captured[0][3].verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(connection.requests[0][0], "GET")
        self.assertTrue(connection.requests[0][1].startswith(BRAVE_SEARCH_API_PATH))
        self.assertTrue(connection.closed)

    def test_resolver_gate_bounds_stalled_workers_and_remains_poisoned(self) -> None:
        valid_query = "q=bounded&count=1&safesearch=strict"
        gate = _ResolverGate()
        entered = Event()
        release = Event()
        resolver_exited = Event()
        resolver_calls: list[int] = []

        def blocking_resolver(_host: str, _port: int) -> tuple[str, ...]:
            resolver_calls.append(1)
            entered.set()
            release.wait(timeout=2.0)
            resolver_exited.set()
            return ("93.184.216.34",)

        def forbidden_factory(*_args: object) -> object:
            raise AssertionError("a timed-out resolver must never connect")

        transport = StandardLibraryGeneralWebTransport(
            resolver=blocking_resolver,
            connection_factory=forbidden_factory,
            resolver_gate=gate,
        )
        failures: list[BaseException] = []

        def first_request() -> None:
            try:
                transport.get(
                    host=BRAVE_SEARCH_API_HOST,
                    path=BRAVE_SEARCH_API_PATH,
                    query_string=valid_query,
                    headers={"Accept": "application/json"},
                    timeout_seconds=0.15,
                    maximum_response_bytes=1024,
                    cancelled=lambda: False,
                    authorize_dispatch=lambda dispatch: dispatch() or True,
                )
            except BaseException as error:
                failures.append(error)

        request_thread = Thread(target=first_request)
        request_thread.start()
        try:
            self.assertTrue(entered.wait(timeout=1.0))
            # While the first getaddrinfo-equivalent call is wedged, another
            # request must fail without creating a second resolver thread.
            with self.assertRaises(GeneralWebTransportError):
                transport.get(
                    host=BRAVE_SEARCH_API_HOST,
                    path=BRAVE_SEARCH_API_PATH,
                    query_string=valid_query,
                    headers={"Accept": "application/json"},
                    timeout_seconds=0.05,
                    maximum_response_bytes=1024,
                    cancelled=lambda: False,
                    authorize_dispatch=lambda dispatch: dispatch() or True,
                )
            self.assertEqual(len(resolver_calls), 1)

            request_thread.join(timeout=1.0)
            self.assertFalse(request_thread.is_alive())
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], GeneralWebTransportError)

            # The timed-out active lease poisons the process gate.  It remains
            # fail-closed even after the uninterruptible worker eventually exits.
            with self.assertRaises(GeneralWebTransportError):
                transport.get(
                    host=BRAVE_SEARCH_API_HOST,
                    path=BRAVE_SEARCH_API_PATH,
                    query_string=valid_query,
                    headers={"Accept": "application/json"},
                    timeout_seconds=0.05,
                    maximum_response_bytes=1024,
                    cancelled=lambda: False,
                )
            self.assertEqual(len(resolver_calls), 1)
        finally:
            release.set()
            request_thread.join(timeout=1.0)

        self.assertTrue(resolver_exited.wait(timeout=1.0))
        with self.assertRaises(GeneralWebTransportError):
            transport.get(
                host=BRAVE_SEARCH_API_HOST,
                path=BRAVE_SEARCH_API_PATH,
                query_string=valid_query,
                headers={"Accept": "application/json"},
                timeout_seconds=0.05,
                maximum_response_bytes=1024,
                cancelled=lambda: False,
            )
        self.assertEqual(len(resolver_calls), 1)


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

    def test_inflight_registry_cancel_and_shutdown_revoke_are_fail_closed(self) -> None:
        entered = Event()
        release = Event()
        transport = FakeTransport(_json_response(_result_payload()))

        def block_return() -> None:
            entered.set()
            self.assertTrue(release.wait(2.0))

        transport.before_return = block_return
        client = GeneralWebSearchClient(
            _config(), transport=transport, maximum_retries=0
        )
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(
                Path(temporary), self._model(), general_web_client=client
            )
            captured: list[BaseException] = []

            def search() -> None:
                try:
                    service.search_web(
                        "bounded cancel",
                        session_online_opt_in=True,
                        request_id="b" * 32,
                    )
                except BaseException as error:
                    captured.append(error)

            worker = Thread(target=search)
            worker.start()
            self.assertTrue(entered.wait(1.0))
            cancellation = service.cancel_web_search("b" * 32)
            self.assertTrue(cancellation["cancelled"])
            release.set()
            worker.join(2.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(captured), 1)
            self.assertIsInstance(captured[0], WorkspaceCapabilityError)
            self.assertEqual(captured[0].code, "WEB_SEARCH_CANCELLED")
            self.assertIsNone(captured[0].__cause__)
            self.assertEqual(
                service.cancel_web_search("b" * 32)["state"], "not_found"
            )

            service.shutdown()
            self.assertEqual(
                service.general_web_client.capability_payload()["state"], "revoked"
            )
            with self.assertRaises(WorkspaceCapabilityError) as invalid:
                service.cancel_web_search("not-an-id")
            self.assertEqual(invalid.exception.code, "INVALID_WEB_SEARCH_REQUEST_ID")


if __name__ == "__main__":
    unittest.main()
