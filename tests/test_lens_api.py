import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from cogni_demo.lens_api import (
    LENS_API_HOST,
    LENS_PATENT_ENDPOINT,
    LENS_SCHOLARLY_ENDPOINT,
    MAX_LENS_RESPONSE_BYTES,
    LensAkasicBridge,
    LensApiClient,
    LensApiConfig,
    LensApiError,
    LensHttpResponse,
    LensSearchKind,
    LensTransportError,
    StandardLibraryLensTransport,
)
from cogni_demo.workspace_capabilities import (
    VerifiedModelMetadata,
    WorkspaceCapabilityError,
    WorkspaceCapabilityService,
)


TOKEN = "lens-secret-token-that-must-never-be-published"
NOW = datetime(2026, 7, 16, 4, 5, 6, tzinfo=timezone.utc)


def _config(
    *,
    online: bool = True,
    allowlist: tuple[str, ...] = (LENS_API_HOST,),
    token: str = TOKEN,
    terms_accepted: bool = True,
) -> LensApiConfig:
    return LensApiConfig(
        online_mode=online,
        allowlist=allowlist,
        token=token,
        terms_accepted=terms_accepted,
    )


def _json_response(payload: object, *, status: int = 200) -> LensHttpResponse:
    return LensHttpResponse(
        status=status,
        headers={"content-type": "application/json; charset=utf-8"},
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


class FakeTransport:
    def __init__(self, *responses: LensHttpResponse | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def post(self, **kwargs: object) -> LensHttpResponse:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected Lens transport call")
        selected = self.responses.pop(0)
        if isinstance(selected, BaseException):
            raise selected
        return selected


class FakeAkasicSink:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def index_document(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "attachment_id": kwargs["attachment_id"],
            "indexed": True,
            "chunks": 1,
        }


class TestLensConfiguration(unittest.TestCase):
    def test_air_gap_allowlist_and_credentials_are_independent_gates(self) -> None:
        self.assertEqual(_config(online=False).state, "disabled_air_gap")
        self.assertEqual(_config(allowlist=()).state, "allowlist_required")
        self.assertEqual(_config(token="").state, "credentials_required")
        self.assertEqual(
            _config(terms_accepted=False).state, "terms_acceptance_required"
        )
        self.assertEqual(_config().state, "ready")

    def test_environment_is_explicit_opt_in_and_public_payload_is_redacted(
        self,
    ) -> None:
        config = LensApiConfig.from_environment(
            {
                "COGNI_OS_ONLINE_MODE": "1",
                "COGNI_OS_WEB_ALLOWLIST": "example.com, API.LENS.ORG",
                "COGNI_OS_LENS_API_TOKEN": TOKEN,
                "COGNI_OS_LENS_TERMS_ACCEPTED": "1",
            }
        )
        self.assertTrue(config.enabled)
        public = json.dumps(config.public_payload(), sort_keys=True)
        self.assertNotIn(TOKEN, public)
        self.assertNotIn(TOKEN, repr(config))
        self.assertTrue(config.public_payload()["executor_implemented"])
        self.assertFalse(config.public_payload()["scraping_allowed"])
        attribution = config.public_payload()["attribution"]
        self.assertEqual(attribution["label"], "Data Sourced from The Lens")
        self.assertEqual(attribution["url"], "https://www.lens.org/")
        self.assertFalse(attribution["logo_asset_bundled"])
        self.assertFalse(attribution["public_deployment_ready"])

    def test_malformed_token_and_allowlist_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "token is malformed"):
            _config(token="secret\nsecond-header")
        with self.assertRaisesRegex(ValueError, "invalid hostname"):
            _config(allowlist=("https://api.lens.org",))

    def test_disabled_clients_never_reach_transport(self) -> None:
        for config, code in (
            (_config(online=False), "LENS_AIR_GAP_BLOCKED"),
            (_config(allowlist=()), "LENS_HOST_NOT_ALLOWLISTED"),
            (_config(token=""), "LENS_TOKEN_REQUIRED"),
            (_config(terms_accepted=False), "LENS_TERMS_REQUIRED"),
        ):
            transport = FakeTransport()
            client = LensApiClient(config, transport=transport)
            with self.assertRaises(LensApiError) as raised:
                client.search("patent", "deep equilibrium")
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(transport.calls, [])
            self.assertNotIn(TOKEN, str(raised.exception))


class TestLensSearch(unittest.TestCase):
    def test_scholarly_search_is_normalized_with_citation_and_provenance(self) -> None:
        upstream = {
            "total": 27,
            "results": 1,
            "data": [
                {
                    "lens_id": "100-004-910-081-14X",
                    "title": "Deep Equilibrium Models",
                    "abstract": "A fixed-point neural architecture.",
                    "authors": [
                        {"display_name": "Alice Example"},
                        {"first_name": "Bob", "last_name": "Researcher"},
                    ],
                    "date_published": "2019-10-01",
                    "year_published": 2019,
                    "publication_type": "journal article",
                    "external_ids": [{"type": "doi", "value": "10.1000/example"}],
                }
            ],
        }
        transport = FakeTransport(_json_response(upstream))
        client = LensApiClient(
            _config(), transport=transport, clock=lambda: NOW, maximum_retries=0
        )

        result = client.search("scholarly", "deep equilibrium", limit=3)

        self.assertEqual(result.kind, LensSearchKind.SCHOLARLY)
        self.assertEqual(result.total, 27)
        self.assertEqual(len(result.results), 1)
        record = result.results[0]
        self.assertEqual(record.title, "Deep Equilibrium Models")
        self.assertEqual(record.contributors, ("Alice Example", "Bob Researcher"))
        self.assertEqual(record.identifiers, ("doi:10.1000/example",))
        self.assertEqual(
            record.provenance.canonical_url,
            "https://lens.org/100-004-910-081-14X",
        )
        payload = result.as_payload()
        citation = payload["results"][0]["citation_links"][0]
        self.assertEqual(citation["url"], record.provenance.canonical_url)
        self.assertEqual(payload["retrieved_at"], NOW.isoformat())

        call = transport.calls[0]
        self.assertEqual(call["host"], LENS_API_HOST)
        self.assertEqual(call["path"], LENS_SCHOLARLY_ENDPOINT)
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {TOKEN}")
        request = json.loads(call["body"].decode("utf-8"))
        self.assertEqual(request["query"], "deep equilibrium")
        self.assertEqual(request["size"], 3)
        self.assertNotIn("token", request)

    def test_patent_search_normalizes_bibliographic_fields(self) -> None:
        upstream = {
            "total": 1,
            "data": [
                {
                    "lens_id": "186-488-232-022-055",
                    "doc_key": "US_123_A1_20260101",
                    "jurisdiction": "US",
                    "doc_number": "123",
                    "kind": "A1",
                    "date_published": "2026-01-01",
                    "year_published": "2026",
                    "publication_type": "Patent Application",
                    "biblio": {
                        "invention_title": [
                            {"lang": "EN", "text": "Bounded AI runtime"}
                        ],
                        "abstracts": [
                            {"lang": "EN", "text": "A local inference system."}
                        ],
                        "parties": {"inventors": [{"name": "Ada Inventor"}]},
                    },
                }
            ],
        }
        transport = FakeTransport(_json_response(upstream))
        client = LensApiClient(
            _config(), transport=transport, clock=lambda: NOW, maximum_retries=0
        )

        response = client.search(LensSearchKind.PATENT, "bounded AI", limit=1)

        record = response.results[0]
        self.assertEqual(record.title, "Bounded AI runtime")
        self.assertEqual(record.abstract, "A local inference system.")
        self.assertEqual(record.contributors, ("Ada Inventor",))
        self.assertEqual(record.publication_year, 2026)
        self.assertEqual(transport.calls[0]["path"], LENS_PATENT_ENDPOINT)

    def test_query_kind_and_limit_are_bounded_before_network(self) -> None:
        transport = FakeTransport()
        client = LensApiClient(_config(), transport=transport)
        for kind, query, limit, code in (
            ("web", "valid", 1, "LENS_INVALID_KIND"),
            ("patent", "", 1, "LENS_INVALID_QUERY"),
            ("patent", "x" * 513, 1, "LENS_INVALID_QUERY"),
            ("patent", "valid", 21, "LENS_INVALID_LIMIT"),
        ):
            with self.assertRaises(LensApiError) as raised:
                client.search(kind, query, limit=limit)
            self.assertEqual(raised.exception.code, code)
        self.assertEqual(transport.calls, [])

    def test_retry_is_bounded_and_retry_after_is_clamped(self) -> None:
        rate_limited = LensHttpResponse(
            status=429,
            headers={"retry-after": "999999"},
            body=b'{"error":"do not return this body"}',
        )
        transport = FakeTransport(
            rate_limited,
            _json_response({"total": 0, "data": []}),
        )
        delays: list[float] = []
        client = LensApiClient(
            _config(),
            transport=transport,
            sleeper=delays.append,
            maximum_retries=1,
            clock=lambda: NOW,
        )

        response = client.search("scholarly", "safe query")

        self.assertEqual(response.external_calls, 2)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(delays, [1.0])

    def test_upstream_error_never_echoes_body_or_token(self) -> None:
        transport = FakeTransport(
            LensHttpResponse(
                status=403,
                headers={"content-type": "application/json"},
                body=f'{{"token":"{TOKEN}","detail":"private"}}'.encode(),
            )
        )
        client = LensApiClient(_config(), transport=transport, maximum_retries=0)

        with self.assertRaises(LensApiError) as raised:
            client.search("patent", "safe")

        self.assertEqual(raised.exception.code, "LENS_AUTH_FAILED")
        self.assertNotIn(TOKEN, str(raised.exception))
        self.assertNotIn("private", str(raised.exception))

    def test_redirects_are_not_followed(self) -> None:
        transport = FakeTransport(
            LensHttpResponse(
                status=302,
                headers={"location": "https://evil.example/collect"},
                body=b"redirect",
            )
        )
        client = LensApiClient(_config(), transport=transport, maximum_retries=0)

        with self.assertRaises(LensApiError) as raised:
            client.search("patent", "safe")

        self.assertEqual(raised.exception.code, "LENS_SERVICE_UNAVAILABLE")
        self.assertEqual(len(transport.calls), 1)

    def test_only_one_outstanding_search_is_admitted(self) -> None:
        transport = FakeTransport(_json_response({"total": 0, "data": []}))
        client = LensApiClient(_config(), transport=transport, maximum_retries=0)
        self.assertTrue(client._search_gate.acquire(blocking=False))
        try:
            with self.assertRaises(LensApiError) as raised:
                client.search("scholarly", "bounded")
        finally:
            client._search_gate.release()
        self.assertEqual(raised.exception.code, "LENS_BUSY")
        self.assertEqual(transport.calls, [])

    def test_invalid_or_oversized_responses_fail_closed(self) -> None:
        cases = (
            LensHttpResponse(
                200, {"content-type": "text/html"}, b"<html>not json</html>"
            ),
            LensHttpResponse(
                200,
                {"content-type": "application/json"},
                b"x" * (MAX_LENS_RESPONSE_BYTES + 1),
            ),
            _json_response(
                {
                    "total": 1,
                    "data": [{"lens_id": "not-a-lens-id", "title": "bad"}],
                }
            ),
        )
        for response in cases:
            client = LensApiClient(
                _config(), transport=FakeTransport(response), maximum_retries=0
            )
            with self.assertRaises(LensApiError) as raised:
                client.search("scholarly", "safe")
            self.assertIn(
                raised.exception.code,
                {"LENS_RESPONSE_TOO_LARGE", "LENS_RESPONSE_INVALID"},
            )

    def test_standard_transport_rejects_any_non_lens_destination_preflight(
        self,
    ) -> None:
        transport = StandardLibraryLensTransport()
        with self.assertRaises(LensTransportError):
            transport.post(
                host="example.com",
                path=LENS_PATENT_ENDPOINT,
                headers={},
                body=b"{}",
                timeout_seconds=1.0,
                maximum_response_bytes=1024,
            )


class TestLensAkasicBridge(unittest.TestCase):
    def test_normalized_result_becomes_bounded_provenance_document(self) -> None:
        response = _json_response(
            {
                "total": 1,
                "data": [
                    {
                        "lens_id": "100-004-910-081-14X",
                        "title": "Deep Equilibrium Models",
                        "abstract": "Evidence, not instructions.",
                        "authors": [{"display_name": "A. Author"}],
                        "year_published": 2019,
                    }
                ],
            }
        )
        client = LensApiClient(
            _config(),
            transport=FakeTransport(response),
            maximum_retries=0,
            clock=lambda: NOW,
        )
        sink = FakeAkasicSink()
        bridge = LensAkasicBridge(client, sink)

        outcome = bridge.search_and_index("scholarly", "equilibrium", limit=1)

        self.assertEqual(len(sink.calls), 1)
        indexed = sink.calls[0]
        self.assertRegex(indexed["attachment_id"], r"[0-9a-f]{24}\Z")
        self.assertEqual(indexed["media_type"], "text/markdown")
        self.assertIn("Provider: Lens.org official API", indexed["text"])
        self.assertIn(
            "Canonical URL: https://lens.org/100-004-910-081-14X", indexed["text"]
        )
        self.assertIn("Trust boundary: untrusted external", indexed["text"])
        document_payload = outcome["indexed"][0]["document"]
        self.assertEqual(
            document_payload["provenance"]["lens_id"], "100-004-910-081-14X"
        )
        self.assertNotIn(TOKEN, json.dumps(outcome))

    def test_workspace_service_exposes_state_search_and_index_bridge(self) -> None:
        search_payload = {
            "total": 1,
            "data": [
                {
                    "lens_id": "100-004-910-081-14X",
                    "title": "Equilibrium evidence",
                    "abstract": "A bounded official API record.",
                }
            ],
        }
        transport = FakeTransport(
            _json_response(search_payload),
            _json_response(search_payload),
        )
        client = LensApiClient(
            _config(),
            transport=transport,
            maximum_retries=0,
            clock=lambda: NOW,
        )
        model = VerifiedModelMetadata(
            model_id="verified",
            label="Gemma 4 e4b",
            architecture="Gemma4ForConditionalGeneration",
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            checkpoint_modalities=("text",),
            runtime_input_modalities=("text",),
        )
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(
                Path(temporary), model, lens_client=client
            )
            lens_capability = service.capability_payload()["web_search"][
                "official_lens_connector"
            ]
            self.assertEqual(lens_capability["state"], "ready")
            self.assertTrue(lens_capability["executor_implemented"])
            self.assertEqual(
                lens_capability["lens_to_akasicdb"], service.akasicdb is not None
            )

            searched = service.search_lens("scholarly", "equilibrium", limit=1)
            self.assertEqual(searched["count"], 1)

            sink = FakeAkasicSink()
            service.akasicdb = sink
            service.akasicdb_error = None
            indexed = service.search_lens(
                "scholarly",
                "equilibrium",
                limit=1,
                index_in_akasicdb=True,
            )
            self.assertEqual(indexed["index_engine"], "AkasicDB")
            self.assertTrue(indexed["provenance_embedded"])
            self.assertEqual(len(sink.calls), 1)

    def test_workspace_service_maps_lens_error_without_secret(self) -> None:
        client = LensApiClient(
            _config(online=False), transport=FakeTransport(), maximum_retries=0
        )
        model = VerifiedModelMetadata(
            model_id="verified",
            label="Gemma 4 e4b",
            architecture="Gemma4ForConditionalGeneration",
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            checkpoint_modalities=("text",),
            runtime_input_modalities=("text",),
        )
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(
                Path(temporary), model, lens_client=client
            )
            with self.assertRaises(WorkspaceCapabilityError) as raised:
                service.search_lens("patent", "equilibrium")
        self.assertEqual(raised.exception.code, "LENS_AIR_GAP_BLOCKED")
        self.assertNotIn(TOKEN, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
