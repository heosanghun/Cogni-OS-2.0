"""Bounded, session-opt-in general web search over an official JSON API.

Cogni-OS remains air-gapped by default.  This module is a deliberately narrow
network exception for Brave Search's documented JSON endpoint.  A request is
admitted only when the operator enables online mode, pins the provider, adds
the fixed API host to the allowlist, configures a token through the process
environment, accepts the provider terms, *and* the caller opts in for the
individual session.

User input can never select the HTTP method, host, path, redirect target, or
response destination.  Search result URLs are provenance only and are never
fetched by this connector.  Provider responses are untrusted external data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import http.client
import ipaddress
import json
import re
import ssl
from threading import BoundedSemaphore, Event, RLock
import time
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


BRAVE_SEARCH_PROVIDER_ID = "brave"
BRAVE_SEARCH_PROVIDER = "Brave Search official API"
BRAVE_SEARCH_API_HOST = "api.search.brave.com"
BRAVE_SEARCH_API_PATH = "/res/v1/web/search"
BRAVE_SEARCH_TERMS_URL = "https://brave.com/search/api/terms-of-service/"

MAX_WEB_QUERY_CHARS = 512
MAX_WEB_QUERY_BYTES = 4 * 1024
MAX_WEB_RESULTS = 10
MAX_WEB_REQUEST_TARGET_BYTES = 8 * 1024
MAX_WEB_RESPONSE_BYTES = 1024 * 1024
MAX_WEB_TOKEN_CHARS = 4 * 1024
MAX_WEB_JSON_DEPTH = 24
MAX_WEB_JSON_NODES = 20_000
MAX_WEB_URL_CHARS = 2_048
MAX_WEB_TITLE_CHARS = 512
MAX_WEB_DESCRIPTION_CHARS = 4_096
DEFAULT_WEB_TIMEOUT_SECONDS = 8.0
MAX_WEB_TIMEOUT_SECONDS = 20.0
MAX_WEB_RETRIES = 1
MAX_WEB_RETRY_DELAY_SECONDS = 0.5

_RETRIABLE_STATUS = frozenset({429, 502, 503, 504})
_HOST_RE = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}\Z"
)
_ERROR_CODE_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")


class GeneralWebSearchError(RuntimeError):
    """A stable, redacted failure safe to expose through the loopback API."""

    def __init__(self, code: str, message: str) -> None:
        if _ERROR_CODE_RE.fullmatch(code) is None:
            raise ValueError("web search error code is invalid")
        if not isinstance(message, str) or not 1 <= len(message) <= 256:
            raise ValueError("web search error message is invalid")
        super().__init__(message)
        self.code = code


class GeneralWebTransportError(RuntimeError):
    """Internal transport marker; its diagnostic text is never published."""


@dataclass(frozen=True, slots=True)
class GeneralWebHttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class GeneralWebTransport(Protocol):
    """The complete network authority available to the search client."""

    def get(
        self,
        *,
        host: str,
        path: str,
        query_string: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        maximum_response_bytes: int,
        cancelled: Callable[[], bool],
    ) -> GeneralWebHttpResponse: ...


class StandardLibraryGeneralWebTransport:
    """TLS-validating fixed-destination transport that never follows redirects."""

    def get(
        self,
        *,
        host: str,
        path: str,
        query_string: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        maximum_response_bytes: int,
        cancelled: Callable[[], bool],
    ) -> GeneralWebHttpResponse:
        _validate_transport_contract(host, path, query_string)
        if cancelled():
            raise GeneralWebTransportError("request cancelled before transport")
        connection = http.client.HTTPSConnection(
            BRAVE_SEARCH_API_HOST,
            port=443,
            timeout=timeout_seconds,
            context=ssl.create_default_context(),
        )
        try:
            connection.request(
                "GET", f"{BRAVE_SEARCH_API_PATH}?{query_string}", headers=dict(headers)
            )
            response = connection.getresponse()
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as exc:
                    raise GeneralWebTransportError("invalid content length") from exc
                if not 0 <= declared_length <= maximum_response_bytes:
                    raise GeneralWebTransportError("response exceeds byte limit")
            chunks: list[bytes] = []
            received = 0
            while True:
                if cancelled():
                    raise GeneralWebTransportError("request cancelled while reading")
                chunk = response.read(
                    min(64 * 1024, maximum_response_bytes + 1 - received)
                )
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
                if received > maximum_response_bytes:
                    raise GeneralWebTransportError("response exceeds byte limit")
            response_headers = {
                name.casefold(): value
                for name, value in response.getheaders()
                if isinstance(name, str) and isinstance(value, str)
            }
            return GeneralWebHttpResponse(
                status=int(response.status),
                headers=response_headers,
                body=b"".join(chunks),
            )
        except GeneralWebTransportError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException, ssl.SSLError) as exc:
            raise GeneralWebTransportError(
                "official search HTTPS request failed"
            ) from exc
        finally:
            connection.close()


class GeneralWebSearchConfig:
    """Secret-bearing provider configuration with a redacted representation."""

    __slots__ = (
        "online_mode",
        "provider_id",
        "allowlist",
        "terms_accepted",
        "_token",
    )

    def __init__(
        self,
        *,
        online_mode: bool,
        provider_id: str,
        allowlist: tuple[str, ...],
        token: str,
        terms_accepted: bool,
    ) -> None:
        if not isinstance(online_mode, bool):
            raise TypeError("online_mode must be bool")
        if not isinstance(provider_id, str):
            raise TypeError("provider_id must be text")
        if not isinstance(allowlist, tuple) or any(
            not isinstance(host, str) for host in allowlist
        ):
            raise TypeError("allowlist must be a tuple of hostnames")
        normalized_allowlist = tuple(
            sorted({host.strip().casefold() for host in allowlist if host.strip()})
        )
        if any(_HOST_RE.fullmatch(host) is None for host in normalized_allowlist):
            raise ValueError("allowlist contains an invalid hostname")
        if not isinstance(token, str):
            raise TypeError("token must be text")
        if not isinstance(terms_accepted, bool):
            raise TypeError("terms_accepted must be bool")
        selected_token = token.strip()
        if len(selected_token) > MAX_WEB_TOKEN_CHARS or any(
            character in selected_token for character in "\r\n\0"
        ):
            raise ValueError("web search token is malformed")
        self.online_mode = online_mode
        self.provider_id = provider_id.strip().casefold()
        self.allowlist = normalized_allowlist
        self.terms_accepted = terms_accepted
        self._token = selected_token

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> GeneralWebSearchConfig:
        if not isinstance(environment, Mapping):
            raise TypeError("environment must be a mapping")
        allowlist = tuple(
            item.strip()
            for item in environment.get("COGNI_OS_WEB_ALLOWLIST", "").split(",")
            if item.strip()
        )
        return cls(
            online_mode=environment.get("COGNI_OS_ONLINE_MODE") == "1",
            provider_id=environment.get("COGNI_OS_GENERAL_WEB_PROVIDER", ""),
            allowlist=allowlist,
            token=environment.get("COGNI_OS_BRAVE_SEARCH_API_TOKEN", ""),
            terms_accepted=(
                environment.get("COGNI_OS_BRAVE_SEARCH_TERMS_ACCEPTED") == "1"
            ),
        )

    def __repr__(self) -> str:
        return (
            "GeneralWebSearchConfig(online_mode="
            f"{self.online_mode!r}, provider_id={self.provider_id!r}, "
            f"allowlist={self.allowlist!r}, "
            f"token_configured={self.token_configured!r}, "
            f"terms_accepted={self.terms_accepted!r})"
        )

    @property
    def token_configured(self) -> bool:
        return bool(self._token)

    @property
    def state(self) -> str:
        if not self.online_mode:
            return "disabled_air_gap"
        if self.provider_id != BRAVE_SEARCH_PROVIDER_ID:
            return "provider_configuration_required"
        if BRAVE_SEARCH_API_HOST not in self.allowlist:
            return "allowlist_required"
        if not self._token:
            return "credentials_required"
        if not self.terms_accepted:
            return "terms_acceptance_required"
        return "ready_for_session_opt_in"

    @property
    def enabled(self) -> bool:
        return self.state == "ready_for_session_opt_in"

    def public_payload(
        self, *, external_calls: int = 0, revoked: bool = False
    ) -> dict[str, object]:
        state = "revoked" if revoked else self.state
        return {
            "provider": BRAVE_SEARCH_PROVIDER,
            "provider_id": BRAVE_SEARCH_PROVIDER_ID,
            "state": state,
            "host": BRAVE_SEARCH_API_HOST,
            "path": BRAVE_SEARCH_API_PATH,
            "method": "GET",
            "official_json_api": True,
            "html_scraping": False,
            "redirects_followed": False,
            "token_configured": self.token_configured,
            "terms_accepted": self.terms_accepted,
            "terms_url": BRAVE_SEARCH_TERMS_URL,
            "session_online_opt_in_required": True,
            "session_online_opt_in_active": False,
            "executor_implemented": True,
            "result_limit": MAX_WEB_RESULTS,
            "request_timeout_seconds": DEFAULT_WEB_TIMEOUT_SECONDS,
            "maximum_retries": MAX_WEB_RETRIES,
            "maximum_response_bytes": MAX_WEB_RESPONSE_BYTES,
            "external_calls": external_calls,
            "provenance": {
                "retrieved_at": True,
                "query_sha256": True,
                "source_record_sha256": True,
                "canonical_url": True,
            },
        }


class GeneralWebSearchSession:
    """One explicit online consent lease with terminal cancel/revoke states."""

    __slots__ = ("_owner", "_lock", "_state", "_abort")

    def __init__(self, owner: object, *, online_opt_in: bool) -> None:
        if not isinstance(online_opt_in, bool):
            raise TypeError("online_opt_in must be bool")
        self._owner = owner
        self._lock = RLock()
        self._state = "active" if online_opt_in else "opt_in_required"
        self._abort = Event()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def cancel(self) -> bool:
        with self._lock:
            if self._state != "active":
                return False
            self._state = "cancelled"
            self._abort.set()
            return True

    def revoke(self) -> bool:
        with self._lock:
            changed = self._state != "revoked"
            self._state = "revoked"
            self._abort.set()
            return changed

    def _aborted(self) -> bool:
        return self._abort.is_set()


@dataclass(frozen=True, slots=True)
class GeneralWebProvenance:
    provider: str
    api_endpoint: str
    retrieved_at: str
    query_sha256: str
    source_record_sha256: str
    canonical_url: str

    def as_payload(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "api_endpoint": self.api_endpoint,
            "retrieved_at": self.retrieved_at,
            "query_sha256": self.query_sha256,
            "source_record_sha256": self.source_record_sha256,
            "canonical_url": self.canonical_url,
        }


@dataclass(frozen=True, slots=True)
class GeneralWebSearchResult:
    title: str
    description: str | None
    age: str | None
    provenance: GeneralWebProvenance

    def as_payload(self) -> dict[str, object]:
        return {
            "title": self.title,
            "description": self.description,
            "age": self.age,
            "url": self.provenance.canonical_url,
            "citation_links": [
                {
                    "label": self.title,
                    "url": self.provenance.canonical_url,
                }
            ],
            "provenance": self.provenance.as_payload(),
        }


@dataclass(frozen=True, slots=True)
class GeneralWebSearchResponse:
    query: str
    results: tuple[GeneralWebSearchResult, ...]
    retrieved_at: str
    external_calls: int

    def as_payload(self) -> dict[str, object]:
        return {
            "provider": BRAVE_SEARCH_PROVIDER,
            "query": self.query,
            "count": len(self.results),
            "results": [result.as_payload() for result in self.results],
            "retrieved_at": self.retrieved_at,
            "endpoint": f"https://{BRAVE_SEARCH_API_HOST}{BRAVE_SEARCH_API_PATH}",
            "external_calls": self.external_calls,
            "trust_boundary": "untrusted_external_search_metadata",
        }


class GeneralWebSearchClient:
    """Official Brave JSON search client with bounded session authority."""

    def __init__(
        self,
        config: GeneralWebSearchConfig,
        *,
        transport: GeneralWebTransport | None = None,
        timeout_seconds: float = DEFAULT_WEB_TIMEOUT_SECONDS,
        maximum_retries: int = MAX_WEB_RETRIES,
        retry_waiter: Callable[[float, Callable[[], bool]], bool] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(config, GeneralWebSearchConfig):
            raise TypeError("config must be GeneralWebSearchConfig")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(
            timeout_seconds, bool
        ):
            raise TypeError("timeout_seconds must be numeric")
        if not 0.1 <= float(timeout_seconds) <= MAX_WEB_TIMEOUT_SECONDS:
            raise ValueError("timeout_seconds is outside the safe bound")
        if not isinstance(maximum_retries, int) or isinstance(maximum_retries, bool):
            raise TypeError("maximum_retries must be int")
        if not 0 <= maximum_retries <= MAX_WEB_RETRIES:
            raise ValueError("maximum_retries is outside the safe bound")
        self.config = config
        self.transport = transport or StandardLibraryGeneralWebTransport()
        self.timeout_seconds = float(timeout_seconds)
        self.maximum_retries = maximum_retries
        self.retry_waiter = retry_waiter or _wait_with_cancellation
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._owner = object()
        self._revoked = Event()
        self._call_lock = RLock()
        self._external_calls = 0
        self._search_gate = BoundedSemaphore(value=1)

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str], **kwargs: Any
    ) -> GeneralWebSearchClient:
        return cls(GeneralWebSearchConfig.from_environment(environment), **kwargs)

    @property
    def external_calls(self) -> int:
        with self._call_lock:
            return self._external_calls

    def capability_payload(self) -> dict[str, object]:
        return self.config.public_payload(
            external_calls=self.external_calls, revoked=self._revoked.is_set()
        )

    def open_session(self, *, online_opt_in: bool) -> GeneralWebSearchSession:
        return GeneralWebSearchSession(self._owner, online_opt_in=online_opt_in)

    def revoke(self) -> bool:
        with self._call_lock:
            if self._revoked.is_set():
                return False
            self._revoked.set()
            return True

    def search(
        self,
        session: GeneralWebSearchSession,
        query: str,
        *,
        limit: int = 5,
    ) -> GeneralWebSearchResponse:
        normalized_query = _query_text(query)
        if len(self.config._token) >= 8 and self.config._token in normalized_query:
            raise GeneralWebSearchError(
                "WEB_SEARCH_INVALID_QUERY",
                "web search query contains configured credential material",
            )
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise GeneralWebSearchError(
                "WEB_SEARCH_INVALID_LIMIT", "web result limit must be an integer"
            )
        if not 1 <= limit <= MAX_WEB_RESULTS:
            raise GeneralWebSearchError(
                "WEB_SEARCH_INVALID_LIMIT",
                f"web result limit must be between 1 and {MAX_WEB_RESULTS}",
            )
        self._require_configured()
        self._require_session(session)
        if not self._search_gate.acquire(blocking=False):
            raise GeneralWebSearchError(
                "WEB_SEARCH_BUSY", "another bounded web search is already running"
            )
        try:
            self._require_session(session)
            query_string = urlencode(
                {
                    "q": normalized_query,
                    "count": str(limit),
                    "safesearch": "strict",
                }
            )
            if len(query_string.encode("ascii")) > MAX_WEB_REQUEST_TARGET_BYTES:
                raise GeneralWebSearchError(
                    "WEB_SEARCH_REQUEST_TOO_LARGE",
                    "web search request exceeds the local byte bound",
                )
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.config._token,
                "User-Agent": "Cogni-OS-General-Web-Connector/1",
            }
            response = self._request_with_retry(
                session,
                query_string=query_string,
                headers=headers,
            )
            self._require_session(session)
            retrieved_at = _retrieved_at(self.clock)
            payload = _decode_response(response, secret_token=self.config._token)
            results = _normalize_results(
                payload,
                query=normalized_query,
                requested_limit=limit,
                retrieved_at=retrieved_at,
            )
            # Revocation/cancellation wins even when it races normalization.
            self._require_session(session)
            return GeneralWebSearchResponse(
                query=normalized_query,
                results=results,
                retrieved_at=retrieved_at,
                external_calls=self.external_calls,
            )
        finally:
            self._search_gate.release()

    def _require_configured(self) -> None:
        if self._revoked.is_set():
            raise GeneralWebSearchError(
                "WEB_SEARCH_REVOKED", "web search network authority was revoked"
            )
        state = self.config.state
        errors = {
            "disabled_air_gap": (
                "WEB_SEARCH_AIR_GAP_BLOCKED",
                "web search requires explicit operator online mode",
            ),
            "provider_configuration_required": (
                "WEB_SEARCH_PROVIDER_REQUIRED",
                "the fixed general web search provider is not configured",
            ),
            "allowlist_required": (
                "WEB_SEARCH_HOST_NOT_ALLOWLISTED",
                "the official search API host is not allowlisted",
            ),
            "credentials_required": (
                "WEB_SEARCH_TOKEN_REQUIRED",
                "the official search API token is not configured",
            ),
            "terms_acceptance_required": (
                "WEB_SEARCH_TERMS_REQUIRED",
                "the official search API terms are not accepted",
            ),
        }
        if state != "ready_for_session_opt_in":
            code, message = errors[state]
            raise GeneralWebSearchError(code, message)

    def _require_session(self, session: GeneralWebSearchSession) -> None:
        if (
            not isinstance(session, GeneralWebSearchSession)
            or session._owner is not self._owner
        ):
            raise GeneralWebSearchError(
                "WEB_SEARCH_SESSION_INVALID",
                "web search requires a session issued by this connector",
            )
        if self._revoked.is_set() or session.state == "revoked":
            raise GeneralWebSearchError(
                "WEB_SEARCH_REVOKED", "web search network authority was revoked"
            )
        if session.state == "cancelled":
            raise GeneralWebSearchError(
                "WEB_SEARCH_CANCELLED", "web search was cancelled"
            )
        if session.state != "active":
            raise GeneralWebSearchError(
                "WEB_SEARCH_SESSION_OPT_IN_REQUIRED",
                "web search requires explicit online opt-in for this session",
            )

    def _request_with_retry(
        self,
        session: GeneralWebSearchSession,
        *,
        query_string: str,
        headers: Mapping[str, str],
    ) -> GeneralWebHttpResponse:
        for attempt in range(self.maximum_retries + 1):
            self._require_session(session)
            try:
                with self._call_lock:
                    self._external_calls += 1
                response = self.transport.get(
                    host=BRAVE_SEARCH_API_HOST,
                    path=BRAVE_SEARCH_API_PATH,
                    query_string=query_string,
                    headers=headers,
                    timeout_seconds=self.timeout_seconds,
                    maximum_response_bytes=MAX_WEB_RESPONSE_BYTES,
                    cancelled=lambda: self._revoked.is_set() or session._aborted(),
                )
            except GeneralWebTransportError as exc:
                self._require_session(session)
                if attempt < self.maximum_retries:
                    if not self.retry_waiter(
                        MAX_WEB_RETRY_DELAY_SECONDS,
                        lambda: self._revoked.is_set() or session._aborted(),
                    ):
                        self._require_session(session)
                    continue
                raise GeneralWebSearchError(
                    "WEB_SEARCH_NETWORK_ERROR",
                    "the official search API could not be reached",
                ) from exc
            except Exception as exc:  # noqa: BLE001 - transport trust boundary
                self._require_session(session)
                raise GeneralWebSearchError(
                    "WEB_SEARCH_NETWORK_ERROR",
                    "the official search API transport failed closed",
                ) from exc
            self._require_session(session)
            if (
                not isinstance(response, GeneralWebHttpResponse)
                or not isinstance(response.status, int)
                or isinstance(response.status, bool)
                or not 100 <= response.status <= 599
                or not isinstance(response.headers, Mapping)
                or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in response.headers.items()
                )
                or type(response.body) is not bytes
            ):
                raise GeneralWebSearchError(
                    "WEB_SEARCH_RESPONSE_INVALID",
                    "web search transport returned an invalid response",
                )
            if response.status == 200:
                return response
            if response.status in _RETRIABLE_STATUS and attempt < self.maximum_retries:
                delay = _retry_delay(response.headers.get("retry-after"))
                if not self.retry_waiter(
                    delay, lambda: self._revoked.is_set() or session._aborted()
                ):
                    self._require_session(session)
                continue
            if response.status in {301, 302, 303, 307, 308}:
                raise GeneralWebSearchError(
                    "WEB_SEARCH_REDIRECT_BLOCKED",
                    "search API redirects are not followed",
                )
            if response.status in {401, 403}:
                raise GeneralWebSearchError(
                    "WEB_SEARCH_AUTH_FAILED",
                    "the official search API rejected the configured credentials",
                )
            if response.status == 429:
                raise GeneralWebSearchError(
                    "WEB_SEARCH_RATE_LIMITED",
                    "the official search API rate limit was reached",
                )
            if 400 <= response.status < 500:
                raise GeneralWebSearchError(
                    "WEB_SEARCH_REQUEST_REJECTED",
                    "the official search API rejected the bounded request",
                )
            raise GeneralWebSearchError(
                "WEB_SEARCH_SERVICE_UNAVAILABLE",
                "the official search API returned an unavailable status",
            )
        raise AssertionError("bounded web search retry loop did not terminate")


def _validate_transport_contract(host: str, path: str, query_string: str) -> None:
    if host != BRAVE_SEARCH_API_HOST or path != BRAVE_SEARCH_API_PATH:
        raise GeneralWebTransportError("fixed web search endpoint contract violated")
    if (
        not isinstance(query_string, str)
        or len(query_string.encode("utf-8")) > MAX_WEB_REQUEST_TARGET_BYTES
    ):
        raise GeneralWebTransportError("web search query string is invalid")
    try:
        pairs = parse_qsl(
            query_string,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=3,
            encoding="utf-8",
            errors="strict",
        )
    except (UnicodeError, ValueError) as exc:
        raise GeneralWebTransportError("web search query string is invalid") from exc
    fields = {key: value for key, value in pairs}
    if (
        len(pairs) != 3
        or set(fields) != {"q", "count", "safesearch"}
        or not fields["q"]
        or not fields["count"].isdigit()
        or not 1 <= int(fields["count"]) <= MAX_WEB_RESULTS
        or fields["safesearch"] != "strict"
    ):
        raise GeneralWebTransportError("web search query contract violated")


def _query_text(query: str) -> str:
    if not isinstance(query, str):
        raise GeneralWebSearchError(
            "WEB_SEARCH_INVALID_QUERY", "web search query must be text"
        )
    selected = " ".join(query.split())
    if not selected or len(selected) > MAX_WEB_QUERY_CHARS:
        raise GeneralWebSearchError(
            "WEB_SEARCH_INVALID_QUERY",
            "web search query is empty or exceeds the character bound",
        )
    if len(selected.encode("utf-8")) > MAX_WEB_QUERY_BYTES:
        raise GeneralWebSearchError(
            "WEB_SEARCH_INVALID_QUERY", "web search query exceeds the byte bound"
        )
    return selected


def _retrieved_at(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise GeneralWebSearchError(
            "WEB_SEARCH_CLOCK_INVALID", "web search clock did not provide UTC time"
        )
    return value.astimezone(timezone.utc).isoformat()


def _decode_response(
    response: GeneralWebHttpResponse, *, secret_token: str
) -> dict[str, Any]:
    if len(response.body) > MAX_WEB_RESPONSE_BYTES:
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_TOO_LARGE",
            "web search response exceeds the byte bound",
        )
    content_type = response.headers.get("content-type", "").casefold()
    if not content_type.startswith("application/json"):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search response is not JSON"
        )
    if secret_token and secret_token.encode("utf-8") in response.body:
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID",
            "web search response reflected configured credential material",
        )
    try:
        payload = json.loads(
            response.body.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, ValueError) as exc:
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search response JSON is invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search response must be an object"
        )
    _validate_json_shape(payload)
    return payload


def _validate_json_shape(payload: object) -> None:
    stack: list[tuple[object, int]] = [(payload, 1)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if depth > MAX_WEB_JSON_DEPTH or nodes > MAX_WEB_JSON_NODES:
            raise GeneralWebSearchError(
                "WEB_SEARCH_RESPONSE_INVALID",
                "web search response JSON exceeds structural bounds",
            )
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def _normalize_results(
    payload: Mapping[str, Any],
    *,
    query: str,
    requested_limit: int,
    retrieved_at: str,
) -> tuple[GeneralWebSearchResult, ...]:
    web = payload.get("web")
    if web is None:
        return ()
    if not isinstance(web, dict) or not isinstance(web.get("results"), list):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result collection is invalid"
        )
    records = web["results"]
    if len(records) > requested_limit or len(records) > MAX_WEB_RESULTS:
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result count exceeds the request"
        )
    query_digest = sha256(query.encode("utf-8")).hexdigest()
    normalized: list[GeneralWebSearchResult] = []
    for record in records:
        if not isinstance(record, dict):
            raise GeneralWebSearchError(
                "WEB_SEARCH_RESPONSE_INVALID", "web search result must be an object"
            )
        title = _bounded_text(record.get("title"), MAX_WEB_TITLE_CHARS, required=True)
        canonical_url = _canonical_result_url(record.get("url"))
        description = _bounded_text(
            record.get("description"), MAX_WEB_DESCRIPTION_CHARS, required=False
        )
        age = _bounded_text(record.get("age"), 128, required=False)
        source_digest = sha256(
            json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        normalized.append(
            GeneralWebSearchResult(
                title=title,
                description=description,
                age=age,
                provenance=GeneralWebProvenance(
                    provider=BRAVE_SEARCH_PROVIDER,
                    api_endpoint=(
                        f"https://{BRAVE_SEARCH_API_HOST}{BRAVE_SEARCH_API_PATH}"
                    ),
                    retrieved_at=retrieved_at,
                    query_sha256=query_digest,
                    source_record_sha256=source_digest,
                    canonical_url=canonical_url,
                ),
            )
        )
    return tuple(normalized)


def _bounded_text(value: object, maximum: int, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result text is invalid"
        )
    selected = " ".join(value.split())
    if (
        (required and not selected)
        or len(selected) > maximum
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159
            for character in selected
        )
    ):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result text exceeds bounds"
        )
    return selected or None


def _canonical_result_url(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= MAX_WEB_URL_CHARS
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
        )
        or "\\" in value
    ):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result URL is invalid"
        )
    parsed = urlsplit(value)
    try:
        port = parsed.port
        host = (parsed.hostname or "").encode("idna").decode("ascii").casefold()
    except (UnicodeError, ValueError) as exc:
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result URL is invalid"
        ) from exc
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result URL host is invalid"
        )
    if (
        parsed.scheme.casefold() != "https"
        or _HOST_RE.fullmatch(host) is None
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID",
            "web search result URL is not canonical HTTPS",
        )
    path = parsed.path or "/"
    canonical = urlunsplit(("https", host, path, parsed.query, ""))
    if len(canonical) > MAX_WEB_URL_CHARS:
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result URL exceeds bounds"
        )
    return canonical


def _retry_delay(value: str | None) -> float:
    if value is None:
        return MAX_WEB_RETRY_DELAY_SECONDS
    try:
        selected = float(value)
    except ValueError:
        return MAX_WEB_RETRY_DELAY_SECONDS
    return max(0.0, min(selected, MAX_WEB_RETRY_DELAY_SECONDS))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is rejected: {value}")


def _wait_with_cancellation(delay: float, cancelled: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + max(0.0, delay)
    while True:
        if cancelled():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(remaining, 0.05))


__all__ = [
    "BRAVE_SEARCH_API_HOST",
    "BRAVE_SEARCH_API_PATH",
    "BRAVE_SEARCH_PROVIDER",
    "GeneralWebHttpResponse",
    "GeneralWebProvenance",
    "GeneralWebSearchClient",
    "GeneralWebSearchConfig",
    "GeneralWebSearchError",
    "GeneralWebSearchResponse",
    "GeneralWebSearchResult",
    "GeneralWebSearchSession",
    "GeneralWebTransportError",
    "MAX_WEB_RESPONSE_BYTES",
    "MAX_WEB_RESULTS",
    "StandardLibraryGeneralWebTransport",
]
