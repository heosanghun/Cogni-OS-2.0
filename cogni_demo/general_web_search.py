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
from queue import Empty, Queue
import re
import socket
import ssl
from threading import BoundedSemaphore, Event, RLock, Thread
import time
from typing import Any, Protocol, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit


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
MIN_WEB_TOKEN_CHARS = 16
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
MAX_WEB_RESOLVED_ADDRESSES = 16

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
        authorize_dispatch: Callable[[Callable[[], None]], bool] | None = None,
    ) -> GeneralWebHttpResponse: ...


class GeneralWebResolver(Protocol):
    """Resolve the fixed API hostname without granting URL-selection authority."""

    def __call__(self, host: str, port: int) -> Sequence[str]: ...


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to one admitted IP while authenticating the fixed DNS hostname."""

    def __init__(
        self,
        host: str,
        pinned_ip: str,
        *,
        port: int,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        # No proxy/tunnel is admitted.  The TCP peer is the literal IP returned
        # by the validated resolver, while TLS SNI and certificate hostname
        # verification remain bound to ``self.host`` (api.search.brave.com).
        raw_socket = socket.create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            raw_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except BaseException:
            raw_socket.close()
            raise


def _system_resolver(host: str, port: int) -> tuple[str, ...]:
    records = socket.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return tuple(record[4][0] for record in records)


def _default_connection_factory(
    host: str,
    pinned_ip: str,
    timeout_seconds: float,
    context: ssl.SSLContext,
) -> _PinnedHTTPSConnection:
    return _PinnedHTTPSConnection(
        host,
        pinned_ip,
        port=443,
        timeout=timeout_seconds,
        context=context,
    )


class _ResolverGate:
    """Bound resolver concurrency and permanently fail closed after a stall.

    ``socket.getaddrinfo`` cannot be interrupted portably.  The resolver runs in
    a daemon thread so the request deadline remains enforceable, but allowing a
    new thread after every timeout would turn a wedged system resolver into an
    unbounded process-resource leak.  One lease is therefore admitted at a
    time.  If its caller times out or is cancelled while that lease is still
    active, the gate is poisoned for the rest of the process.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._active_lease: int | None = None
        self._next_lease = 0
        self._poisoned = False

    def claim(self) -> int:
        with self._lock:
            if self._poisoned:
                raise GeneralWebTransportError(
                    "official search DNS resolver is unavailable"
                )
            if self._active_lease is not None:
                raise GeneralWebTransportError(
                    "official search DNS resolver is already active"
                )
            self._next_lease += 1
            self._active_lease = self._next_lease
            return self._next_lease

    def release(self, lease: int) -> None:
        with self._lock:
            if self._active_lease == lease:
                self._active_lease = None

    def poison_if_active(self, lease: int) -> None:
        with self._lock:
            # The lease comparison prevents a late caller cancellation from
            # poisoning a newer lookup that claimed the slot after this worker
            # had already completed.
            if self._active_lease == lease:
                self._poisoned = True


_GLOBAL_RESOLVER_GATE = _ResolverGate()


class StandardLibraryGeneralWebTransport:
    """TLS-validating fixed-destination transport that never follows redirects."""

    def __init__(
        self,
        *,
        resolver: GeneralWebResolver | None = None,
        connection_factory: Callable[[str, str, float, ssl.SSLContext], Any]
        | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        resolver_gate: _ResolverGate | None = None,
    ) -> None:
        self._resolver = resolver or _system_resolver
        self._connection_factory = connection_factory or _default_connection_factory
        self._monotonic = monotonic_clock or time.monotonic
        # All production/default transports share one process-wide slot.  The
        # injectable gate exists only so deterministic tests cannot poison the
        # production singleton for later cases in the same interpreter.
        self._resolver_gate = resolver_gate or _GLOBAL_RESOLVER_GATE

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
        authorize_dispatch: Callable[[Callable[[], None]], bool] | None = None,
    ) -> GeneralWebHttpResponse:
        _validate_transport_contract(host, path, query_string)
        if cancelled():
            raise GeneralWebTransportError("request cancelled before transport")
        deadline = self._monotonic() + float(timeout_seconds)
        addresses = _resolve_public_addresses(
            self._resolver,
            host=BRAVE_SEARCH_API_HOST,
            port=443,
            deadline=deadline,
            monotonic_clock=self._monotonic,
            cancelled=cancelled,
            resolver_gate=self._resolver_gate,
        )
        remaining = _transport_remaining(deadline, self._monotonic, cancelled)
        tls_context = ssl.create_default_context()
        # ``create_default_context`` currently enables both checks, but assert
        # the invariant explicitly so a custom runtime cannot silently weaken it.
        tls_context.check_hostname = True
        tls_context.verify_mode = ssl.CERT_REQUIRED
        connection = self._connection_factory(
            BRAVE_SEARCH_API_HOST,
            addresses[0],
            remaining,
            tls_context,
        )
        try:
            def send_request() -> None:
                _set_connection_timeout(
                    connection,
                    _transport_remaining(deadline, self._monotonic, cancelled),
                )
                connection.request(
                    "GET",
                    f"{BRAVE_SEARCH_API_PATH}?{query_string}",
                    headers=dict(headers),
                )

            if authorize_dispatch is None or not authorize_dispatch(send_request):
                raise GeneralWebTransportError(
                    "request authority was revoked before dispatch"
                )
            _set_connection_timeout(
                connection,
                _transport_remaining(deadline, self._monotonic, cancelled),
            )
            response = connection.getresponse()
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    raise GeneralWebTransportError("invalid content length") from None
                if not 0 <= declared_length <= maximum_response_bytes:
                    raise GeneralWebTransportError("response exceeds byte limit")
            chunks: list[bytes] = []
            received = 0
            while True:
                _set_connection_timeout(
                    connection,
                    _transport_remaining(deadline, self._monotonic, cancelled),
                )
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
        except (OSError, TimeoutError, http.client.HTTPException, ssl.SSLError):
            raise GeneralWebTransportError(
                "official search HTTPS request failed"
            ) from None
        finally:
            connection.close()


def _resolve_public_addresses(
    resolver: GeneralWebResolver,
    *,
    host: str,
    port: int,
    deadline: float,
    monotonic_clock: Callable[[], float],
    cancelled: Callable[[], bool],
    resolver_gate: _ResolverGate,
) -> tuple[str, ...]:
    """Resolve once and reject the complete answer if any address is non-public."""

    result_queue: Queue[tuple[bool, object]] = Queue(maxsize=1)
    lease = resolver_gate.claim()

    def resolve() -> None:
        try:
            result: tuple[bool, object] = (True, resolver(host, port))
        except BaseException:  # Resolver diagnostics are an untrusted boundary.
            result = (False, None)
        # Release before publishing: receipt of a completed lookup must never
        # leave the global slot looking busy.  A lease id ensures a late timeout
        # cannot poison a subsequent lookup that acquired the freed slot.
        resolver_gate.release(lease)
        try:
            result_queue.put_nowait(result)
        except Exception:
            return

    try:
        Thread(target=resolve, name="cogni-web-dns", daemon=True).start()
    except Exception:
        resolver_gate.release(lease)
        raise GeneralWebTransportError(
            "official search DNS worker failed to start"
        ) from None
    while True:
        if cancelled():
            resolver_gate.poison_if_active(lease)
            raise GeneralWebTransportError("request cancelled during resolution")
        remaining = deadline - monotonic_clock()
        if remaining <= 0.0:
            resolver_gate.poison_if_active(lease)
            raise GeneralWebTransportError("official search DNS deadline exceeded")
        try:
            succeeded, raw_addresses = result_queue.get(timeout=min(remaining, 0.05))
            break
        except Empty:
            continue
    if not succeeded or isinstance(raw_addresses, (str, bytes)) or not isinstance(
        raw_addresses, Sequence
    ):
        raise GeneralWebTransportError("official search DNS resolution failed")
    if not 1 <= len(raw_addresses) <= MAX_WEB_RESOLVED_ADDRESSES:
        raise GeneralWebTransportError("official search DNS answer is outside bounds")
    normalized: set[str] = set()
    for raw_address in raw_addresses:
        if not isinstance(raw_address, str):
            raise GeneralWebTransportError("official search DNS answer is invalid")
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError:
            raise GeneralWebTransportError(
                "official search DNS answer is invalid"
            ) from None
        if (
            not address.is_global
            or address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise GeneralWebTransportError(
                "official search DNS answer contains a non-public address"
            )
        normalized.add(address.compressed)
    if not normalized:
        raise GeneralWebTransportError("official search DNS answer is empty")
    return tuple(sorted(normalized))


def _transport_remaining(
    deadline: float,
    monotonic_clock: Callable[[], float],
    cancelled: Callable[[], bool],
) -> float:
    if cancelled():
        raise GeneralWebTransportError("request cancelled")
    remaining = deadline - monotonic_clock()
    if remaining <= 0.0:
        raise GeneralWebTransportError("official search request deadline exceeded")
    return remaining


def _set_connection_timeout(connection: object, timeout_seconds: float) -> None:
    sock = getattr(connection, "sock", None)
    setter = getattr(sock, "settimeout", None)
    if callable(setter):
        setter(timeout_seconds)
    if hasattr(connection, "timeout"):
        connection.timeout = timeout_seconds


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
        if token and (
            token != selected_token
            or not MIN_WEB_TOKEN_CHARS <= len(selected_token) <= MAX_WEB_TOKEN_CHARS
            or any(not 33 <= ord(character) <= 126 for character in selected_token)
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

    __slots__ = (
        "_owner",
        "_authority_lock",
        "_state",
        "_abort",
        "_epoch",
        "_wire_epoch",
    )

    def __init__(
        self,
        owner: object,
        authority_lock: RLock,
        *,
        online_opt_in: bool,
    ) -> None:
        if not isinstance(online_opt_in, bool):
            raise TypeError("online_opt_in must be bool")
        self._owner = owner
        self._authority_lock = authority_lock
        self._state = "active" if online_opt_in else "opt_in_required"
        self._abort = Event()
        self._epoch = 0
        self._wire_epoch: int | None = None

    @property
    def state(self) -> str:
        with self._authority_lock:
            return self._state

    def cancel(self) -> bool:
        with self._authority_lock:
            if self._state != "active":
                return False
            self._epoch += 1
            self._state = "cancelled"
            self._abort.set()
            return True

    def revoke(self) -> bool:
        with self._authority_lock:
            if self._state == "published":
                return False
            changed = self._state != "revoked"
            self._epoch += 1
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
        monotonic_clock: Callable[[], float] | None = None,
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
        self.monotonic = monotonic_clock or time.monotonic
        self._owner = object()
        # Session cancel/revoke, wire dispatch, and response publication all
        # linearize on this single authority lock and session epoch.
        self._authority_lock = RLock()
        self._revoked = False
        self._external_calls = 0
        self._search_gate = BoundedSemaphore(value=1)

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str], **kwargs: Any
    ) -> GeneralWebSearchClient:
        return cls(GeneralWebSearchConfig.from_environment(environment), **kwargs)

    @property
    def external_calls(self) -> int:
        with self._authority_lock:
            return self._external_calls

    def capability_payload(self) -> dict[str, object]:
        with self._authority_lock:
            return self.config.public_payload(
                external_calls=self._external_calls,
                revoked=self._revoked,
            )

    def open_session(self, *, online_opt_in: bool) -> GeneralWebSearchSession:
        with self._authority_lock:
            session = GeneralWebSearchSession(
                self._owner,
                self._authority_lock,
                online_opt_in=online_opt_in,
            )
            if self._revoked:
                session.revoke()
            return session

    def revoke(self) -> bool:
        with self._authority_lock:
            if self._revoked:
                return False
            self._revoked = True
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
            deadline = self.monotonic() + self.timeout_seconds
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
            response, response_epoch = self._request_with_retry(
                session,
                query_string=query_string,
                headers=headers,
                deadline=deadline,
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
            result = GeneralWebSearchResponse(
                query=normalized_query,
                results=results,
                retrieved_at=retrieved_at,
                external_calls=self.external_calls,
            )
            self._remaining(deadline)
            # This is the publication linearization point.  A cancellation or
            # global revoke that wins the same lock first prevents the result;
            # once publication wins, a later cancel returns False.
            self._claim_publication(session, response_epoch)
            return result
        finally:
            self._search_gate.release()

    def _require_configured(self) -> None:
        with self._authority_lock:
            if self._revoked:
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
        with self._authority_lock:
            self._require_session_locked(session)

    def _require_session_locked(self, session: GeneralWebSearchSession) -> None:
        if (
            not isinstance(session, GeneralWebSearchSession)
            or session._owner is not self._owner
            or session._authority_lock is not self._authority_lock
        ):
            raise GeneralWebSearchError(
                "WEB_SEARCH_SESSION_INVALID",
                "web search requires a session issued by this connector",
            )
        if self._revoked or session._state == "revoked":
            raise GeneralWebSearchError(
                "WEB_SEARCH_REVOKED", "web search network authority was revoked"
            )
        if session._state == "cancelled":
            raise GeneralWebSearchError(
                "WEB_SEARCH_CANCELLED", "web search was cancelled"
            )
        if session._state == "published":
            raise GeneralWebSearchError(
                "WEB_SEARCH_SESSION_CONSUMED",
                "web search session authority was already consumed",
            )
        if session._state != "active":
            raise GeneralWebSearchError(
                "WEB_SEARCH_SESSION_OPT_IN_REQUIRED",
                "web search requires explicit online opt-in for this session",
            )

    def _begin_attempt(self, session: GeneralWebSearchSession) -> int:
        with self._authority_lock:
            self._require_session_locked(session)
            session._epoch += 1
            session._wire_epoch = None
            return session._epoch

    def _authorize_dispatch(
        self,
        session: GeneralWebSearchSession,
        epoch: int,
        dispatch: Callable[[], None],
    ) -> bool:
        with self._authority_lock:
            if (
                self._revoked
                or session._owner is not self._owner
                or session._state != "active"
                or session._epoch != epoch
                or session._wire_epoch is not None
            ):
                return False
            session._wire_epoch = epoch
            self._external_calls += 1
            # Keep the authority fence held through the actual HTTP request
            # dispatch.  Therefore cancel/revoke cannot return and then allow
            # a request to be sent afterward.
            dispatch()
            return True

    def _claim_publication(
        self, session: GeneralWebSearchSession, epoch: int
    ) -> None:
        with self._authority_lock:
            self._require_session_locked(session)
            if session._epoch != epoch or session._wire_epoch != epoch:
                raise GeneralWebSearchError(
                    "WEB_SEARCH_SESSION_INVALID",
                    "web search response authority fence is invalid",
                )
            session._state = "published"

    def _cancelled(self, session: GeneralWebSearchSession) -> bool:
        with self._authority_lock:
            return self._revoked or session._state != "active"

    def _request_with_retry(
        self,
        session: GeneralWebSearchSession,
        *,
        query_string: str,
        headers: Mapping[str, str],
        deadline: float,
    ) -> tuple[GeneralWebHttpResponse, int]:
        for attempt in range(self.maximum_retries + 1):
            self._require_session(session)
            attempt_epoch = self._begin_attempt(session)
            remaining = self._remaining(deadline)
            try:
                response = self.transport.get(
                    host=BRAVE_SEARCH_API_HOST,
                    path=BRAVE_SEARCH_API_PATH,
                    query_string=query_string,
                    headers=headers,
                    timeout_seconds=remaining,
                    maximum_response_bytes=MAX_WEB_RESPONSE_BYTES,
                    cancelled=lambda: self._cancelled(session),
                    authorize_dispatch=lambda dispatch: self._authorize_dispatch(
                        session, attempt_epoch, dispatch
                    ),
                )
                with self._authority_lock:
                    if session._wire_epoch != attempt_epoch:
                        raise GeneralWebTransportError(
                            "transport bypassed the dispatch authority fence"
                        )
                self._remaining(deadline)
            except GeneralWebTransportError:
                self._require_session(session)
                if attempt < self.maximum_retries:
                    remaining = self._remaining(deadline)
                    if not self.retry_waiter(
                        min(MAX_WEB_RETRY_DELAY_SECONDS, remaining),
                        lambda: self._cancelled(session),
                    ):
                        self._require_session(session)
                    self._remaining(deadline)
                    continue
                raise GeneralWebSearchError(
                    "WEB_SEARCH_NETWORK_ERROR",
                    "the official search API could not be reached",
                ) from None
            except GeneralWebSearchError:
                raise
            except Exception:  # noqa: BLE001 - transport trust boundary
                self._require_session(session)
                raise GeneralWebSearchError(
                    "WEB_SEARCH_NETWORK_ERROR",
                    "the official search API transport failed closed",
                ) from None
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
                return response, attempt_epoch
            if response.status in _RETRIABLE_STATUS and attempt < self.maximum_retries:
                delay = _retry_delay(response.headers.get("retry-after"))
                remaining = self._remaining(deadline)
                if not self.retry_waiter(
                    min(delay, remaining), lambda: self._cancelled(session)
                ):
                    self._require_session(session)
                self._remaining(deadline)
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

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self.monotonic()
        if remaining <= 0.0:
            raise GeneralWebSearchError(
                "WEB_SEARCH_TIMEOUT", "web search exceeded its total wall deadline"
            )
        return remaining


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
    except (UnicodeError, ValueError):
        raise GeneralWebTransportError("web search query string is invalid") from None
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
    secret_fragments = _secret_fragments(secret_token)
    if any(fragment.encode("utf-8") in response.body for fragment in secret_fragments):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID",
            "web search response reflected configured credential material",
        )
    try:
        payload = json.loads(
            response.body.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, ValueError):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search response JSON is invalid"
        ) from None
    if not isinstance(payload, dict):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search response must be an object"
        )
    _validate_json_shape(payload)
    if _json_contains_secret(payload, secret_fragments):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID",
            "web search response reflected configured credential material",
        )
    return payload


def _secret_fragments(secret_token: str) -> frozenset[str]:
    if not secret_token:
        return frozenset()
    return frozenset(
        {
            secret_token,
            quote(secret_token, safe=""),
            quote(secret_token, safe="", encoding="utf-8", errors="strict"),
        }
    )


def _json_contains_secret(payload: object, fragments: frozenset[str]) -> bool:
    if not fragments:
        return False
    stack = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            if any(fragment and fragment in value for fragment in fragments):
                return True
        elif isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return False


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
    except (UnicodeError, ValueError):
        raise GeneralWebSearchError(
            "WEB_SEARCH_RESPONSE_INVALID", "web search result URL is invalid"
        ) from None
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
    "GeneralWebResolver",
    "GeneralWebSearchClient",
    "GeneralWebSearchConfig",
    "GeneralWebSearchError",
    "GeneralWebSearchResponse",
    "GeneralWebSearchResult",
    "GeneralWebSearchSession",
    "GeneralWebTransportError",
    "MAX_WEB_RESPONSE_BYTES",
    "MAX_WEB_RESULTS",
    "MIN_WEB_TOKEN_CHARS",
    "StandardLibraryGeneralWebTransport",
]
