from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from time import time
from typing import Iterator

from .harness import FailureTrace


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    timestamp: float
    kind: str
    subject: str
    detail: str


class LogDB:
    """Local append-oriented failure and audit store; no network dependency."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    test_id TEXT NOT NULL,
                    exception_type TEXT NOT NULL,
                    verifier_code TEXT NOT NULL,
                    mechanism TEXT NOT NULL,
                    excerpt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
                """
            )

    def record_failure(
        self, trace: FailureTrace, timestamp: float | None = None
    ) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO failures(timestamp,test_id,exception_type,verifier_code,mechanism,excerpt) "
                "VALUES(?,?,?,?,?,?)",
                (
                    timestamp or time(),
                    trace.test_id,
                    trace.exception_type,
                    trace.verifier_code,
                    trace.mechanism,
                    trace.excerpt,
                ),
            )
            return int(cursor.lastrowid)

    def failures_since(self, timestamp: float) -> list[FailureTrace]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT test_id,exception_type,verifier_code,mechanism,excerpt "
                "FROM failures WHERE timestamp>=? ORDER BY id",
                (timestamp,),
            ).fetchall()
        return [FailureTrace(*row) for row in rows]

    def audit(self, kind: str, subject: str, detail: str) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO audit(timestamp,kind,subject,detail) VALUES(?,?,?,?)",
                (time(), kind, subject, detail),
            )
            return int(cursor.lastrowid)

    def audit_events(self) -> list[AuditEvent]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT sequence,timestamp,kind,subject,detail FROM audit ORDER BY sequence"
            ).fetchall()
        return [AuditEvent(*row) for row in rows]
