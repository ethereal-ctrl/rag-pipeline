"""Persist reservations across processes and throttle remote calls over rolling windows."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from importlib.resources import files
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, PositiveInt, TypeAdapter

MINUTE = 60
DAY = 86400


class Clock(Protocol):
    def time(self) -> float: ...

    def sleep(self, seconds: float, /) -> None: ...


class RateLimits(BaseModel):
    requests_per_minute: PositiveInt
    tokens_per_minute: PositiveInt
    requests_per_day: PositiveInt
    tokens_per_day: PositiveInt | None = None


class _PublishedLimits(RateLimits):
    source: str
    looked_up: str


def groq_limits(model: str) -> _PublishedLimits:
    data = yaml.safe_load(files("rag_pipeline").joinpath("rate_limits.yaml").read_text("utf-8"))
    limits = TypeAdapter(dict[str, _PublishedLimits]).validate_python(data)
    if model not in limits or limits[model].tokens_per_day is None:
        raise ValueError(f"no verified Groq Free Plan limits for {model!r}")
    return limits[model]


def _wait(
    events: list[tuple[float, int]],
    now: float,
    window: int,
    request_limit: int,
    token_limit: int | None,
    tokens: int,
) -> float:
    relevant = [(at, n) for at, n in events if at > now - window]
    remaining_tokens = sum(n for _, n in relevant) + tokens
    remaining_requests = len(relevant) + 1
    delay = 0.0
    for at, used_tokens in relevant:
        if remaining_requests <= request_limit and (
            token_limit is None or remaining_tokens <= token_limit
        ):
            break
        delay = at + window - now
        remaining_requests -= 1
        remaining_tokens -= used_tokens
    return delay


class RateLimiter:
    def __init__(self, path: Path, limits: RateLimits, *, clock: Clock = time) -> None:
        self._path = path
        self.limits = limits
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS remote_requests ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, "
                "at REAL NOT NULL, tokens INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS remote_requests_by_model ON remote_requests(model, at)"
            )

    def reserve(self, model: str, estimated_tokens: int) -> int:
        limits = self.limits
        # A large estimate takes the entire minute's allowance. Reconcile against usage on success.
        tokens = min(
            estimated_tokens, limits.tokens_per_minute, limits.tokens_per_day or estimated_tokens
        )
        while True:
            with closing(sqlite3.connect(self._path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                now = self.clock.time()
                db.execute(
                    "DELETE FROM remote_requests WHERE model = ? AND at <= ?", (model, now - DAY)
                )
                events: list[tuple[float, int]] = db.execute(
                    "SELECT at, tokens FROM remote_requests WHERE model = ? ORDER BY at",
                    (model,),
                ).fetchall()
                delay = max(
                    _wait(
                        events,
                        now,
                        MINUTE,
                        limits.requests_per_minute,
                        limits.tokens_per_minute,
                        tokens,
                    ),
                    _wait(events, now, DAY, limits.requests_per_day, limits.tokens_per_day, tokens),
                )
                if delay <= 0:
                    request_id = db.execute(
                        "INSERT INTO remote_requests(model, at, tokens) VALUES (?, ?, ?)",
                        (model, now, tokens),
                    ).lastrowid
                    assert request_id is not None
                    return request_id
            self.clock.sleep(delay)

    def record_usage(self, request_id: int, tokens: int) -> None:
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute("UPDATE remote_requests SET tokens = ? WHERE id = ?", (tokens, request_id))
