"""Persist reservations across processes and throttle remote calls over rolling windows."""

from __future__ import annotations

import math
import os
import sqlite3
import time
from contextlib import closing
from importlib.resources import files
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, PositiveInt, TypeAdapter

MINUTE = 60
DAY = 86400
MONTH = 31 * DAY  # Conservative rolling window across calendar months.
type RemoteProvider = Literal["groq", "mistral"]


class Clock(Protocol):
    def time(self) -> float: ...

    def sleep(self, seconds: float, /) -> None: ...


class RateLimits(BaseModel):
    requests_per_second: PositiveInt | None = None
    requests_per_minute: PositiveInt | None = None
    tokens_per_minute: PositiveInt | None = None
    requests_per_day: PositiveInt | None = None
    tokens_per_day: PositiveInt | None = None
    requests_per_month: PositiveInt | None = None
    tokens_per_month: PositiveInt | None = None

    def windows(self) -> list[tuple[int, int | None, int | None]]:
        return [
            (seconds, requests, tokens)
            for seconds, requests, tokens in (
                (1, self.requests_per_second, None),
                (MINUTE, self.requests_per_minute, self.tokens_per_minute),
                (DAY, self.requests_per_day, self.tokens_per_day),
                (MONTH, self.requests_per_month, self.tokens_per_month),
            )
            if requests is not None or tokens is not None
        ]


class _PublishedLimits(RateLimits):
    provider: RemoteProvider = "groq"
    source: str
    looked_up: str


_MISTRAL_REQUIRED = ("requests_per_second", "tokens_per_minute", "tokens_per_month")


def remote_limits(provider: str, model: str) -> _PublishedLimits:
    data = yaml.safe_load(files("rag_pipeline").joinpath("rate_limits.yaml").read_text("utf-8"))
    limits = TypeAdapter(dict[str, _PublishedLimits]).validate_python(data)
    selected = limits.get(model)
    if selected is None or selected.provider != provider:
        raise ValueError(f"no verified {provider.title()} limits for {model!r}")
    if provider == "mistral":
        fields = selected.model_dump()
        for field in (*_MISTRAL_REQUIRED, "requests_per_month"):
            setting = f"MISTRAL_{field.upper()}"
            if value := os.getenv(setting, "").strip():
                fields[field] = value
        missing = [
            f"MISTRAL_{field.upper()}" for field in _MISTRAL_REQUIRED if fields[field] is None
        ]
        if missing:
            raise ValueError(
                f"no verified Mistral limits for {model!r}. "
                f"Confirm API access and set {', '.join(missing)} from your organization limits."
            )
        return _PublishedLimits.model_validate(fields)
    if selected.tokens_per_day is None:
        raise ValueError(f"no verified Groq Free Plan limits for {model!r}")
    return selected


def groq_limits(model: str) -> _PublishedLimits:
    return remote_limits("groq", model)


def quota_days(limits: RateLimits, calls: int, tokens: int) -> int:
    """Estimate whole quota days for an uncached workload."""
    return math.ceil(
        max(
            (
                seconds
                / DAY
                * max(calls / (request_limit or math.inf), tokens / (token_limit or math.inf))
                for seconds, request_limit, token_limit in limits.windows()
            ),
            default=0.0,
        )
    )


def _wait(
    events: list[tuple[float, int]],
    now: float,
    window: int,
    request_limit: int | None,
    token_limit: int | None,
    tokens: int,
) -> float:
    relevant = [(at, n) for at, n in events if at > now - window]
    remaining_tokens = sum(n for _, n in relevant) + tokens
    remaining_requests = len(relevant) + 1
    delay = 0.0
    for at, used_tokens in relevant:
        if (request_limit is None or remaining_requests <= request_limit) and (
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
        windows = self.limits.windows()
        horizon = max((seconds for seconds, _, _ in windows), default=DAY)
        # A large estimate takes the entire minute's allowance. Reconcile against usage on success.
        tokens = min([estimated_tokens, *(cap for _, _, cap in windows if cap is not None)])
        while True:
            with closing(sqlite3.connect(self._path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                now = self.clock.time()
                db.execute(
                    "DELETE FROM remote_requests WHERE model = ? AND at <= ?",
                    (model, now - horizon),
                )
                events: list[tuple[float, int]] = db.execute(
                    "SELECT at, tokens FROM remote_requests WHERE model = ? ORDER BY at",
                    (model,),
                ).fetchall()
                delay = max(
                    (
                        _wait(events, now, seconds, requests, cap, tokens)
                        for seconds, requests, cap in windows
                    ),
                    default=0.0,
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
