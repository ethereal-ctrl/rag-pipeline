"""Persistent model responses and their original call durations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

CACHE_FILE = Path("data/cache/models.sqlite3")


def cache_key(*parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelResponse[T]:
    value: T
    seconds: float
    cache_hit: bool = False


def measure_call[T](call: Callable[[], T]) -> ModelResponse[T]:
    started = time.perf_counter()
    value = call()
    return ModelResponse(value, time.perf_counter() - started)


class ResponseCache:
    def __init__(self, path: Path = CACHE_FILE) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "key TEXT PRIMARY KEY, payload BLOB NOT NULL, seconds REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS generation_plans ("
                "key TEXT PRIMARY KEY, response_key TEXT NOT NULL)"
            )

    def contains(self, key: str) -> bool:
        with closing(sqlite3.connect(self.path)) as db:
            return (
                db.execute("SELECT 1 FROM responses WHERE key = ?", (key,)).fetchone() is not None
            )

    def contains_plan(self, key: str) -> bool:
        with closing(sqlite3.connect(self.path)) as db:
            return (
                db.execute(
                    "SELECT 1 FROM generation_plans JOIN responses "
                    "ON responses.key = generation_plans.response_key "
                    "WHERE generation_plans.key = ?",
                    (key,),
                ).fetchone()
                is not None
            )

    def remember_plan(self, key: str, response_key: str) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO generation_plans VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET response_key = excluded.response_key",
                (key, response_key),
            )

    def get_or_call[T](
        self, key: str, adapter: TypeAdapter[T], call: Callable[[], ModelResponse[T]]
    ) -> ModelResponse[T]:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT payload, seconds FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is not None:
            return ModelResponse(adapter.validate_json(row[0]), row[1], cache_hit=True)
        response = call()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO responses VALUES (?, ?, ?)",
                (key, adapter.dump_json(response.value), response.seconds),
            )
        return response
