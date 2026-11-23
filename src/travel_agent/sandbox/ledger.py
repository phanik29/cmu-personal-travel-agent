"""Sandbox booking ledger: the only place sandbox bookings, price changes, and disruptions live."""
from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS offers (
    option_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    provider TEXT NOT NULL,
    base_price REAL NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS price_overrides (
    option_id TEXT PRIMARY KEY,
    price REAL NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS availability (
    option_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bookings (
    booking_ref TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    option_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    total_price REAL NOT NULL,
    currency TEXT NOT NULL,
    approval_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS once_flags (
    name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
"""

_REF_PREFIX = {"flight": "SBX-FL", "hotel": "SBX-HT", "car": "SBX-CR", "transfer": "SBX-TR"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SandboxLedger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_offers(self, options: list[dict[str, Any]]) -> None:
        now = _now()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO offers(option_id, kind, provider, base_price, payload, created_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(option_id) DO UPDATE SET payload=excluded.payload, base_price=excluded.base_price",
                [
                    (o["option_id"], o["kind"], o["source"]["provider"], o["price_total"], json.dumps(o, default=str), now)
                    for o in options
                ],
            )

    def get_offer(self, option_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM offers WHERE option_id=?", (option_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def price_override(self, option_id: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute("SELECT price FROM price_overrides WHERE option_id=?", (option_id,)).fetchone()
        return float(row["price"]) if row else None

    def set_price_override(self, option_id: str, price: float, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO price_overrides(option_id, price, reason, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(option_id) DO UPDATE SET price=excluded.price, reason=excluded.reason",
                (option_id, round(price, 2), reason, _now()),
            )

    def unavailable_reason(self, option_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT reason FROM availability WHERE option_id=?", (option_id,)).fetchone()
        return row["reason"] if row else None

    def mark_unavailable(self, option_id: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO availability(option_id, status, reason, created_at) VALUES (?,?,?,?)",
                (option_id, "unavailable", reason, _now()),
            )

    def first_time(self, name: str) -> bool:
        """True only the first time a named one-off event is triggered (used by test scenarios)."""
        with self._connect() as conn:
            cur = conn.execute("INSERT OR IGNORE INTO once_flags(name, created_at) VALUES (?,?)", (name, _now()))
            return cur.rowcount == 1

    def create_booking(
        self,
        *,
        kind: str,
        option_id: str,
        provider: str,
        total_price: float,
        currency: str,
        approval_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        ref = f"{_REF_PREFIX[kind]}-{secrets.token_hex(3).upper()}"
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bookings(booking_ref, kind, option_id, provider, status, total_price, currency, approval_id,"
                " payload, note, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ref, kind, option_id, provider, "confirmed", round(total_price, 2), currency, approval_id,
                 json.dumps(payload, default=str), None, now, now),
            )
        return self.get_booking(ref)  # type: ignore[return-value]

    def get_booking(self, booking_ref: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM bookings WHERE booking_ref=?", (booking_ref,)).fetchone()
        if row is None:
            return None
        booking = dict(row)
        booking["payload"] = json.loads(booking["payload"])
        return booking

    def update_booking_status(self, booking_ref: str, status: str, note: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE bookings SET status=?, note=COALESCE(?, note), updated_at=? WHERE booking_ref=?",
                (status, note, _now(), booking_ref),
            )
