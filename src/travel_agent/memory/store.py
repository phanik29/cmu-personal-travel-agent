"""Long-term memory in SQLite: trips, preferences, option lifecycle, approvals, bookings, and traces."""
from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent.models.planning import ApprovalSnapshot, OptionStatus
from travel_agent.models.results import TraceEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trips (
    trip_id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL, request_json TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
    payload_json TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY, value_json TEXT NOT NULL, source TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS itinerary_items (
    trip_id TEXT NOT NULL, option_id TEXT NOT NULL, kind TEXT NOT NULL, provider TEXT NOT NULL, label TEXT NOT NULL,
    status TEXT NOT NULL, price_total REAL NOT NULL, currency TEXT NOT NULL, option_json TEXT NOT NULL,
    retrieved_at TEXT, booking_ref TEXT, updated_at TEXT NOT NULL, PRIMARY KEY (trip_id, option_id));
CREATE TABLE IF NOT EXISTS status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id TEXT NOT NULL, option_id TEXT NOT NULL, from_status TEXT,
    to_status TEXT NOT NULL, reason TEXT, approval_id TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY, trip_id TEXT NOT NULL, run_id TEXT, action TEXT NOT NULL, option_id TEXT NOT NULL,
    booking_ref TEXT, snapshot_json TEXT NOT NULL, decision TEXT NOT NULL, created_at TEXT NOT NULL,
    used_at TEXT, expired_at TEXT);
CREATE TABLE IF NOT EXISTS bookings (
    booking_ref TEXT PRIMARY KEY, trip_id TEXT NOT NULL, kind TEXT NOT NULL, option_id TEXT NOT NULL,
    provider TEXT NOT NULL, status TEXT NOT NULL, total_price REAL NOT NULL, currency TEXT NOT NULL,
    record_json TEXT NOT NULL, verified_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id TEXT, kind TEXT NOT NULL, subject TEXT NOT NULL,
    sentiment TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id TEXT, run_id TEXT, ts TEXT NOT NULL, event TEXT NOT NULL,
    node TEXT, server TEXT, tool TEXT, status TEXT, duration_ms INTEGER, detail_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_trace_trip ON trace_events(trip_id, id);
CREATE INDEX IF NOT EXISTS idx_items_trip ON itinerary_items(trip_id, status);
"""

S = OptionStatus
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    S.SEARCHED: {S.RECOMMENDED, S.SUPERSEDED},
    S.RECOMMENDED: {S.APPROVED, S.REJECTED, S.SUPERSEDED, S.SEARCHED},
    S.APPROVED: {S.BOOKED, S.FAILED, S.SUPERSEDED},
    S.BOOKED: {S.CANCELLED},
    S.REJECTED: {S.RECOMMENDED, S.SUPERSEDED, S.SEARCHED},
    S.FAILED: {S.RECOMMENDED, S.SUPERSEDED, S.SEARCHED},
    S.SUPERSEDED: {S.RECOMMENDED, S.SEARCHED},
    S.CANCELLED: set(),
}


class LifecycleError(ValueError):
    pass


class ApprovalError(PermissionError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(5)}"


def option_label(option: Mapping[str, Any]) -> str:
    kind = option.get("kind")
    if kind == "flight":
        out = option["outbound"]
        return f"{option['airline']} {out['flight_number']}, {out['origin']} to {out['destination']}"
    if kind == "hotel":
        return option["name"]
    if kind == "car":
        return f"{option['vendor']} {option['vehicle_class']}"
    if kind == "transfer":
        return f"{option['vendor']} {option['service']}"
    return str(option.get("option_id"))


class MemoryStore:
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

    # Trips and messages ------------------------------------------------------------------------------------------
    def create_trip(self, title: str = "New trip") -> str:
        trip_id = new_id("trip")
        now = _now()
        with self._connect() as conn:
            conn.execute("INSERT INTO trips(trip_id, title, status, request_json, created_at, updated_at) "
                         "VALUES (?,?,?,?,?,?)", (trip_id, title, "planning", None, now, now))
        return trip_id

    def get_trip(self, trip_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trips WHERE trip_id=?", (trip_id,)).fetchone()
        if row is None:
            return None
        trip = dict(row)
        trip["request"] = json.loads(trip.pop("request_json")) if trip["request_json"] else None
        return trip

    def list_trips(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT t.trip_id, t.title, t.status, t.updated_at, "
                "(SELECT COUNT(*) FROM itinerary_items i WHERE i.trip_id=t.trip_id AND i.status='booked') AS booked "
                "FROM trips t ORDER BY t.updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def update_trip(self, trip_id: str, *, title: str | None = None, status: str | None = None,
                    request: Mapping[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trips SET title=COALESCE(?, title), status=COALESCE(?, status), "
                "request_json=COALESCE(?, request_json), updated_at=? WHERE trip_id=?",
                (title, status, json.dumps(request, default=str) if request is not None else None, _now(), trip_id))

    def add_message(self, trip_id: str, role: str, content: str, payload: Mapping[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO messages(trip_id, role, content, payload_json, created_at) VALUES (?,?,?,?,?)",
                         (trip_id, role, content, json.dumps(payload, default=str) if payload else None, _now()))

    def list_messages(self, trip_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT role, content, payload_json, created_at FROM messages WHERE trip_id=? "
                                "ORDER BY id DESC LIMIT ?", (trip_id, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"],
                 "payload": json.loads(r["payload_json"]) if r["payload_json"] else None} for r in reversed(rows)]

    # Preferences and feedback ------------------------------------------------------------------------------------
    def set_preference(self, key: str, value: Any, source: str = "stated") -> None:
        if source not in {"stated", "learned"}:
            raise ValueError("source must be 'stated' or 'learned'")
        with self._connect() as conn:
            row = conn.execute("SELECT value_json, source, count FROM preferences WHERE key=?", (key,)).fetchone()
            if row and row["source"] == "stated" and source == "learned":
                return  # never let a learned preference overwrite something the user said
            count = row["count"] + 1 if row and json.loads(row["value_json"]) == value else 1
            conn.execute("INSERT OR REPLACE INTO preferences(key, value_json, source, count, updated_at) "
                         "VALUES (?,?,?,?,?)", (key, json.dumps(value), source, count, _now()))

    def get_preferences(self) -> dict[str, dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM preferences ORDER BY key").fetchall()
        return {r["key"]: {"value": json.loads(r["value_json"]), "source": r["source"], "count": r["count"],
                           "updated_at": r["updated_at"]} for r in rows}

    def delete_preference(self, key: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM preferences WHERE key=?", (key,)).rowcount == 1

    def add_feedback(self, trip_id: str | None, kind: str, subject: str, sentiment: str, note: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO feedback(trip_id, kind, subject, sentiment, note, created_at) VALUES (?,?,?,?,?,?)",
                         (trip_id, kind, subject, sentiment, note, _now()))

    def list_feedback(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # Itinerary items and lifecycle -------------------------------------------------------------------------------
    def record_options(self, trip_id: str, options: Iterable[Mapping[str, Any]]) -> None:
        """Record searched options; existing items keep their status but get the latest price."""
        now = _now()
        with self._connect() as conn:
            for option in options:
                source = option.get("source", {})
                existing = conn.execute("SELECT status FROM itinerary_items WHERE trip_id=? AND option_id=?",
                                        (trip_id, option["option_id"])).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO itinerary_items(trip_id, option_id, kind, provider, label, status, price_total,"
                        " currency, option_json, retrieved_at, booking_ref, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (trip_id, option["option_id"], option["kind"], source.get("provider", "unknown"),
                         option_label(option), S.SEARCHED.value, option["price_total"], option.get("currency", "USD"),
                         json.dumps(option, default=str), source.get("retrieved_at"), None, now))
                    conn.execute("INSERT INTO status_history(trip_id, option_id, from_status, to_status, reason,"
                                 " approval_id, created_at) VALUES (?,?,?,?,?,?,?)",
                                 (trip_id, option["option_id"], None, S.SEARCHED.value, "search result", None, now))
                elif existing["status"] in {S.SEARCHED, S.RECOMMENDED, S.REJECTED, S.SUPERSEDED, S.FAILED}:
                    conn.execute("UPDATE itinerary_items SET option_json=?, price_total=?, retrieved_at=?, updated_at=? "
                                 "WHERE trip_id=? AND option_id=?",
                                 (json.dumps(option, default=str), option["price_total"], source.get("retrieved_at"),
                                  now, trip_id, option["option_id"]))

    def get_item(self, trip_id: str, option_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM itinerary_items WHERE trip_id=? AND option_id=?",
                               (trip_id, option_id)).fetchone()
        return self._item(row) if row else None

    def list_items(self, trip_id: str, statuses: Iterable[str] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM itinerary_items WHERE trip_id=?"
        params: list[Any] = [trip_id]
        if statuses is not None:
            wanted = [str(getattr(s, "value", s)) for s in statuses]
            query += f" AND status IN ({','.join('?' * len(wanted))})"
            params += wanted
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY updated_at", params).fetchall()
        return [self._item(row) for row in rows]

    @staticmethod
    def _item(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["option"] = json.loads(item.pop("option_json"))
        return item

    def status_history(self, trip_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM status_history WHERE trip_id=? ORDER BY id", (trip_id,)).fetchall()
        return [dict(r) for r in rows]

    def set_item_status(self, trip_id: str, option_id: str, status: str | OptionStatus, *, reason: str = "",
                        approval_id: str | None = None, booking_ref: str | None = None,
                        by_provider: bool = False) -> None:
        new = OptionStatus(status)
        with self._connect() as conn:
            row = conn.execute("SELECT status, booking_ref FROM itinerary_items WHERE trip_id=? AND option_id=?",
                               (trip_id, option_id)).fetchone()
            if row is None:
                raise LifecycleError(f"Unknown item {option_id}")
            current = OptionStatus(row["status"])
            if current == new:
                return
            if new not in ALLOWED_TRANSITIONS[current]:
                raise LifecycleError(f"Cannot change {option_id} from {current.value} to {new.value}")
            if new == S.BOOKED:
                self._require_used_approval(conn, approval_id, trip_id, option_id, prefix="book_")
            if new == S.CANCELLED and not by_provider:
                self._require_used_approval(conn, approval_id, trip_id, option_id, prefix="cancel_")
            now = _now()
            conn.execute("UPDATE itinerary_items SET status=?, booking_ref=COALESCE(?, booking_ref), updated_at=? "
                         "WHERE trip_id=? AND option_id=?", (new.value, booking_ref, now, trip_id, option_id))
            conn.execute("INSERT INTO status_history(trip_id, option_id, from_status, to_status, reason, approval_id,"
                         " created_at) VALUES (?,?,?,?,?,?,?)",
                         (trip_id, option_id, current.value, new.value, reason, approval_id, now))

    @staticmethod
    def _require_used_approval(conn: sqlite3.Connection, approval_id: str | None, trip_id: str, option_id: str,
                               prefix: str) -> None:
        if not approval_id:
            raise LifecycleError(f"Changing {option_id} requires an approval")
        row = conn.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
        if row is None or row["trip_id"] != trip_id or row["option_id"] != option_id:
            raise LifecycleError("The approval does not match this item")
        if not row["action"].startswith(prefix) or row["used_at"] is None:
            raise LifecycleError("The approval was not used for this action")

    # Approvals ---------------------------------------------------------------------------------------------------
    def create_approval(self, trip_id: str, snapshot: ApprovalSnapshot, run_id: str | None = None) -> str:
        approval_id = new_id("apv")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO approvals(approval_id, trip_id, run_id, action, option_id, booking_ref, snapshot_json,"
                " decision, created_at, used_at, expired_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (approval_id, trip_id, run_id, snapshot.action, snapshot.option_id, snapshot.booking_ref,
                 snapshot.model_dump_json(), "approved", _now(), None, None))
        return approval_id

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
        if row is None:
            return None
        approval = dict(row)
        approval["snapshot"] = json.loads(approval.pop("snapshot_json"))
        return approval

    def list_approvals(self, trip_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT approval_id FROM approvals WHERE trip_id=? ORDER BY created_at",
                                (trip_id,)).fetchall()
        return [self.get_approval(r["approval_id"]) for r in rows]  # type: ignore[misc]

    def verify_and_consume(self, *, approval_id: str, action: str, args: Mapping[str, Any],
                           trip_id: str | None) -> dict[str, Any]:
        """Gateway hook: the approval must exist, be unused, and match the exact item, price, and booking."""
        approval = self.get_approval(approval_id) if approval_id else None
        if approval is None:
            raise ApprovalError("Unknown approval id")
        if trip_id is not None and approval["trip_id"] != trip_id:
            raise ApprovalError("The approval belongs to a different trip")
        if approval["decision"] != "approved" or approval["expired_at"]:
            raise ApprovalError("The approval is no longer valid")
        if approval["used_at"]:
            raise ApprovalError("The approval was already used")
        snapshot = ApprovalSnapshot.model_validate(approval["snapshot"])
        expected = {"cancel_hotel", "cancel_car", "cancel_transfer"} if action == "cancel_booking" else {action}
        if snapshot.action not in expected:
            raise ApprovalError(f"The approval covers {snapshot.action}, not {action}")
        if action.startswith("book_"):
            option_id = args.get("offer_id") or args.get("option_id")
            if option_id != snapshot.option_id:
                raise ApprovalError("The approval covers a different option")
            if abs(float(args.get("expected_total_price", -1)) - snapshot.total_price) > 0.009:
                raise ApprovalError("The price differs from the approved price")
            external = args.get("option_snapshot")
            if external:
                for key, value in snapshot.dates.items():
                    if str(external.get(key)) != value:
                        raise ApprovalError(f"The {key} differs from the approved dates")
        elif args.get("booking_ref") != snapshot.booking_ref:
            raise ApprovalError("The approval covers a different booking")
        with self._connect() as conn:
            used = conn.execute("UPDATE approvals SET used_at=? WHERE approval_id=? AND used_at IS NULL",
                                (_now(), approval_id)).rowcount
        if used != 1:
            raise ApprovalError("The approval was already used")
        return snapshot.model_dump()

    def expire_unused_approvals(self, trip_id: str) -> int:
        with self._connect() as conn:
            return conn.execute("UPDATE approvals SET expired_at=? WHERE trip_id=? AND used_at IS NULL "
                                "AND expired_at IS NULL", (_now(), trip_id)).rowcount

    # Bookings ----------------------------------------------------------------------------------------------------
    def save_booking(self, trip_id: str, record: Mapping[str, Any], *, verified: bool) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bookings(booking_ref, trip_id, kind, option_id, provider, status, total_price, currency,"
                " record_json, verified_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(booking_ref) DO UPDATE SET status=excluded.status, record_json=excluded.record_json, "
                "verified_at=COALESCE(excluded.verified_at, bookings.verified_at), updated_at=excluded.updated_at",
                (record["booking_ref"], trip_id, record["kind"], record["option_id"], record["provider"],
                 record["status"], record["total_price"], record["currency"], json.dumps(record, default=str),
                 now if verified else None, now, now))

    def list_bookings(self, trip_id: str | None = None, statuses: Iterable[str] | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM bookings WHERE 1=1", []
        if trip_id is not None:
            query += " AND trip_id=?"
            params.append(trip_id)
        if statuses is not None:
            wanted = list(statuses)
            query += f" AND status IN ({','.join('?' * len(wanted))})"
            params += wanted
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY created_at", params).fetchall()
        bookings = []
        for row in rows:
            booking = dict(row)
            booking["record"] = json.loads(booking.pop("record_json"))
            bookings.append(booking)
        return bookings

    def get_booking(self, booking_ref: str) -> dict[str, Any] | None:
        found = [b for b in self.list_bookings() if b["booking_ref"] == booking_ref]
        return found[0] if found else None

    # Trace -------------------------------------------------------------------------------------------------------
    def add_trace(self, event: TraceEvent) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO trace_events(trip_id, run_id, ts, event, node, server, tool, status, duration_ms,"
                         " detail_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (event.trip_id, event.run_id, event.ts.isoformat(), event.event, event.node, event.server,
                          event.tool, event.status, event.duration_ms, json.dumps(event.detail, default=str)))

    def list_trace(self, trip_id: str, run_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM trace_events WHERE trip_id=?", [trip_id]
        if run_id is not None:
            query += " AND run_id=?"
            params.append(run_id)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY id DESC LIMIT ?", [*params, limit]).fetchall()
        events = []
        for row in reversed(rows):
            event = dict(row)
            event["detail"] = json.loads(event.pop("detail_json"))
            events.append(event)
        return events
