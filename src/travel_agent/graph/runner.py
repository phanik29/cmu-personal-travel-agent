"""Runs the graph for a trip: one run per chat message or decision, streamed as progress events.

A per-trip lock keeps runs on the same checkpoint thread sequential. If a run crashes, completed work stays in the
checkpoint and “try again” resumes from the failed step instead of starting over.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from travel_agent import text
from travel_agent.config import Settings
from travel_agent.graph.build import build_graph
from travel_agent.graph.deps import Deps
from travel_agent.memory.store import MemoryStore, new_id
from travel_agent.security import sanitize_untrusted_text
from travel_agent.tool_gateway import ToolGateway

RECURSION_LIMIT = 100
MAX_MESSAGE_CHARS = 2000
_RESUME_WORDS = re.compile(r"\b(try again|retry|resume|continue)\b", re.I)


class TripRunner:
    def __init__(self, settings: Settings, gateway: ToolGateway, store: MemoryStore, llm: Any | None = None,
                 checkpointer: Any | None = None) -> None:
        self.settings = settings
        self.gateway = gateway
        self.store = store
        self.llm = llm
        self._checkpointer = checkpointer
        self._stack: AsyncExitStack | None = None
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.graph: Any = None

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        if self._checkpointer is None:
            self.settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
            self._checkpointer = await self._stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(self.settings.checkpoint_db)))
        self.graph = build_graph(self._checkpointer)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    async def __aenter__(self) -> TripRunner:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @staticmethod
    def config(trip_id: str, run_id: str = "inspect") -> dict[str, Any]:
        return {"configurable": {"thread_id": trip_id, "run_id": run_id}, "recursion_limit": RECURSION_LIMIT}

    async def snapshot(self, trip_id: str) -> Any:
        return await self.graph.aget_state(self.config(trip_id))

    async def pending_interrupt(self, trip_id: str) -> dict[str, Any] | None:
        snapshot = await self.snapshot(trip_id)
        for task in snapshot.tasks:
            for item in task.interrupts:
                return item.value
        return None

    async def is_resumable(self, trip_id: str) -> bool:
        """True when a run stopped part-way (e.g., a crash) and has unfinished steps to resume."""
        snapshot = await self.snapshot(trip_id)
        return bool(snapshot.next) and not any(task.interrupts for task in snapshot.tasks)

    async def values(self, trip_id: str) -> dict[str, Any]:
        return dict((await self.snapshot(trip_id)).values or {})

    # Entry points ------------------------------------------------------------------------------------------------
    async def send_message(self, trip_id: str, text: str) -> AsyncIterator[dict[str, Any]]:
        text = sanitize_untrusted_text(text, MAX_MESSAGE_CHARS)
        async with self._locks[trip_id]:
            self.store.add_message(trip_id, "user", text)
            pending = await self.pending_interrupt(trip_id)
            if pending is not None:
                payload: Any = Command(resume={"decision": "message", "text": text})
            elif _RESUME_WORDS.search(text) and await self.is_resumable(trip_id):
                payload = None
            else:
                payload = {"user_message": text, "intent": None, "decision": None}
            async for event in self._run(trip_id, payload):
                yield event

    async def decide(self, trip_id: str, decision: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Start the search for a shown plan (kind "start"), approve or reject options, or approve or decline a
        cancellation (kind "cancel")."""
        decision = {k: v for k, v in decision.items() if k in {"decision", "candidate_id", "feedback", "kind"}}
        if decision.get("feedback"):
            decision["feedback"] = sanitize_untrusted_text(decision["feedback"], MAX_MESSAGE_CHARS)
        async with self._locks[trip_id]:
            proposal = (await self.values(trip_id)).get("proposal") or {}
            self.store.add_message(trip_id, "user", describe_decision(decision, proposal.get("candidates") or []),
                                   {"decision": decision})
            pending = await self.pending_interrupt(trip_id)
            expected = {"cancel": "cancel_approval", "start": "plan_approval"}.get(decision.get("kind"), "approval")
            if pending is not None and pending.get("type") == expected:
                payload: Any = Command(resume=decision)
            else:
                payload = {"intent": "decision", "decision": decision, "user_message": decision.get("feedback") or ""}
            async for event in self._run(trip_id, payload):
                yield event

    async def collect(self, events: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
        return [event async for event in events]

    # Internals ---------------------------------------------------------------------------------------------------
    async def _run(self, trip_id: str, payload: Any) -> AsyncIterator[dict[str, Any]]:
        run_id = new_id("run")
        deps = Deps(self.settings, self.gateway, self.store, trip_id, run_id, self.llm)
        config = self.config(trip_id, run_id)
        started = time.perf_counter()
        reply, reply_kind = "", ""
        yield {"type": "start", "run_id": run_id}
        try:
            async for mode, chunk in self.graph.astream(payload, config, context=deps,
                                                        stream_mode=["custom", "updates"]):
                if mode == "custom":
                    if chunk.get("type") == "message":
                        self.store.add_message(trip_id, "assistant", chunk["text"], {"kind": chunk.get("kind")})
                    yield {**{k: v for k, v in chunk.items() if k != "type"}, "type": "progress",
                           "event": chunk.get("type")}
                    continue
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        yield {"type": "interrupt", "value": [getattr(i, "value", i) for i in update]}
                        continue
                    if isinstance(update, dict) and update.get("reply"):
                        reply, reply_kind = update["reply"], update.get("reply_kind", "")
                    yield {"type": "node", "node": node}
        except Exception as exc:  # noqa: BLE001 - report, keep the checkpoint, allow "try again"
            deps.trace("run", "runner", type(exc).__name__, int((time.perf_counter() - started) * 1000),
                       error=sanitize_untrusted_text(str(exc), 300))
            message = text.paragraphs("Sorry, something went wrong on my side. Your progress is saved.",
                                      'Next step: say "try again", and I\'ll continue from where I stopped.')
            self.store.add_message(trip_id, "assistant", message, {"kind": "error"})
            yield {"type": "error", "message": message, "run_id": run_id}
            return
        if reply:
            self.store.add_message(trip_id, "assistant", reply, {"kind": reply_kind})
        pending = await self.pending_interrupt(trip_id)
        duration_ms = int((time.perf_counter() - started) * 1000)
        deps.trace("run", "runner", "ok", duration_ms, reply_kind=reply_kind, tool_calls=self.gateway.calls_used(run_id))
        yield {"type": "done", "run_id": run_id, "reply": reply, "reply_kind": reply_kind,
               "awaiting": (pending or {}).get("type"), "duration_ms": duration_ms,
               "tool_calls": self.gateway.calls_used(run_id)}


def describe_decision(decision: dict[str, Any], candidates: list[dict[str, Any]] | None = None) -> str:
    """The user's button press, written as what they would have said."""
    action = decision.get("decision", "")
    if decision.get("kind") == "start":
        return "Go ahead and search" if action == "approve" else "Don't search yet"
    if decision.get("kind") == "cancel":
        return "Yes, cancel it" if action == "approve" else "No, keep my bookings"
    if action == "approve":
        for index, candidate in enumerate(candidates or [], 1):
            if candidate.get("candidate_id") == decision.get("candidate_id"):
                return f"Book option {index} ({text.money(candidate['total_price'])})"
        return "Book this option"
    if action == "change":
        return decision.get("feedback") or "I'd like to change something"
    return "None of these, thanks" + (f". {decision['feedback']}" if decision.get("feedback") else "")
