"""Runtime dependencies passed to every graph node as LangGraph runtime context (never checkpointed)."""
from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer

from travel_agent.agents.common import AgentDeps
from travel_agent.config import Settings
from travel_agent.memory.store import MemoryStore
from travel_agent.models.results import TraceEvent
from travel_agent.tool_gateway import RunContext, ToolGateway


@dataclass
class Deps:
    settings: Settings
    gateway: ToolGateway
    store: MemoryStore
    trip_id: str
    run_id: str
    llm: Any | None = None

    @property
    def run(self) -> RunContext:
        return RunContext(self.trip_id, self.run_id)

    def agent_deps(self) -> AgentDeps:
        return AgentDeps(self.gateway, self.run)

    def trace(self, event: str, node: str, status: str = "ok", duration_ms: int | None = None, **detail: Any) -> None:
        self.store.add_trace(TraceEvent(trip_id=self.trip_id, run_id=self.run_id, event=event, node=node,
                                        status=status, duration_ms=duration_ms, detail=detail))


def emit(event_type: str, **data: Any) -> None:
    """Send a progress event to the UI (LangGraph custom stream); a no-op outside streaming runs."""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - not inside a streaming graph run
        return
    writer({"type": event_type, **data})


NodeFn = Callable[..., Awaitable[dict[str, Any]]]


def traced(name: str) -> Callable[[NodeFn], NodeFn]:
    """Record each node's duration and outcome in the trace table."""

    def wrap(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def node(state: Any, runtime: Any) -> dict[str, Any]:
            deps: Deps = runtime.context
            started = time.perf_counter()
            status = "ok"
            try:
                return await fn(state, runtime)
            except Exception as exc:
                status = type(exc).__name__
                raise
            finally:
                if status != "GraphInterrupt":
                    deps.trace("node", name, status, int((time.perf_counter() - started) * 1000))

        return node

    return wrap
