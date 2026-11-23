"""In-memory MCP servers for tests and the evaluation: the same tools as the stdio servers, without subprocesses."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.shared.memory import create_connected_server_and_client_session

from travel_agent.config import Settings
from travel_agent.mcp_servers import flights, travel_sandbox, trip_context
from travel_agent.memory.store import MemoryStore
from travel_agent.tool_gateway import ToolGateway


@asynccontextmanager
async def memory_gateway(settings: Settings, store: MemoryStore | None = None, **kwargs: Any) -> AsyncIterator[ToolGateway]:
    servers = {
        "flights": flights.create_server(settings),
        "travel-sandbox": travel_sandbox.create_server(settings),
        "trip-context": trip_context.create_server(settings),
    }
    factories = {name: (lambda server=server: create_connected_server_and_client_session(server))
                 for name, server in servers.items()}
    gateway = ToolGateway(settings, connections={}, session_factories=factories, approval_verifier=store,
                          trace_sink=store.add_trace if store else None, **kwargs)
    async with gateway:
        yield gateway
