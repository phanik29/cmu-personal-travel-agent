"""ToolGateway: the application's only MCP client. Every tool call passes through these checks:

role allowlist -> read-only name check -> pinned tool definition -> strict arguments -> call quota ->
approval check (action tools only) -> timeout and limited retries -> result validation -> trace log.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import jsonschema
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, types
from pydantic import BaseModel, ValidationError
from pydantic_core import to_jsonable_python

from travel_agent.config import Settings
from travel_agent.mcp_servers import LOCAL_SERVERS
from travel_agent.models.options import WeatherReport, WeatherRisk
from travel_agent.models.results import TraceEvent
from travel_agent.providers.base import (
    AirportLookupResult,
    BookingLookup,
    BookingResult,
    CancellationResult,
    CarSearchResult,
    FlightSearchResult,
    HotelSearchResult,
    OfferRefresh,
    TransferSearchResult,
)
from travel_agent.security import assert_tool_allowed_for_role, redact_secrets, sanitize_untrusted_text
from travel_agent.tool_registry import fingerprint, load_registry

SEARCH_TOOLS = frozenset({
    ("flights", "search_flights"), ("flights", "refresh_offer"),
    ("travel-sandbox", "search_hotels"), ("travel-sandbox", "search_cars"), ("travel-sandbox", "search_transfers"),
    ("trip-context", "lookup_airport"), ("trip-context", "nearby_airports"),
    ("trip-context", "get_weather"), ("trip-context", "weather_risk"),
})
ROLE_TOOLS: dict[str, frozenset[tuple[str, str]]] = {
    "search": SEARCH_TOOLS,
    "qa": frozenset(SEARCH_TOOLS - {("flights", "refresh_offer")}),
    "executor": frozenset({
        ("flights", "refresh_offer"), ("flights", "book_flight"), ("flights", "get_flight_order"),
        ("flights", "cancel_flight_order"), ("travel-sandbox", "book_hotel"), ("travel-sandbox", "book_car"),
        ("travel-sandbox", "book_transfer"), ("travel-sandbox", "get_booking"), ("travel-sandbox", "cancel_booking"),
    }),
}
# Tools that change sandbox state, mapped to the approval action they require.
ACTION_TOOLS: dict[tuple[str, str], str] = {
    ("flights", "book_flight"): "book_flight",
    ("travel-sandbox", "book_hotel"): "book_hotel",
    ("travel-sandbox", "book_car"): "book_car",
    ("travel-sandbox", "book_transfer"): "book_transfer",
    ("flights", "cancel_flight_order"): "cancel_flight",
    ("travel-sandbox", "cancel_booking"): "cancel_booking",
}
RESULT_MODELS: dict[tuple[str, str], type[BaseModel]] = {
    ("flights", "search_flights"): FlightSearchResult,
    ("flights", "refresh_offer"): OfferRefresh,
    ("flights", "book_flight"): BookingResult,
    ("flights", "get_flight_order"): BookingLookup,
    ("flights", "cancel_flight_order"): CancellationResult,
    ("travel-sandbox", "search_hotels"): HotelSearchResult,
    ("travel-sandbox", "search_cars"): CarSearchResult,
    ("travel-sandbox", "book_hotel"): BookingResult,
    ("travel-sandbox", "book_car"): BookingResult,
    ("travel-sandbox", "search_transfers"): TransferSearchResult,
    ("travel-sandbox", "book_transfer"): BookingResult,
    ("travel-sandbox", "get_booking"): BookingLookup,
    ("travel-sandbox", "cancel_booking"): CancellationResult,
    ("trip-context", "lookup_airport"): AirportLookupResult,
    ("trip-context", "nearby_airports"): AirportLookupResult,
    ("trip-context", "get_weather"): WeatherReport,
    ("trip-context", "weather_risk"): WeatherRisk,
}
_TRANSIENT_HINTS = ("503", "502", "504", "429", "temporar", "unavailable", "timed out", "connection reset")


class ToolCallError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


class ApprovalVerifier(Protocol):
    def verify_and_consume(self, *, approval_id: str, action: str, args: Mapping[str, Any],
                           trip_id: str | None) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RunContext:
    trip_id: str | None
    run_id: str
    node: str | None = None

    def at(self, node: str) -> RunContext:
        return RunContext(self.trip_id, self.run_id, node)


@dataclass
class ToolSpec:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    fingerprint: str
    status: str  # enabled | unreviewed | changed


@dataclass
class CallRecord:
    server: str
    tool: str
    role: str
    run_id: str
    args: dict[str, Any]
    status: str
    duration_ms: int


@dataclass
class _ServerState:
    connected: bool = False
    error: str | None = None
    tools: list[types.Tool] = field(default_factory=list)


SessionFactory = Callable[[], AbstractAsyncContextManager[ClientSession]]


def build_connections(settings: Settings) -> dict[str, dict[str, Any]]:
    env = {
        "TRAVEL_AGENT_DATA_DIR": str(settings.data_dir.resolve()),
        "TRAVEL_AGENT_SCENARIO": settings.scenario,
        "TRAVEL_AGENT_SIMULATED_DELAY_S": str(settings.simulated_delay_s),
        "TRAVEL_AGENT_MAX_OPTIONS_PER_SEARCH": str(settings.max_options_per_search),
        "TRAVEL_AGENT_TOOL_TIMEOUT_S": str(settings.tool_timeout_s),
        "TRAVEL_AGENT_WEATHER_SOURCE": settings.weather_source,
    }
    connections: dict[str, dict[str, Any]] = {}
    for name, module in LOCAL_SERVERS.items():
        server_env = dict(env)
        if name == "flights" and settings.duffel_access_token:
            server_env["DUFFEL_ACCESS_TOKEN"] = settings.duffel_access_token  # only this server sees the token
        connections[name] = {"transport": "stdio", "command": sys.executable, "args": ["-m", module], "env": server_env}
    return connections


def current_run_context(default_node: str | None = None) -> RunContext:
    """Run identifiers for calls made by LangChain tools inside a LangGraph node."""
    try:
        from langgraph.config import get_config

        configurable = get_config().get("configurable", {})
        return RunContext(configurable.get("thread_id"), configurable.get("run_id", "adhoc"), default_node)
    except Exception:  # noqa: BLE001 - outside a graph run
        return RunContext(None, "adhoc", default_node)


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_untrusted_text(value, max_length=1500)
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_payload(item) for key, item in value.items()}
    return value


def _summarize_result(data: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if isinstance(data.get("options"), list):
        summary["option_ids"] = [o.get("option_id") for o in data["options"][:20] if isinstance(o, dict)]
        summary["count"] = len(data["options"])
    for key in ("status", "level", "found", "price_changed", "current_price"):
        if key in data:
            summary[key] = data[key]
    if isinstance(data.get("booking"), dict):
        summary["booking_ref"] = data["booking"].get("booking_ref")
        summary["booking_status"] = data["booking"].get("status")
    return summary


class ToolGateway:
    def __init__(
        self,
        settings: Settings,
        *,
        connections: dict[str, dict[str, Any]] | None = None,
        session_factories: dict[str, SessionFactory] | None = None,
        registry: dict[str, dict[str, str]] | None = None,
        approval_verifier: ApprovalVerifier | None = None,
        trace_sink: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.settings = settings
        self.connections = build_connections(settings) if connections is None else connections
        self.session_factories = session_factories or {}
        self.registry = load_registry() if registry is None else registry
        self.approval_verifier = approval_verifier
        self.trace_sink = trace_sink
        self.calls: list[CallRecord] = []
        self._client = MultiServerMCPClient(self.connections)
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._servers: dict[str, _ServerState] = {}
        self._specs: dict[tuple[str, str], ToolSpec] = {}
        self._run_calls: dict[str, int] = defaultdict(int)
        self._role_tools = {role: set(tools) for role, tools in ROLE_TOOLS.items()}

    async def __aenter__(self) -> ToolGateway:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _session_cm(self, name: str) -> AbstractAsyncContextManager[ClientSession]:
        if name in self.session_factories:
            return self.session_factories[name]()
        return self._client.session(name)

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        names = list(dict.fromkeys([*self.session_factories, *self.connections]))
        for name in names:
            state = _ServerState()
            self._servers[name] = state
            try:
                session = await self._stack.enter_async_context(self._session_cm(name))
                state.tools = await self._list_tools(session)
            except Exception as exc:  # noqa: BLE001 - one unavailable server must not stop the others
                state.error = sanitize_untrusted_text(str(exc) or type(exc).__name__, 200)
                continue
            state.connected = True
            self._sessions[name] = session
            approved = self.registry.get(name, {})
            for tool in state.tools:
                digest = fingerprint(tool)
                status = "enabled" if approved.get(tool.name) == digest else (
                    "changed" if tool.name in approved else "unreviewed")
                self._specs[(name, tool.name)] = ToolSpec(
                    server=name, name=tool.name, description=tool.description or "",
                    input_schema=tool.inputSchema or {"type": "object"}, fingerprint=digest, status=status,
                )

    async def close(self) -> None:
        if self._stack is not None:
            stack, self._stack = self._stack, None
            await stack.aclose()
        self._sessions.clear()

    @staticmethod
    async def _list_tools(session: ClientSession) -> list[types.Tool]:
        tools: list[types.Tool] = []
        cursor: str | None = None
        for _ in range(20):
            result = await session.list_tools(cursor) if cursor else await session.list_tools()
            tools.extend(result.tools)
            cursor = result.nextCursor
            if not cursor:
                break
        return tools

    def listed_tools(self) -> dict[str, list[types.Tool]]:
        return {name: state.tools for name, state in self._servers.items() if state.connected}

    def is_available(self, server: str) -> bool:
        return self._servers.get(server, _ServerState()).connected

    def status(self) -> dict[str, Any]:
        servers = {}
        for name, state in self._servers.items():
            tools = []
            for (server, tool), spec in sorted(self._specs.items()):
                if server == name:
                    roles = sorted(role for role, allowed in self._role_tools.items() if (server, tool) in allowed)
                    tools.append({"name": tool, "status": spec.status, "roles": roles,
                                  "action": (server, tool) in ACTION_TOOLS})
            servers[name] = {"connected": state.connected, "error": state.error, "tools": tools}
        return servers

    def tools_for(self, role: str) -> list[ToolSpec]:
        allowed = self._role_tools.get(role, set())
        return [spec for key, spec in sorted(self._specs.items()) if key in allowed and spec.status == "enabled"]

    def _check_access(self, server: str, tool: str, role: str) -> ToolSpec:
        if (server, tool) not in self._role_tools.get(role, set()):
            raise ToolCallError("blocked", f"{server}.{tool} is not allowed for the {role!r} role")
        try:
            assert_tool_allowed_for_role(tool, role)
        except PermissionError as exc:
            raise ToolCallError("blocked", str(exc)) from exc
        if not self.is_available(server):
            raise ToolCallError("unavailable", f"MCP server {server!r} is not connected", retryable=True)
        spec = self._specs.get((server, tool))
        if spec is None:
            raise ToolCallError("not_found", f"{server}.{tool} does not exist")
        if spec.status != "enabled":
            raise ToolCallError("blocked", f"{server}.{tool} is {spec.status}; review it with "
                                           "`python -m travel_agent.tool_registry` before use")
        return spec

    @staticmethod
    def _validate_args(spec: ToolSpec, args: Mapping[str, Any]) -> dict[str, Any]:
        payload = to_jsonable_python(dict(args))
        allowed = set(spec.input_schema.get("properties", {}))
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ToolCallError("invalid_args", f"Unknown arguments for {spec.name}: {unknown}")
        try:
            jsonschema.validate(payload, spec.input_schema)
        except jsonschema.ValidationError as exc:
            raise ToolCallError("invalid_args", f"Invalid arguments for {spec.name}: {exc.message}") from exc
        return payload

    def _charge_quota(self, run: RunContext) -> None:
        self._run_calls[run.run_id] += 1
        if self._run_calls[run.run_id] > self.settings.max_tool_calls_per_run:
            raise ToolCallError("quota", f"Tool-call budget of {self.settings.max_tool_calls_per_run} per run reached")

    def calls_used(self, run_id: str) -> int:
        return self._run_calls.get(run_id, 0)

    def _parse_result(self, server: str, tool: str, result: types.CallToolResult) -> dict[str, Any]:
        text = "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))
        if result.isError:
            message = sanitize_untrusted_text(text or "tool error", 300)
            transient = any(hint in message.lower() for hint in _TRANSIENT_HINTS)
            raise ToolCallError("tool_error", message, retryable=transient)
        if result.structuredContent is not None:
            data: Any = result.structuredContent
        else:
            try:
                data = json.loads(text)
            except (TypeError, ValueError):
                data = {"text": text}
            if not isinstance(data, dict):
                data = {"result": data}
        model = RESULT_MODELS.get((server, tool))
        if model is None:
            return _sanitize_payload(data)
        try:
            return model.model_validate(data).model_dump(mode="json")
        except ValidationError as exc:
            raise ToolCallError("invalid_result", f"{server}.{tool} returned data that failed validation: "
                                                  f"{exc.error_count()} errors") from exc

    async def call(
        self,
        server: str,
        tool: str,
        args: Mapping[str, Any],
        *,
        role: str,
        run: RunContext,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        status = "ok"
        payload: dict[str, Any] = {}
        result_summary: dict[str, Any] = {}
        try:
            spec = self._check_access(server, tool, role)
            payload = self._validate_args(spec, args)
            action = ACTION_TOOLS.get((server, tool))
            if action is not None:
                if role != "executor":
                    raise ToolCallError("blocked", f"{server}.{tool} changes bookings and needs the executor role")
                if self.approval_verifier is None:
                    raise ToolCallError("approval", "No approval store is configured")
                try:
                    self.approval_verifier.verify_and_consume(
                        approval_id=str(payload.get("approval_id", "")), action=action, args=payload, trip_id=run.trip_id)
                except Exception as exc:  # noqa: BLE001 - any verification failure blocks the action
                    raise ToolCallError("approval", str(exc)) from exc
            attempts = 1 if action is not None else 1 + self.settings.max_tool_retries
            timeout = timeout_s or self.settings.tool_timeout_s
            for attempt in range(attempts):
                self._charge_quota(run)
                try:
                    result = await asyncio.wait_for(self._sessions[server].call_tool(tool, payload), timeout)
                    data = self._parse_result(server, tool, result)
                    result_summary = _summarize_result(data)
                    return data
                except TimeoutError as exc:
                    raise ToolCallError("timeout", f"{server}.{tool} did not answer within {timeout:.0f}s") from exc
                except ToolCallError as exc:
                    if not exc.retryable or attempt == attempts - 1:
                        raise
                except Exception as exc:  # noqa: BLE001 - transport-level failure
                    if attempt == attempts - 1:
                        raise ToolCallError("transport", sanitize_untrusted_text(str(exc), 200), retryable=True) from exc
                await asyncio.sleep(0.2 * (attempt + 1))
            raise ToolCallError("tool_error", f"{server}.{tool} failed after {attempts} attempts")
        except ToolCallError as exc:
            status = exc.code
            result_summary = {"error": exc.message}
            raise
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            safe_args = redact_secrets(payload)
            self.calls.append(CallRecord(server, tool, role, run.run_id, safe_args, status, duration_ms))
            if self.trace_sink is not None:
                self.trace_sink(TraceEvent(
                    trip_id=run.trip_id, run_id=run.run_id, event="tool_call", node=run.node, server=server,
                    tool=tool, status=status, duration_ms=duration_ms,
                    detail={"role": role, "args": safe_args, "result": result_summary},
                ))

    def _interceptor(self, role: str):
        gateway = self

        async def intercept(request, handler):  # noqa: ARG001 - the gateway performs the call itself
            run = current_run_context(default_node="qa_tools")
            try:
                data = await gateway.call(request.server_name, request.name, request.args, role=role, run=run)
            except ToolCallError as exc:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Tool call refused or failed ({exc.code}): {exc.message}")],
                    isError=True,
                )
            text = json.dumps(data, default=str)
            return types.CallToolResult(content=[types.TextContent(type="text", text=text[:12000])],
                                        structuredContent=data, isError=False)

        return intercept

    async def langchain_tools(self, role: str = "qa") -> list[BaseTool]:
        """LangChain tools for LLM tool calling; every call is routed back through this gateway."""
        allowed = {(spec.server, spec.name) for spec in self.tools_for(role)}
        tools: list[BaseTool] = []
        seen: set[str] = set()
        for server, session in self._sessions.items():
            if not any(key[0] == server for key in allowed):
                continue
            for tool in await load_mcp_tools(session, tool_interceptors=[self._interceptor(role)], server_name=server):
                if (server, tool.name) in allowed and tool.name not in seen:
                    seen.add(tool.name)
                    tools.append(tool)
        return tools


@asynccontextmanager
async def open_gateway(settings: Settings, **kwargs: Any) -> AsyncIterator[ToolGateway]:
    gateway = ToolGateway(settings, **kwargs)
    await gateway.start()
    try:
        yield gateway
    finally:
        await gateway.close()
