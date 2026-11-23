"""Context step: weather risk and airport details for the trip (tools, not an agent)."""
from __future__ import annotations

import time
from typing import Any

from travel_agent.agents.common import AgentDeps
from travel_agent.models.request import TravelRequest
from travel_agent.models.results import AgentResult, AgentStatus
from travel_agent.tool_gateway import ToolCallError


async def run_context_agent(deps: AgentDeps, request: TravelRequest) -> tuple[AgentResult, dict[str, Any]]:
    run = deps.run.at("context_agent")
    started = time.perf_counter()
    context: dict[str, Any] = {}
    notes: list[str] = []
    failures = 0
    calls = 0
    end = request.return_date or request.departure_date
    try:
        calls += 1
        context["weather"] = await deps.gateway.call(
            "trip-context", "weather_risk",
            {"city_code": request.destination, "start_date": request.departure_date.isoformat(),
             "end_date": end.isoformat(), "trip_type": request.trip_type},
            role="search", run=run)
    except ToolCallError as exc:
        failures += 1
        notes.append(f"weather: {exc.code} – {exc.message}")
    airports: dict[str, Any] = {}
    for code in (request.origin, request.destination):
        try:
            calls += 1
            matches = (await deps.gateway.call("trip-context", "lookup_airport", {"query": code, "limit": 1},
                                               role="search", run=run))["matches"]
            if matches:
                airports[code] = matches[0]
        except ToolCallError as exc:
            failures += 1
            notes.append(f"airport {code}: {exc.code}")
    context["airports"] = airports
    status = AgentStatus.SUCCESS if failures == 0 else (AgentStatus.PARTIAL if failures < calls else AgentStatus.FAILED)
    result = AgentResult(agent="context", status=status, notes=notes, calls=calls,
                         duration_ms=int((time.perf_counter() - started) * 1000))
    return result, context
