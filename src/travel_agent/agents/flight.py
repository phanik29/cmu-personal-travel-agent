"""Flight Agent: searches flights for each directive and drops results that don't match the request."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from travel_agent.agents.common import AgentDeps, drop_none, run_directives
from travel_agent.models.planning import SearchDirective
from travel_agent.models.request import TravelRequest
from travel_agent.models.results import AgentResult
from travel_agent.providers.airports import get_airport_index


def flight_args(request: TravelRequest, directive: SearchDirective) -> dict[str, Any]:
    max_stops = directive.max_stops if directive.max_stops is not None else request.max_stops
    if directive.strategy == "nonstop":
        max_stops = 0
    return drop_none({
        "origin": request.origin, "destination": request.destination,
        "departure_date": request.departure_date.isoformat(),
        "return_date": request.return_date.isoformat() if request.return_date else None,
        "passengers": request.travelers, "cabin_class": request.cabin_class, "max_stops": max_stops,
        "max_price_total": directive.max_price_total, "strategy": directive.strategy,
    })


def validate_flight(option: dict[str, Any], request: TravelRequest) -> str | None:
    airports = get_airport_index()
    out = option["outbound"]
    if not airports.matches(request.origin, out["origin"]) or not airports.matches(request.destination, out["destination"]):
        return "wrong airports"
    if date.fromisoformat(out["departure_time"][:10]) != request.departure_date:
        return "wrong departure date"
    inbound = option.get("inbound")
    if request.return_date is not None:
        if not inbound or date.fromisoformat(inbound["departure_time"][:10]) != request.return_date:
            return "wrong return date"
    if option["passengers"] < request.travelers:
        return "not enough passengers"
    seats = option.get("seats_available")
    if seats is not None and seats < request.travelers:
        return "not enough seats"
    return None


async def run_flight_agent(deps: AgentDeps, request: TravelRequest,
                           directives: Iterable[SearchDirective]) -> AgentResult:
    run = deps.run.at("flight_agent")

    async def search(directive: SearchDirective) -> list[dict[str, Any]]:
        data = await deps.gateway.call("flights", "search_flights", flight_args(request, directive), role="search", run=run)
        return data["options"]

    return await run_directives("flight", directives, request, search, validate_flight)
