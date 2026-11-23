"""Hotel Agent: searches hotels for the stay dates, area, and check-in needs the Planner asks for."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from travel_agent.agents.common import AgentDeps, drop_none, run_directives
from travel_agent.models.planning import SearchDirective
from travel_agent.models.request import TravelRequest
from travel_agent.models.results import AgentResult
from travel_agent.providers.airports import get_airport_index


def hotel_args(request: TravelRequest, directive: SearchDirective) -> dict[str, Any]:
    return drop_none({
        "city_code": request.destination, "check_in": request.departure_date.isoformat(),
        "check_out": request.return_date.isoformat() if request.return_date else None,
        "guests": request.travelers, "area": directive.area if directive.area is not None else request.hotel_area,
        "max_price_total": directive.max_price_total,
        "require_late_checkin": directive.require_late_checkin or None, "strategy": directive.strategy,
    })


def validate_hotel(option: dict[str, Any], request: TravelRequest) -> str | None:
    if not get_airport_index().matches(request.destination, option["city_code"]) and option["city_code"] != request.destination:
        return "wrong city"
    if request.return_date is None:
        return "no stay dates"
    if option["check_in"] != request.departure_date.isoformat() or option["check_out"] != request.return_date.isoformat():
        return "wrong dates"
    if option["max_guests"] < request.travelers:
        return "too small"
    return None


async def run_hotel_agent(deps: AgentDeps, request: TravelRequest,
                          directives: Iterable[SearchDirective]) -> AgentResult:
    run = deps.run.at("hotel_agent")

    async def search(directive: SearchDirective) -> list[dict[str, Any]]:
        data = await deps.gateway.call("travel-sandbox", "search_hotels", hotel_args(request, directive),
                                       role="search", run=run)
        return data["options"]

    return await run_directives("hotel", directives, request, search, validate_hotel)
