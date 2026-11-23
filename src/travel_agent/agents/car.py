"""Car Rental Agent: searches rental cars for the trip dates and the desk hours the Planner asks for."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, time
from typing import Any

from travel_agent.agents.common import AgentDeps, drop_none, run_directives
from travel_agent.models.planning import SearchDirective
from travel_agent.models.request import TravelRequest
from travel_agent.models.results import AgentResult

NOMINAL_PICKUP = time(12, 0)
NOMINAL_DROPOFF = time(10, 0)


def car_args(request: TravelRequest, directive: SearchDirective) -> dict[str, Any]:
    if request.return_date is None:
        raise ValueError("Cars need a return date")
    return drop_none({
        "city_code": request.destination,
        "pickup_time": datetime.combine(request.departure_date, NOMINAL_PICKUP).isoformat(timespec="minutes"),
        "dropoff_time": datetime.combine(request.return_date, NOMINAL_DROPOFF).isoformat(timespec="minutes"),
        "passengers": request.travelers,
        "car_class": directive.car_class if directive.car_class is not None else request.car_class,
        "max_price_total": directive.max_price_total, "need_desk_hours": directive.need_desk_hours or None,
        "strategy": directive.strategy,
    })


def validate_car(option: dict[str, Any], request: TravelRequest) -> str | None:
    if request.return_date is None:
        return "no rental dates"
    if option["seats"] < request.travelers:
        return "too few seats"
    if option["pickup_time"][:10] != request.departure_date.isoformat():
        return "wrong pickup date"
    if option["dropoff_time"][:10] != request.return_date.isoformat():
        return "wrong drop-off date"
    return None


async def run_car_agent(deps: AgentDeps, request: TravelRequest,
                        directives: Iterable[SearchDirective]) -> AgentResult:
    run = deps.run.at("car_agent")

    async def search(directive: SearchDirective) -> list[dict[str, Any]]:
        data = await deps.gateway.call("travel-sandbox", "search_cars", car_args(request, directive),
                                       role="search", run=run)
        return data["options"]

    return await run_directives("car", directives, request, search, validate_car)
