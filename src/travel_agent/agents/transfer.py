"""Transfer Agent: searches airport pickups (shared shuttles and private cars) for the arrival day."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from travel_agent.agents.common import AgentDeps, drop_none, run_directives
from travel_agent.models.planning import SearchDirective
from travel_agent.models.request import TravelRequest
from travel_agent.models.results import AgentResult


def transfer_args(request: TravelRequest, directive: SearchDirective) -> dict[str, Any]:
    return drop_none({
        "city_code": request.destination, "arrival_date": request.departure_date.isoformat(),
        "passengers": request.travelers, "max_price_total": directive.max_price_total, "strategy": directive.strategy,
    })


def validate_transfer(option: dict[str, Any], request: TravelRequest) -> str | None:
    if option["pickup_date"] != request.departure_date.isoformat():
        return "wrong date"
    if option["max_passengers"] < request.travelers:
        return "too few seats"
    return None


async def run_transfer_agent(deps: AgentDeps, request: TravelRequest,
                             directives: Iterable[SearchDirective]) -> AgentResult:
    run = deps.run.at("transfer_agent")

    async def search(directive: SearchDirective) -> list[dict[str, Any]]:
        data = await deps.gateway.call("travel-sandbox", "search_transfers", transfer_args(request, directive),
                                       role="search", run=run)
        return data["options"]

    return await run_directives("transfer", directives, request, search, validate_transfer)
