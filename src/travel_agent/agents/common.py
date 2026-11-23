"""Shared search-agent plumbing: run each directive through the gateway and classify the outcome."""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from travel_agent.models.planning import SearchDirective
from travel_agent.models.request import TravelRequest
from travel_agent.models.results import AgentResult, AgentStatus
from travel_agent.tool_gateway import RunContext, ToolCallError, ToolGateway

SearchFn = Callable[[SearchDirective], Awaitable[list[dict[str, Any]]]]
Validator = Callable[[dict[str, Any], TravelRequest], str | None]


def drop_none(args: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in args.items() if value is not None}


async def run_directives(
    agent: str,
    directives: Iterable[SearchDirective],
    request: TravelRequest,
    search: SearchFn,
    validate: Validator,
) -> AgentResult:
    started = time.perf_counter()
    options: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    errors: list[ToolCallError] = []
    calls = 0
    for directive in directives:
        calls += 1
        try:
            found = await search(directive)
        except ToolCallError as exc:
            errors.append(exc)
            notes.append(f"{directive.strategy}: {exc.code} – {exc.message}")
            continue
        dropped = 0
        for option in found:
            option = {**option, "strategy": option.get("strategy") or directive.strategy}
            if validate(option, request) is not None:
                dropped += 1
                continue
            options.setdefault(option["option_id"], option)
        if dropped:
            notes.append(f"{directive.strategy}: dropped {dropped} result(s) that did not match the request")
    duration_ms = int((time.perf_counter() - started) * 1000)
    if errors and not options:
        status = AgentStatus.TIMEOUT if any(e.code == "timeout" for e in errors) else AgentStatus.FAILED
        return AgentResult(agent=agent, status=status, notes=notes, error=errors[0].message, calls=calls,
                           duration_ms=duration_ms)
    status = AgentStatus.PARTIAL if errors else (AgentStatus.SUCCESS if options else AgentStatus.NO_RESULTS)
    return AgentResult(agent=agent, status=status, options=list(options.values()), notes=notes,
                       error=errors[0].message if errors else None, calls=calls, duration_ms=duration_ms)


class AgentDeps:
    """What a search agent needs: the gateway (its only way to reach data) and run identifiers."""

    def __init__(self, gateway: ToolGateway, run: RunContext) -> None:
        self.gateway = gateway
        self.run = run
