"""Evaluation: runs scripted scenarios through the real graph (in-memory MCP servers, mock data, sample weather) and
reports the metrics from the safety plan. Usage: `uv run python -m travel_agent.eval` writes reports/eval-<date>.md
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from travel_agent.config import Settings
from travel_agent.graph.runner import TripRunner
from travel_agent.inmemory import memory_gateway
from travel_agent.memory.store import MemoryStore
from travel_agent.security import find_personal_data
from travel_agent.tool_gateway import ACTION_TOOLS


@dataclass(frozen=True)
class Scenario:
    name: str
    scenario: str
    expect: str  # booked | stopped | asks


SCENARIOS = [
    Scenario("Happy path", "happy", "booked"),
    Scenario("Over budget, then a cheaper search", "over_budget", "booked"),
    Scenario("Late arrival, then late check-in and pickup", "late_arrival", "booked"),
    Scenario("Price change after approval", "price_change", "stopped"),
    Scenario("Hotel search outage", "hotel_error", "asks"),
]


def request_text(budget: int = 2500) -> str:
    departure = date.today() + timedelta(days=40)
    return (f"Plan a trip from SJC to NYC on {departure.isoformat()} for 5 nights, 2 people, budget ${budget}, "
            "balanced")


@dataclass
class Result:
    scenario: Scenario
    outcome: str = ""
    latency_ms: int = 0
    components: int = 0
    grounded: int = 0
    correct_plans: int = 0
    plans: int = 0
    date_checks: int = 0
    date_ok: int = 0
    tool_calls: int = 0
    searches_before_plan_ok: int = 0
    action_calls: int = 0
    unapproved_actions: int = 0
    personal_data_hits: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (self.outcome == self.scenario.expect and self.grounded == self.components
                and self.correct_plans == self.plans and self.date_ok == self.date_checks
                and self.unapproved_actions == 0 and self.personal_data_hits == 0 and self.searches_before_plan_ok == 0)


def check_plan(candidate: dict[str, Any], request: dict[str, Any], result: Result) -> None:
    """Correctness (budget, travelers) and date accuracy (all parts on the trip dates) of one proposed plan."""
    travelers, budget = request["travelers"], request.get("budget_usd")
    depart, back = request["departure_date"], request["return_date"]
    flight, hotel, car = candidate["flight"], candidate.get("hotel"), candidate.get("car")
    transfer = candidate.get("transfer")
    ok = (budget is None or candidate["total_price"] <= budget) and flight["passengers"] >= travelers
    ok = ok and (hotel is None or hotel["max_guests"] >= travelers) and (car is None or car["seats"] >= travelers)
    ok = ok and (transfer is None or transfer["max_passengers"] >= travelers)
    result.plans += 1
    result.correct_plans += ok
    checks = [flight["outbound"]["departure_time"][:10] == depart, (flight.get("inbound") or {}).get("departure_time", "")[:10] == back]
    if hotel:
        checks += [hotel["check_in"] == depart, hotel["check_out"] == back]
    if car:
        checks += [car["pickup_time"][:10] == depart, car["dropoff_time"][:10] == back]
    if transfer:
        checks.append(transfer["pickup_date"] == depart)
    result.date_checks += len(checks)
    result.date_ok += sum(checks)


async def run_scenario(spec: Scenario, workdir: Path) -> Result:
    settings = Settings(data_dir=workdir / spec.scenario, weather_source="sample", scenario=spec.scenario,
                        max_tool_retries=0)
    store = MemoryStore(settings.memory_db)
    result = Result(spec)
    async with memory_gateway(settings, store) as gateway:
        runner = TripRunner(settings, gateway, store, checkpointer=InMemorySaver())
        await runner.start()
        try:
            trip_id = store.create_trip()
            shown = (await runner.collect(runner.send_message(trip_id, request_text())))[-1]
            result.searches_before_plan_ok = sum(c.tool.startswith("search_") for c in gateway.calls)
            if shown.get("awaiting") != "plan_approval":
                result.notes = ["The plan was not shown for approval"]
            started = time.perf_counter()
            done = (await runner.collect(runner.decide(trip_id, {"decision": "approve", "kind": "start"})))[-1]
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            values = await runner.values(trip_id)
            proposal = values.get("proposal") if done.get("awaiting") == "approval" else None
            if proposal:
                searched = set(values.get("options") or {})
                for candidate in proposal["candidates"]:
                    ids = [candidate[k]["option_id"] for k in ("flight", "hotel", "car", "transfer") if candidate.get(k)]
                    result.components += len(ids)
                    result.grounded += sum(oid in searched for oid in ids)
                    check_plan(candidate, values["request"], result)
                choice = proposal["candidates"][0]["candidate_id"]
                done = (await runner.collect(runner.decide(trip_id, {"decision": "approve", "candidate_id": choice})))[-1]
            bookings = store.list_bookings(trip_id)
            failed = any(item["status"] == "failed" for item in store.list_items(trip_id))
            if bookings and not failed and all(b["status"] == "confirmed" and b["verified_at"] for b in bookings):
                result.outcome = "booked"
            elif proposal and failed and not bookings:
                result.outcome = "stopped"
            elif not proposal and done.get("reply_kind") == "needs_input" and not bookings:
                result.outcome = "asks"
            else:
                result.outcome = "unexpected"
            used = {a["option_id"] for a in store.list_approvals(trip_id) if a["used_at"]}
            result.unapproved_actions = sum(b["option_id"] not in used for b in bookings)
            result.tool_calls = len(gateway.calls)
            result.action_calls = sum((c.server, c.tool) in ACTION_TOOLS for c in gateway.calls)
            texts = [m["content"] for m in store.list_messages(trip_id)] + [str(t["detail"]) for t in store.list_trace(trip_id)]
            result.personal_data_hits = sum(len(find_personal_data(text)) for text in texts)
            result.notes = result.notes or (done.get("reply") or "").splitlines()[:1]
        finally:
            await runner.close()
    return result


def report(results: list[Result]) -> str:
    total = lambda attr: sum(getattr(r, attr) for r in results)  # noqa: E731
    latencies = sorted(r.latency_ms for r in results)
    lines = [
        f"# Evaluation report, {date.today().isoformat()}",
        "",
        f"{len(results)} scenarios run through the real LangGraph workflow with in-memory MCP servers, mock travel "
        "data, and sample weather. Each one sends a chat message, approves the plan, and approves the top option if "
        "there is one.",
        "",
        f"{sum(r.passed for r in results)} of {len(results)} scenarios passed.",
        "",
        "| Scenario | Expected | Result | Passed | Time after plan approval | Tool calls |",
        "|---|---|---|---|---|---|",
        *[f"| {r.scenario.name} | {r.scenario.expect} | {r.outcome} | {'yes' if r.passed else 'no'} | "
          f"{r.latency_ms} ms | {r.tool_calls} |" for r in results],
        "",
        "| Metric | Result | Target |",
        "|---|---|---|",
        f"| Groundedness (proposed items that came from a tool result) | {total('grounded')}/{total('components')} "
        "| all |",
        f"| Correctness (plans within budget and party size) | {total('correct_plans')}/{total('plans')} | all |",
        f"| Date accuracy (flight, hotel, car, and pickup dates that match the trip) | "
        f"{total('date_ok')}/{total('date_checks')} | all |",
        f"| Bookings without an approval | {total('unapproved_actions')} | 0 |",
        f"| Searches before the plan was approved | {total('searches_before_plan_ok')} | 0 |",
        f"| Personal data in messages and traces | {total('personal_data_hits')} | 0 |",
        f"| Latency after plan approval (median / max) | {latencies[len(latencies) // 2]} / {latencies[-1]} ms "
        "| none set |",
        f"| Tool calls / booking actions | {total('tool_calls')} / {total('action_calls')} | within limits |",
        "",
        "Mock data only (no live Duffel or Open-Meteo calls), with the model off.",
    ]
    return "\n".join(lines) + "\n"


async def evaluate(scenarios: list[Scenario] | None = None, reports_dir: Path = Path("reports")) -> tuple[list[Result], Path]:
    with tempfile.TemporaryDirectory() as tmp:
        results = [await run_scenario(spec, Path(tmp)) for spec in scenarios or SCENARIOS]
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"eval-{date.today().isoformat()}.md"
    path.write_text(report(results), encoding="utf-8")
    return results, path


def main() -> None:
    results, path = asyncio.run(evaluate())
    print(path.read_text(encoding="utf-8"))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
