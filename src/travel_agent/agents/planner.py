"""Planner Agent (rules; an optional local LLM adds a critic and explanations).

Plan-and-Execute: the Planner turns a validated request into a TaskPlan (which agents run, with which search
strategies). After each round of Tree-of-Thought results it decides the next step:
- a complete plan exists: propose it;
- every plan is over budget: ask the agents for cheaper options (once);
- flights land late: ask only the Hotel agent for late check-in and only the Car agent for open rental desks;
- anything else (no results, a failed search, limits reached): explain what happened and ask the traveler.
Every follow-up is a real search, so plans stay grounded in tool results.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from travel_agent import text
from travel_agent.models.itinerary import Issue, ItineraryCandidate
from travel_agent.models.planning import SearchDirective, TaskPlan
from travel_agent.models.request import TravelRequest

AGENT_NAMES = {"flight": "flight", "hotel": "hotel", "car": "car rental", "transfer": "airport pickup"}
SEARCH_NAMES = {"flight": "flight", "hotel": "hotel", "car": "rental car", "transfer": "airport pickup"}
CHEAPER = {"flight": "cheapest", "hotel": "budget", "car": "budget", "transfer": "budget"}
MAX_STRATEGIES_PER_AGENT = 2
SUGGESTIONS = {
    "no_flights": "Try different dates or airports",
    "flight_rules": "Allow more stops or a different cabin",
    "no_hotels": 'Say "no hotel" to plan without one, or choose another area',
    "no_cars": 'Say "no car" to plan without a rental car',
    "late_arrival_hotel": 'Say "no hotel" if you don\'t need one, or pick other dates',
    "car_desk_hours": 'Say "no car", or pick other dates',
    "no_transfers": 'Say "no airport pickup" to plan without one',
    "no_more_options": "Change your dates, budget, stops, or hotel area to see different trips",
    "landing_time": 'Say "any landing time", or choose another time of day or date',
    "rules": "Relax one of your requirements",
}


def describe_request(request: TravelRequest) -> str:
    """One plain sentence about the trip (used in replies and as context for the optional LLM)."""
    parts = [f"{text.place(request.origin)} to {text.place(request.destination)}",
             text.dates(request.departure_date, request.return_date), text.plural(request.travelers, "traveler")]
    if request.budget_usd:
        parts.append(f"budget {text.money(request.budget_usd)}")
    if request.max_stops == 0:
        parts.append("nonstop only")
    elif request.max_stops is not None:
        parts.append(f"up to {text.plural(request.max_stops, 'stop')}")
    if request.cabin_class != "economy":
        parts.append(f"{request.cabin_class.replace('_', ' ')} class")
    if request.return_date is None:
        parts.append("flight only")
    else:
        if not request.need_hotel:
            parts.append("no hotel")
        elif request.hotel_area:
            parts.append(f"hotel in the {request.hotel_area.replace('-', ' ')} area")
        if not request.require_car:
            parts.append("no car")
        elif request.car_class:
            parts.append(f"{request.car_class} car")
    if not request.need_transfer:
        parts.append("no airport pickup")
    if request.arrival_window:
        parts.append(f"landing {text.LANDING_TEXT[request.arrival_window].split(' (')[0]}")
    parts.append(f"optimizing for {text.PRIORITY_TEXT[request.effective_priority]}")
    return ", ".join(parts)


def trip_title(request: TravelRequest) -> str:
    when = f"{request.departure_date:%b} {request.departure_date.day}"
    if request.return_date is not None:
        when += f" to {request.return_date:%b} {request.return_date.day}"
    return f"{request.origin} to {request.destination}, {when}"


def trip_agents(request: TravelRequest) -> list[str]:
    agents = ["flight"]
    if request.wants_hotel:
        agents.append("hotel")
    if request.wants_car:
        agents.append("car")
    if request.wants_transfer:
        agents.append("transfer")
    return agents


def initial_strategies(request: TravelRequest) -> dict[str, list[str]]:
    """Starting branches for each agent, chosen from the traveler's priority and trip type."""
    cost = request.effective_priority == "cost"
    flight = ["cheapest" if cost else "best"]
    if request.effective_priority in {"time", "comfort"} or (request.flight_preference or "").lower() == "nonstop":
        flight.append("nonstop")
    hotel = ["budget" if cost else "preferred"]
    if request.trip_type == "family":
        hotel.append("family")
    car = ["budget" if cost else "standard"]
    if request.trip_type == "family" or request.travelers >= 4:
        car.append("larger")
    transfer = ["budget" if cost else "private" if request.effective_priority in {"time", "comfort"} else "standard"]
    strategies = {"flight": flight, "hotel": hotel, "car": car, "transfer": transfer}
    plan = {agent: strategies[agent][:MAX_STRATEGIES_PER_AGENT] for agent in trip_agents(request)}
    if request.arrival_window:
        plan["flight"] = list(BROADER["flight"])  # only some flights land then, so search with every style
    return plan


def directives_for(strategies: dict[str, list[str]], reason: str = "initial search") -> list[SearchDirective]:
    return [SearchDirective(agent=agent, strategy=name, reason=reason)  # type: ignore[arg-type]
            for agent, names in strategies.items() for name in names]


BROADER = {"flight": ["best", "nonstop", "cheapest"], "hotel": ["preferred", "budget"],
           "car": ["standard", "budget", "larger"], "transfer": ["standard", "budget", "private"]}


def broader_strategies(request: TravelRequest) -> dict[str, list[str]]:
    """For "more options": every search style for each agent, so new options can come up."""
    strategies = {agent: list(BROADER[agent]) for agent in trip_agents(request)}
    if "hotel" in strategies and request.trip_type == "family":
        strategies["hotel"].append("family")
    return strategies


KINDS = ("flight", "hotel", "car", "transfer")
KIND_PLURAL = {"flight": "flights", "hotel": "hotels", "car": "cars", "transfer": "airport pickups"}


def build_task_plan(request: TravelRequest, beam_width: int = 3, broader: bool = False, only: str | None = None,
                    kept: Iterable[str] = ()) -> TaskPlan:
    """The plan's steps and searches. With `only`, just that part is searched again and the `kept` parts stay."""
    strategies = broader_strategies(request) if broader else initial_strategies(request)
    if only:
        strategies = {only: strategies.get(only) or list(BROADER[only])}
    steps = [f"Check the weather and airports for {text.place(request.destination)}"]
    if "flight" in strategies:
        landing = (f", keeping only flights that land {text.LANDING_TEXT[request.arrival_window]}"
                   if request.arrival_window else "")
        steps.append(f"Flight agent: search flights from {request.origin} to {request.destination} "
                     f"({text.strategies(strategies['flight'])}){landing}")
    if "hotel" in strategies:
        area = f" in the {request.hotel_area.replace('-', ' ')} area" if request.hotel_area else ""
        steps.append(f"Hotel agent: search hotels{area} for {text.plural(request.nights or 0, 'night')} "
                     f"({text.strategies(strategies['hotel'])})")
    if "car" in strategies:
        steps.append(f"Car rental agent: search rental cars ({text.strategies(strategies['car'])})")
    if "transfer" in strategies:
        steps.append(f"Transfer agent: search airport pickups for when you land "
                     f"({text.strategies(strategies['transfer'])})")
    kept_kinds = [kind for kind in KINDS if kind in set(kept)]
    if only and kept_kinds:
        steps.append(f"Combine the {KIND_PLURAL[only]} I find with the "
                     f"{text.join_words(text.KIND_NAME[k] for k in kept_kinds)} I'm keeping, and keep the best "
                     f"{beam_width} (Tree-of-Thought beam search)")
    else:
        parts = text.join_words(KIND_PLURAL[kind] for kind in KINDS if kind in strategies)
        steps.append(f"Combine {parts} and keep the best {beam_width} at each step (Tree-of-Thought beam search)")
    checks = ["dates", "availability", "arrival time against hotel check-in and rental desk hours"]
    if request.budget_usd:
        checks.insert(0, "budget")
    steps.append(f"Check {text.join_words(checks)}, then rank by {text.PRIORITY_TEXT[request.effective_priority]}")
    steps.append("If something doesn't fit, run a follow-up search or ask you")
    steps.append("Show you the best options; nothing is booked until you approve")
    goal = f"Plan a {request.trip_type} trip: {describe_request(request)}"
    return TaskPlan(goal=goal, priority=request.effective_priority, trip_type=request.trip_type, steps=steps,
                    agents=[*strategies, "context"], directives=directives_for(strategies))


@dataclass
class ReviewInput:
    request: TravelRequest
    candidates: list[ItineraryCandidate]
    issues: list[Issue]
    agent_status: dict[str, dict[str, Any]]
    tried: set[str]
    round: int
    max_rounds: int
    near_misses: list[ItineraryCandidate] = field(default_factory=list)
    locked: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)


@dataclass
class ReviewDecision:
    action: Literal["refine", "propose", "ask"]
    directives: list[SearchDirective] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    message: str = ""
    limit_reached: bool = False

    def to_state(self) -> dict[str, Any]:
        return {"action": self.action, "directives": [d.model_dump() for d in self.directives], "notes": self.notes,
                "message": self.message, "limit_reached": self.limit_reached}


def review(inp: ReviewInput) -> ReviewDecision:
    if inp.candidates:
        return ReviewDecision(action="propose")
    batch: dict[str, SearchDirective] = {}
    notes: list[str] = []

    def add(agent: str, strategy: str, reason: str, **extra: Any) -> bool:
        if agent in inp.locked:
            return False
        directive = SearchDirective(agent=agent, strategy=strategy, reason=reason, **extra)  # type: ignore[arg-type]
        signature = directive.signature()
        if signature in inp.tried or signature in batch:
            return False
        batch[signature] = directive
        return True

    for issue in inp.issues:
        if issue.code == "late_arrival_hotel":
            if add("hotel", "late_checkin", "Flight lands late: find hotels with late check-in", require_late_checkin=True):
                notes.append("The flights land late, so I asked the hotel agent for hotels with late check-in.")
        elif issue.code == "car_desk_hours":
            hours = sorted({int(h) for h in issue.data.get("hours", [])})
            if add("car", "late_pickup", "Rental desks closed at arrival: find desks open then", need_desk_hours=hours):
                notes.append("The rental desks were closed when you land, so I asked the car rental agent for desks "
                             "that are open then.")
        elif issue.code == "over_budget":
            added = [agent for agent in trip_agents(inp.request)
                     if add(agent, CHEAPER[agent], "Over budget: search cheaper options")]
            if added:
                notes.append("The first results were over your budget, so I asked the "
                             + text.join_words(AGENT_NAMES[a] for a in added) + " agents for cheaper options.")
    if batch and inp.round < inp.max_rounds:
        return ReviewDecision(action="refine", directives=list(batch.values()), notes=notes)
    limit_reached = bool(batch)
    return ReviewDecision(action="ask", message=needs_input_message(inp, limit_reached), limit_reached=limit_reached)


def needs_input_message(inp: ReviewInput, limit_reached: bool) -> str:
    failed = {agent: status for agent, status in inp.agent_status.items()
              if agent != "context" and status.get("status") in {"failed", "timeout"}}
    happened: list[str] = []
    for agent, status in failed.items():
        happened.append(f"The {SEARCH_NAMES.get(agent, agent)} search "
                        f"{'took too long to answer' if status['status'] == 'timeout' else 'is not working right now'}.")
    over_budget = next((i for i in inp.issues if i.code == "over_budget"), None)
    for issue in inp.issues:
        if issue.agent in failed and issue.code.startswith("no_"):
            continue
        detail = issue.detail
        if issue is over_budget and inp.near_misses:
            detail += f" The cheapest one costs {text.money(inp.near_misses[0].total_price)}."
        happened.append(detail)
    happened += inp.history[-2:]
    if limit_reached:
        happened.append(f"I stopped after {text.plural(inp.round, 'follow-up search')}.")
    options: list[str] = []
    if failed:
        options.append('Say "try again" to run the search again')
    for issue in inp.issues:
        if issue.code == "over_budget":
            minimum = inp.near_misses[0].total_price if inp.near_misses else issue.data.get("min_total")
            options.append(f"Raise your budget to about {text.money(minimum)}" if minimum else "Raise your budget")
            if inp.request.wants_car:
                options.append('Say "no car" to save on the rental car')
        elif issue.code in SUGGESTIONS and not (issue.agent in failed and issue.code.startswith("no_")):
            options.append(SUGGESTIONS[issue.code])
    if not options:
        options.append("Change your dates, budget, or requirements")
    budget = inp.request.budget_usd
    headline = (f"I couldn't find a trip within your {text.money(budget)} budget."
                if over_budget and budget and len(inp.issues) == 1 and not failed else
                "I don't have any other options that fit." if [i.code for i in inp.issues] == ["no_more_options"] else
                "I couldn't put together a trip that fits yet.")
    return text.paragraphs(
        headline,
        "What happened:\n" + text.bullets(dict.fromkeys(happened)),
        "What you can do:\n" + text.bullets(dict.fromkeys(options)),
        "Next step: tell me what to change, and I'll show you an updated plan. Nothing has been booked.",
    )
