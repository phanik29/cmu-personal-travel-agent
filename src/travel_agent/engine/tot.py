"""Tree-of-Thought over itineraries with beam search.

A thought is a partial itinerary built only from real search results. The tree's depth is the planning stages
(flights, then hotels, then cars, then airport pickup); its branches are the options found at each stage. At every stage the search expands
each kept plan with every option of the next kind, prunes branches that break a hard rule, scores the rest, and
keeps the top `beam_width` (beam search). If no complete plan survives, it backtracks into plans the beam dropped,
so the only feasible plan is never lost.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from travel_agent.engine.constraints import (
    car_checks,
    describe,
    flight_violations,
    hotel_checks,
    needed_desk_hours,
    transfer_checks,
)
from travel_agent import text
from travel_agent.engine.scoring import PriceRange, score_plan
from travel_agent.models.itinerary import Issue, ItineraryCandidate, PruningRecord
from travel_agent.models.options import CarOption, FlightOption, HotelOption, TransferOption
from travel_agent.models.request import TravelRequest

MAX_PRUNE_RECORDS = 300
LATE_CHECKIN_CODE = "late_arrival_no_late_checkin"
DESK_CODES = frozenset({"car_desk_closed_at_arrival", "car_desk_closed_at_dropoff"})
TIMING_CODES = DESK_CODES | {LATE_CHECKIN_CODE}


@dataclass(frozen=True)
class ToTConfig:
    beam_width: int = 3
    max_stages: int = 5
    max_expansions: int = 200
    tie_margin: float = 2.0


@dataclass
class _Node:
    flight: FlightOption
    hotel: HotelOption | None = None
    car: CarOption | None = None
    transfer: TransferOption | None = None
    warnings: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def parts(self) -> list[Any]:
        return [o for o in (self.flight, self.hotel, self.car, self.transfer) if o is not None]

    @property
    def total(self) -> float:
        return sum(o.price_total for o in self.parts)

    @property
    def node_id(self) -> str:
        return "|".join(o.option_id for o in self.parts)

    def extend(self, option: Any, warnings: list[str]) -> _Node:
        return replace(self, warnings=[*self.warnings, *warnings], score=0.0, **{option.kind: option})


@dataclass
class ToTResult:
    candidates: list[ItineraryCandidate] = field(default_factory=list)
    near_misses: list[ItineraryCandidate] = field(default_factory=list)
    pruned: list[PruningRecord] = field(default_factory=list)
    pruned_count: int = 0
    stages: list[dict[str, Any]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    expansions: int = 0

    def to_state(self) -> dict[str, Any]:
        return {
            "candidates": [c.model_dump(mode="json") for c in self.candidates],
            "near_misses": [c.model_dump(mode="json") for c in self.near_misses],
            "pruned": [p.model_dump() for p in self.pruned[:60]],
            "pruned_count": self.pruned_count,
            "stages": self.stages,
            "issues": [i.model_dump() for i in self.issues],
            "expansions": self.expansions,
        }


class _Search:
    def __init__(self, request: TravelRequest, flights: list[FlightOption], hotels: list[HotelOption],
                 cars: list[CarOption], transfers: list[TransferOption], locked: dict[str, Any], config: ToTConfig,
                 result: ToTResult) -> None:
        self.request = request
        self.flights = flights
        self.hotels = hotels
        self.cars = cars
        self.transfers = transfers
        self.locked = locked
        self.config = config
        self.result = result
        self.codes: Counter[str] = Counter()
        self.flight_codes: Counter[str] = Counter()
        self.late_checkin_blocked = 0
        self.budget_only = 0
        self.desk_hours: set[int] = set()
        self.flight_survivors = 0
        self.min_over_budget: float | None = None
        self.stage_names = ["flights"]
        if request.wants_hotel:
            self.stage_names.append("hotels")
        if request.wants_car:
            self.stage_names.append("cars")
        if request.wants_transfer:
            self.stage_names.append("transfers")
        self.stage_names = self.stage_names[: max(1, config.max_stages - 1)]
        self.reserves: dict[int, list[_Node]] = {}

    def _prune(self, node_id: str, stage: str, codes: list[str]) -> None:
        self.codes.update(codes)
        self.result.pruned_count += 1
        if len(self.result.pruned) < MAX_PRUNE_RECORDS:
            reason = "; ".join(describe(code) for code in codes)
            self.result.pruned.append(PruningRecord(node_id=node_id, stage=stage, reason=reason))

    def _over_budget(self, total: float) -> bool:
        budget = self.request.budget_usd
        if budget is not None and total > budget:
            self.min_over_budget = total if self.min_over_budget is None else min(self.min_over_budget, total)
            return True
        return False

    def _score(self, nodes: list[_Node], final: bool) -> None:
        if not nodes:
            return
        totals = [n.total for n in nodes]
        price_range = PriceRange(min(totals), max(totals))
        for node in nodes:
            node.score = score_plan(self.request, node.flight, node.hotel, node.car, price_range, final=final,
                                    transfer=node.transfer).total

    def _summary(self, stage: str, considered: int, kept: int, pruned_rules: int, pruned_beam: int, note: str = "") -> None:
        self.result.stages.append({"stage": stage, "considered": considered, "kept": kept,
                                   "pruned_by_rules": pruned_rules, "pruned_by_beam": pruned_beam, "note": note})

    def _flight_level(self) -> list[_Node]:
        nodes = []
        for flight in self.flights:
            codes = [] if "flight" in self.locked else flight_violations(flight, self.request)
            if codes:
                if "over_budget" in codes:
                    self._over_budget(flight.price_total)
                if codes == ["over_budget"]:
                    self.budget_only += 1
                self.flight_codes.update(codes)
                self._prune(flight.option_id, "flights", codes)
                continue
            nodes.append(_Node(flight=flight))
        return nodes

    def _expand(self, nodes: list[_Node], stage: str) -> tuple[list[_Node], int]:
        children: list[_Node] = []
        pruned = 0
        options = {"hotels": self.hotels, "cars": self.cars, "transfers": self.transfers}[stage]
        for node in nodes:
            for option in options:
                if self.result.expansions >= self.config.max_expansions:
                    return children, pruned
                self.result.expansions += 1
                if stage == "hotels":
                    codes, warnings = hotel_checks(option, self.request, node.flight)  # type: ignore[arg-type]
                elif stage == "cars":
                    codes, warnings = car_checks(option, self.request, node.flight)  # type: ignore[arg-type]
                else:
                    codes, warnings = transfer_checks(option, self.request)  # type: ignore[arg-type]
                child = node.extend(option, warnings)
                over = self._over_budget(child.total)
                if codes and set(codes) <= TIMING_CODES and not over:
                    # Blocked only by timing: a targeted follow-up search (late check-in, open desk) may fix it.
                    if LATE_CHECKIN_CODE in codes:
                        self.late_checkin_blocked += 1
                    if DESK_CODES & set(codes):
                        self.desk_hours.update(needed_desk_hours(node.flight))
                if over:
                    self.budget_only += not codes
                    codes = [*codes, "over_budget"]
                if codes:
                    pruned += 1
                    self._prune(child.node_id, stage, codes)
                    continue
                children.append(child)
        return children, pruned

    def _forward(self, frontier: list[_Node], start: int, note: str = "") -> list[_Node]:
        for index in range(start, len(self.stage_names)):
            stage = self.stage_names[index]
            children, pruned = self._expand(frontier, stage)
            final = index == len(self.stage_names) - 1
            self._score(children, final)
            children.sort(key=lambda n: n.score, reverse=True)
            frontier, reserve = children[: self.config.beam_width], children[self.config.beam_width:]
            self.reserves.setdefault(index, []).extend(reserve)
            self._record_beam(reserve, stage)
            self._summary(stage, len(children) + pruned, len(frontier), pruned, len(reserve), note)
            if not frontier:
                return []
        return frontier

    def _record_beam(self, nodes: list[_Node], stage: str) -> None:
        for node in nodes:
            self.result.pruned_count += 1
            if len(self.result.pruned) < MAX_PRUNE_RECORDS:
                self.result.pruned.append(PruningRecord(
                    node_id=node.node_id, stage=stage,
                    reason=f"Ranked below the top {self.config.beam_width} (kept for backtracking)"))

    def _precheck_car_desks(self, frontier: list[_Node]) -> None:
        """Share the best flight's times with the car stage early: if no rental desk is open then, the Planner can
        ask the Car agent in the same round as any hotel follow-up."""
        if not self.request.wants_car or not self.cars or not frontier or "car" in self.locked:
            return
        hours = needed_desk_hours(frontier[0].flight)
        if not any(all(car.desk_open_at(hour) for hour in hours) for car in self.cars):
            self.desk_hours.update(hours)

    def run(self) -> list[_Node]:
        level = self._flight_level()
        self.flight_survivors = len(level)
        final_single = len(self.stage_names) == 1
        self._score(level, final=final_single)
        level.sort(key=lambda n: n.score, reverse=True)
        frontier, reserve = level[: self.config.beam_width], level[self.config.beam_width:]
        self._precheck_car_desks(frontier)
        self.reserves[0] = reserve
        self._record_beam(reserve, "flights")
        self._summary("flights", len(self.flights), len(frontier), len(self.flights) - len(level), len(reserve))
        finals = frontier if final_single else self._forward(frontier, 1)
        # Backtracking: revisit branches the beam dropped at earlier stages.
        for index in range(0, len(self.stage_names) - 1):
            if finals or self.result.expansions >= self.config.max_expansions:
                break
            reserve = self.reserves.get(index, [])
            if reserve:
                self.reserves[index] = []
                finals = self._forward(reserve, index + 1, note="backtrack")
        return finals

    def candidate(self, node: _Node, price_range: PriceRange) -> ItineraryCandidate:
        breakdown = score_plan(self.request, node.flight, node.hotel, node.car, price_range, final=True,
                               transfer=node.transfer)
        budget = self.request.budget_usd
        return ItineraryCandidate(
            candidate_id="C-" + hashlib.sha256(node.node_id.encode()).hexdigest()[:8],
            flight=node.flight, hotel=node.hotel, car=node.car, transfer=node.transfer, total_price=round(node.total, 2),
            within_budget=(node.total <= budget) if budget is not None else None,
            warnings=list(dict.fromkeys(node.warnings)), score=breakdown,
            locked=[kind for kind in ("flight", "hotel", "car", "transfer") if kind in self.locked],
        )

    def issues(self, finals: list[_Node]) -> list[Issue]:
        """Why no complete plan survived, in a form the Planner can act on."""
        missing = []
        if not self.flights:
            missing.append(Issue(code="no_flights", agent="flight", detail="I couldn't find any flights for these dates."))
        if self.request.wants_hotel and not self.hotels:
            missing.append(Issue(code="no_hotels", agent="hotel", detail="I couldn't find any hotels for these dates."))
        if self.request.wants_car and not self.cars:
            missing.append(Issue(code="no_cars", agent="car", detail="I couldn't find any rental cars for these dates."))
        if self.request.wants_transfer and not self.transfers:
            missing.append(Issue(code="no_transfers", agent="transfer", detail="I couldn't find any airport pickups for these dates."))
        if missing or finals:
            return missing
        if self.flight_survivors == 0 and self.request.arrival_window and "landing_time_mismatch" in self.flight_codes:
            return [Issue(code="landing_time", agent="flight", detail="None of the flights I found land "
                          f"{text.LANDING_TEXT[self.request.arrival_window]} on {text.day(self.request.departure_date)}.")]
        if self.flight_survivors == 0 and not self.budget_only:
            top = "; ".join(describe(c).lower() for c, _ in self.flight_codes.most_common(2))
            return [Issue(code="flight_rules", agent="flight", detail=f"None of the flights fit your requirements ({top}).",
                          data={"codes": dict(self.flight_codes)})]
        issues: list[Issue] = []
        if self.late_checkin_blocked:
            issues.append(Issue(code="late_arrival_hotel", agent="hotel",
                                detail="The flights land after the hotels' front desks close."))
        if self.desk_hours:
            issues.append(Issue(code="car_desk_hours", agent="car",
                                detail="The rental desks are closed when you land or when you fly home.",
                                data={"hours": sorted(self.desk_hours)}))
        if self.budget_only:
            issues.append(Issue(code="over_budget", detail="Every complete trip I found is over your budget.",
                                data={"min_total": self.min_over_budget, "budget": self.request.budget_usd}))
        if not issues:
            other = {c: n for c, n in self.codes.items() if c != "over_budget" and c not in TIMING_CODES}
            detail = ("No combination passed my checks (" + "; ".join(describe(c).lower() for c in other) + ").") \
                if other else "No combination passed my checks."
            issues.append(Issue(code="rules", detail=detail, data={"codes": other}))
        return issues


def _models(options: list[dict[str, Any]], kind: str, model: type) -> list[Any]:
    return [model.model_validate(o) for o in options if o.get("kind") == kind]


def run_tot(
    request: TravelRequest,
    flights: list[dict[str, Any]],
    hotels: list[dict[str, Any]],
    cars: list[dict[str, Any]],
    transfers: list[dict[str, Any]] | None = None,
    *,
    locked: dict[str, dict[str, Any]] | None = None,
    config: ToTConfig | None = None,
) -> ToTResult:
    config = config or ToTConfig()
    locked = locked or {}
    result = ToTResult()
    f = [FlightOption.model_validate(locked["flight"])] if "flight" in locked else _models(flights, "flight", FlightOption)
    h = [HotelOption.model_validate(locked["hotel"])] if "hotel" in locked else _models(hotels, "hotel", HotelOption)
    c = [CarOption.model_validate(locked["car"])] if "car" in locked else _models(cars, "car", CarOption)
    t = [TransferOption.model_validate(locked["transfer"])] if "transfer" in locked else \
        _models(transfers or [], "transfer", TransferOption)
    search = _Search(request, f, h, c, t, locked, config, result)
    finals = search.run()
    if finals:
        totals = [n.total for n in finals]
        price_range = PriceRange(min(totals), max(totals))
        result.candidates = [search.candidate(node, price_range) for node in finals]
    result.issues = search.issues(finals)
    over_budget = next((i for i in result.issues if i.code == "over_budget"), None)
    if over_budget is not None and request.budget_usd is not None:
        # Find the cheapest complete plan ignoring the budget, so the traveler sees how far over it is.
        relaxed = TravelRequest.model_validate({**request.model_dump(), "budget_usd": None, "priority": "cost"})
        probe = _Search(relaxed, f, h, c, t, locked, config, ToTResult())
        near = probe.run()
        if near:
            cheapest = min(near, key=lambda n: n.total)
            miss = probe.candidate(cheapest, PriceRange(cheapest.total, cheapest.total))
            miss.within_budget = False
            miss.violations = [f"Over budget by ${cheapest.total - request.budget_usd:,.0f}"]
            result.near_misses = [miss]
            over_budget.data["min_total"] = round(cheapest.total, 2)
    result.candidates.sort(key=lambda c: -c.score.total)
    mark_ties(result.candidates, config.tie_margin)
    return result


def mark_ties(candidates: list[ItineraryCandidate], margin: float = 2.0) -> None:
    """Plans within `margin` points of the best are a tie; they're shown together for the traveler to choose."""
    for candidate in candidates:
        candidate.tie_group = None
    if len(candidates) >= 2 and candidates[0].score.total - candidates[1].score.total <= margin:
        for candidate in candidates:
            if candidates[0].score.total - candidate.score.total <= margin:
                candidate.tie_group = 1


def _clamp(value: float) -> float:
    return round(max(-5.0, min(5.0, float(value))), 2)


def _resort(candidates: list[ItineraryCandidate], margin: float) -> list[ItineraryCandidate]:
    candidates.sort(key=lambda c: -c.score.total)
    mark_ties(candidates, margin)
    return candidates


def apply_critic(candidates: list[ItineraryCandidate], adjustments: dict[str, tuple[float, str]],
                 margin: float = 2.0) -> list[ItineraryCandidate]:
    """Apply bounded LLM-critic adjustments. Unknown ids are ignored; the critic can't add or revive plans."""
    for candidate in candidates:
        if candidate.candidate_id in adjustments:
            value, reason = adjustments[candidate.candidate_id]
            candidate.score.llm_adjustment = _clamp(value)
            candidate.score.llm_reason = str(reason)[:200]
            candidate.score.recompute()
    return _resort(candidates, margin)


def apply_preferences(candidates: list[ItineraryCandidate], preferences: dict[str, Any],
                      weather_level: str | None = None, margin: float = 2.0) -> list[ItineraryCandidate]:
    """Small, bounded nudges from long-term memory (past approved choices) and weather risk."""
    airline = preferences.get("preferred_airline")
    area = preferences.get("preferred_hotel_area")
    car_class = preferences.get("preferred_car_class")
    for candidate in candidates:
        value, reasons = 0.0, []
        if airline and candidate.flight and candidate.flight.airline == airline:
            value += 1.5
            reasons.append(f"You chose {airline} before")
        if area and candidate.hotel and candidate.hotel.area == area:
            value += 1.0
            reasons.append(f"You chose a {area} hotel before")
        if car_class and candidate.car and candidate.car.vehicle_class == car_class:
            value += 0.5
            reasons.append(f"You chose a {car_class} car before")
        if weather_level == "high":
            if candidate.flight and candidate.flight.refundable:
                value += 1.0
                reasons.append("Refundable fare (high weather risk)")
            if candidate.hotel and candidate.hotel.free_cancellation_until:
                value += 1.0
                reasons.append("Free hotel cancellation (high weather risk)")
        candidate.score.pref_adjustment = _clamp(value)
        candidate.score.pref_reasons = reasons
        candidate.score.recompute()
    return _resort(candidates, margin)
