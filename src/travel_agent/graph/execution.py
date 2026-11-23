"""Proposal → human approval (interrupt) → sandbox execution → verification → reflection, plus rejection,
cancellation, and a rule-based Q&A fallback. Every booking or cancellation needs a single-use approval that is
bound to the exact item, price, and dates; the gateway verifies it before any action tool runs."""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import ValidationError

from travel_agent import text
from travel_agent.agents.planner import describe_request
from travel_agent.graph.deps import Deps, emit, traced
from travel_agent.graph.nodes import next_step, stated_trip_type, waiting_node
from travel_agent.graph.state import TravelState
from travel_agent.memory.store import LifecycleError, new_id
from travel_agent.models.itinerary import ItineraryCandidate
from travel_agent.models.options import CarOption, FlightOption, HotelOption, TransferOption
from travel_agent.models.planning import ApprovalDecision, ApprovalSnapshot
from travel_agent.models.request import TravelRequest
from travel_agent.tool_gateway import ToolCallError

MAX_SHOWN = 3
KINDS = ("flight", "hotel", "car", "transfer")
BOOK_TOOLS = {"flight": ("flights", "book_flight"), "hotel": ("travel-sandbox", "book_hotel"),
              "car": ("travel-sandbox", "book_car"), "transfer": ("travel-sandbox", "book_transfer")}
_KIND_WORDS = {"flight": r"\b(flights?|airfare|plane|tickets?|fare)\b",
               "hotel": r"\b(hotels?|rooms?|stay|accommodation)\b",
               "car": r"\b(cars?|rental|vehicle)\b",
               "transfer": r"\b(transfers?|pickups?|shuttles?|rides?)\b"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def plan_lines(candidate: ItineraryCandidate) -> list[str]:
    """One plain line per part of the trip, for chat messages."""
    flight, hotel = candidate.flight, candidate.hotel
    lines = []
    for kind in KINDS:
        option = getattr(candidate, kind)
        if option is None:
            continue
        if kind == "flight":
            detail = text.flight_text(option)
        elif kind == "hotel":
            detail = text.hotel_text(option)
        elif kind == "car":
            detail = text.car_text(option)
        else:
            detail = text.transfer_text(option, flight, hotel)
        booked = " (already booked)" if kind in candidate.locked else ""
        lines.append(f"{text.KIND_LABEL[kind]}: {detail} ({text.money(option.price_total)}){booked}")
    if flight is not None:
        lines.append(f"Airports: {text.airport(flight.outbound.origin)} to {text.airport(flight.outbound.destination)}")
    return lines


def _set_status(deps: Deps, option_id: str, status: str, reason: str, **kwargs: Any) -> None:
    try:
        deps.store.set_item_status(deps.trip_id, option_id, status, reason=reason, **kwargs)
    except LifecycleError:
        pass  # already in a state where this change doesn't apply (e.g., a booked component)


# Proposal and approval ---------------------------------------------------------------------------------------------
@traced("prepare_proposal")
async def prepare_proposal(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    tot = state.get("tot") or {}
    request = TravelRequest.model_validate(state["request"])
    shown = [ItineraryCandidate.model_validate(c) for c in tot.get("candidates", [])][:MAX_SHOWN]
    locked_ids = {o.get("option_id") for o in (state.get("locked") or {}).values()}
    shown_ids = {oid for c in shown for oid in c.option_ids()} - locked_ids
    for item in deps.store.list_items(deps.trip_id, ["recommended"]):
        if item["option_id"] not in shown_ids:
            _set_status(deps, item["option_id"], "superseded", "replaced by a newer proposal")
    for option_id in shown_ids:
        item = deps.store.get_item(deps.trip_id, option_id)
        if item and item["status"] in {"searched", "rejected", "failed", "superseded"}:
            _set_status(deps, option_id, "recommended", "shown to the traveler")

    best = shown[0]
    part, kept = state.get("part"), state.get("kept") or {}
    if part:
        count = f"{text.plural(len(shown), 'option')} with a different {text.KIND_NAME[part]}"
    else:
        count = text.plural(len(shown), "new option" if state.get("exclude_seen") else "option")
    top = f"My top pick costs {text.money(best.total_price)} in total"
    if request.budget_usd:
        top += f", {text.money(request.budget_usd - best.total_price)} under your budget"
    tied = [c for c in shown if c.tie_group]
    if len(tied) > 1:
        compare = f"The top {len(tied)} options are very close, so pick the one you like best. All of them are on the right."
    elif len(shown) > 1:
        compare = f"You can compare it with {text.plural(len(shown) - 1, 'other option')} on the right."
    else:
        compare = None
    notes = state.get("planner_notes") or []
    explanation = await deps.llm.explain(request, best, trip_type=stated_trip_type(state)) if deps.llm else None
    weather = (state.get("context") or {}).get("weather")
    message = text.paragraphs(
        f"I found {count} for your trip: {describe_request(request)}.",
        f"I kept the same {text.join_words(text.KIND_NAME[k] for k in KINDS if k in kept)}." if kept else None,
        f"{top}:\n" + text.bullets(plan_lines(best)),
        compare,
        ("What I adjusted along the way:\n" + text.bullets(notes[-3:])) if notes else None,
        text.weather_sentence(weather, text.city(request.destination)),
        text.rain_sentence(weather) if "no_rain" in (state.get("wishes") or []) else None,
        explanation,
        text.NEXT_OPTIONS,
    )
    proposal = {
        "proposal_id": new_id("prop"), "status": "pending", "created_at": _now(),
        "request": state["request"], "summary": describe_request(request),
        "candidates": [c.model_dump(mode="json") for c in shown], "recommended_id": best.candidate_id,
        "locked": sorted((state.get("locked") or {}).keys()), "message": message,
    }
    deps.store.update_trip(deps.trip_id, status="proposed", request=state["request"])
    emit("proposal", proposal_id=proposal["proposal_id"], candidates=[c.candidate_id for c in shown])
    seen = list(dict.fromkeys([*(state.get("seen_candidates") or []), *(c.candidate_id for c in shown)]))
    seen_options = {kind: list(ids) for kind, ids in (state.get("seen_options") or {}).items()}
    for candidate in shown:
        for kind in KINDS:
            option = getattr(candidate, kind)
            if option is not None and option.option_id not in seen_options.setdefault(kind, []):
                seen_options[kind].append(option.option_id)
    return {"proposal": proposal, "reply": proposal["message"], "reply_kind": "proposal", "seen_candidates": seen,
            "seen_options": seen_options,
            "top_pick": {"request": state["request"], "candidate": best.model_dump(mode="json")}}


async def await_approval(state: TravelState) -> dict[str, Any]:
    """Pause until the traveler approves, asks for a change, or rejects. No side effects before `interrupt()`."""
    proposal = state["proposal"] or {}
    answer = interrupt({"type": "approval", "proposal_id": proposal.get("proposal_id"),
                        "candidate_ids": [c["candidate_id"] for c in proposal.get("candidates", [])]})
    decision = answer if isinstance(answer, dict) else {"decision": str(answer)}
    if decision.get("decision") == "message":
        return {"decision": None, "intent": None, "user_message": decision.get("text", "")}
    update: dict[str, Any] = {"decision": decision}
    if decision.get("decision") == "change":
        update["user_message"] = decision.get("feedback") or ""
    return update


def route_after_approval(state: TravelState) -> str:
    decision = state.get("decision")
    if decision is None:
        return "route_intent"
    return {"approve": "record_approval", "reject": "handle_rejection", "change": "intake"}.get(
        decision.get("decision", ""), "__end__")


def approval_dates(option: FlightOption | HotelOption | CarOption | TransferOption) -> dict[str, str]:
    if isinstance(option, FlightOption):
        dates = {"departure_time": option.outbound.departure_time.isoformat()}
        if option.inbound is not None:
            dates["return_departure_time"] = option.inbound.departure_time.isoformat()
        return dates
    if isinstance(option, HotelOption):
        return {"check_in": option.check_in.isoformat(), "check_out": option.check_out.isoformat()}
    if isinstance(option, TransferOption):
        return {"pickup_date": option.pickup_date.isoformat()}
    return {"pickup_time": option.pickup_time.isoformat(), "dropoff_time": option.dropoff_time.isoformat()}


def learn_preferences(deps: Deps, candidate: ItineraryCandidate) -> None:
    if candidate.flight is not None:
        deps.store.set_preference("preferred_airline", candidate.flight.airline, "learned")
    if candidate.hotel is not None:
        deps.store.set_preference("preferred_hotel_area", candidate.hotel.area, "learned")
    if candidate.car is not None:
        deps.store.set_preference("preferred_car_class", candidate.car.vehicle_class, "learned")


@traced("record_approval")
async def record_approval(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    proposal = state.get("proposal")
    raw = state.get("decision") or {}
    try:
        decision = ApprovalDecision.model_validate({k: raw.get(k) for k in ("decision", "candidate_id", "feedback")})
    except ValidationError:
        return {"approvals": [], "reply_kind": "info", "reply": text.paragraphs(
            "Sorry, I couldn't read that choice.", text.NEXT_OPTIONS)}
    if not proposal or proposal.get("status") != "pending":
        return {"approvals": [], "reply_kind": "info", "reply": text.paragraphs(
            "Those options are no longer available.", 'Next step: say "search again" and I\'ll find fresh ones.')}
    chosen = next((c for c in proposal["candidates"] if c["candidate_id"] == decision.candidate_id), None)
    if chosen is None:
        return {"approvals": [], "reply_kind": "info",
                "reply": text.paragraphs("Please choose one of the options on the right.", text.NEXT_OPTIONS)}
    candidate = ItineraryCandidate.model_validate(chosen)
    deps.store.expire_unused_approvals(deps.trip_id)
    approvals = []
    for kind in KINDS:
        option = getattr(candidate, kind)
        if option is None or kind in candidate.locked:
            continue
        snapshot = ApprovalSnapshot(
            action=f"book_{kind}", kind=kind, option_id=option.option_id, provider=option.source.provider,  # type: ignore[arg-type]
            total_price=option.price_total, currency=option.currency, dates=approval_dates(option), label=option.label(),
        )
        approval_id = deps.store.create_approval(deps.trip_id, snapshot, deps.run_id)
        deps.store.set_item_status(deps.trip_id, option.option_id, "approved", reason="approved by the traveler",
                                   approval_id=approval_id)
        approvals.append({"approval_id": approval_id, **snapshot.model_dump()})
    chosen_ids = set(candidate.option_ids())
    for item in deps.store.list_items(deps.trip_id, ["recommended"]):
        if item["option_id"] not in chosen_ids:
            _set_status(deps, item["option_id"], "superseded", "the traveler chose another option")
    learn_preferences(deps, candidate)
    emit("approval", candidate_id=candidate.candidate_id, approvals=[a["approval_id"] for a in approvals])
    return {"approvals": approvals, "execution": [],
            "proposal": {**proposal, "status": "approved", "approved_candidate_id": candidate.candidate_id}}


def route_after_record(state: TravelState) -> str:
    return "execute" if state.get("approvals") else "__end__"


# Execute → verify → reflect ----------------------------------------------------------------------------------------
async def _book_one(deps: Deps, approval: dict[str, Any]) -> dict[str, Any]:
    run = deps.run.at("execute")
    kind, option_id, price = approval["kind"], approval["option_id"], approval["total_price"]
    base = {"kind": kind, "option_id": option_id, "label": approval["label"], "approved_price": price,
            "approval_id": approval["approval_id"]}

    def failed(status: str, message: str, current_price: float | None = None) -> dict[str, Any]:
        _set_status(deps, option_id, "failed", message)
        return {**base, "status": status, "message": message, "current_price": current_price}

    try:
        if kind == "flight":
            refresh = await deps.gateway.call("flights", "refresh_offer", {"offer_id": option_id}, role="executor", run=run)
            if not refresh["found"] or not refresh["available"]:
                return failed("unavailable", refresh.get("message") or "The fare is no longer available")
            current = refresh.get("current_price")
            if current is not None and abs(current - price) > 0.009:
                return failed("price_changed", f"The fare changed from ${price:,.2f} to ${current:,.2f} after you "
                                               "approved it", current)
            result = await deps.gateway.call("flights", "book_flight", {
                "offer_id": option_id, "approval_id": approval["approval_id"], "expected_total_price": price},
                role="executor", run=run)
        else:
            args: dict[str, Any] = {"option_id": option_id, "approval_id": approval["approval_id"],
                                    "expected_total_price": price}
            item = deps.store.get_item(deps.trip_id, option_id)
            if kind == "hotel" and item and item["provider"] != "mock":
                args["option_snapshot"] = item["option"]
            server, tool = BOOK_TOOLS[kind]
            result = await deps.gateway.call(server, tool, args, role="executor", run=run)
    except ToolCallError as exc:
        return failed("error", f"{exc.code}: {exc.message}")
    if result["status"] != "confirmed":
        current = result.get("current_price")
        message = result.get("message") or result["status"]
        if result["status"] == "price_changed" and current is not None:
            message = f"The price changed from ${price:,.2f} to ${current:,.2f} after you approved it"
        return failed(result["status"], message, current)
    booking = result["booking"]
    deps.store.save_booking(deps.trip_id, booking, verified=False)
    deps.store.set_item_status(deps.trip_id, option_id, "booked", reason="sandbox booking confirmed",
                               approval_id=approval["approval_id"], booking_ref=booking["booking_ref"])
    return {**base, "status": "booked", "booking_ref": booking["booking_ref"], "price": booking["total_price"],
            "message": result.get("message", "")}


@traced("execute")
async def execute(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    results: list[dict[str, Any]] = []
    stopped = False
    for approval in state.get("approvals") or []:
        if stopped:
            _set_status(deps, approval["option_id"], "superseded", "not attempted because an earlier step failed")
            results.append({"kind": approval["kind"], "option_id": approval["option_id"], "label": approval["label"],
                            "approved_price": approval["total_price"], "status": "skipped",
                            "message": "Not attempted because an earlier step failed"})
            continue
        emit("execute", kind=approval["kind"], status="running", label=approval["label"])
        outcome = await _book_one(deps, approval)
        emit("execute", **{k: v for k, v in outcome.items() if k != "approval_id"})
        results.append(outcome)
        stopped = outcome["status"] != "booked"
    if stopped:
        deps.store.expire_unused_approvals(deps.trip_id)
    return {"execution": results}


def compare_booking(booking: dict[str, Any], approval: dict[str, Any]) -> list[str]:
    problems = []
    if booking["status"] != "confirmed":
        problems.append(f"status is {booking['status']}")
    if booking["option_id"] != approval["option_id"]:
        problems.append("the provider booked a different option")
    if abs(float(booking["total_price"]) - float(approval["total_price"])) > 0.009:
        problems.append(f"price ${booking['total_price']:,.2f} differs from the approved ${approval['total_price']:,.2f}")
    details = booking.get("details") or {}
    for key, value in (approval.get("dates") or {}).items():
        if str(details.get(key)) != value:
            problems.append(f"{key.replace('_', ' ')} differs from what you approved")
    return problems


@traced("verify")
async def verify(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    approvals = {a["option_id"]: a for a in state.get("approvals") or []}
    updated = []
    for result in state.get("execution") or []:
        if result["status"] != "booked":
            updated.append(result)
            continue
        server, tool = ("flights", "get_flight_order") if result["kind"] == "flight" else ("travel-sandbox", "get_booking")
        problems: list[str] = []
        booking = None
        try:
            lookup = await deps.gateway.call(server, tool, {"booking_ref": result["booking_ref"]}, role="executor",
                                             run=deps.run.at("verify"))
            booking = lookup.get("booking") if lookup.get("found") else None
        except ToolCallError as exc:
            problems.append(f"could not check the booking ({exc.code})")
        if booking is None and not problems:
            problems.append("the provider has no record of this booking")
        if booking is not None:
            problems += compare_booking(booking, approvals[result["option_id"]])
            deps.store.save_booking(deps.trip_id, booking, verified=not problems)
        updated.append({**result, "verified": not problems, "problems": problems})
    emit("verify", results=[{"kind": r["kind"], "verified": r.get("verified")} for r in updated if r["status"] == "booked"])
    return {"execution": updated}


@traced("reflect")
async def reflect(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    results = state.get("execution") or []
    booked = [r for r in results if r["status"] == "booked"]
    failed = [r for r in results if r["status"] not in {"booked", "skipped"}]
    skipped = [r for r in results if r["status"] == "skipped"]
    unverified = [r for r in booked if not r.get("verified")]
    proposal = state.get("proposal") or {}
    lines_booked = [f"{text.KIND_LABEL[r['kind']]}: {r['label']}, confirmation {r['booking_ref']} ({text.money(r['price'])})"
                    for r in booked]
    if booked and not failed and not skipped and not unverified:
        deps.store.update_trip(deps.trip_id, status="booked")
        reply = text.paragraphs(
            "All set. Your trip is booked in the sandbox (no real payment):\n" + text.bullets(lines_booked),
            "I checked each booking with the provider, and everything matches what you approved.",
            text.NEXT_BOOKED)
        return {"reply": reply, "reply_kind": "execution", "proposal": {**proposal, "status": "executed"}}
    deps.store.update_trip(deps.trip_id, status="booked" if booked else "needs_input")
    not_booked = [text.KIND_NAME[r["kind"]] for r in failed + skipped]
    reply = text.paragraphs(
        f"I stopped booking because {failure_reason(failed[0])}." if failed else None,
        ("Already booked:\n" + text.bullets(lines_booked)) if booked else None,
        "".join(f"I couldn't confirm the {text.KIND_NAME[r['kind']]} booking with the provider "
                f"({'; '.join(r.get('problems') or [])}). " for r in unverified) or None,
        f"Not booked: {text.join_words(not_booked)}." if not_booked else None,
        'Next step: say "search again" and I\'ll find new options' + (" for the rest." if booked else "."),
    )
    return {"reply": reply, "reply_kind": "execution", "proposal": {**proposal, "status": "failed"}}


def failure_reason(result: dict[str, Any]) -> str:
    name = text.KIND_NAME[result["kind"]]
    if result["status"] == "price_changed" and result.get("current_price") is not None:
        direction = "went up" if result["current_price"] > result["approved_price"] else "changed"
        return (f"the {name} price {direction} from {text.money(result['approved_price'])} to "
                f"{text.money(result['current_price'])} after you approved it")
    if result["status"] == "unavailable":
        return f"the {name} ({result['label']}) is no longer available"
    return f"the {name} booking didn't go through"


@traced("handle_rejection")
async def handle_rejection(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    proposal = state.get("proposal")
    if not proposal or proposal.get("status") != "pending":
        return {"reply_kind": "info", "reply": text.paragraphs("There are no options waiting for you right now.",
                                                               text.NEXT_CHANGE)}
    for item in deps.store.list_items(deps.trip_id, ["recommended"]):
        _set_status(deps, item["option_id"], "rejected", "rejected by the traveler")
    feedback = (state.get("decision") or {}).get("feedback")
    if feedback:
        deps.store.add_feedback(deps.trip_id, "proposal", proposal["proposal_id"], "negative", feedback[:500])
    deps.store.expire_unused_approvals(deps.trip_id)
    deps.store.update_trip(deps.trip_id, status="rejected")
    return {"proposal": {**proposal, "status": "rejected"}, "reply_kind": "info", "reply": text.paragraphs(
        "OK, I won't book any of these.",
        'Next step: say "more options" to see others, or tell me what to change (dates, budget, stops, hotel area, '
        "rental car, or airport pickup), and I'll show you an updated plan.")}


# Cancellation ------------------------------------------------------------------------------------------------------
def refund_note(booking: dict[str, Any]) -> str:
    details = booking["record"].get("details") or {}
    if booking["kind"] == "flight":
        if "refundable" not in details:  # e.g. Duffel: the airline's fare rules decide, shown after cancelling
            return "the airline decides any refund"
        return "refundable fare" if details["refundable"] else "non-refundable fare, no refund"
    if booking["kind"] == "hotel":
        deadline = details.get("free_cancellation_until")
        if deadline and date.fromisoformat(deadline) >= date.today():
            day = date.fromisoformat(deadline)
            return f"free cancellation until {day:%b} {day.day}"
        return "past the free-cancellation date, no refund" if deadline else "non-refundable rate, no refund"
    return "free cancellation" if details.get("free_cancellation") else "no refund"


@traced("prepare_cancellation")
async def prepare_cancellation(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    said = state.get("user_message") or ""
    kinds = {kind for kind, pattern in _KIND_WORDS.items() if re.search(pattern, said, re.I)} or set(KINDS)
    targets = [b for b in deps.store.list_bookings(deps.trip_id, statuses=["confirmed"]) if b["kind"] in kinds]
    if not targets:
        names = text.join_words(text.KIND_NAME[k] for k in KINDS if k in kinds).replace(" and ", " or ")
        has_bookings = bool(deps.store.list_bookings(deps.trip_id, statuses=["confirmed"]))
        return {"cancellation": None, "reply_kind": "info", "reply": text.paragraphs(
            f"You don't have a confirmed {names} booking to cancel.", next_step(state, has_bookings))}
    items = []
    for booking in targets:
        item = deps.store.get_item(deps.trip_id, booking["option_id"])
        items.append({"booking_ref": booking["booking_ref"], "kind": booking["kind"], "option_id": booking["option_id"],
                      "provider": booking["provider"], "total_price": booking["total_price"],
                      "currency": booking["currency"], "label": item["label"] if item else booking["booking_ref"],
                      "refund_note": refund_note(booking)})
    which = "this booking" if len(items) == 1 else "these bookings"
    message = text.paragraphs(
        f"Do you want me to cancel {which}?\n" + text.bullets(
            f"{text.KIND_LABEL[i['kind']]}: {i['label']}, confirmation {i['booking_ref']}, "
            f"{text.money(i['total_price'])} ({i['refund_note']})" for i in items),
        text.NEXT_CANCEL)
    cancellation = {"cancellation_id": new_id("cxl"), "status": "pending", "targets": items, "created_at": _now(),
                    "message": message}
    return {"cancellation": cancellation, "reply": cancellation["message"], "reply_kind": "cancel_proposal"}


def route_after_prepare_cancellation(state: TravelState) -> str:
    return "await_cancel_approval" if (state.get("cancellation") or {}).get("status") == "pending" else "__end__"


async def await_cancel_approval(state: TravelState) -> dict[str, Any]:
    cancellation = state["cancellation"] or {}
    answer = interrupt({"type": "cancel_approval", "cancellation_id": cancellation.get("cancellation_id"),
                        "booking_refs": [t["booking_ref"] for t in cancellation.get("targets", [])]})
    decision = answer if isinstance(answer, dict) else {"decision": str(answer)}
    if decision.get("decision") == "message":
        return {"decision": None, "intent": None, "user_message": decision.get("text", "")}
    return {"decision": {**decision, "kind": "cancel"}}


def route_after_cancel_approval(state: TravelState) -> str:
    decision = state.get("decision")
    if decision is None:
        return "route_intent"
    return "execute_cancellation" if decision.get("decision") == "approve" else "cancel_declined"


@traced("execute_cancellation")
async def execute_cancellation(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    cancellation = state.get("cancellation")
    if not cancellation or cancellation.get("status") != "pending":
        return {"reply_kind": "info", "reply": text.paragraphs("There's no cancellation waiting for you.",
                                                               next_step(state, has_bookings=True))}
    run = deps.run.at("execute_cancellation")
    results = []
    for target in cancellation["targets"]:
        kind = target["kind"]
        snapshot = ApprovalSnapshot(
            action=f"cancel_{kind}", kind=kind, option_id=target["option_id"], provider=target["provider"],
            total_price=target["total_price"], currency=target["currency"], booking_ref=target["booking_ref"],
            label=target["label"],
        )
        approval_id = deps.store.create_approval(deps.trip_id, snapshot, deps.run_id)
        server, tool, lookup_tool = (("flights", "cancel_flight_order", "get_flight_order") if kind == "flight"
                                     else ("travel-sandbox", "cancel_booking", "get_booking"))
        try:
            outcome = await deps.gateway.call(server, tool, {"booking_ref": target["booking_ref"],
                                                             "approval_id": approval_id}, role="executor", run=run)
        except ToolCallError as exc:
            results.append({**target, "status": "error", "message": f"{exc.code}: {exc.message}"})
            continue
        if outcome["status"] != "cancelled":
            results.append({**target, "status": outcome["status"], "message": outcome.get("message", "")})
            continue
        verified = False
        try:
            lookup = await deps.gateway.call(server, lookup_tool, {"booking_ref": target["booking_ref"]},
                                             role="executor", run=run)
            booking = lookup.get("booking") or outcome.get("booking")
            verified = bool(lookup.get("found") and booking and booking["status"] == "cancelled")
        except ToolCallError:
            booking = outcome.get("booking")
        if booking:
            deps.store.save_booking(deps.trip_id, booking, verified=verified)
        deps.store.set_item_status(deps.trip_id, target["option_id"], "cancelled", approval_id=approval_id,
                                   reason="cancelled by the traveler (sandbox)")
        results.append({**target, "status": "cancelled", "refund": outcome.get("refund_amount"), "verified": verified})
    deps.store.expire_unused_approvals(deps.trip_id)
    remaining = deps.store.list_bookings(deps.trip_id, statuses=["confirmed"])
    deps.store.update_trip(deps.trip_id, status="booked" if remaining else "cancelled")
    done = [r for r in results if r["status"] == "cancelled"]
    problems = [r for r in results if r["status"] != "cancelled"]
    reply = text.paragraphs(
        ("Cancelled:\n" + text.bullets(
            f"{text.KIND_LABEL[r['kind']]}: {r['label']}"
            + (f", refund {text.money(r['refund'])}" if r.get("refund") is not None else "") for r in done)) if done else None,
        ("I couldn't cancel:\n" + text.bullets(
            f"{text.KIND_LABEL[r['kind']]}: {r['label']} ({r.get('message') or r['status']})" for r in problems))
        if problems else None,
        "I checked with the provider, and the cancellation went through." if done and all(r["verified"] for r in done)
        else None,
        text.NEXT_BOOKED if remaining else "Next step: tell me if you'd like to plan this trip again or start a new one.",
    )
    return {"cancellation": {**cancellation, "status": "done", "results": results}, "reply_kind": "cancel_result",
            "reply": reply}


@traced("cancel_declined")
async def cancel_declined(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    cancellation = state.get("cancellation")
    return {"cancellation": {**cancellation, "status": "declined"} if cancellation else None, "reply_kind": "info",
            "reply": text.paragraphs("OK, I kept your bookings. Nothing was cancelled.", text.NEXT_BOOKED)}


# Questions: ReAct with read-only tools when the LLM is on; simple rules otherwise ----------------------------------
WEATHER_WORDS = re.compile(r"\b(weather|forecast\w*|rain\w*|snow\w*|temperature\w*|pack\w*|jacket|coat|cold|warm|hot"
                           r"|umbrella|wear)\b", re.I)
QA_TRIP_FIELDS = {"origin", "destination", "departure_date", "return_date", "travelers", "trip_type"}


@traced("qa_assistant")
async def qa_assistant(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    question = state.get("user_message") or ""
    request = TravelRequest.model_validate(state["request"]) if state.get("request") else None
    footer = next_step(state, bool(deps.store.list_bookings(deps.trip_id, statuses=["confirmed"])))
    answer = None
    # Weather questions about the trip get the weather tool's answer; the LLM answers the rest.
    if deps.llm is not None and not (request is not None and WEATHER_WORDS.search(question)):
        # Exact codes and ISO dates, so the model calls tools with the right year.
        context = json.dumps(request.model_dump(mode="json", include=QA_TRIP_FIELDS)) if request else ""
        answer = await deps.llm.answer(question, await deps.gateway.langchain_tools("qa"), context)
    if answer is None:
        answer = await rule_answer(deps, question, request)
    return {"reply": text.paragraphs(answer, footer), "reply_kind": "answer"}


async def rule_answer(deps: Deps, question: str, request: TravelRequest | None) -> str:
    run = deps.run.at("qa_assistant")
    if WEATHER_WORDS.search(question):
        if request is None:
            return "Tell me where and when you're going first, and I'll check the weather."
        try:
            risk = await deps.gateway.call("trip-context", "weather_risk", {
                "city_code": request.destination, "start_date": request.departure_date.isoformat(),
                "end_date": (request.return_date or request.departure_date).isoformat(), "trip_type": request.trip_type},
                role="qa", run=run)
        except ToolCallError:
            return "Sorry, I couldn't get the weather right now. Please try again in a moment."
        report = risk["report"]
        kind = {"forecast": "This is the latest forecast.", "sample": "This is sample data, not a real forecast."}
        return text.paragraphs(
            f"Weather in {text.city(request.destination)}, {text.dates(request.departure_date, request.return_date)}:",
            f"{report['summary']} Overall risk: {risk['level']}. {kind.get(report['kind'], '')}".strip(),
            " ".join([a for a in risk.get("advice", []) if "not a forecast" not in a][:2]) or None)
    if re.search(r"airport", question, re.I) and request is not None:
        try:
            data = await deps.gateway.call("trip-context", "nearby_airports", {"iata": request.destination},
                                           role="qa", run=run)
        except ToolCallError:
            return "Sorry, I couldn't look up airports right now. Please try again in a moment."
        if not data["matches"]:
            return f"I didn't find other large airports near {text.place(request.destination)}."
        return f"Other large airports near {text.place(request.destination)}:\n" + text.bullets(
            f"{m['name']} ({m['iata']})" for m in data["matches"])
    return ("I can answer general travel questions when the local AI model is turned on. Right now I can plan or "
            "change your trip, check the weather for it, and list nearby airports.")


def route_after_qa(state: TravelState) -> str:
    return waiting_node(state)
