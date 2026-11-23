"""Graph nodes for intent routing, intake, Plan-and-Execute planning (with plan approval), the parallel search
agents, Tree-of-Thought evaluation, and the Planner's review loop."""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Send, interrupt
from pydantic import ValidationError

from travel_agent import intake as rules
from travel_agent import text
from travel_agent.agents import (
    run_car_agent,
    run_context_agent,
    run_flight_agent,
    run_hotel_agent,
    run_transfer_agent,
)
from travel_agent.agents.planner import ReviewInput, build_task_plan, describe_request, review, trip_title
from travel_agent.engine.tot import ToTConfig, ToTResult, apply_preferences, run_tot
from travel_agent.graph.deps import Deps, emit, traced
from travel_agent.graph.state import RESET, AgentTask, TravelState, replace
from travel_agent.models.itinerary import Issue, ItineraryCandidate
from travel_agent.models.planning import SearchDirective
from travel_agent.models.request import TravelRequest
from travel_agent.models.results import AgentResult, AgentStatus
from travel_agent.tool_gateway import ToolCallError

AGENT_RUNNERS = {"flight": run_flight_agent, "hotel": run_hotel_agent, "car": run_car_agent,
                 "transfer": run_transfer_agent}
REMEMBERED_FIELDS = ("priority", "max_stops", "cabin_class")
MORE_BEAM_WIDTH = 5  # "more options" keeps more trips at each step, so some are new
# Which request fields a booked component depends on; changing them after booking needs a cancellation first.
BOOKED_DEPENDENCIES = {
    "flight": {"origin", "destination", "departure_date", "return_date", "travelers", "cabin_class", "max_stops",
               "one_way", "arrival_window"},
    "hotel": {"destination", "departure_date", "return_date", "travelers", "hotel_area", "need_hotel"},
    "car": {"destination", "departure_date", "return_date", "travelers", "car_class", "require_car"},
    "transfer": {"destination", "departure_date", "travelers", "need_transfer"},
}
_YES = re.compile(r"^(?:(?:yes|yep|yeah|sure|ok(?:ay)?|go(?: ahead| for it)?|start(?: the search| searching)?|search|"
                  r"proceed|continue|do it|sounds good|looks good|let'?s go|please do|confirm(?:ed)?|approved?)"
                  r"\b[\s,.!]*(?:(?:and|then)\s+)?)+(?:please|thanks|thank you)?[\s.!]*$", re.I)
_NO = re.compile(r"^(?:no|nope|not now|stop|wait|hold on|don'?t)\b[\s,.!]*(?:thanks|thank you)?[\s.!]*$", re.I)

_QUESTION_START = re.compile(
    r"^\s*(what|what's|whats|how|is|are|will|would|should|when|where|which|why|do|does|did|can you tell|tell me|any)\b", re.I)
_QA_TOPICS = re.compile(
    r"\b(weather|forecast|rain|snow|temperature|things to do|attractions?|visa|safe|safety|restaurants?|"
    r"neighbou?rhoods?|pack(?:ing)?|airports? near|nearby airports?|time zone|currency|sights?|museums?)\b", re.I)
_CHANGE_HINTS = re.compile(
    r"\b(cheaper|less expensive|nonstop|direct|earlier|later|another|different|instead|change|switch|swap|add|"
    r"remove|without)\b", re.I)
_CANCEL = re.compile(r"\bcancel\b", re.I)
_RETRY = re.compile(r"\b(try again|retry|search again|run (?:it|the search) again|re-?search)\b", re.I)
_RESET = re.compile(r"\b(start over|new trip|reset|forget (?:it|this trip))\b", re.I)
_MORE = re.compile(r"\b(?:more|other|different|another|new)\s+(?:options?|choices?|ones?|results?)\b|"
                   r"\b(?:show|give|find|get)\s+(?:me\s+)?(?:some\s+)?(?:more|others)\b(?:\s+please)?[\s.!?]*$|"
                   r"\bsomething else\b|\balternatives?\b", re.I)
# One part of the trip: "a different hotel", "other flights", "hotel options", "change the car".
_KIND = r"(airport pickups?|rental cars?|hotels?|flights?|cars?|pickups?|shuttles?|transfers?)"
_PART = re.compile(r"\b(?:more|other|different|another|new|change|switch|swap|replace)\s+(?:the\s+|my\s+|a\s+|your\s+)?"
                   + _KIND + r"\b|\b" + _KIND + r"\s+(?:options?|choices?|alternatives?)\b", re.I)
KINDS = ("flight", "hotel", "car", "transfer")
# What the optional LLM's intents mean for the graph.
LLM_INTENTS = {"change_trip": "plan", "question": "question", "more_options": "more", "search_again": "retry",
               "go_ahead": "go_ahead", "decline": "decline", "cancel_booking": "cancel", "new_trip": "reset",
               "unclear": "unclear"}
GO_AHEAD_REPLY = {"await_approval": "Which option would you like to book?", "await_cancel_approval": "Got it."}
EXAMPLES = [
    '"more options", "a different hotel", or "search again"',
    '"make it nonstop", "cheaper", or "budget $3000"',
    '"land at night", "no car", or "no airport pickup"',
    '"leave Dec 12" or "return Dec 20"',
    'questions such as "what\'s the weather like?"',
]


# Intent ------------------------------------------------------------------------------------------------------------
def part_of(message: str) -> str | None:
    """Which single part of the trip the traveler wants changed ("a different hotel" -> hotel), if any."""
    match = _PART.search(message)
    if not match:
        return None
    word = (match[1] or match[2]).lower()
    return "transfer" if re.search(r"pickup|shuttle|transfer", word) else "car" if "car" in word else \
        "hotel" if "hotel" in word else "flight"


def stated_trip_type(state: TravelState) -> str | None:
    """The trip type only if the traveler said it (the request defaults to leisure)."""
    return (state.get("fields") or {}).get("trip_type")


def classify_intent(text: str, *, has_request: bool, has_bookings: bool) -> tuple[str, bool]:
    """Keyword rules. Returns (intent, whether the rules really understood the message)."""
    stripped = text.strip()
    if not stripped:
        return "noop", False
    if _CANCEL.search(stripped) and has_bookings:
        return "cancel", True
    if _RESET.search(stripped):
        return "reset", True
    if _RETRY.search(stripped) and has_request:
        return "retry", True
    if _YES.match(stripped):
        return "go_ahead", True
    if _NO.match(stripped):
        return "decline", True
    extraction = rules.extract(stripped)
    found = bool(extraction.fields or extraction.places or extraction.wishes)
    if (_MORE.search(stripped) or _PART.search(stripped)) and has_request and not found:
        return "more", True
    trip_signal = found or bool(_CHANGE_HINTS.search(stripped))
    asks = stripped.endswith("?") or bool(_QUESTION_START.search(stripped))
    if asks and (_QA_TOPICS.search(stripped) or not trip_signal):
        return "question", bool(_QA_TOPICS.search(stripped))
    return "plan", found


def waiting_for(state: TravelState) -> str:
    return {"await_plan_approval": "the traveler's go-ahead for the search plan",
            "await_approval": "the traveler to choose one of the proposed options",
            "await_cancel_approval": "the traveler to confirm a cancellation"}.get(waiting_node(state), "nothing")


@traced("route_intent")
async def route_intent(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    intent = state.get("intent")
    update: dict[str, Any] = {"reply": "", "reply_kind": "", "understood": None}
    if intent != "decision":
        message = state.get("user_message") or ""
        has_bookings = bool(deps.store.list_bookings(deps.trip_id, statuses=["confirmed"]))
        intent, understood = classify_intent(message, has_request=bool(state.get("request")), has_bookings=has_bookings)
        part = part_of(message)
        if deps.llm is not None and not understood:
            result = await deps.llm.understand(message, state.get("fields") or {}, waiting_for(state))
            if result is not None:
                intent, part = LLM_INTENTS[result["intent"]], result.get("part")
                update["understood"] = {**result, "fields": fields_to_state(result["fields"])}
                emit("understood", intent=result["intent"])
        update["change_part"] = part if intent == "more" else None
        waiting = waiting_node(state)
        if intent == "go_ahead":
            if waiting == "await_plan_approval":
                update["plan_status"] = "approved"
            elif waiting == "__end__" and state.get("plan_status") == "declined" and state.get("request"):
                intent = "retry"  # the traveler changed their mind about the plan they declined
            else:
                # Booking and cancelling still need the buttons, so a typed "yes" never books or cancels anything.
                update.update(reply_kind="info", reply=text.paragraphs(
                    GO_AHEAD_REPLY.get(waiting, "Sure."), next_step(state, has_bookings)))
        elif intent == "decline":
            if waiting == "await_plan_approval":
                update.update(plan_status="declined", reply_kind="info",
                              reply=text.paragraphs("OK, I won't search yet.", text.NEXT_CHANGE))
            elif waiting == "__end__":
                update.update(reply_kind="info", reply=text.paragraphs("OK.", next_step(state, has_bookings)))
        elif intent == "unclear":
            if state.get("request"):
                update.update(reply_kind="info", reply=not_understood(state, deps, has_bookings))
            else:
                intent = "plan"  # intake asks for the missing trip details
    emit("intent", intent=intent)
    return {**update, "intent": intent}


def not_understood(state: TravelState, deps: Deps, has_bookings: bool) -> str:
    intro = ("Sorry, I didn't understand that. Could you say it another way? For example:" if deps.llm is not None else
             "Sorry, I'm not sure what you mean. The AI model is off, so I understand short phrases like these:")
    return text.paragraphs(intro + "\n" + text.bullets(EXAMPLES), next_step(state, has_bookings))


def route_after_intent(state: TravelState) -> list[Send] | str:
    intent = state.get("intent")
    if intent in {"plan", "reset"}:
        return "intake"
    if intent in {"retry", "more"}:
        return "planner" if state.get("request") else "intake"
    if intent == "go_ahead" and state.get("plan_status") == "approved" and state.get("reply_kind") != "info":
        return dispatch(state)
    if intent == "decline":
        return {"await_approval": "handle_rejection", "await_cancel_approval": "cancel_declined"}.get(
            waiting_node(state), "__end__")
    if intent in {"go_ahead", "unclear"}:
        return waiting_node(state)
    if intent == "question":
        return "qa_assistant"
    if intent == "cancel":
        return "prepare_cancellation"
    if intent == "decision":
        decision = state.get("decision") or {}
        if decision.get("kind") == "cancel":
            return "execute_cancellation" if decision.get("decision") == "approve" else "cancel_declined"
        if decision.get("kind") == "start":
            return "__end__"  # no plan is waiting (for example, a stale button press)
        return {"approve": "record_approval", "reject": "handle_rejection"}.get(decision.get("decision", ""), "intake")
    return "__end__"


def waiting_node(state: TravelState) -> str:
    """After a side question or a message that changed nothing, go back to whatever was waiting for the traveler."""
    if (state.get("cancellation") or {}).get("status") == "pending":
        return "await_cancel_approval"
    if (state.get("proposal") or {}).get("status") == "pending":
        return "await_approval"
    if state.get("plan_status") == "pending" and state.get("request"):
        return "await_plan_approval"
    return "__end__"


def next_step(state: TravelState, has_bookings: bool) -> str:
    waiting = waiting_node(state)
    if waiting == "await_cancel_approval":
        return text.NEXT_CANCEL
    if waiting == "await_approval":
        return text.NEXT_OPTIONS
    if waiting == "await_plan_approval":
        return text.NEXT_PLAN
    if has_bookings:
        return text.NEXT_BOOKED
    if state.get("request"):
        return 'Next step: say "search again" to see options, or tell me what to change.'
    return "Next step: tell me where and when you'd like to travel."


# Intake ------------------------------------------------------------------------------------------------------------
def fields_to_state(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value.isoformat() if isinstance(value, date) else value for key, value in fields.items()}


def fields_from_state(fields: dict[str, Any]) -> dict[str, Any]:
    restored = dict(fields)
    for key in ("departure_date", "return_date"):
        if isinstance(restored.get(key), str):
            restored[key] = date.fromisoformat(restored[key])
    return restored


def _norm(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", folded)).strip()


async def resolve_place(deps: Deps, place: str) -> tuple[str | None, str | None]:
    """Resolve a place name to an airport or metro code via the trip-context MCP server.
    Returns (code, question to ask if it can't be resolved)."""
    queries = [place]
    if re.search(r"\s+city$", place, re.I):
        queries.append(re.sub(r"\s+city$", "", place, flags=re.I))
    matches: list[dict[str, Any]] = []
    for query in queries:
        try:
            data = await deps.gateway.call("trip-context", "lookup_airport", {"query": query[:80], "limit": 5},
                                           role="search", run=deps.run.at("intake"))
        except ToolCallError:
            return None, f'Sorry, I couldn\'t look up "{place}" right now. Which airport code should I use?'
        matches = data["matches"]
        if matches:
            place = query
            break
    if not matches:
        return None, f'I couldn\'t find an airport for "{place}". Which city or airport code do you mean?'
    key = _norm(place)
    exact = [m for m in matches if _norm(m["city"]) == key or m["iata"] == place.upper()]
    if not exact:
        choices = ", ".join(f"{m['iata']} ({m['city']}, {m['country']})" for m in matches[:3])
        return None, f'Which airport do you mean for "{place}": {choices}?'
    return exact[0]["iata"], None


def _booked_conflicts(deps: Deps, previous: dict[str, Any] | None, request: TravelRequest) -> list[str]:
    if not previous:
        return []
    new = request.model_dump(mode="json")
    changed = {key for key in new if new.get(key) != previous.get(key)}
    booked = {item["kind"] for item in deps.store.list_items(deps.trip_id, ["booked"])}
    return sorted(kind for kind in booked if changed & BOOKED_DEPENDENCIES[kind])


def saved_preference_text(key: str, value: Any) -> str:
    if key == "priority":
        return f"optimize for {text.PRIORITY_TEXT.get(value, value)}"
    if key == "max_stops":
        return "nonstop only" if value == 0 else f"up to {text.plural(value, 'stop')}"
    return f"{str(value).replace('_', ' ')} class"


@traced("intake")
async def intake(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    message = state.get("user_message") or ""
    reset = state.get("intent") == "reset"
    current = {} if reset else fields_from_state(state.get("fields") or {})
    extraction = rules.extract(message)
    understood = state.get("understood") or {}
    for key, value in fields_from_state(understood.get("fields") or {}).items():
        extraction.fields.setdefault(key, value)
    for role, place in (understood.get("places") or {}).items():
        if role not in extraction.fields:
            extraction.places.setdefault(role, place)
    questions: list[str] = []
    for role, place in extraction.places.items():
        code, question = await resolve_place(deps, place)
        if code:
            extraction.fields[role] = code
        elif question:
            questions.append(question)
    merged, changed = rules.merge(current, extraction.fields)

    for key in REMEMBERED_FIELDS:
        if key in extraction.fields:
            if extraction.fields[key] is None:
                deps.store.delete_preference(key)
            else:
                deps.store.set_preference(key, extraction.fields[key], "stated")
    stated = {k: v["value"] for k, v in deps.store.get_preferences().items() if v["source"] == "stated"}
    assumptions: list[str] = []
    for key in REMEMBERED_FIELDS:
        if key not in merged and key in stated:
            merged[key] = stated[key]
            assumptions.append(saved_preference_text(key, stated[key]))

    wishes = [] if reset else list(state.get("wishes") or [])
    new_wishes = [wish for wish in extraction.wishes if wish not in wishes]
    update: dict[str, Any] = {"fields": fields_to_state(merged), "assumptions": assumptions,
                              "wishes": [*wishes, *new_wishes]}
    if reset:
        update.update({"request": None, "proposal": None, "tot": None, "review": None, "plan_status": None,
                       "seen_candidates": [], "seen_options": {}, "top_pick": None})
    missing = rules.missing_fields(merged)
    if questions and not missing and not reset and state.get("request"):
        # Keep the current trip; only the new place needs an answer.
        has_bookings = bool(deps.store.list_bookings(deps.trip_id, statuses=["confirmed"]))
        return {**update, "reply_kind": "question", "reply": text.paragraphs(
            *questions, "I kept the rest of your trip as it was.", next_step(state, has_bookings))}
    if questions or missing:
        known = _known_summary(merged)
        if current and not changed and not questions:
            intro = "Sorry, I didn't catch that."
        else:
            intro = f"Got it: {known}." if known else "Happy to help you plan a trip."
        still = [m for m in missing if not questions or m not in extraction.places]
        reply = text.paragraphs(
            intro, *questions,
            ("To plan your trip, I still need:\n" + rules.question_for(still)) if still else None,
            f"I'll use your saved preferences: {text.join_words(assumptions)}." if assumptions else None,
            text.NEXT_DETAILS,
        )
        return {**update, "missing": missing, "request": None, "reply": reply, "reply_kind": "question"}
    try:
        request = rules.to_request(merged)
    except ValidationError as exc:
        return {**update, "missing": [], "request": None, "reply_kind": "question",
                "reply": text.paragraphs(f"I can't plan that yet: {rules.explain_validation_error(exc)}.",
                                         "Next step: tell me what to change.")}
    previous = None if reset else state.get("request")
    if previous == request.model_dump(mode="json") and not changed and not new_wishes and state.get("intent") != "retry":
        has_bookings = bool(deps.store.list_bookings(deps.trip_id, statuses=["confirmed"]))
        return {**update, "reply_kind": "info", "reply": not_understood(state, deps, has_bookings)}
    if conflicts := _booked_conflicts(deps, previous, request):
        names = text.join_words(text.KIND_NAME[k] for k in conflicts)
        return {**update, "reply_kind": "info", "reply": text.paragraphs(
            f"Your {names} {'is' if len(conflicts) == 1 else 'are'} already booked, and this change would affect "
            f"{'it' if len(conflicts) == 1 else 'them'}.",
            f'Next step: cancel it first (say "cancel my {text.KIND_NAME[conflicts[0]]}"), or start a new trip.')}
    if previous != request.model_dump(mode="json"):
        update.update({"seen_candidates": [], "seen_options": {}, "top_pick": None})
    return {**update, "missing": [], "request": request.model_dump(mode="json")}


def _known_summary(fields: dict[str, Any]) -> str:
    parts = []
    origin, destination = fields.get("origin"), fields.get("destination")
    if origin and destination:
        parts.append(f"{text.place(origin)} to {text.place(destination)}")
    elif destination:
        parts.append(f"a trip to {text.place(destination)}")
    elif origin:
        parts.append(f"a trip from {text.place(origin)}")
    departure = fields.get("departure_date")
    if departure:
        if fields.get("return_date") or fields.get("one_way"):
            parts.append(text.dates(departure, fields.get("return_date")))
        else:
            parts.append(f"leaving {text.day(departure)}")
    if fields.get("travelers"):
        parts.append(text.plural(fields["travelers"], "traveler"))
    if fields.get("budget_usd"):
        parts.append(f"budget {text.money(fields['budget_usd'])}")
    if fields.get("priority"):
        parts.append(f"optimizing for {text.PRIORITY_TEXT.get(fields['priority'], fields['priority'])}")
    return ", ".join(parts)


def route_after_intake(state: TravelState) -> str:
    if state.get("request") and not state.get("missing") and state.get("reply_kind") not in {"info", "question"}:
        return "planner"
    return waiting_node(state)


# Plan-and-Execute: show the plan, then wait for the traveler's go-ahead ----------------------------------------------
AGENT_STEP = {"flight": "flight agent", "hotel": "hotel agent", "car": "car rental agent", "transfer": "transfer agent"}


def plan_message(request: TravelRequest, steps: list[str], locked: list[str], assumptions: list[str], *,
                 seen: int = 0, part: str | None = None, kept: list[str] | None = None,
                 notes: list[str] | None = None) -> str:
    summary = describe_request(request)
    if part:
        name = text.KIND_NAME[part]
        intro = f"Here's my plan to find a different {name} for your trip: {summary}."
        keep = f"I'll keep the {text.join_words(text.KIND_NAME[k] for k in kept)} from my top pick" if kept else ""
        skip = f"skip the {text.plural(seen, name)} I already showed you" if seen else ""
        if keep or skip:
            intro += " " + (f"{keep} and {skip}" if keep and skip else keep or skip[0].upper() + skip[1:]) + "."
    elif seen:
        intro = (f"Here's my plan to find more options for your trip: {summary}. I'll search more broadly and skip "
                 f"the {text.plural(seen, 'option')} I already showed you.")
    else:
        intro = f"Here's my plan for your trip: {summary}."
    return text.paragraphs(
        intro,
        "Airports:\n" + text.bullets([f"From: {text.airport(request.origin)}", f"To: {text.airport(request.destination)}"]),
        "What I'll do:\n" + "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1)),
        f"Already booked, so I'll keep it: {text.join_words(text.KIND_NAME[k] for k in locked)}." if locked else None,
        f"I used your saved preferences: {text.join_words(assumptions)}." if assumptions else None,
        *(notes or []),
        ("This is a one-way trip, so I'll search a flight and an airport pickup only. If you also need a hotel and a "
         "rental car, tell me your return date or how many nights you'll stay.") if request.return_date is None else None,
        text.NEXT_PLAN,
    )


def plan_notes(state: TravelState, request: TravelRequest) -> list[str]:
    """What I assumed, and wishes I can't act on, so the plan doesn't ignore them."""
    notes = []
    if "travelers" not in (state.get("fields") or {}):
        notes.append(f"I assumed {text.plural(request.travelers, 'traveler')}. Tell me if more people are going.")
    if "no_rain" in (state.get("wishes") or []):
        notes.append("You asked for no rain. I can't choose the weather, but I'll check it for your dates and tell "
                     "you what to expect.")
    return notes


@traced("planner")
async def planner(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    request = TravelRequest.model_validate(state["request"])
    locked = {item["kind"]: item["option"] for item in deps.store.list_items(deps.trip_id, ["booked"])}
    # "A different hotel": search only the hotel again and keep the rest of the top pick shown for this trip.
    part, kept = None, {}
    asked = state.get("change_part") if state.get("intent") == "more" else None
    top = state.get("top_pick") or {}
    if asked and top.get("request") == state["request"] and (top.get("candidate") or {}).get(asked):
        if asked in locked:
            name = text.KIND_NAME[asked]
            return {"reply_kind": "info", "reply": text.paragraphs(
                f"Your {name} is already booked.", f'Next step: to change it, cancel it first (say "cancel my {name}").')}
        part = asked
        kept = {k: top["candidate"][k] for k in KINDS if k != part and top["candidate"].get(k) and k not in locked}
    more = state.get("intent") == "more" and not part and bool(state.get("seen_candidates"))
    plan = build_task_plan(request, MORE_BEAM_WIDTH if more else deps.settings.beam_width, broader=more or bool(part),
                           only=part, kept=kept)
    directives = [d for d in plan.directives if d.agent not in locked]
    deps.store.update_trip(deps.trip_id, title=trip_title(request), request=state["request"])
    if not directives:
        return {"pending": [], "plan_status": None, "reply_kind": "info", "reply": text.paragraphs(
            "Everything on this trip is already booked.",
            'Next step: to change something, cancel it first (for example, say "cancel my flight").')}
    retry = state.get("intent") == "retry"
    emit("plan", task_plan=plan.model_dump(mode="json"), summary=describe_request(request), locked=sorted(locked))
    update: dict[str, Any] = {
        "task_plan": plan.model_dump(mode="json"),
        "pending": [d.model_dump() for d in directives],
        "tried": sorted(d.signature() for d in directives),
        "round": 0, "planner_notes": [], "locked": locked, "plan_status": "approved" if retry else "pending",
        "exclude_seen": more, "part": part, "kept": kept,
        "options": replace({option["option_id"]: option for option in kept.values()}), "agent_status": replace(), "agent_runs": [RESET], "context": replace(),
        "tot": None, "review": None, "proposal": None, "decision": None, "approvals": [], "execution": [],
    }
    if not retry:
        steps = [step for step in plan.steps if not any(step.lower().startswith(AGENT_STEP[k]) for k in locked)]
        seen = (len((state.get("seen_options") or {}).get(part, [])) if part else
                len(state.get("seen_candidates") or []) if more else 0)
        update.update({"reply": plan_message(request, steps, sorted(locked), state.get("assumptions") or [], seen=seen,
                                             part=part, kept=list(kept), notes=plan_notes(state, request)),
                       "reply_kind": "plan"})
    deps.store.update_trip(deps.trip_id, status="planning" if retry else "plan_ready")
    return update


def route_after_planner(state: TravelState) -> list[Send] | str:
    if state.get("reply_kind") == "info":
        return "__end__"
    return dispatch(state) if state.get("plan_status") == "approved" else "await_plan_approval"


async def await_plan_approval(state: TravelState) -> dict[str, Any]:
    """Pause until the traveler says "go ahead" (or presses Start search). No side effects before `interrupt()`."""
    answer = interrupt({"type": "plan_approval"})
    decision = answer if isinstance(answer, dict) else {"decision": str(answer)}
    said = (decision.get("text") or "").strip() if decision.get("decision") == "message" else ""
    if decision.get("decision") == "approve" or (said and _YES.match(said)):
        return {"plan_status": "approved", "decision": None}
    if decision.get("decision") == "reject" or (said and _NO.match(said)):
        return {"plan_status": "declined", "decision": None, "reply_kind": "info",
                "reply": text.paragraphs("OK, I won't search yet.", text.NEXT_CHANGE)}
    return {"decision": None, "intent": None, "user_message": said}


def route_after_plan_approval(state: TravelState) -> list[Send] | str:
    status = state.get("plan_status")
    if status == "approved":
        return dispatch(state)
    return "route_intent" if status == "pending" else "__end__"


def dispatch(state: TravelState) -> list[Send] | str:
    """Fan out this round's directives: one `Send` per agent, so the agents search in parallel."""
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for directive in state.get("pending") or []:
        by_agent.setdefault(directive["agent"], []).append(directive)
    round_ = state.get("round", 0)
    sends = [Send(f"{agent}_agent", AgentTask(agent=agent, directives=directives, request=state["request"], round=round_))
             for agent, directives in by_agent.items()]
    if round_ == 0 and not state.get("context"):
        sends.append(Send("context_agent", AgentTask(agent="context", directives=[], request=state["request"], round=0)))
    return sends or "evaluate"


async def _search_agent(task: AgentTask, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    agent = task["agent"]
    directives = [SearchDirective.model_validate(d) for d in task["directives"]]
    request = TravelRequest.model_validate(task["request"])
    strategies = [d.strategy for d in directives]
    emit("agent", agent=agent, status="running", round=task["round"], strategies=strategies)
    try:
        result = await AGENT_RUNNERS[agent](deps.agent_deps(), request, directives)
    except Exception as exc:  # noqa: BLE001 - one crashing agent must not stop the others
        result = AgentResult(agent=agent, status=AgentStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:200])
    if result.options:
        deps.store.record_options(deps.trip_id, result.options)
    summary = {"agent": agent, "round": task["round"], "status": result.status.value, "count": len(result.options),
               "strategies": strategies, "notes": result.notes, "error": result.error, "calls": result.calls,
               "duration_ms": result.duration_ms}
    emit("agent", **summary)
    return {"options": {o["option_id"]: o for o in result.options}, "agent_status": {agent: summary},
            "agent_runs": [summary]}


@traced("flight_agent")
async def flight_agent(task: AgentTask, runtime: Runtime[Deps]) -> dict[str, Any]:
    return await _search_agent(task, runtime)


@traced("hotel_agent")
async def hotel_agent(task: AgentTask, runtime: Runtime[Deps]) -> dict[str, Any]:
    return await _search_agent(task, runtime)


@traced("car_agent")
async def car_agent(task: AgentTask, runtime: Runtime[Deps]) -> dict[str, Any]:
    return await _search_agent(task, runtime)


@traced("transfer_agent")
async def transfer_agent(task: AgentTask, runtime: Runtime[Deps]) -> dict[str, Any]:
    return await _search_agent(task, runtime)


@traced("context_agent")
async def context_agent(task: AgentTask, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    request = TravelRequest.model_validate(task["request"])
    emit("agent", agent="context", status="running", round=0, strategies=["weather", "airports"])
    result, context = await run_context_agent(deps.agent_deps(), request)
    summary = {"agent": "context", "round": 0, "status": result.status.value, "count": len(context),
               "strategies": ["weather", "airports"], "notes": result.notes, "error": result.error,
               "calls": result.calls, "duration_ms": result.duration_ms}
    emit("agent", **summary)
    return {"context": context, "agent_status": {"context": summary}, "agent_runs": [summary]}


# Tree-of-Thought evaluation and review -----------------------------------------------------------------------------
@traced("evaluate")
async def evaluate(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    request = TravelRequest.model_validate(state["request"])
    options = list((state.get("options") or {}).values())
    beam = MORE_BEAM_WIDTH if state.get("exclude_seen") else deps.settings.beam_width
    config = ToTConfig(beam_width=beam, max_stages=deps.settings.max_tot_stages)
    by_kind = {kind: [o for o in options if o["kind"] == kind] for kind in KINDS}
    part = state.get("part")
    if part:
        shown = set((state.get("seen_options") or {}).get(part, []))
        fresh = [o for o in by_kind[part] if o["option_id"] not in shown]
        if by_kind[part] and not fresh:
            no_more = Issue(code="no_more_options", agent=part,
                            detail=f"I've already shown you every {text.KIND_NAME[part]} I found.")
            emit("tot", round=state.get("round", 0), stages=[], candidates=0, pruned=0, issues=[no_more.code])
            return {"tot": ToTResult(issues=[no_more]).to_state()}
        by_kind[part] = fresh
    result = run_tot(request, by_kind["flight"], by_kind["hotel"], by_kind["car"], by_kind["transfer"],
                     locked=state.get("locked") or {}, config=config)
    if state.get("exclude_seen"):
        seen = set(state.get("seen_candidates") or [])
        fresh = [c for c in result.candidates if c.candidate_id not in seen]
        if result.candidates and not fresh:
            result.issues.append(Issue(code="no_more_options", detail="I've already shown you every trip that fits."))
        result.candidates = fresh
    learned = {k: v["value"] for k, v in deps.store.get_preferences().items() if k.startswith("preferred_")}
    weather_level = ((state.get("context") or {}).get("weather") or {}).get("level")
    apply_preferences(result.candidates, learned, weather_level, config.tie_margin)
    if deps.llm is not None and result.candidates:
        await deps.llm.critique(request, result.candidates, config.tie_margin, trip_type=stated_trip_type(state))
    emit("tot", round=state.get("round", 0), stages=result.stages, candidates=len(result.candidates),
         pruned=result.pruned_count, issues=[i.code for i in result.issues])
    return {"tot": result.to_state()}


@traced("planner_review")
async def planner_review(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    tot = state["tot"] or {}
    inp = ReviewInput(
        request=TravelRequest.model_validate(state["request"]),
        candidates=[ItineraryCandidate.model_validate(c) for c in tot.get("candidates", [])],
        issues=[Issue.model_validate(i) for i in tot.get("issues", [])],
        agent_status=dict(state.get("agent_status") or {}), tried=set(state.get("tried") or []),
        round=state.get("round", 0), max_rounds=deps.settings.max_refinement_rounds,
        near_misses=[ItineraryCandidate.model_validate(c) for c in tot.get("near_misses", [])],
        locked=sorted({*(state.get("locked") or {}), *(state.get("kept") or {})}),
        history=list(state.get("planner_notes") or []),
    )
    decision = review(inp)
    emit("review", action=decision.action, notes=decision.notes, round=inp.round)
    update: dict[str, Any] = {"review": decision.to_state(),
                              "planner_notes": [*(state.get("planner_notes") or []), *decision.notes]}
    if decision.action == "refine":
        update.update({"pending": [d.model_dump() for d in decision.directives], "round": inp.round + 1,
                       "tried": sorted(inp.tried | {d.signature() for d in decision.directives})})
    return update


def route_after_review(state: TravelState) -> list[Send] | str:
    action = (state.get("review") or {}).get("action")
    if action == "refine":
        return dispatch(state)
    return "prepare_proposal" if action == "propose" else "ask_user"


@traced("ask_user")
async def ask_user(state: TravelState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    deps.store.update_trip(deps.trip_id, status="needs_input")
    return {"reply": (state.get("review") or {}).get("message") or "I need more information to continue.",
            "reply_kind": "needs_input"}
