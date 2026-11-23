"""Rule-based extraction of trip details from chat text. It never guesses required details: anything missing
becomes a follow-up question. Place names are resolved to airport codes by the caller (trip-context MCP)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError

from travel_agent.models.request import PRIORITIES, TravelRequest
from travel_agent.providers.airports import get_airport_index

MONTHS = {name: index for index, names in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
     ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
     ("dec", "december")], start=1) for name in names}
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "zero": 0}
_NOT_CODES = {"USD", "EUR", "GBP", "AND", "THE", "FOR", "SUV", "MAX", "ONE", "WAY", "NON", "AIR", "CAR", "AM", "PM"}
_PLACE_STOP = (r"on|in|for|from|to|with|under|below|next|this|and|at|by|around|between|leaving|departing|returning|"
               r"round|one|via|budget|max|we|i|my|our|the|a|visit|visiting|go|going|fly|flying|travel|traveling|"
               r"travelling|see|plan|trip|change|try|book|find|show|get|keep|make|add|remove|switch|swap|replace|"
               r"cancel|pick|choose|use|look|check|stay|land|landing|arrive|arriving|search|compare|know|be|do|have|"
               r"take|rent|hotels?|cars?|flights?|options?|" + _MONTH[3:-1])
_PLACE = rf"(?P<place>(?:(?!(?:{_PLACE_STOP})\b)[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.'’-]*)(?:\s+(?!(?:{_PLACE_STOP})\b)[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.'’-]*){{0,3}})"

REQUIRED = ("origin", "destination", "departure_date")
MISSING_TEXT = {
    "route": "Where you're flying from and to",
    "origin": "Where you're flying from",
    "destination": "Where you're going",
    "departure_date": "Your departure date (for example, Dec 10)",
    "return_date": "Your return date (I also use it for the hotel and rental car)",
    "priority": "What matters most: lowest cost, shortest travel time, comfort, or convenience",
}
FULL_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december")
FIELD_NAMES = {"budget_usd": "budget", "travelers": "number of travelers", "origin": "origin airport",
               "destination": "destination airport"}


@dataclass
class Extraction:
    fields: dict[str, Any] = field(default_factory=dict)
    places: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    wishes: list[str] = field(default_factory=list)  # things we can't control but should acknowledge ("no rain")


def _year_for(month: int, day: int, today: date, explicit: int | None) -> date | None:
    try:
        if explicit:
            return date(explicit, month, day)
        candidate = date(today.year, month, day)
        return candidate if candidate >= today else date(today.year + 1, month, day)
    except ValueError:
        return None


def fix_month_typos(text: str) -> str:
    """Correct misspelled month names ("decemember" becomes "december") so dates are read correctly."""
    def fix(match: re.Match[str]) -> str:
        word = match.group(0).lower()
        candidates = [m for m in FULL_MONTHS if m[:3] == word[:3] and not word.startswith(m)]
        if word in MONTHS or not candidates:
            return match.group(0)
        close = difflib.get_close_matches(word, candidates, n=1, cutoff=0.8)
        return close[0] if close else match.group(0)

    return re.sub(r"\b[A-Za-z]{5,12}\b", fix, text)


def _dates(text: str, today: date) -> list[tuple[int, date]]:
    found: list[tuple[int, date]] = []
    for m in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text):
        try:
            found.append((m.start(), date(int(m[1]), int(m[2]), int(m[3]))))
        except ValueError:
            continue
    for m in re.finditer(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d/])", text):
        year = int(m[3]) if m[3] else None
        year = year + 2000 if year is not None and year < 100 else year
        if (parsed := _year_for(int(m[1]), int(m[2]), today, year)) is not None:
            found.append((m.start(), parsed))
    range_tail = r"(?:\s*(?:-|–|—|to|through|thru|until|till)\s*(?:(?P<m2>" + _MONTH + r")\.?\s+)?(?P<d2>\d{1,2})(?:st|nd|rd|th)?\b(?!\s*(?:people|persons|travel|adults|guests|nights?|days?|stops?|:|am|pm)))?"
    month_first = re.compile(r"\b(?P<m1>" + _MONTH + r")\.?\s+(?P<d1>\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*(?P<y1>20\d{2}))?"
                             + range_tail + r"(?:,?\s*(?P<y2>20\d{2}))?", re.I)
    day_first = re.compile(r"\b(?P<d1>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m1>" + _MONTH + r")\b\.?(?:,?\s*(?P<y1>20\d{2}))?", re.I)
    for m in month_first.finditer(text):
        month = MONTHS[m["m1"].lower()[:3] if m["m1"].lower()[:4] != "sept" else "sep"]
        year = int(m["y1"] or m["y2"]) if (m["y1"] or m["y2"]) else None
        start = _year_for(month, int(m["d1"]), today, year)
        if start is None:
            continue
        found.append((m.start(), start))
        if m["d2"]:
            end_month = MONTHS[m["m2"].lower()[:3]] if m["m2"] else month
            end = _year_for(end_month, int(m["d2"]), today, year)
            if end is not None and end < start:
                end = _year_for(end_month, int(m["d2"]), start + timedelta(days=1), None)
            if end is not None:
                found.append((m.start("d2"), end))
    for m in day_first.finditer(text):
        if any(abs(pos - m.start()) < 3 for pos, _ in found):
            continue
        month = MONTHS[m["m1"].lower()[:3]]
        if (parsed := _year_for(month, int(m["d1"]), today, int(m["y1"]) if m["y1"] else None)) is not None:
            found.append((m.start(), parsed))
    lowered = text.lower()
    for word, delta in (("today", 0), ("tomorrow", 1)):
        for m in re.finditer(rf"\b{word}\b", lowered):
            found.append((m.start(), today + timedelta(days=delta)))
    found.sort(key=lambda item: item[0])
    unique: list[tuple[int, date]] = []
    for pos, value in found:
        if not unique or unique[-1][1] != value or pos - unique[-1][0] > 25:
            unique.append((pos, value))
    return unique


def _number(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _clean_place(raw: str) -> str:
    return re.sub(r"[\s.,]+$", "", raw).strip()


def _route(text: str, extraction: Extraction) -> None:
    codes = re.search(r"\b([A-Za-z]{3})\s*(?:-|–|->|\bto\b)\s*([A-Za-z]{3})\b", text)
    if codes and (codes[1].isupper() or codes[2].isupper()) and all(
            c.upper() not in _NOT_CODES and get_airport_index().get(c) for c in (codes[1], codes[2])):
        extraction.fields["origin"], extraction.fields["destination"] = codes[1].upper(), codes[2].upper()
        return
    origin = re.search(r"\b(?:from|out of|leaving|departing)\s+" + _PLACE, text, re.I)
    destination = re.search(r"\b(?:to|visit(?:ing)?|in(?:to)?)\s+" + _PLACE, text, re.I)
    if origin:
        extraction.places["origin"] = _clean_place(origin["place"])
    if destination:
        place = _clean_place(destination["place"])
        if not origin or destination.start() != origin.start():
            extraction.places["destination"] = place
    for role in ("origin", "destination"):
        place = extraction.places.get(role)
        if place and re.fullmatch(r"[A-Za-z]{3}", place) and place.upper() not in _NOT_CODES:
            extraction.fields[role] = place.upper()
            extraction.places.pop(role)
    if "origin" not in extraction.fields and "origin" not in extraction.places and \
            "destination" not in extraction.fields and "destination" not in extraction.places:
        bare = [c for c in re.findall(r"\b([A-Z]{3})\b", text) if c not in _NOT_CODES]
        if len(bare) == 2:
            extraction.fields["origin"], extraction.fields["destination"] = bare


def extract(text: str, today: date | None = None) -> Extraction:
    today = today or date.today()
    extraction = Extraction()
    fields = extraction.fields
    lowered = text.lower()
    _route(text, extraction)

    fixed = fix_month_typos(text)
    found = _dates(fixed, today)
    dates = [value for _, value in found]
    if len(dates) == 1 and re.search(r"\b(?:return(?:ing)?|back|until|till)\b(?:\s+\w+)?\s*(?:on\s+)?$",
                                     fixed[:found[0][0]], re.I):
        fields["return_date"] = dates[0]  # "return Dec 20", "coming back on Dec 20"
    elif dates:
        fields["departure_date"] = dates[0]
    if len(dates) >= 2:
        fields["return_date"] = dates[1]
    if re.search(r"\bone[- ]way\b", lowered):
        fields["one_way"] = True
        fields["return_date"] = None
    elif re.search(r"\bround[- ]trip\b", lowered):
        fields["one_way"] = False
    if (m := re.search(r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")[- ]nights?\b", lowered)):
        fields["_nights"] = _number(m[1])
    elif (m := re.search(r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")[- ]days?\b", lowered)):
        fields["_days"] = _number(m[1])

    if (m := re.search(r"\bfamily of (\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\b", lowered)):
        fields["travelers"] = _number(m[1])
        fields["trip_type"] = "family"
    elif (m := re.search(r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\s+(?:people|persons|travell?ers|adults|passengers|guests|of us|pax)\b", lowered)):
        fields["travelers"] = _number(m[1])
    elif (m := re.search(r"\bfor (\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\b(?!\s*(?:nights?|days?|stops?|hours?|k\b|usd|dollars|%))", lowered)):
        fields["travelers"] = _number(m[1])
    elif re.search(r"\b(solo|just me|by myself|alone)\b", lowered):
        fields["travelers"] = 1
    elif re.search(r"\b(a couple|two of us|me and my (?:wife|husband|partner|friend))\b", lowered):
        fields["travelers"] = 2

    budget = re.search(r"(?:under|below|less than|max(?:imum)?|budget(?:\s+(?:of|is))?:?|up to|within|no more than)\s*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", lowered) \
        or re.search(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", lowered) \
        or re.search(r"\b(\d[\d,]*(?:\.\d+)?)\s*(k\b)?\s*(?:usd|dollars)\b", lowered)
    if budget:
        amount = float(budget[1].replace(",", "")) * (1000 if budget[2] else 1)
        if amount >= 50:
            fields["budget_usd"] = amount

    if (m := re.search(r"\b(?:max(?:imum)?|at most|up to|no more than)\s+(\d|one|two|zero)\s+stops?\b", lowered)) or \
            (m := re.search(r"\b(\d|one|two)\s+stops?\s+max\b", lowered)):
        fields["max_stops"] = _number(m[1])
    elif re.search(r"\b(non[- ]?stop|direct)\b", lowered):
        fields["flight_preference"] = "nonstop"
        if re.search(r"\b(only|must|have to|need)\b[^.]*\b(non[- ]?stop|direct)\b|\b(non[- ]?stop|direct)\b[^.]*\bonly\b", lowered):
            fields["max_stops"] = 0
    elif re.search(r"\b(any (?:number of )?stops|stops are (?:fine|ok))\b", lowered):
        fields["max_stops"] = None

    if re.search(r"\b(no|without(?: a)?|skip(?: the)?|don'?t need(?: a)?|do not need(?: a)?|not renting(?: a)?)\s+(?:rental\s+|rent-a-)?car\b", lowered):
        fields["require_car"] = False
    elif re.search(r"\b(need|want|add|with|rent|include|get)(?: a| the)?\s+(?:rental\s+)?car\b", lowered):
        fields["require_car"] = True
    if (m := re.search(r"\b(suv|minivan|compact|midsize|mid-size|full-?size|economy car|premium car)\b", lowered)):
        fields["car_class"] = {"mid-size": "midsize", "full-size": "fullsize", "fullsize": "fullsize",
                               "economy car": "economy", "premium car": "premium"}.get(m[1], m[1])
        fields["require_car"] = True

    if re.search(r"\b(no|without(?: an?)?|skip(?: the)?|don'?t need(?: an?)?)\s+(?:airport\s+)?(?:transfers?|pickups?|shuttles?)\b", lowered):
        fields["need_transfer"] = False
    elif re.search(r"\b(need|want|add|book)(?: an?| the)?\s+(?:airport\s+)?(?:transfer|pickup|shuttle)\b", lowered):
        fields["need_transfer"] = True
    if re.search(r"\b(no hotel|don'?t need a hotel|do not need a hotel|staying with (?:friends|family))\b", lowered):
        fields["need_hotel"] = False
    elif re.search(r"\b(need|want|add)(?: a)? hotel\b", lowered):
        fields["need_hotel"] = True
    if (m := re.search(r"\b(downtown|midtown|waterfront|old[- ]town|near the airport|airport hotel|beach(?:front)?)\b", lowered)):
        fields["hotel_area"] = {"near the airport": "airport", "airport hotel": "airport", "old town": "old-town",
                                "beach": "waterfront", "beachfront": "waterfront"}.get(m[1], m[1])

    if re.search(r"\b(?:any (?:landing|arrival) time|(?:land|arrive)(?:ing)? (?:at )?any ?time)\b", lowered):
        fields["arrival_window"] = None
    elif (m := re.search(r"\b(?:land(?:s|ing)?|arriv(?:e|es|al|ing)|get(?:ting)? in)\b(?:\W+\w+){0,4}?\W+"
                         r"(morning|afternoon|evening|night(?:time)?s?|late)\b", lowered)) or \
            (m := re.search(r"\b(morning|afternoon|evening|night)[- ](?:landings?|arrivals?)\b", lowered)):
        fields["arrival_window"] = "night" if m[1].startswith("night") or m[1] == "late" else m[1]
    if re.search(r"\b(?:no|not|without|avoid(?:ing)?)\s+(?:be\s+|any\s+|the\s+)?rain(?:ing|y|s)?\b|"
                 r"\b(?:dry|sunny|good|nice|clear)\s+weather\b|\b(?:don'?t|do not)\s+want\s+(?:any\s+)?rain\b", lowered):
        extraction.wishes.append("no_rain")

    if re.search(r"\bbusiness class\b", lowered):
        fields["cabin_class"] = "business"
    elif re.search(r"\bfirst class\b", lowered):
        fields["cabin_class"] = "first"
    elif re.search(r"\bpremium economy\b", lowered):
        fields["cabin_class"] = "premium_economy"

    priority_patterns = {
        "cost": r"\b(cheapest|cheaper|cheap|less expensive|lower (?:price|cost)|lowest (?:price|cost|fare)|save money|"
                r"budget[- ]friendly|optimi[sz]e (?:for )?cost|cost)\b",
        "time": r"\b(fastest|quickest|shortest (?:travel|trip|flight)|least travel time|save time|optimi[sz]e (?:for )?time|travel time)\b",
        "comfort": r"\b(comfort(?:able)?|luxur(?:y|ious))\b",
        "convenience": r"\b(convenien(?:t|ce)|easiest|hassle[- ]free)\b",
        "balanced": r"\b(balanced?|best overall|best value|a balance)\b",
    }
    for priority, pattern in priority_patterns.items():
        if re.search(pattern, lowered):
            fields["priority"] = priority
            break
    if re.fullmatch(r"\s*(time|travel time)\s*[.!]?\s*", lowered):
        fields["priority"] = "time"
    if "priority" not in fields and re.search(
            r"\b(assume|you (?:decide|choose|pick)|up to you|(?:doesn'?t|does not) matter|no preference|"
            r"anything(?: is)? fine|whatever|defaults?)\b", lowered):
        fields["priority"] = "balanced"

    trip_patterns = {
        "family": r"\b(family|kids|children|child|toddler)\b",
        "business": r"\b(business trip|work trip|for work|conference|client meeting|business travel|business)\b(?!\s+class)",
        "adventure": r"\b(adventure|hiking|hike|outdoors?|ski(?:ing)?|camping|trek(?:king)?)\b",
        "leisure": r"\b(vacation|holiday|leisure|getaway|relax(?:ing)?)\b",
    }
    if "trip_type" not in fields:
        for trip_type, pattern in trip_patterns.items():
            if re.search(pattern, lowered):
                fields["trip_type"] = trip_type
                break
    return extraction


def merge(current: dict[str, Any], new: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    merged = dict(current)
    changed: list[str] = []
    for key, value in new.items():
        if key.startswith("_"):
            continue
        if merged.get(key) != value or key not in merged:
            if merged.get(key) != value:
                changed.append(key)
            merged[key] = value
    departure = merged.get("departure_date")
    if new.get("_nights") or new.get("_days"):
        merged["one_way"] = False  # a stay length means a round trip with a hotel
    if departure and not merged.get("return_date") and not merged.get("one_way"):
        if new.get("_nights"):
            merged["return_date"] = departure + timedelta(days=new["_nights"])
            changed.append("return_date")
        elif new.get("_days"):
            merged["return_date"] = departure + timedelta(days=max(0, new["_days"] - 1))
            changed.append("return_date")
    if merged.get("return_date"):
        merged["one_way"] = False
    return merged, sorted(set(changed))


def missing_fields(fields: dict[str, Any], saved_priority: str | None = None) -> list[str]:
    missing = [name for name in REQUIRED if not fields.get(name)]
    if not fields.get("return_date") and not fields.get("one_way"):
        missing.append("return_date")
    if not fields.get("priority") and not saved_priority:
        missing.append("priority")
    return missing


def question_for(missing: list[str]) -> str:
    """The details still needed, as a simple list."""
    keys = list(missing)
    if "origin" in keys and "destination" in keys:
        keys[keys.index("origin")] = "route"
        keys.remove("destination")
    return "\n".join(f"- {MISSING_TEXT[key]}" for key in keys if key in MISSING_TEXT)


def to_request(fields: dict[str, Any], saved_priority: str | None = None) -> TravelRequest:
    allowed = set(TravelRequest.model_fields)
    data = {key: value for key, value in fields.items() if key in allowed and value is not None}
    if "priority" not in data and saved_priority in PRIORITIES:
        data["priority"] = saved_priority
    return TravelRequest.model_validate(data)


def explain_validation_error(error: ValidationError) -> str:
    messages = []
    for item in error.errors():
        message = str(item.get("msg", "")).removeprefix("Value error, ")
        if item.get("type") == "value_error":
            messages.append(message)  # our own validators already name the field
            continue
        field = ".".join(str(part) for part in item.get("loc", ()))
        messages.append(f"{FIELD_NAMES.get(field, field.replace('_', ' '))}: {message.lower()}")
    return "; ".join(messages)
