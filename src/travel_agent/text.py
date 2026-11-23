"""Plain-language text for chat replies: short paragraphs and simple "- " lists. No icons, internal ids, or scores.

The web page shows blank-line-separated paragraphs and "- " lines as lists (always as plain text, never HTML).
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

from travel_agent.models.options import CarOption, FlightOption, HotelOption, TransferOption
from travel_agent.providers.airports import get_airport_index

PRIORITY_TEXT = {"cost": "lowest cost", "time": "shortest travel time", "comfort": "comfort",
                 "convenience": "convenience", "balanced": "a balance of price and convenience"}
KIND_LABEL = {"flight": "Flight", "hotel": "Hotel", "car": "Car", "transfer": "Airport pickup"}
KIND_NAME = {"flight": "flight", "hotel": "hotel", "car": "rental car", "transfer": "airport pickup"}
STRATEGY_TEXT = {"best": "best overall", "nonstop": "nonstop", "cheapest": "cheapest", "preferred": "best rated",
                 "budget": "lowest price", "family": "family friendly", "late_checkin": "late check-in",
                 "standard": "standard", "larger": "larger cars", "late_pickup": "late pickup", "private": "private car"}
NEXT_DETAILS = "Next step: once I have these details, I'll show you my plan before I search."
NEXT_PLAN = ('Next step: say "go ahead" (or press Start search), and the agents will search at the same time. '
             "To change anything, just tell me.")
NEXT_OPTIONS = ("Next step: choose an option on the right to book it in the sandbox (no real payment), "
                "or tell me what to change.")
NEXT_CANCEL = "Next step: confirm on the right. Nothing is cancelled until you do."
NEXT_BOOKED = ('Next steps: ask me anything about your trip, cancel a booking (for example, "cancel my car"), '
               "or start a new trip.")
NEXT_CHANGE = "Next step: tell me what you'd like to change, and I'll show you an updated plan."
LANDING_TEXT = {"morning": "in the morning (5 AM to noon)", "afternoon": "in the afternoon (noon to 5 PM)",
                "evening": "in the evening (5 to 9 PM)", "night": "at night (9 PM to 5 AM)"}
WEATHER_KIND = {"forecast": "forecast", "typical": "based on past years, not a forecast",
                "sample": "sample data, not a real forecast"}


def money(amount: float) -> str:
    """Dollars, with cents only when there are any, so a trip's parts always add up to the total shown."""
    cents = round(amount * 100)
    return f"${cents // 100:,}" if cents % 100 == 0 else f"${cents / 100:,.2f}"


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def join_words(words: Iterable[str]) -> str:
    items = list(words)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def day(value: date | datetime) -> str:
    return f"{value:%a}, {value:%b} {value.day}"


def clock(moment: datetime) -> str:
    return f"{moment.hour % 12 or 12}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def hour_text(hour: int) -> str:
    hour %= 24
    if hour == 0:
        return "midnight"
    if hour == 12:
        return "noon"
    return f"{hour % 12} {'AM' if hour < 12 else 'PM'}"


def city(code: str) -> str:
    name = get_airport_index().city_of(code)
    return name.split(" (")[0].split(",")[0].strip() or code


def place(code: str) -> str:
    name = city(code)
    return code if name.upper() == code.upper() else f"{name} ({code})"


def airport(code: str) -> str:
    """Full airport name, or the airports of a metro code ("New York: JFK, LGA, or EWR")."""
    info = get_airport_index().get(code)
    if info is None:
        return code
    if info.kind == "metro":
        return f"{info.city}, any airport ({join_words(info.members).replace(' and ', ' or ')})"
    return f"{info.name} ({code})"


def dates(departure: date, return_date: date | None) -> str:
    return f"{day(departure)}, one-way" if return_date is None else f"{day(departure)} to {day(return_date)}"


def bullets(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def paragraphs(*blocks: str | None) -> str:
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


def strategies(names: Iterable[str]) -> str:
    return ", ".join(STRATEGY_TEXT.get(name, name) for name in names)


def stops(flight: FlightOption) -> str:
    if flight.max_stops == 0:
        return "nonstop"
    airports = [layover.split(" ")[0] for layover in flight.outbound.layovers]
    return plural(flight.max_stops, "stop") + (f" in {join_words(airports)}" if airports else "")


def flight_text(flight: FlightOption) -> str:
    out = flight.outbound
    arrives = clock(out.arrival_time) if out.arrival_time.date() == out.departure_time.date() else \
        f"{day(out.arrival_time)} at {clock(out.arrival_time)}"
    text = (f"{flight.airline} {out.flight_number}, {stops(flight)}, leaves {day(out.departure_time)} at "
            f"{clock(out.departure_time)} and arrives {arrives}")
    if flight.inbound is not None:
        text += f"; returns {day(flight.inbound.departure_time)} at {clock(flight.inbound.departure_time)}"
    return text


def hotel_text(hotel: HotelOption) -> str:
    area = "near the airport" if hotel.area == "airport" else f"{hotel.area.replace('-', ' ')} area"
    parts = [hotel.name, area]
    if hotel.rating:
        parts.append(f"rated {hotel.rating:.1f}")
    parts.append(plural(hotel.nights, "night"))
    if hotel.free_cancellation_until:
        parts.append(f"free cancellation until {hotel.free_cancellation_until:%b} {hotel.free_cancellation_until.day}")
    return ", ".join(parts)


def car_text(car: CarOption) -> str:
    desk = "desk open 24 hours" if car.open_24h else \
        f"desk open {hour_text(car.desk_open_hour)} to {hour_text(car.desk_close_hour)}"
    pickup = "" if car.pickup_at_airport else f", pick up at {car.pickup_location}"
    return f"{car.vendor} {car.vehicle_class}, {desk}{pickup}"


def transfer_text(transfer: TransferOption, flight: FlightOption | None, hotel: HotelOption | None) -> str:
    start = f"from {flight.outbound.destination}" if flight is not None else "from the airport"
    end = f"to {hotel.name}" if hotel is not None else "to the city center"
    return f"{transfer.vendor} {transfer.service} {start} {end} when you land, about {transfer.duration_minutes} minutes"


def rain_sentence(weather: dict | None) -> str:
    """For a traveler who asked for no rain: the agent can't choose the weather, so say what the data shows."""
    report = (weather or {}).get("report") or {}
    days = report.get("days") or []
    if not days:
        return "You asked for no rain. I can't choose the weather, and I don't have weather data for your dates."
    wet = sum(1 for d in days if (d.get("precipitation_mm") or 0) >= 1 or (d.get("precipitation_probability_pct") or 0) >= 50)
    share = f"{wet} of {len(days)} days"
    if report.get("kind") == "typical":
        detail = (f"on the same dates last year, it rained on {share}" if wet else
                  "it didn't rain on the same dates last year") + " (it's too early for a forecast)"
    elif report.get("kind") == "forecast":
        detail = f"the forecast shows rain on {share}" if wet else "the forecast shows no rain"
    else:
        detail = f"the sample weather shows rain on {share}" if wet else "the sample weather shows no rain"
    return f"You asked for no rain. I can't choose the weather, but {detail}."


def weather_sentence(weather: dict | None, where: str) -> str | None:
    """One sentence about weather risk at the destination, from the weather_risk tool result."""
    if not weather:
        return None
    kind = WEATHER_KIND.get(weather["report"]["kind"], weather["report"]["kind"])
    if weather["level"] == "low":
        return f"Weather in {where}: no problems expected ({kind})."
    reason = f" {weather['reasons'][0]}." if weather.get("reasons") else ""
    return f"Weather in {where}: {weather['level']} risk ({kind}).{reason}"
