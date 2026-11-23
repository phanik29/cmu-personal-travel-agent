"""Deterministic mock travel inventory and sandbox booking, with scenarios for testing."""
from __future__ import annotations

import asyncio
import hashlib
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from travel_agent.models.options import CarOption, FlightLeg, FlightOption, HotelOption, SourceMetadata, TransferOption
from travel_agent.models.planning import BookingRecord
from travel_agent.providers.base import (
    BookingLookup,
    BookingResult,
    CancellationResult,
    CarQuery,
    CarSearchResult,
    FlightQuery,
    FlightSearchResult,
    HotelQuery,
    HotelSearchResult,
    OfferRefresh,
    TransferQuery,
    TransferSearchResult,
)
from travel_agent.providers.airports import METRO_CODES, utc_offset
from travel_agent.sandbox.ledger import SandboxLedger

PROVIDER = "mock"
LABEL = "Mock"
TIMEOUT_SLEEP_S = 120

# carrier, airline, outbound departure (h, m), minutes, stops, layovers, base price per person per leg,
# refundable, seats, return departure (h, m)
FLIGHT_TEMPLATES: list[tuple[Any, ...]] = [
    ("AP", "Apex Air", (8, 30), 330, 0, [], 310, False, 9, (17, 30)),
    ("AP", "Apex Air", (11, 15), 320, 0, [], 365, True, 5, (19, 10)),
    ("VJ", "ValueJet", (6, 15), 455, 1, ["DEN (1h05m)"], 215, False, 12, (14, 20)),
    ("BW", "BlueWing", (13, 40), 345, 0, [], 289, False, 4, (10, 0)),
    ("SK", "SkyLink", (7, 5), 520, 1, ["ORD (1h50m)"], 198, False, 20, (7, 40)),
    ("NW", "NightWing", (16, 20), 335, 0, [], 245, False, 7, (12, 15)),
    ("CW", "CoastWay", (22, 10), 330, 0, [], 179, False, 15, (15, 30)),
    ("PR", "Premier", (9, 45), 325, 0, [], 640, True, 3, (18, 45)),
]
BUDGET_FLIGHT_TEMPLATES: list[tuple[Any, ...]] = [
    ("ZS", "ZipSaver", (9, 5), 470, 1, ["PHX (1h20m)"], 139, False, 14, (11, 45)),
    ("LF", "LowFare Air", (12, 20), 505, 1, ["LAS (1h35m)"], 129, False, 18, (13, 5)),
    ("EC", "EconoJet", (7, 40), 335, 0, [], 165, False, 6, (16, 25)),
]
# name, area, rating, base per room-night, late check-in, early check-in, check-in hour,
# free-cancellation days before arrival, amenities, guests per room
HOTEL_TEMPLATES: list[tuple[Any, ...]] = [
    ("Metro Central Hotel", "downtown", 4.5, 210, True, False, 15, 2, ["wifi", "gym", "24h-desk"], 2),
    ("Harbor View Suites", "waterfront", 4.3, 185, True, True, 14, 3, ["wifi", "breakfast", "pool"], 3),
    ("Civic Boutique Stay", "downtown", 4.7, 289, True, True, 15, 1, ["wifi", "spa", "24h-desk"], 2),
    ("Terminal Express Inn", "airport", 3.6, 118, True, False, 15, None, ["wifi", "shuttle", "24h-desk"], 2),
    ("Midtown Garden Hotel", "midtown", 4.1, 160, False, False, 16, 2, ["wifi", "breakfast"], 2),
    ("Family Fun Resort", "midtown", 4.2, 199, False, True, 15, 5, ["wifi", "pool", "family-rooms", "breakfast"], 4),
    ("Old Town Lodge", "old-town", 4.0, 142, True, False, 15, 7, ["wifi", "breakfast"], 2),
]
BUDGET_HOTEL_TEMPLATES: list[tuple[Any, ...]] = [
    ("Budget Stay Downtown", "downtown", 3.4, 99, False, False, 16, None, ["wifi"], 2),
    ("Saver Inn Midtown", "midtown", 3.6, 89, True, False, 15, 1, ["wifi", "24h-desk"], 2),
]
# vendor, class, seats, base per day, desk opens, desk closes (24 = round the clock), at airport, free cancellation
CAR_TEMPLATES: list[tuple[Any, ...]] = [
    ("Enterprise Mock", "compact", 5, 48, 6, 23, True, True),
    ("Hertz Mock", "midsize", 5, 64, 0, 24, True, True),
    ("Avis Mock", "suv", 7, 89, 7, 22, True, False),
    ("National Mock", "fullsize", 5, 72, 0, 24, True, True),
    ("Sixt Mock", "minivan", 8, 105, 8, 20, False, True),
    ("Alamo Mock", "suv", 5, 79, 6, 22, True, True),
]
BUDGET_CAR_TEMPLATES: list[tuple[Any, ...]] = [
    ("Budget Mock", "economy", 4, 39, 7, 21, True, True),
    ("Fox Mock", "compact", 5, 35, 8, 20, True, False),
]
# vendor, service, max passengers, base price, priced per person?, minutes, shared, free cancellation
TRANSFER_TEMPLATES: list[tuple[Any, ...]] = [
    ("Airport Shuttle Mock", "shared shuttle", 10, 24, True, 80, True, True),
    ("City Car Mock", "private sedan", 3, 89, False, 45, False, True),
    ("City Car Mock", "private SUV", 6, 129, False, 45, False, True),
]
BUDGET_TRANSFER_TEMPLATES: list[tuple[Any, ...]] = [
    ("Saver Ride Mock", "shared van", 8, 16, True, 95, True, False),
]
CABIN_MULTIPLIER = {"economy": 1.0, "premium_economy": 1.6, "business": 3.2, "first": 4.5}


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _factor(*parts: object, spread: float = 0.1) -> float:
    unit = int(_hash(*parts)[:8], 16) / 0xFFFFFFFF
    return 1 + (unit * 2 - 1) * spread


def _source(option_id: str) -> SourceMetadata:
    return SourceMetadata(provider=PROVIDER, label=LABEL, source_id=option_id, retrieved_at=datetime.now(timezone.utc))


def _leg(origin: str, destination: str, day: date, dep: tuple[int, int], minutes: int, carrier: str,
         number: int, stops: int, layovers: list[str]) -> FlightLeg:
    departure = datetime.combine(day, time(*dep))
    shift = utc_offset(destination, day) - utc_offset(origin, day)  # times are local at each airport
    return FlightLeg(
        origin=origin, destination=destination, departure_time=departure,
        arrival_time=departure + timedelta(minutes=minutes) + shift, carrier=carrier, flight_number=f"{carrier}{number}",
        stops=stops, duration_minutes=minutes, layovers=layovers,
    )


def airport_for(code: str, index: int) -> str:
    """Mock flights use real airports: a metro code such as NYC becomes one of its airports (JFK, LGA, EWR)."""
    members = METRO_CODES.get(code.upper(), ("", ()))[1]
    return members[index % len(members)] if members else code.upper()


def build_flights(q: FlightQuery, scenario: str) -> list[FlightOption]:
    templates = [(t, False) for t in FLIGHT_TEMPLATES]
    if q.strategy == "cheapest" or q.max_price_total is not None:
        templates += [(t, True) for t in BUDGET_FLIGHT_TEMPLATES]
    options: list[FlightOption] = []
    legs = 2 if q.return_date else 1
    for index, (template, is_budget) in enumerate(templates):
        carrier, airline, dep, minutes, stops, layovers, base, refundable, seats, ret_dep = template
        origin, destination = airport_for(q.origin, index), airport_for(q.destination, index)
        if scenario == "late_arrival" and carrier != "PR":
            arrival = 22 * 60 + 40 + index * 7  # local time at the destination
            shift = utc_offset(destination, q.departure_date) - utc_offset(origin, q.departure_date)
            start = (arrival - minutes - int(shift.total_seconds() // 60)) % (24 * 60)
            dep = (start // 60, start % 60)
        number = 100 + int(_hash(carrier, dep)[:4], 16) % 800
        outbound = _leg(origin, destination, q.departure_date, dep, minutes, carrier, number, stops, layovers)
        inbound = (
            _leg(destination, origin, q.return_date, ret_dep, minutes, carrier, number + 1, stops, layovers)
            if q.return_date else None
        )
        multiplier = 1.9 if scenario == "over_budget" and not is_budget else 1.0
        price = base * q.passengers * legs * CABIN_MULTIPLIER.get(q.cabin_class, 1.0) * multiplier
        price *= _factor(q.origin, q.destination, q.departure_date, carrier, dep)
        option_id = "FL-" + _hash(q.origin, q.destination, q.departure_date, q.return_date, carrier, dep,
                                  q.passengers, q.cabin_class)[:10]
        options.append(FlightOption(
            option_id=option_id, strategy=q.strategy, airline=airline, outbound=outbound,
            inbound=inbound, cabin_class=q.cabin_class, passengers=q.passengers, price_total=round(price, 2),
            seats_available=seats, refundable=refundable,
            passenger_ids=[f"pas_mock_{n + 1}" for n in range(q.passengers)], source=_source(option_id),
        ))
    return options


def filter_flights(options: list[FlightOption], q: FlightQuery) -> list[FlightOption]:
    result = []
    for option in options:
        if q.strategy == "nonstop" and option.max_stops > 0:
            continue
        if q.max_stops is not None and option.max_stops > q.max_stops:
            continue
        if q.max_price_total is not None and option.price_total > q.max_price_total:
            continue
        result.append(option)
    result.sort(key=lambda o: o.price_total if q.strategy == "cheapest" else (o.max_stops, o.price_total))
    return result


def build_hotels(q: HotelQuery, scenario: str) -> list[HotelOption]:
    nights = (q.check_out - q.check_in).days
    templates = [(t, False) for t in HOTEL_TEMPLATES]
    if q.strategy == "budget" or q.max_price_total is not None:
        templates += [(t, True) for t in BUDGET_HOTEL_TEMPLATES]
    options = []
    for template, is_budget in templates:
        name, area, rating, base, late, early, from_hour, cancel_days, amenities, per_room = template
        rooms = max(1, math.ceil(q.guests / per_room))
        multiplier = 1.8 if scenario == "over_budget" and not is_budget else 1.0
        nightly = base * rooms * multiplier * _factor(q.city_code, q.check_in, name)
        option_id = "HT-" + _hash(q.city_code, q.check_in, q.check_out, name, rooms)[:10]
        options.append(HotelOption(
            option_id=option_id, strategy=q.strategy, name=name, city_code=q.city_code,
            area=area, check_in=q.check_in, check_out=q.check_out, nights=nights, check_in_from_hour=from_hour,
            late_check_in=late, early_check_in=early, rating=rating, rooms=rooms, max_guests=rooms * per_room,
            price_per_night=round(nightly, 2), price_total=round(nightly * nights, 2),
            free_cancellation_until=(q.check_in - timedelta(days=cancel_days)) if cancel_days else None,
            amenities=amenities, source=_source(option_id),
        ))
    return options


def filter_hotels(options: list[HotelOption], q: HotelQuery, scenario: str) -> list[HotelOption]:
    result = []
    for option in options:
        if scenario == "late_arrival" and not q.require_late_checkin and option.late_check_in:
            continue  # in this scenario only the follow-up search reveals hotels with late check-in
        if q.require_late_checkin and not option.late_check_in:
            continue
        if q.area and option.area != q.area.lower():
            continue
        if q.min_rating is not None and (option.rating or 0) < q.min_rating:
            continue
        if q.max_price_total is not None and option.price_total > q.max_price_total:
            continue
        if q.strategy == "family" and "family-rooms" not in option.amenities and "pool" not in option.amenities:
            continue
        result.append(option)
    result.sort(key=lambda o: o.price_total if q.strategy == "budget" else (-(o.rating or 0), o.price_total))
    return result


def build_cars(q: CarQuery, scenario: str) -> list[CarOption]:
    days = max(1, math.ceil((q.dropoff_time - q.pickup_time).total_seconds() / 86400))
    templates = list(CAR_TEMPLATES)
    if q.strategy == "budget" or q.max_price_total is not None:
        templates += BUDGET_CAR_TEMPLATES
    options = []
    for vendor, vclass, seats, base, opens, closes, airport, free_cancel in templates:
        price = base * days * _factor(q.city_code, q.pickup_time.date(), vendor)
        option_id = "CR-" + _hash(q.city_code, q.pickup_time, q.dropoff_time, vendor, vclass)[:10]
        options.append(CarOption(
            option_id=option_id, strategy=q.strategy, vendor=vendor, vehicle_class=vclass,
            seats=seats, pickup_location=f"{q.city_code} {'Airport' if airport else 'Downtown'}",
            pickup_at_airport=airport, pickup_time=q.pickup_time, dropoff_time=q.dropoff_time, desk_open_hour=opens,
            desk_close_hour=closes, price_total=round(price, 2), free_cancellation=free_cancel, source=_source(option_id),
        ))
    return options


def filter_cars(options: list[CarOption], q: CarQuery, scenario: str) -> list[CarOption]:
    result = []
    for option in options:
        if scenario == "late_arrival" and not q.need_desk_hours and option.open_24h:
            continue  # in this scenario only the follow-up search reveals 24-hour desks
        if any(not option.desk_open_at(hour) for hour in q.need_desk_hours):
            continue
        if q.passengers > option.seats:
            continue
        if q.car_class and q.strategy != "budget" and option.vehicle_class != q.car_class:
            continue
        if q.strategy == "larger" and option.seats < max(q.passengers + 1, 6):
            continue
        if q.max_price_total is not None and option.price_total > q.max_price_total:
            continue
        result.append(option)
    result.sort(key=lambda o: o.price_total)
    return result


def build_transfers(q: TransferQuery, scenario: str) -> list[TransferOption]:
    templates = list(TRANSFER_TEMPLATES) + (BUDGET_TRANSFER_TEMPLATES if q.strategy == "budget" else [])
    options = []
    for vendor, service, capacity, base, per_person, minutes, shared, free_cancel in templates:
        vehicles = 1 if per_person else max(1, math.ceil(q.passengers / capacity))
        price = base * (q.passengers if per_person else vehicles) * _factor(q.city_code, q.arrival_date, service)
        option_id = "TR-" + _hash(q.city_code, q.arrival_date, vendor, service, q.passengers)[:10]
        options.append(TransferOption(
            option_id=option_id, strategy=q.strategy, vendor=vendor, service=service, city_code=q.city_code,
            pickup_date=q.arrival_date, max_passengers=capacity * vehicles, duration_minutes=minutes, shared=shared,
            price_total=round(price, 2), free_cancellation=free_cancel, source=_source(option_id),
        ))
    return options


def filter_transfers(options: list[TransferOption], q: TransferQuery) -> list[TransferOption]:
    result = [o for o in options if o.max_passengers >= q.passengers
              and (q.max_price_total is None or o.price_total <= q.max_price_total)
              and (q.strategy != "private" or not o.shared)]
    result.sort(key=lambda o: o.price_total if q.strategy == "budget" else (o.shared, o.price_total))
    return result


class MockInventory:
    """Searches and sandbox bookings for flights, hotels, and cars, backed by the sandbox ledger."""

    def __init__(self, ledger: SandboxLedger, scenario: str = "happy", delay_s: float = 0.0, max_options: int = 10):
        self.ledger = ledger
        self.scenario = scenario
        self.delay_s = delay_s
        self.max_options = max_options

    async def _pause(self, kind: str) -> None:
        if self.scenario == f"{kind}_timeout":
            await asyncio.sleep(TIMEOUT_SLEEP_S)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.scenario == f"{kind}_error":
            raise RuntimeError(f"The {kind} provider returned HTTP 503 (simulated outage)")

    def _available(self, options: list[Any]) -> list[Any]:
        available = []
        for option in options:
            if self.ledger.unavailable_reason(option.option_id):
                continue
            if (override := self.ledger.price_override(option.option_id)) is not None:
                option = option.model_copy(update={"price_total": override})
            available.append(option)
        return available

    async def search_flights(self, q: FlightQuery) -> FlightSearchResult:
        await self._pause("flight")
        options = [] if self.scenario == "no_flights" else self._available(build_flights(q, self.scenario))
        options = filter_flights(options, q)[: self.max_options]
        self.ledger.save_offers([o.model_dump(mode="json") for o in options])
        return FlightSearchResult(options=options, provider=PROVIDER, label=LABEL)

    async def search_hotels(self, q: HotelQuery) -> HotelSearchResult:
        await self._pause("hotel")
        options = [] if self.scenario == "no_hotels" else self._available(build_hotels(q, self.scenario))
        options = filter_hotels(options, q, self.scenario)[: self.max_options]
        self.ledger.save_offers([o.model_dump(mode="json") for o in options])
        return HotelSearchResult(options=options, provider=PROVIDER, label=LABEL)

    async def search_cars(self, q: CarQuery) -> CarSearchResult:
        await self._pause("car")
        options = [] if self.scenario == "no_cars" else self._available(build_cars(q, self.scenario))
        options = filter_cars(options, q, self.scenario)[: self.max_options]
        self.ledger.save_offers([o.model_dump(mode="json") for o in options])
        return CarSearchResult(options=options, provider=PROVIDER, label=LABEL)

    async def search_transfers(self, q: TransferQuery) -> TransferSearchResult:
        await self._pause("transfer")
        options = self._available(build_transfers(q, self.scenario))
        options = filter_transfers(options, q)[: self.max_options]
        self.ledger.save_offers([o.model_dump(mode="json") for o in options])
        return TransferSearchResult(options=options, provider=PROVIDER, label=LABEL)

    def _current(self, option_id: str) -> tuple[dict[str, Any] | None, float | None, str | None]:
        offer = self.ledger.get_offer(option_id)
        if offer is None:
            return None, None, "Unknown offer; search again."
        override = self.ledger.price_override(option_id)
        return offer, override if override is not None else offer["price_total"], self.ledger.unavailable_reason(option_id)

    async def refresh_offer(self, offer_id: str) -> OfferRefresh:
        offer, price, unavailable = self._current(offer_id)
        if offer is None:
            return OfferRefresh(found=False, available=False, message="Offer not found")
        option = FlightOption.model_validate({**offer, "price_total": price}) if offer["kind"] == "flight" else None
        return OfferRefresh(
            found=True, available=unavailable is None, option=option, previous_price=offer["price_total"],
            current_price=price, price_changed=abs(price - offer["price_total"]) > 0.009, message=unavailable or "",
        )

    async def book(self, kind: str, option_id: str, approval_id: str, expected_total_price: float,
                   option_snapshot: dict[str, Any] | None = None) -> BookingResult:
        if not approval_id.startswith("apv-"):
            return BookingResult(status="error", message="A valid approval id is required")
        offer = self.ledger.get_offer(option_id)
        if offer is None and option_snapshot and option_snapshot.get("option_id") == option_id:
            self.ledger.save_offers([option_snapshot])
            offer = option_snapshot
        if offer is None or offer["kind"] != kind:
            return BookingResult(status="unavailable", message="This option is no longer offered; search again.")
        if kind == "flight" and self.scenario == "price_change" and self.ledger.first_time(f"price_change:{option_id}"):
            self.ledger.set_price_override(option_id, offer["price_total"] * 1.12, "simulated fare increase")
        if kind == "hotel" and self.scenario == "sold_out" and self.ledger.first_time(f"sold_out:{option_id}"):
            self.ledger.mark_unavailable(option_id, "sold out after approval (simulated)")
        _, price, unavailable = self._current(option_id)
        if unavailable:
            return BookingResult(status="unavailable", message=f"Not available: {unavailable}")
        if abs(price - expected_total_price) > 0.009:
            return BookingResult(status="price_changed", current_price=price,
                                 message=f"Price changed from {expected_total_price:.2f} to {price:.2f}")
        booking = self.ledger.create_booking(
            kind=kind, option_id=option_id, provider=offer["source"]["provider"], total_price=price,
            currency=offer.get("currency", "USD"), approval_id=approval_id, payload={**offer, "price_total": price},
        )
        return BookingResult(status="confirmed", booking=to_record(booking), current_price=price,
                             message="Sandbox booking confirmed (no real reservation or payment)")

    async def get_booking(self, booking_ref: str) -> BookingLookup:
        booking = self.ledger.get_booking(booking_ref)
        return BookingLookup(found=booking is not None, booking=to_record(booking) if booking else None)

    async def cancel(self, booking_ref: str, approval_id: str) -> CancellationResult:
        if not approval_id.startswith("apv-"):
            return CancellationResult(status="error", message="A valid approval id is required")
        booking = self.ledger.get_booking(booking_ref)
        if booking is None:
            return CancellationResult(status="not_found", message="Booking not found")
        if booking["status"] == "cancelled":
            return CancellationResult(status="not_cancellable", booking=to_record(booking), message="Already cancelled")
        refund = refund_amount(booking)
        self.ledger.update_booking_status(booking_ref, "cancelled", "Cancelled by the traveler (sandbox)")
        updated = self.ledger.get_booking(booking_ref)
        return CancellationResult(status="cancelled", booking=to_record(updated), refund_amount=refund,
                                  message="Sandbox cancellation confirmed")


def refund_amount(booking: dict[str, Any]) -> float:
    payload = booking["payload"]
    if booking["status"] == "cancelled_by_provider":
        return booking["total_price"]
    if booking["kind"] == "flight":
        return booking["total_price"] if payload.get("refundable") else 0.0
    if booking["kind"] == "hotel":
        deadline = payload.get("free_cancellation_until")
        return booking["total_price"] if deadline and date.fromisoformat(deadline) >= date.today() else 0.0
    return booking["total_price"] if payload.get("free_cancellation") else 0.0  # cars and airport pickups


def to_record(booking: dict[str, Any]) -> BookingRecord:
    payload = booking["payload"]
    return BookingRecord(
        booking_ref=booking["booking_ref"], kind=booking["kind"], option_id=booking["option_id"],
        provider=booking["provider"], status=booking["status"], total_price=booking["total_price"],
        currency=booking["currency"], sandbox=True, created_at=datetime.fromisoformat(booking["created_at"]),
        note=booking.get("note"), details=_details(payload),
    )


def _details(payload: dict[str, Any]) -> dict[str, Any]:
    kind = payload.get("kind")
    if kind == "flight":
        out = payload["outbound"]
        details = {"airline": payload["airline"], "flight_number": out["flight_number"],
                   "departure_time": out["departure_time"], "arrival_time": out["arrival_time"],
                   "refundable": payload.get("refundable", False)}
        if payload.get("inbound"):
            details["return_departure_time"] = payload["inbound"]["departure_time"]
        return details
    if kind == "hotel":
        return {"name": payload["name"], "check_in": payload["check_in"], "check_out": payload["check_out"],
                "free_cancellation_until": payload.get("free_cancellation_until")}
    if kind == "car":
        return {"vendor": payload["vendor"], "vehicle_class": payload["vehicle_class"],
                "pickup_time": payload["pickup_time"], "dropoff_time": payload["dropoff_time"],
                "free_cancellation": payload.get("free_cancellation", False)}
    if kind == "transfer":
        return {"vendor": payload["vendor"], "service": payload["service"], "pickup_date": payload["pickup_date"],
                "free_cancellation": payload.get("free_cancellation", False)}
    return {}
