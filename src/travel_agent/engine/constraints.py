"""Hard rules (prune) and soft warnings (score) for flights, hotels, cars, and complete plans."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from travel_agent.models.options import CarOption, FlightOption, HotelOption, TransferOption
from travel_agent.models.request import ARRIVAL_WINDOWS, TravelRequest
from travel_agent.providers.airports import AirportIndex, get_airport_index

LATE_ARRIVAL_HOUR = 22
OVERNIGHT_CUTOFF_HOUR = 5  # landing after midnight but before 5 AM still counts as the night of the departure day
CAR_PICKUP_BUFFER = timedelta(minutes=45)
CAR_DROPOFF_BUFFER = timedelta(minutes=120)
EARLY_ARRIVAL_GAP_HOURS = 3

VIOLATION_TEXT = {
    "route_mismatch": "Flight airports don't match your trip",
    "lands_next_day": "Lands the next morning, after your first hotel night",
    "landing_time_mismatch": "Lands outside the time you asked for",
    "departure_date_mismatch": "Flight leaves on a different date",
    "return_date_mismatch": "Return flight is on a different date",
    "not_enough_seats": "Not enough seats for all travelers",
    "too_many_stops": "More stops than you allow",
    "currency_unsupported": "Price is not in USD",
    "offer_expired": "The offer has expired",
    "over_budget": "Over your budget",
    "stay_dates_mismatch": "Hotel dates don't match your trip",
    "hotel_too_small": "Rooms don't fit all travelers",
    "late_arrival_no_late_checkin": "Flight lands after the hotel's front desk closes",
    "car_too_small": "Car has too few seats",
    "car_desk_closed_at_arrival": "Rental desk is closed when your flight lands",
    "car_desk_closed_at_dropoff": "Rental desk is closed before your return flight",
    "transfer_too_small": "Airport pickup has too few seats",
}


def describe(code: str) -> str:
    return VIOLATION_TEXT.get(code, code.replace("_", " "))


def hour_of(moment: datetime) -> float:
    return moment.hour + moment.minute / 60


def lands_in_window(moment: datetime, window: str) -> bool:
    start, end = ARRIVAL_WINDOWS[window]
    hour = hour_of(moment)
    return start <= hour < end or start <= hour + 24 < end


def needs_late_checkin(flight: FlightOption) -> bool:
    arrival = flight.outbound.arrival_time
    return arrival.date() > flight.outbound.departure_time.date() or arrival.hour >= LATE_ARRIVAL_HOUR


def car_pickup_hour(flight: FlightOption) -> float:
    return hour_of(flight.outbound.arrival_time + CAR_PICKUP_BUFFER)


def car_dropoff_hour(flight: FlightOption) -> float | None:
    if flight.inbound is None:
        return None
    return hour_of(flight.inbound.departure_time - CAR_DROPOFF_BUFFER)


def flight_violations(flight: FlightOption, request: TravelRequest, airports: AirportIndex | None = None) -> list[str]:
    airports = airports or get_airport_index()
    out = flight.outbound
    codes: list[str] = []
    if not airports.matches(request.origin, out.origin) or not airports.matches(request.destination, out.destination):
        codes.append("route_mismatch")
    if out.departure_time.date() != request.departure_date:
        codes.append("departure_date_mismatch")
    days_later = (out.arrival_time.date() - out.departure_time.date()).days
    if days_later > 1 or (days_later == 1 and out.arrival_time.hour >= OVERNIGHT_CUTOFF_HOUR):
        codes.append("lands_next_day")  # simple POC: the hotel night and car pickup start on the departure day
    if request.arrival_window and not lands_in_window(out.arrival_time, request.arrival_window):
        codes.append("landing_time_mismatch")
    if request.return_date is not None and (
        flight.inbound is None or flight.inbound.departure_time.date() != request.return_date
    ):
        codes.append("return_date_mismatch")
    if flight.passengers < request.travelers or (
        flight.seats_available is not None and flight.seats_available < request.travelers
    ):
        codes.append("not_enough_seats")
    if request.max_stops is not None and flight.max_stops > request.max_stops:
        codes.append("too_many_stops")
    if flight.currency != "USD":
        codes.append("currency_unsupported")
    if flight.expires_at is not None:
        expires = flight.expires_at if flight.expires_at.tzinfo else flight.expires_at.replace(tzinfo=timezone.utc)
        if expires < datetime.now(timezone.utc):
            codes.append("offer_expired")
    if request.budget_usd is not None and flight.price_total > request.budget_usd:
        codes.append("over_budget")
    return codes


def hotel_checks(hotel: HotelOption, request: TravelRequest, flight: FlightOption | None) -> tuple[list[str], list[str]]:
    codes: list[str] = []
    warnings: list[str] = []
    if hotel.check_in != request.departure_date or hotel.check_out != request.return_date:
        codes.append("stay_dates_mismatch")
    if hotel.max_guests < request.travelers:
        codes.append("hotel_too_small")
    if hotel.currency != "USD":
        codes.append("currency_unsupported")
    if flight is not None:
        arrival = flight.outbound.arrival_time
        if needs_late_checkin(flight):
            if not hotel.late_check_in:
                codes.append("late_arrival_no_late_checkin")
        elif hour_of(arrival) < hotel.check_in_from_hour - EARLY_ARRIVAL_GAP_HOURS and not hotel.early_check_in:
            wait = hotel.check_in_from_hour - hour_of(arrival)
            warnings.append(f"You land about {wait:.0f} hours before check-in, and this hotel has no early check-in.")
    if request.hotel_area and hotel.area not in {request.hotel_area, "unknown"}:
        warnings.append(f"Hotel is in {hotel.area}, not your preferred {request.hotel_area}.")
    return codes, warnings


def car_checks(car: CarOption, request: TravelRequest, flight: FlightOption | None) -> tuple[list[str], list[str]]:
    codes: list[str] = []
    warnings: list[str] = []
    if car.seats < request.travelers:
        codes.append("car_too_small")
    if car.currency != "USD":
        codes.append("currency_unsupported")
    if flight is not None:
        if not car.desk_open_at(car_pickup_hour(flight)):
            codes.append("car_desk_closed_at_arrival")
        dropoff = car_dropoff_hour(flight)
        if dropoff is not None and not car.desk_open_at(dropoff):
            codes.append("car_desk_closed_at_dropoff")
    if not car.pickup_at_airport:
        warnings.append(f"Car pickup is at {car.pickup_location}, not at the airport.")
    return codes, warnings


def transfer_checks(transfer: TransferOption, request: TravelRequest) -> tuple[list[str], list[str]]:
    codes: list[str] = []
    warnings: list[str] = []
    if transfer.max_passengers < request.travelers:
        codes.append("transfer_too_small")
    if transfer.currency != "USD":
        codes.append("currency_unsupported")
    if transfer.shared:
        warnings.append(f"The shared shuttle takes about {transfer.duration_minutes} minutes and stops on the way.")
    return codes, warnings


def needed_desk_hours(flight: FlightOption) -> list[int]:
    hours = {int(car_pickup_hour(flight))}
    dropoff = car_dropoff_hour(flight)
    if dropoff is not None:
        hours.add(int(dropoff))
    return sorted(hours)
