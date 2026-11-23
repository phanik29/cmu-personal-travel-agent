"""Soft-preference scoring weighted by the traveler's priority and trip type, with explanations."""
from __future__ import annotations

from dataclasses import dataclass

from travel_agent.engine.constraints import car_pickup_hour, hour_of
from travel_agent.models.itinerary import ScoreBreakdown
from travel_agent.models.options import CarOption, FlightOption, HotelOption, TransferOption
from travel_agent.models.request import TravelRequest

WEIGHTS: dict[str, dict[str, float]] = {
    "cost":        {"price": .40, "budget": .20, "flight": .10, "schedule": .10, "hotel": .10, "location": .05, "car": .05,
                    "transfer": .03},
    "time":        {"price": .10, "budget": .05, "flight": .35, "schedule": .25, "hotel": .10, "location": .10, "car": .05,
                    "transfer": .05},
    "comfort":     {"price": .10, "budget": .05, "flight": .25, "schedule": .10, "hotel": .30, "location": .10, "car": .10,
                    "transfer": .05},
    "convenience": {"price": .10, "budget": .05, "flight": .15, "schedule": .30, "hotel": .10, "location": .20, "car": .10,
                    "transfer": .05},
    "balanced":    {"price": .25, "budget": .10, "flight": .15, "schedule": .15, "hotel": .15, "location": .10, "car": .10,
                    "transfer": .04},
}
AMENITY_FIT = {
    "family": {"pool", "family-rooms", "breakfast"},
    "business": {"wifi", "gym", "24h-desk"},
    "adventure": {"breakfast", "shuttle"},
    "leisure": {"pool", "breakfast", "spa"},
}
AREA_FIT = {
    "business": {"downtown": 1.0, "midtown": 0.85, "airport": 0.6},
    "family": {"waterfront": 0.95, "midtown": 0.9, "downtown": 0.85, "old-town": 0.85, "airport": 0.4},
    "adventure": {"old-town": 0.9, "downtown": 0.85, "midtown": 0.8, "waterfront": 0.85, "airport": 0.6},
    "leisure": {"downtown": 0.9, "waterfront": 0.95, "old-town": 0.9, "midtown": 0.8, "airport": 0.4},
}


@dataclass(frozen=True)
class PriceRange:
    low: float
    high: float

    def score(self, value: float) -> float:
        """Ratio to the cheapest option: 1.0 for the cheapest, 0.5 for twice the price."""
        if value <= 0:
            return 1.0
        return max(0.0, min(1.0, self.low / value))


def flight_quality(flight: FlightOption, request: TravelRequest) -> float:
    stop_scores = {0: 1.0, 1: 0.55, 2: 0.2}
    stops = stop_scores.get(flight.max_stops, 0.0)
    if request.trip_type == "family" and flight.max_stops:
        stops *= 0.7
    legs = 2 if flight.inbound is not None else 1
    per_leg_minutes = flight.travel_minutes / legs
    duration = max(0.0, min(1.0, 1 - (per_leg_minutes - 240) / 480))
    score = 0.55 * stops + 0.25 * duration + (0.15 if flight.refundable else 0.0)
    if request.trip_type == "business" and flight.refundable:
        score += 0.05
    if request.flight_preference and "nonstop" in request.flight_preference.lower() and flight.max_stops == 0:
        score += 0.05
    return min(1.0, score)


def schedule_fit(flight: FlightOption, request: TravelRequest, hotel: HotelOption | None,
                 car: CarOption | None) -> float:
    departure = hour_of(flight.outbound.departure_time)
    arrival = flight.outbound.arrival_time
    dep_score = 1.0 if 7 <= departure <= 20 else (0.6 if 6 <= departure < 7 or 20 < departure <= 22 else 0.3)
    if arrival.date() > flight.outbound.departure_time.date():
        arr_score = 0.3
    else:
        hour = hour_of(arrival)
        arr_score = 1.0 if 9 <= hour <= 20 else (0.6 if hour < 9 or hour <= 22 else 0.35)
        if request.trip_type == "business" and hour <= 16:
            arr_score = min(1.0, arr_score + 0.1)
    score = 0.4 * dep_score + 0.6 * arr_score
    if hotel is not None and arrival.date() == hotel.check_in:
        wait = hotel.check_in_from_hour - hour_of(arrival)
        if wait > 3 and not hotel.early_check_in:
            score -= 0.15
    if car is not None and not car.open_24h:
        margin = car.desk_close_hour - car_pickup_hour(flight)
        if margin < 1.5:
            score -= 0.1
    return max(0.0, min(1.0, score))


def hotel_quality(hotel: HotelOption, request: TravelRequest) -> float:
    rating = (hotel.rating or 3.0) / 5
    fit = AMENITY_FIT.get(request.trip_type, set())
    amenities = len(fit & set(hotel.amenities)) / max(1, len(fit))
    cancel = 1.0 if hotel.free_cancellation_until else 0.0
    return min(1.0, 0.6 * rating + 0.2 * amenities + 0.2 * cancel)


def location_fit(hotel: HotelOption, request: TravelRequest) -> float:
    if request.hotel_area:
        return 1.0 if hotel.area == request.hotel_area else 0.4
    return AREA_FIT.get(request.trip_type, AREA_FIT["leisure"]).get(hotel.area, 0.5)


def car_fit(car: CarOption, request: TravelRequest) -> float:
    if request.car_class:
        cls = 1.0 if car.vehicle_class == request.car_class else 0.5
    elif request.trip_type == "family" or request.travelers >= 4:
        cls = 1.0 if car.vehicle_class in {"suv", "minivan"} or car.seats >= request.travelers + 1 else 0.6
    elif request.trip_type == "adventure":
        cls = 1.0 if car.vehicle_class == "suv" else 0.6
    else:
        cls = 0.9 if car.vehicle_class in {"compact", "midsize", "economy"} else 0.75
    return min(1.0, 0.6 * cls + (0.2 if car.free_cancellation else 0) + (0.1 if car.pickup_at_airport else 0)
               + (0.1 if car.open_24h else 0.05))


def transfer_fit(transfer: TransferOption, request: TravelRequest) -> float:
    """Private rides suit time, comfort, convenience, and families; shared shuttles are fine when cost matters."""
    if request.effective_priority in {"time", "comfort", "convenience"} or request.trip_type == "family":
        return 0.55 if transfer.shared else 1.0
    return 0.75 if transfer.shared else 0.85


def budget_fit(total: float, budget: float | None) -> float | None:
    if budget is None:
        return None
    if total <= 0.8 * budget:
        return 1.0
    return max(0.0, 1 - 0.5 * (total - 0.8 * budget) / (0.2 * budget))


def score_plan(
    request: TravelRequest,
    flight: FlightOption,
    hotel: HotelOption | None,
    car: CarOption | None,
    price_range: PriceRange,
    *,
    final: bool,
    transfer: TransferOption | None = None,
) -> ScoreBreakdown:
    total = sum(item.price_total for item in (flight, hotel, car, transfer) if item is not None)
    factors: dict[str, float] = {
        "price": price_range.score(total),
        "flight": flight_quality(flight, request),
        "schedule": schedule_fit(flight, request, hotel, car),
    }
    if hotel is not None:
        factors["hotel"] = hotel_quality(hotel, request)
        factors["location"] = location_fit(hotel, request)
    if car is not None:
        factors["car"] = car_fit(car, request)
    if transfer is not None:
        factors["transfer"] = transfer_fit(transfer, request)
    if final and (fit := budget_fit(total, request.budget_usd)) is not None:
        factors["budget"] = fit
    base_weights = WEIGHTS[request.effective_priority]
    weights = {name: base_weights[name] for name in factors}
    norm = sum(weights.values())
    weights = {name: round(value / norm, 4) for name, value in weights.items()}
    base = 100 * sum(weights[name] * factors[name] for name in factors)
    return ScoreBreakdown(
        factors={k: round(v, 3) for k, v in factors.items()}, weights=weights, base_score=round(base, 2),
        total=round(base, 2), rationale=rationale(request, flight, hotel, car, total, transfer),
    )


def rationale(request: TravelRequest, flight: FlightOption, hotel: HotelOption | None, car: CarOption | None,
              total: float, transfer: TransferOption | None = None) -> list[str]:
    out = flight.outbound
    stops = "Nonstop" if flight.max_stops == 0 else f"{flight.max_stops} stop ({', '.join(out.layovers) or 'connection'})"
    bullets = [f"{stops} on {flight.airline} {out.flight_number}, lands {out.arrival_time:%a %H:%M}"]
    if flight.refundable:
        bullets.append("Refundable fare")
    if hotel is not None:
        stars = f"rated {hotel.rating:.1f}" if hotel.rating else "unrated"
        preferred = " (your preferred area)" if request.hotel_area == hotel.area else ""
        bullets.append(f"{hotel.name}, {stars}, {hotel.area}{preferred}")
        if hotel.free_cancellation_until:
            bullets.append(f"Free hotel cancellation until {hotel.free_cancellation_until:%b %d}")
    if car is not None:
        desk = "open 24 hours" if car.open_24h else f"open {car.desk_open_hour}:00–{car.desk_close_hour}:00"
        bullets.append(f"{car.vendor} {car.vehicle_class}, rental desk {desk}")
    if transfer is not None:
        bullets.append(f"Airport pickup: {transfer.vendor} {transfer.service}")
    if request.budget_usd is not None:
        bullets.append(f"Total ${total:,.0f}, ${request.budget_usd - total:,.0f} under your ${request.budget_usd:,.0f} budget")
    else:
        bullets.append(f"Total ${total:,.0f}")
    return bullets
