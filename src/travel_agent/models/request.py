from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AirportCode = Annotated[str, Field(min_length=3, max_length=3)]
Priority = Literal["cost", "time", "comfort", "convenience", "balanced"]
TripType = Literal["leisure", "family", "business", "adventure"]
CabinClass = Literal["economy", "premium_economy", "business", "first"]
CarClass = Literal["economy", "compact", "midsize", "fullsize", "suv", "minivan", "premium"]
ArrivalWindow = Literal["morning", "afternoon", "evening", "night"]
# When the outbound flight should land, in local hours at the destination (night runs past midnight).
ARRIVAL_WINDOWS: dict[str, tuple[int, int]] = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 21),
                                               "night": (21, 29)}

PRIORITIES: tuple[str, ...] = ("cost", "time", "comfort", "convenience", "balanced")
TRIP_TYPES: tuple[str, ...] = ("leisure", "family", "business", "adventure")
SEARCH_FIELDS = frozenset({
    "origin", "destination", "departure_date", "return_date", "travelers", "cabin_class",
    "require_car", "need_hotel", "car_class", "need_transfer",
})


class TravelRequest(BaseModel):
    """Normalized, validated user travel request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    origin: AirportCode = Field(description="Three-letter origin airport or city code")
    destination: AirportCode = Field(description="Three-letter destination airport or city code")
    departure_date: date
    return_date: date | None = None
    travelers: int = Field(default=1, ge=1, le=20)
    budget_usd: float | None = Field(default=None, gt=0)
    flight_preference: str | None = None
    hotel_preference: str | None = None
    require_car: bool = True
    need_hotel: bool = True
    need_transfer: bool = True
    priority: Priority | None = None
    trip_type: TripType = "leisure"
    max_stops: int | None = Field(default=None, ge=0, le=3)
    cabin_class: CabinClass = "economy"
    hotel_area: str | None = None
    car_class: CarClass | None = None
    arrival_window: ArrivalWindow | None = None

    @field_validator("origin", "destination")
    @classmethod
    def normalize_airport_codes(cls, value: str) -> str:
        code = value.upper()
        if not code.isalpha():
            raise ValueError("Airport codes must contain letters only")
        return code

    @field_validator("departure_date")
    @classmethod
    def departure_must_not_be_in_past(cls, value: date) -> date:
        if value < date.today():
            raise ValueError("Departure date cannot be in the past")
        return value

    @field_validator("hotel_area")
    @classmethod
    def normalize_area(cls, value: str | None) -> str | None:
        return value.lower() if value else None

    @model_validator(mode="after")
    def validate_trip_dates_and_airports(self):
        if self.origin == self.destination:
            raise ValueError("Origin and destination must be different airports")
        if self.return_date is not None and self.return_date < self.departure_date:
            raise ValueError("Return date must be on or after departure date")
        return self

    @property
    def nights(self) -> int | None:
        if self.return_date is None:
            return None
        return (self.return_date - self.departure_date).days

    @property
    def wants_hotel(self) -> bool:
        """Hotels are searched only for round trips with at least one night."""
        return self.need_hotel and bool(self.nights)

    @property
    def wants_car(self) -> bool:
        """Cars need a return date to know the rental length."""
        return self.require_car and self.return_date is not None

    @property
    def wants_transfer(self) -> bool:
        """An airport pickup on arrival (to the hotel, or to the city center for one-way trips)."""
        return self.need_transfer

    @property
    def effective_priority(self) -> str:
        return self.priority or "balanced"

