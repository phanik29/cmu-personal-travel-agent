"""Tool input queries and result contracts shared by the MCP servers and the gateway."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.models.options import (
    AirportInfo,
    CarOption,
    FlightOption,
    HotelOption,
    TransferOption,
    WeatherReport,
    WeatherRisk,
)
from travel_agent.models.planning import BookingRecord


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FlightQuery(Strict):
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    departure_date: date
    return_date: date | None = None
    passengers: int = Field(default=1, ge=1, le=20)
    cabin_class: str = "economy"
    max_stops: int | None = Field(default=None, ge=0, le=3)
    max_price_total: float | None = Field(default=None, gt=0)
    strategy: str = "best"


class HotelQuery(Strict):
    city_code: str = Field(min_length=3, max_length=3)
    check_in: date
    check_out: date
    guests: int = Field(default=1, ge=1, le=20)
    area: str | None = None
    max_price_total: float | None = Field(default=None, gt=0)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    require_late_checkin: bool = False
    strategy: str = "preferred"


class CarQuery(Strict):
    city_code: str = Field(min_length=3, max_length=3)
    pickup_time: datetime
    dropoff_time: datetime
    passengers: int = Field(default=1, ge=1, le=20)
    car_class: str | None = None
    max_price_total: float | None = Field(default=None, gt=0)
    need_desk_hours: list[int] = Field(default_factory=list)
    strategy: str = "standard"


class TransferQuery(Strict):
    city_code: str = Field(min_length=3, max_length=3)
    arrival_date: date
    passengers: int = Field(default=1, ge=1, le=20)
    max_price_total: float | None = Field(default=None, gt=0)
    strategy: str = "standard"


class FlightSearchResult(Strict):
    options: list[FlightOption]
    provider: str
    label: str
    notes: list[str] = Field(default_factory=list)


class HotelSearchResult(Strict):
    options: list[HotelOption]
    provider: str
    label: str
    notes: list[str] = Field(default_factory=list)


class CarSearchResult(Strict):
    options: list[CarOption]
    provider: str
    label: str
    notes: list[str] = Field(default_factory=list)


class TransferSearchResult(Strict):
    options: list[TransferOption]
    provider: str
    label: str
    notes: list[str] = Field(default_factory=list)


class OfferRefresh(Strict):
    found: bool
    available: bool
    option: FlightOption | None = None
    previous_price: float | None = None
    current_price: float | None = None
    price_changed: bool = False
    message: str = ""


class BookingResult(Strict):
    status: Literal["confirmed", "price_changed", "unavailable", "error"]
    booking: BookingRecord | None = None
    current_price: float | None = None
    message: str = ""


class BookingLookup(Strict):
    found: bool
    booking: BookingRecord | None = None


class CancellationResult(Strict):
    status: Literal["cancelled", "not_cancellable", "not_found", "error"]
    booking: BookingRecord | None = None
    refund_amount: float | None = None
    message: str = ""


class AirportLookupResult(Strict):
    query: str
    matches: list[AirportInfo]


# Re-exported so the gateway can validate weather tool results.
WeatherReportResult = WeatherReport
WeatherRiskResult = WeatherRisk
