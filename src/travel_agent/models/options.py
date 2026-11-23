"""Normalized travel options returned by the MCP tools."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceMetadata(Strict):
    provider: str
    label: str
    source_id: str
    retrieved_at: datetime


class FlightLeg(Strict):
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    departure_time: datetime
    arrival_time: datetime
    carrier: str
    flight_number: str
    stops: int = Field(ge=0)
    duration_minutes: int = Field(gt=0)
    layovers: list[str] = Field(default_factory=list)


class FlightOption(Strict):
    kind: Literal["flight"] = "flight"
    option_id: str
    strategy: str | None = None
    airline: str
    outbound: FlightLeg
    inbound: FlightLeg | None = None
    cabin_class: str = "economy"
    passengers: int = Field(ge=1)
    price_total: float = Field(gt=0)
    currency: str = "USD"
    seats_available: int | None = Field(default=None, ge=0)
    refundable: bool = False
    expires_at: datetime | None = None
    booking_url: str | None = None
    passenger_ids: list[str] = Field(default_factory=list)
    source: SourceMetadata

    @property
    def max_stops(self) -> int:
        return max(self.outbound.stops, self.inbound.stops if self.inbound else 0)

    @property
    def travel_minutes(self) -> int:
        return self.outbound.duration_minutes + (self.inbound.duration_minutes if self.inbound else 0)

    def label(self) -> str:
        return f"{self.airline} {self.outbound.flight_number}, {self.outbound.origin} to {self.outbound.destination}"


class HotelOption(Strict):
    kind: Literal["hotel"] = "hotel"
    option_id: str
    strategy: str | None = None
    name: str
    city_code: str
    area: str
    check_in: date
    check_out: date
    nights: int = Field(ge=1)
    check_in_from_hour: int = Field(default=15, ge=0, le=23)
    late_check_in: bool = False
    early_check_in: bool = False
    rating: float | None = Field(default=None, ge=0, le=5)
    rooms: int = Field(default=1, ge=1)
    max_guests: int = Field(ge=1)
    price_per_night: float = Field(gt=0)
    price_total: float = Field(gt=0)
    currency: str = "USD"
    free_cancellation_until: date | None = None
    amenities: list[str] = Field(default_factory=list)
    source: SourceMetadata

    def label(self) -> str:
        return self.name


class CarOption(Strict):
    kind: Literal["car"] = "car"
    option_id: str
    strategy: str | None = None
    vendor: str
    vehicle_class: str
    seats: int = Field(ge=1)
    pickup_location: str
    pickup_at_airport: bool = True
    pickup_time: datetime
    dropoff_time: datetime
    desk_open_hour: int = Field(ge=0, le=23)
    desk_close_hour: int = Field(ge=1, le=24)
    price_total: float = Field(gt=0)
    currency: str = "USD"
    free_cancellation: bool = True
    source: SourceMetadata

    @property
    def open_24h(self) -> bool:
        return self.desk_open_hour == 0 and self.desk_close_hour == 24

    def desk_open_at(self, hour: float) -> bool:
        return self.open_24h or self.desk_open_hour <= hour < self.desk_close_hour

    def label(self) -> str:
        return f"{self.vendor} {self.vehicle_class}"


class TransferOption(Strict):
    """Airport pickup on arrival: a ride from the arrival airport to the hotel or city center."""

    kind: Literal["transfer"] = "transfer"
    option_id: str
    strategy: str | None = None
    vendor: str
    service: str
    city_code: str
    pickup_date: date
    max_passengers: int = Field(ge=1)
    duration_minutes: int = Field(gt=0)
    shared: bool = False
    price_total: float = Field(gt=0)
    currency: str = "USD"
    free_cancellation: bool = True
    source: SourceMetadata

    def label(self) -> str:
        return f"{self.vendor} {self.service}"


OPTION_MODELS: dict[str, type[Strict]] = {"flight": FlightOption, "hotel": HotelOption, "car": CarOption,
                                          "transfer": TransferOption}


def parse_option(data: dict) -> FlightOption | HotelOption | CarOption | TransferOption:
    return OPTION_MODELS[data["kind"]].model_validate(data)


class AirportInfo(Strict):
    iata: str
    name: str
    city: str
    country: str
    latitude: float
    longitude: float
    kind: Literal["airport", "metro"] = "airport"
    members: list[str] = Field(default_factory=list)


class WeatherDay(Strict):
    day: date
    temp_max_c: float
    temp_min_c: float
    precipitation_mm: float = Field(ge=0)
    precipitation_probability_pct: int | None = Field(default=None, ge=0, le=100)
    wind_max_kmh: float = Field(ge=0)


class WeatherReport(Strict):
    city_code: str
    start_date: date
    end_date: date
    kind: Literal["forecast", "typical", "sample"]
    days: list[WeatherDay]
    summary: str
    source: SourceMetadata


class WeatherRisk(Strict):
    level: Literal["low", "moderate", "high"]
    reasons: list[str] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    report: WeatherReport
