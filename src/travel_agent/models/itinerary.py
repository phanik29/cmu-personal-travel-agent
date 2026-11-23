"""Itinerary candidates produced by Tree-of-Thought search."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .options import CarOption, FlightOption, HotelOption, TransferOption


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScoreBreakdown(Strict):
    factors: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    base_score: float = 0.0
    pref_adjustment: float = Field(default=0.0, ge=-5.0, le=5.0)
    pref_reasons: list[str] = Field(default_factory=list)
    llm_adjustment: float = Field(default=0.0, ge=-5.0, le=5.0)
    llm_reason: str | None = None
    total: float = 0.0
    rationale: list[str] = Field(default_factory=list)

    def recompute(self) -> None:
        self.total = round(self.base_score + self.pref_adjustment + self.llm_adjustment, 2)


class ItineraryCandidate(Strict):
    candidate_id: str
    flight: FlightOption | None = None
    hotel: HotelOption | None = None
    car: CarOption | None = None
    transfer: TransferOption | None = None
    total_price: float = 0.0
    currency: str = "USD"
    within_budget: bool | None = None
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    tie_group: int | None = None
    locked: list[str] = Field(default_factory=list)

    def components(self) -> list[FlightOption | HotelOption | CarOption | TransferOption]:
        return [item for item in (self.flight, self.hotel, self.car, self.transfer) if item is not None]

    def option_ids(self) -> list[str]:
        return [item.option_id for item in self.components()]


class PruningRecord(Strict):
    node_id: str
    stage: str
    reason: str


class Issue(Strict):
    """A problem found while evaluating plans, used by the Planner to decide follow-up searches."""

    code: str
    agent: str | None = None
    detail: str
    data: dict[str, Any] = Field(default_factory=dict)
