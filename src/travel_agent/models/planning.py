"""Planning, approval, and booking models."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AgentName = Literal["flight", "hotel", "car", "transfer"]
ActionName = Literal["book_flight", "book_hotel", "book_car", "book_transfer", "cancel_flight", "cancel_hotel",
                     "cancel_car", "cancel_transfer"]

FLIGHT_STRATEGIES = ("best", "nonstop", "cheapest")
HOTEL_STRATEGIES = ("preferred", "budget", "late_checkin", "family")
CAR_STRATEGIES = ("standard", "budget", "late_pickup", "larger")
TRANSFER_STRATEGIES = ("standard", "budget", "private")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchDirective(Strict):
    """One concrete search an agent should run (a ToT branch backed by a real tool call)."""

    agent: AgentName
    strategy: str
    reason: str = ""
    max_price_total: float | None = Field(default=None, gt=0)
    max_stops: int | None = Field(default=None, ge=0, le=3)
    area: str | None = None
    require_late_checkin: bool = False
    car_class: str | None = None
    need_desk_hours: list[int] = Field(default_factory=list)

    def signature(self) -> str:
        data = self.model_dump(exclude={"reason"}, exclude_defaults=True)
        return "|".join(f"{key}={data[key]}" for key in sorted(data))


class TaskPlan(Strict):
    goal: str
    priority: str
    trip_type: str
    steps: list[str]
    agents: list[str]
    directives: list[SearchDirective]
    source: Literal["rules", "llm"] = "rules"


class OptionStatus(str, Enum):
    SEARCHED = "searched"
    RECOMMENDED = "recommended"
    APPROVED = "approved"
    BOOKED = "booked"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class ApprovalSnapshot(Strict):
    """Exactly what the user approved; an approval can be used once, only for this item and price."""

    action: ActionName
    kind: AgentName
    option_id: str
    provider: str
    total_price: float
    currency: str
    dates: dict[str, str] = Field(default_factory=dict)
    booking_ref: str | None = None
    label: str = ""


class BookingRecord(Strict):
    booking_ref: str
    kind: AgentName
    option_id: str
    provider: str
    status: str
    total_price: float
    currency: str
    sandbox: bool = True
    created_at: datetime
    note: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(Strict):
    decision: Literal["approve", "change", "reject"]
    candidate_id: str | None = None
    feedback: str | None = None
