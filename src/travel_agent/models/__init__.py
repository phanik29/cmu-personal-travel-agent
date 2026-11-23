"""Validated data models for the travel agent."""
from .itinerary import Issue, ItineraryCandidate, PruningRecord, ScoreBreakdown
from .options import (
    AirportInfo,
    CarOption,
    FlightLeg,
    FlightOption,
    HotelOption,
    SourceMetadata,
    TransferOption,
    WeatherDay,
    WeatherReport,
    WeatherRisk,
    parse_option,
)
from .planning import (
    ApprovalDecision,
    ApprovalSnapshot,
    BookingRecord,
    OptionStatus,
    SearchDirective,
    TaskPlan,
)
from .request import TravelRequest
from .results import AgentResult, AgentStatus, TraceEvent

__all__ = [
    "AgentResult",
    "AgentStatus",
    "AirportInfo",
    "ApprovalDecision",
    "ApprovalSnapshot",
    "BookingRecord",
    "CarOption",
    "FlightLeg",
    "FlightOption",
    "HotelOption",
    "Issue",
    "ItineraryCandidate",
    "OptionStatus",
    "PruningRecord",
    "ScoreBreakdown",
    "SearchDirective",
    "SourceMetadata",
    "TaskPlan",
    "TraceEvent",
    "TransferOption",
    "TravelRequest",
    "WeatherDay",
    "WeatherReport",
    "WeatherRisk",
    "parse_option",
]
