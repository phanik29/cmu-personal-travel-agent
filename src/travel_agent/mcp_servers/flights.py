"""MCP server `flights`: flight search (mock data or Duffel test mode) and flight orders, which always stay in the
local sandbox."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Protocol

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from travel_agent.config import Settings
from travel_agent.providers.base import (
    BookingLookup,
    BookingResult,
    CancellationResult,
    FlightQuery,
    FlightSearchResult,
    OfferRefresh,
)
from travel_agent.providers.mock import MockInventory
from travel_agent.sandbox.ledger import SandboxLedger

INSTRUCTIONS = (
    "Flight search over sandbox data. Search results are real tool data; never invent flights. "
    "book_flight and cancel_flight_order are sandbox actions that require an approval id issued after the "
    "traveler explicitly approved the exact offer and price."
)


class FlightBackend(Protocol):
    async def search_flights(self, q: FlightQuery) -> FlightSearchResult: ...
    async def refresh_offer(self, offer_id: str) -> OfferRefresh: ...
    async def book(self, kind: str, option_id: str, approval_id: str, expected_total_price: float,
                   option_snapshot: dict[str, Any] | None = None) -> BookingResult: ...
    async def get_booking(self, booking_ref: str) -> BookingLookup: ...
    async def cancel(self, booking_ref: str, approval_id: str) -> CancellationResult: ...


class DuffelSearchSandboxBooking:
    """Duffel test mode finds flights and re-checks prices; booking, lookup, and cancelling stay in the local sandbox,
    so no Duffel order is ever created."""

    def __init__(self, duffel: Any, sandbox: MockInventory, ledger: SandboxLedger) -> None:
        self.duffel, self.sandbox, self.ledger = duffel, sandbox, ledger

    async def search_flights(self, q: FlightQuery) -> FlightSearchResult:
        result = await self.duffel.search_flights(q)
        self.ledger.save_offers([o.model_dump(mode="json") for o in result.options])  # so the sandbox can book them
        return result

    async def refresh_offer(self, offer_id: str) -> OfferRefresh:
        return await self.duffel.refresh_offer(offer_id)

    async def book(self, kind: str, option_id: str, approval_id: str, expected_total_price: float,
                   option_snapshot: dict[str, Any] | None = None) -> BookingResult:
        return await self.sandbox.book(kind, option_id, approval_id, expected_total_price, option_snapshot)

    async def get_booking(self, booking_ref: str) -> BookingLookup:
        return await self.sandbox.get_booking(booking_ref)

    async def cancel(self, booking_ref: str, approval_id: str) -> CancellationResult:
        return await self.sandbox.cancel(booking_ref, approval_id)


def make_backend(settings: Settings) -> FlightBackend:
    ledger = SandboxLedger(settings.sandbox_db)
    sandbox = MockInventory(ledger, settings.scenario, settings.simulated_delay_s, settings.max_options_per_search)
    if settings.duffel_access_token:
        from travel_agent.providers.duffel import DuffelFlights

        duffel = DuffelFlights(settings.duffel_access_token, timeout_s=settings.tool_timeout_s,
                               max_options=settings.max_options_per_search)
        return DuffelSearchSandboxBooking(duffel, sandbox, ledger)
    return sandbox


def create_server(settings: Settings, backend: FlightBackend | None = None) -> FastMCP:
    backend = backend or make_backend(settings)
    server = FastMCP("flights", instructions=INSTRUCTIONS, log_level="WARNING")

    @server.tool()
    async def search_flights(
        origin: Annotated[str, Field(min_length=3, max_length=3, description="IATA airport or city code")],
        destination: Annotated[str, Field(min_length=3, max_length=3, description="IATA airport or city code")],
        departure_date: Annotated[date, Field(description="Outbound date, YYYY-MM-DD")],
        return_date: Annotated[date | None, Field(description="Return date for round trips")] = None,
        passengers: Annotated[int, Field(ge=1, le=20)] = 1,
        cabin_class: Annotated[str, Field(description="economy, premium_economy, business, or first")] = "economy",
        max_stops: Annotated[int | None, Field(ge=0, le=3)] = None,
        max_price_total: Annotated[float | None, Field(gt=0, description="Upper limit for the total price")] = None,
        strategy: Annotated[str, Field(description="best, nonstop, or cheapest")] = "best",
    ) -> FlightSearchResult:
        """Search flight offers with prices, times, stops, and layovers. Read-only."""
        query = FlightQuery(
            origin=origin.upper(), destination=destination.upper(), departure_date=departure_date,
            return_date=return_date, passengers=passengers, cabin_class=cabin_class, max_stops=max_stops,
            max_price_total=max_price_total, strategy=strategy,
        )
        return await backend.search_flights(query)

    @server.tool()
    async def refresh_offer(offer_id: Annotated[str, Field(min_length=3)]) -> OfferRefresh:
        """Re-check a flight offer's current price and availability. Read-only."""
        return await backend.refresh_offer(offer_id)

    @server.tool()
    async def book_flight(
        offer_id: Annotated[str, Field(min_length=3)],
        approval_id: Annotated[str, Field(min_length=5, description="Single-use approval issued by the traveler")],
        expected_total_price: Annotated[float, Field(gt=0, description="The exact price the traveler approved")],
    ) -> BookingResult:
        """Create a SANDBOX flight order (no real ticket or payment). Requires the traveler's approval id.
        Refuses if the current price differs from the approved price."""
        return await backend.book("flight", offer_id, approval_id, expected_total_price)

    @server.tool()
    async def get_flight_order(booking_ref: Annotated[str, Field(min_length=3)]) -> BookingLookup:
        """Look up a sandbox flight order and its current status. Read-only."""
        return await backend.get_booking(booking_ref)

    @server.tool()
    async def cancel_flight_order(
        booking_ref: Annotated[str, Field(min_length=3)],
        approval_id: Annotated[str, Field(min_length=5, description="Single-use cancellation approval")],
    ) -> CancellationResult:
        """Cancel a SANDBOX flight order. Requires the traveler's cancellation approval id."""
        return await backend.cancel(booking_ref, approval_id)

    return server


def main() -> None:
    settings = Settings.from_env(dotenv_path=None)
    create_server(settings).run(transport="stdio")


if __name__ == "__main__":
    main()
