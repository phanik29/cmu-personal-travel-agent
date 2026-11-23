"""MCP server `travel-sandbox`: hotel and rental-car search plus sandbox bookings."""
from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from travel_agent.config import Settings
from travel_agent.providers.base import (
    BookingLookup,
    BookingResult,
    CancellationResult,
    CarQuery,
    CarSearchResult,
    HotelQuery,
    HotelSearchResult,
    TransferQuery,
    TransferSearchResult,
)
from travel_agent.providers.mock import MockInventory
from travel_agent.sandbox.ledger import SandboxLedger

INSTRUCTIONS = (
    "Hotel, rental-car, and airport-pickup search over mock inventory. Booking and cancellation tools change only the sandbox "
    "ledger and require an approval id issued after the traveler explicitly approved the exact option and price."
)


def create_server(settings: Settings, inventory: MockInventory | None = None) -> FastMCP:
    inventory = inventory or MockInventory(
        SandboxLedger(settings.sandbox_db), settings.scenario, settings.simulated_delay_s,
        settings.max_options_per_search,
    )
    server = FastMCP("travel-sandbox", instructions=INSTRUCTIONS, log_level="WARNING")

    @server.tool()
    async def search_hotels(
        city_code: Annotated[str, Field(min_length=3, max_length=3, description="IATA city or airport code")],
        check_in: Annotated[date, Field(description="YYYY-MM-DD")],
        check_out: Annotated[date, Field(description="YYYY-MM-DD")],
        guests: Annotated[int, Field(ge=1, le=20)] = 1,
        area: Annotated[str | None, Field(description="Preferred area, e.g. downtown")] = None,
        max_price_total: Annotated[float | None, Field(gt=0)] = None,
        min_rating: Annotated[float | None, Field(ge=0, le=5)] = None,
        require_late_checkin: bool = False,
        strategy: Annotated[str, Field(description="preferred, budget, late_checkin, or family")] = "preferred",
    ) -> HotelSearchResult:
        """Search hotels with total price, rating, area, check-in times, and cancellation policy. Read-only."""
        if check_out <= check_in:
            raise ValueError("check_out must be after check_in")
        query = HotelQuery(
            city_code=city_code.upper(), check_in=check_in, check_out=check_out, guests=guests, area=area,
            max_price_total=max_price_total, min_rating=min_rating, require_late_checkin=require_late_checkin,
            strategy=strategy,
        )
        return await inventory.search_hotels(query)

    @server.tool()
    async def search_cars(
        city_code: Annotated[str, Field(min_length=3, max_length=3)],
        pickup_time: Annotated[datetime, Field(description="Local pickup time, YYYY-MM-DDTHH:MM")],
        dropoff_time: Annotated[datetime, Field(description="Local drop-off time")],
        passengers: Annotated[int, Field(ge=1, le=20)] = 1,
        car_class: Annotated[str | None, Field(description="economy, compact, midsize, fullsize, suv, minivan")] = None,
        max_price_total: Annotated[float | None, Field(gt=0)] = None,
        need_desk_hours: Annotated[list[int] | None, Field(description="Hours (0-23) the rental desk must be open")] = None,
        strategy: Annotated[str, Field(description="standard, budget, late_pickup, or larger")] = "standard",
    ) -> CarSearchResult:
        """Search rental cars with price, class, seats, and rental-desk opening hours. Read-only."""
        if dropoff_time <= pickup_time:
            raise ValueError("dropoff_time must be after pickup_time")
        query = CarQuery(
            city_code=city_code.upper(), pickup_time=pickup_time, dropoff_time=dropoff_time, passengers=passengers,
            car_class=car_class, max_price_total=max_price_total, need_desk_hours=need_desk_hours or [],
            strategy=strategy,
        )
        return await inventory.search_cars(query)

    @server.tool()
    async def search_transfers(
        city_code: Annotated[str, Field(min_length=3, max_length=3, description="Arrival city or airport code")],
        arrival_date: Annotated[date, Field(description="YYYY-MM-DD")],
        passengers: Annotated[int, Field(ge=1, le=20)] = 1,
        max_price_total: Annotated[float | None, Field(gt=0)] = None,
        strategy: Annotated[str, Field(description="standard, budget, or private")] = "standard",
    ) -> TransferSearchResult:
        """Search airport pickups (shared shuttles and private cars) from the arrival airport. Read-only."""
        query = TransferQuery(city_code=city_code.upper(), arrival_date=arrival_date, passengers=passengers,
                              max_price_total=max_price_total, strategy=strategy)
        return await inventory.search_transfers(query)

    @server.tool()
    async def book_hotel(
        option_id: Annotated[str, Field(min_length=3)],
        approval_id: Annotated[str, Field(min_length=5)],
        expected_total_price: Annotated[float, Field(gt=0)],
        option_snapshot: Annotated[dict[str, Any] | None, Field(description="Option details for external providers")] = None,
    ) -> BookingResult:
        """Create a SANDBOX hotel booking (no real reservation or payment). Requires the traveler's approval id."""
        return await inventory.book("hotel", option_id, approval_id, expected_total_price, option_snapshot)

    @server.tool()
    async def book_car(
        option_id: Annotated[str, Field(min_length=3)],
        approval_id: Annotated[str, Field(min_length=5)],
        expected_total_price: Annotated[float, Field(gt=0)],
    ) -> BookingResult:
        """Create a SANDBOX car rental booking (no real reservation or payment). Requires the traveler's approval id."""
        return await inventory.book("car", option_id, approval_id, expected_total_price)

    @server.tool()
    async def book_transfer(
        option_id: Annotated[str, Field(min_length=3)],
        approval_id: Annotated[str, Field(min_length=5)],
        expected_total_price: Annotated[float, Field(gt=0)],
    ) -> BookingResult:
        """Create a SANDBOX airport-pickup booking (no real reservation or payment). Requires the traveler's approval id."""
        return await inventory.book("transfer", option_id, approval_id, expected_total_price)

    @server.tool()
    async def get_booking(booking_ref: Annotated[str, Field(min_length=3)]) -> BookingLookup:
        """Look up a sandbox hotel, car, or airport-pickup booking and its current status. Read-only."""
        return await inventory.get_booking(booking_ref)

    @server.tool()
    async def cancel_booking(
        booking_ref: Annotated[str, Field(min_length=3)],
        approval_id: Annotated[str, Field(min_length=5)],
    ) -> CancellationResult:
        """Cancel a SANDBOX hotel, car, or airport-pickup booking. Requires the traveler's cancellation approval id."""
        return await inventory.cancel(booking_ref, approval_id)

    return server


def main() -> None:
    settings = Settings.from_env(dotenv_path=None)
    create_server(settings).run(transport="stdio")


if __name__ == "__main__":
    main()
