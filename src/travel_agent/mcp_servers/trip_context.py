"""MCP server `trip-context`: airport lookup and weather (sample data or Open-Meteo)."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from travel_agent.config import Settings
from travel_agent.models.options import WeatherReport, WeatherRisk
from travel_agent.providers.airports import get_airport_index
from travel_agent.providers.base import AirportLookupResult
from travel_agent.providers.weather import evaluate_risk, sample_weather

INSTRUCTIONS = "Airport lookup (OurAirports open data) and trip weather. All tools are read-only."
MAX_WEATHER_DAYS = 31


def create_server(settings: Settings, weather_client=None) -> FastMCP:
    index = get_airport_index(str(settings.airports_csv) if settings.airports_csv.is_file() else None)
    if weather_client is None and settings.weather_source == "open-meteo":
        from travel_agent.providers.open_meteo import OpenMeteoClient

        weather_client = OpenMeteoClient(timeout_s=min(settings.tool_timeout_s, 20))
    server = FastMCP("trip-context", instructions=INSTRUCTIONS, log_level="WARNING")

    async def load_weather(city_code: str, start_date: date, end_date: date) -> WeatherReport:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        if (end_date - start_date).days >= MAX_WEATHER_DAYS:
            end_date = start_date + timedelta(days=MAX_WEATHER_DAYS - 1)
        airport = index.get(city_code)
        if weather_client is None:
            return sample_weather(city_code.upper(), start_date, end_date, airport, settings.scenario)
        if airport is None:
            raise ValueError(f"Unknown airport or city code {city_code}")
        return await weather_client.weather(city_code.upper(), airport.latitude, airport.longitude, start_date, end_date)

    @server.tool()
    async def lookup_airport(
        query: Annotated[str, Field(min_length=2, max_length=80, description="City, airport name, or IATA code")],
        limit: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> AirportLookupResult:
        """Find airports or metro-area codes by city, airport name, or IATA code. Read-only."""
        return AirportLookupResult(query=query, matches=index.search(query, limit))

    @server.tool()
    async def nearby_airports(
        iata: Annotated[str, Field(min_length=3, max_length=3)],
        radius_km: Annotated[float, Field(gt=0, le=400)] = 150,
        limit: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> AirportLookupResult:
        """List large airports within a radius of an airport or city code. Read-only."""
        return AirportLookupResult(query=iata.upper(), matches=index.nearby(iata, radius_km, limit))

    @server.tool()
    async def get_weather(
        city_code: Annotated[str, Field(min_length=3, max_length=3)],
        start_date: date,
        end_date: date,
    ) -> WeatherReport:
        """Daily weather for the trip dates: a forecast within 16 days, otherwise typical weather. Read-only."""
        return await load_weather(city_code, start_date, end_date)

    @server.tool()
    async def weather_risk(
        city_code: Annotated[str, Field(min_length=3, max_length=3)],
        start_date: date,
        end_date: date,
        trip_type: Literal["leisure", "family", "business", "adventure"] = "leisure",
    ) -> WeatherRisk:
        """Assess weather risk (rain, wind, heat, cold) for the trip type and dates. Read-only."""
        return evaluate_risk(await load_weather(city_code, start_date, end_date), trip_type)

    return server


def main() -> None:
    settings = Settings.from_env(dotenv_path=None)
    create_server(settings).run(transport="stdio")


if __name__ == "__main__":
    main()
