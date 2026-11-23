"""Open-Meteo weather: free for non-commercial use, no API key; data licensed CC BY 4.0 (attribution: Open-Meteo.com).

Trips within the 16-day forecast range get a forecast. Later trips get the same dates last year from the archive,
labeled "typical, not a forecast". A failed request raises an error; sample data is never substituted silently.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from travel_agent.models.options import SourceMetadata, WeatherDay, WeatherReport
from travel_agent.providers.weather import FORECAST_HORIZON_DAYS, summarize

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]


def year_earlier(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:  # Feb 29
        return day.replace(year=day.year - 1, day=28)


class OpenMeteoClient:
    def __init__(self, timeout_s: float = 20.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.timeout_s = timeout_s
        self.transport = transport

    async def _daily(self, url: str, params: dict[str, Any]) -> dict[str, list[Any]]:
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as client:
            try:
                response = await client.get(url, params=params)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Open-Meteo request failed ({type(exc).__name__})") from exc
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        if response.status_code != 200 or data.get("error"):
            raise RuntimeError(f"Open-Meteo returned HTTP {response.status_code}: {data.get('reason', '')[:150]}")
        return data["daily"]

    async def weather(self, city_code: str, lat: float, lon: float, start: date, end: date) -> WeatherReport:
        forecast = end <= date.today() + timedelta(days=FORECAST_HORIZON_DAYS)
        query_start, query_end = (start, end) if forecast else (year_earlier(start), year_earlier(end))
        params: dict[str, Any] = {"latitude": round(lat, 3), "longitude": round(lon, 3), "timezone": "auto",
                                  "start_date": query_start.isoformat(), "end_date": query_end.isoformat(),
                                  "daily": ",".join(DAILY + (["precipitation_probability_max"] if forecast else []))}
        daily = await self._daily(FORECAST_URL if forecast else ARCHIVE_URL, params)
        days: list[WeatherDay] = []
        for index, _ in enumerate(daily.get("time", [])):
            high, low = daily["temperature_2m_max"][index], daily["temperature_2m_min"][index]
            if high is None or low is None:
                continue
            probability = (daily.get("precipitation_probability_max") or [None] * (index + 1))[index]
            days.append(WeatherDay(
                day=start + timedelta(days=index), temp_max_c=high, temp_min_c=low,
                precipitation_mm=daily["precipitation_sum"][index] or 0.0,
                precipitation_probability_pct=probability, wind_max_kmh=daily["wind_speed_10m_max"][index] or 0.0,
            ))
        kind = "forecast" if forecast else "typical"
        summary = summarize(days) + ("" if forecast else f" Based on the same dates in {query_start.year}, not a forecast.")
        source = SourceMetadata(provider="open-meteo", label="Open-Meteo.com (CC BY 4.0)", source_id=f"{lat:.3f},{lon:.3f}",
                                retrieved_at=datetime.now(timezone.utc))
        return WeatherReport(city_code=city_code, start_date=start, end_date=end, kind=kind, days=days, summary=summary,
                             source=source)
