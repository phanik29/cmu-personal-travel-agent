"""Weather risk rules and the deterministic sample-weather source."""
from __future__ import annotations

import hashlib
import math
from datetime import date, datetime, timedelta, timezone

from travel_agent.models.options import AirportInfo, SourceMetadata, WeatherDay, WeatherReport, WeatherRisk

# Thresholds differ by trip type: adventure trips are more sensitive to rain and wind than business trips.
RISK_THRESHOLDS: dict[str, dict[str, float]] = {
    "leisure": {"rain_mm": 15, "rain_prob": 80, "wind_kmh": 50, "heat_c": 36, "cold_c": -15},
    "family": {"rain_mm": 12, "rain_prob": 75, "wind_kmh": 45, "heat_c": 34, "cold_c": -12},
    "business": {"rain_mm": 25, "rain_prob": 90, "wind_kmh": 60, "heat_c": 38, "cold_c": -20},
    "adventure": {"rain_mm": 8, "rain_prob": 60, "wind_kmh": 40, "heat_c": 35, "cold_c": -10},
}
FORECAST_HORIZON_DAYS = 16


def _unit(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def summarize(days: list[WeatherDay]) -> str:
    if not days:
        return "No weather data."
    wet = sum(1 for d in days if d.precipitation_mm >= 2 or (d.precipitation_probability_pct or 0) >= 60)
    return (f"Highs {_span([d.temp_max_c for d in days])}, lows {_span([d.temp_min_c for d in days])}. "
            f"Rain likely on {wet} of {len(days)} days. Wind up to {max(d.wind_max_kmh for d in days):.0f} km/h.")


def _span(values: list[float]) -> str:
    low, high = round(min(values)), round(max(values))
    return f"around {low}°C" if low == high else f"{low} to {high}°C"


def sample_weather(
    city_code: str,
    start: date,
    end: date,
    airport: AirportInfo | None,
    scenario: str = "happy",
) -> WeatherReport:
    latitude = airport.latitude if airport else 35.0
    peak_month = 7 if latitude >= 0 else 1
    annual_mean = 28 - 0.42 * abs(latitude)
    amplitude = 0.28 * abs(latitude)
    days: list[WeatherDay] = []
    trip_days = date_range(start, end)
    for index, day in enumerate(trip_days):
        seasonal = math.cos(2 * math.pi * (day.month - peak_month) / 12)
        tmax = annual_mean + amplitude * seasonal + 5 + (_unit(city_code, day, "t") - 0.5) * 4
        tmin = tmax - 8 - _unit(city_code, day, "n") * 3
        roll = _unit(city_code, day, "p") * 100
        precipitation = 0.0 if roll < 55 else round((roll - 55) * 0.25, 1)
        probability = int(roll * 0.85)
        wind = 10 + _unit(city_code, day, "w") * 25
        if scenario == "bad_weather" and index == len(trip_days) // 2:
            precipitation, probability, wind = 32.0, 95, 68.0
        days.append(WeatherDay(
            day=day, temp_max_c=round(tmax, 1), temp_min_c=round(tmin, 1), precipitation_mm=precipitation,
            precipitation_probability_pct=probability, wind_max_kmh=round(wind, 1),
        ))
    return WeatherReport(
        city_code=city_code, start_date=start, end_date=end, kind="sample", days=days, summary=summarize(days),
        source=SourceMetadata(provider="sample-weather", label="Sample data", source_id=f"sample:{city_code}:{start}",
                              retrieved_at=datetime.now(timezone.utc)),
    )


def evaluate_risk(report: WeatherReport, trip_type: str = "leisure") -> WeatherRisk:
    limits = RISK_THRESHOLDS.get(trip_type, RISK_THRESHOLDS["leisure"])
    reasons: list[str] = []
    severe = False
    risky_days = 0
    for day in report.days:
        problems = []
        if day.precipitation_mm >= limits["rain_mm"] or (day.precipitation_probability_pct or 0) >= limits["rain_prob"]:
            problems.append(f"heavy rain likely ({day.precipitation_mm:.0f} mm)")
            severe |= day.precipitation_mm >= 2 * limits["rain_mm"]
        if day.wind_max_kmh >= limits["wind_kmh"]:
            problems.append(f"strong wind ({day.wind_max_kmh:.0f} km/h)")
            severe |= day.wind_max_kmh >= limits["wind_kmh"] + 15
        if day.temp_max_c >= limits["heat_c"]:
            problems.append(f"heat ({day.temp_max_c:.0f}°C)")
        if day.temp_min_c <= limits["cold_c"]:
            problems.append(f"severe cold ({day.temp_min_c:.0f}°C)")
        if problems:
            risky_days += 1
            reasons.append(f"{day.day:%a}, {day.day:%b} {day.day.day}: " + ", ".join(problems))
    if severe or (report.days and risky_days * 2 >= len(report.days)):
        level = "high"
    elif risky_days:
        level = "moderate"
    else:
        level = "low"
    advice: list[str] = []
    if level != "low":
        advice.append("Prefer refundable or free-cancellation options.")
        if trip_type in {"adventure", "family"}:
            advice.append("Plan indoor alternatives for the risky days.")
        if level == "high":
            advice.append("Consider other dates if you can.")
    if report.kind == "typical":
        advice.append("This is typical weather from past years, not a forecast.")
    return WeatherRisk(level=level, reasons=reasons, advice=advice, report=report)
