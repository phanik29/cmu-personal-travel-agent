"""Duffel flights in TEST MODE (tokens must start with `duffel_test_`), used only to search flights and re-check
offer prices. The app never creates Duffel orders: every booking stays in the local sandbox.

Endpoints and field names follow Duffel's docs and its official Python client. Checked with a test token on
2026-11-22: `uv run python scripts/check_duffel.py`.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from travel_agent.models.options import FlightLeg, FlightOption, SourceMetadata
from travel_agent.providers.base import (
        FlightQuery,
    FlightSearchResult,
    OfferRefresh,
)

API_URL = "https://api.duffel.com"
API_VERSION = "v2"
PROVIDER = "duffel-test"
LABEL = "Duffel test"


class DuffelError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Duffel HTTP {status}: {message}")
        self.status = status


def minutes(iso_duration: str | None) -> int:
    """'PT5H30M' or 'P1DT2H' → minutes (0 if missing)."""
    match = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:\d+S)?)?", iso_duration or "")
    if not match:
        return 0
    days, hours, mins = (int(part or 0) for part in match.groups())
    return days * 1440 + hours * 60 + mins


def leg_from_slice(slice_: dict[str, Any]) -> FlightLeg:
    segments = slice_["segments"]
    first, last = segments[0], segments[-1]
    carrier = first["marketing_carrier"]["iata_code"]
    layovers = []
    for previous, following in zip(segments, segments[1:]):
        gap = datetime.fromisoformat(following["departing_at"]) - datetime.fromisoformat(previous["arriving_at"])
        hours, mins = divmod(int(gap.total_seconds() // 60), 60)
        layovers.append(f"{previous['destination']['iata_code']} ({hours}h{mins:02d}m)")
    duration = minutes(slice_.get("duration")) or sum(minutes(s.get("duration")) for s in segments)
    return FlightLeg(
        origin=first["origin"]["iata_code"], destination=last["destination"]["iata_code"],
        departure_time=first["departing_at"], arrival_time=last["arriving_at"], carrier=carrier,
        flight_number=f"{carrier}{first['marketing_carrier_flight_number']}", stops=len(segments) - 1,
        duration_minutes=max(duration, 1), layovers=layovers,
    )


def offer_to_option(offer: dict[str, Any], query: FlightQuery | None = None) -> FlightOption:
    slices = offer["slices"]
    refund = (offer.get("conditions") or {}).get("refund_before_departure") or {}
    return FlightOption(
        option_id=offer["id"], strategy=query.strategy if query else None, airline=offer["owner"]["name"],
        outbound=leg_from_slice(slices[0]), inbound=leg_from_slice(slices[1]) if len(slices) > 1 else None,
        cabin_class=query.cabin_class if query else "economy", passengers=len(offer["passengers"]),
        price_total=float(offer["total_amount"]), currency=offer["total_currency"], refundable=bool(refund.get("allowed")),
        expires_at=offer.get("expires_at"), passenger_ids=[p["id"] for p in offer["passengers"]],
        source=SourceMetadata(provider=PROVIDER, label=LABEL, source_id=offer["id"], retrieved_at=datetime.now(timezone.utc)),
    )


class DuffelFlights:
    def __init__(self, token: str, timeout_s: float = 45.0, max_options: int = 10,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not token.startswith("duffel_test_"):
            raise ValueError("Only Duffel test-mode tokens (duffel_test_...) are allowed")
        self.max_options = max_options
        self._client = httpx.AsyncClient(
            base_url=API_URL, timeout=timeout_s, transport=transport,
            headers={"Authorization": f"Bearer {token}", "Duffel-Version": API_VERSION, "Accept": "application/json"})
        self._offer_ids: dict[str, str] = {}  # order id → offer id

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Duffel request failed ({type(exc).__name__})") from exc
        if response.status_code >= 400:
            try:
                errors = response.json().get("errors") or [{}]
                message = errors[0].get("message") or errors[0].get("code") or "error"
            except ValueError:
                message = response.text[:150]
            raise DuffelError(response.status_code, str(message)[:200])
        return response.json()["data"]

    async def search_flights(self, q: FlightQuery) -> FlightSearchResult:
        slices = [{"origin": q.origin, "destination": q.destination, "departure_date": q.departure_date.isoformat()}]
        if q.return_date:
            slices.append({"origin": q.destination, "destination": q.origin, "departure_date": q.return_date.isoformat()})
        body: dict[str, Any] = {"slices": slices, "passengers": [{"type": "adult"} for _ in range(q.passengers)],
                                "cabin_class": q.cabin_class}
        max_stops = 0 if q.strategy == "nonstop" else q.max_stops
        if max_stops is not None:
            body["max_connections"] = max_stops
        data = await self._request("POST", "/air/offer_requests", params={"return_offers": "true"}, json={"data": body})
        options, skipped = [], 0
        for offer in data.get("offers") or []:
            try:
                option = offer_to_option(offer, q)
            except (KeyError, IndexError, TypeError, ValueError):
                skipped += 1
                continue
            if q.max_price_total is None or option.price_total <= q.max_price_total:
                options.append(option)
        options.sort(key=lambda o: o.price_total if q.strategy == "cheapest" else (o.max_stops, o.price_total))
        notes = [f"Skipped {skipped} offer(s) with an unexpected format"] if skipped else []
        currencies = sorted({o.currency for o in options} - {"USD"})
        if currencies:
            notes.append(f"Duffel returned prices in {', '.join(currencies)}; this POC compares prices in USD only")
        return FlightSearchResult(options=options[: self.max_options], provider=PROVIDER, label=LABEL, notes=notes)

    async def refresh_offer(self, offer_id: str) -> OfferRefresh:
        try:
            offer = await self._request("GET", f"/air/offers/{offer_id}")
        except DuffelError as exc:
            return OfferRefresh(found=False, available=False, message=str(exc))
        option = offer_to_option(offer)
        return OfferRefresh(found=True, available=True, option=option, current_price=option.price_total)
