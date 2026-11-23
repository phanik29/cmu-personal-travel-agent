"""Check the Duffel TEST-mode integration with your token (read from .env as DUFFEL_ACCESS_TOKEN=duffel_test_...).

    uv run python scripts/check_duffel.py     # one test-mode search and a price re-check

The token is never printed. The app only searches Duffel and re-checks prices; it never creates Duffel orders.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

from travel_agent.config import Settings
from travel_agent.providers.base import FlightQuery
from travel_agent.providers.duffel import DuffelFlights


async def main(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if not settings.duffel_access_token:
        raise SystemExit("Set DUFFEL_ACCESS_TOKEN=duffel_test_... in .env first.")
    duffel = DuffelFlights(settings.duffel_access_token, max_options=5)
    departure = date.today() + timedelta(days=args.days)
    back = departure + timedelta(days=5)
    result = await duffel.search_flights(FlightQuery(origin=args.origin, destination=args.destination,
                                                     departure_date=departure, return_date=back, strategy="cheapest"))
    print(f"Search {args.origin}→{args.destination}, {departure} to {back}: {len(result.options)} offer(s)")
    for note in result.notes:
        print("  note:", note)
    for o in result.options:
        print(f"  {o.option_id}  {o.airline:<18} {o.outbound.departure_time:%Y-%m-%d %H:%M} → "
              f"{o.outbound.arrival_time:%H:%M}  stops={o.max_stops}  {o.price_total:.2f} {o.currency}")
    if result.options:
        refresh = await duffel.refresh_offer(result.options[0].option_id)
        print("price re-check:", refresh.found, refresh.current_price, refresh.message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origin", default="SFO")
    parser.add_argument("--destination", default="JFK")
    parser.add_argument("--days", type=int, default=30, help="days from today until departure")
    asyncio.run(main(parser.parse_args()))
