"""Download OurAirports open data (public domain) and write a compact airport list.

Usage:
    uv run python scripts/download_airports.py                       # writes data/airports.csv
    uv run python scripts/download_airports.py --output src/travel_agent/data/airports.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from pathlib import Path

SOURCE_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
FIELDS = ["iata", "name", "city", "country", "latitude", "longitude", "type"]
TYPES = {"large_airport", "medium_airport"}


def build_rows(source_text: str, include_small: bool = False) -> list[list[str]]:
    types = TYPES | ({"small_airport"} if include_small else set())
    rows = []
    seen: set[str] = set()
    for row in csv.DictReader(io.StringIO(source_text)):
        code = row["iata_code"].strip().upper()
        if len(code) != 3 or code in seen or row["scheduled_service"] != "yes" or row["type"] not in types:
            continue
        seen.add(code)
        rows.append([
            code, row["name"], row["municipality"], row["iso_country"],
            f"{float(row['latitude_deg']):.4f}", f"{float(row['longitude_deg']):.4f}", row["type"],
        ])
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/airports.csv")
    parser.add_argument("--source", help="Read a previously downloaded airports.csv instead of downloading")
    parser.add_argument("--include-small", action="store_true", help="Include small airports with scheduled service")
    args = parser.parse_args(argv)

    if args.source:
        text = Path(args.source).read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:  # noqa: S310 - fixed https URL
            text = response.read().decode("utf-8")

    rows = build_rows(text, include_small=args.include_small)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} airports to {output}")


if __name__ == "__main__":
    main()
