"""Airport lookup backed by OurAirports open data (public domain)."""
from __future__ import annotations

import csv
import math
import unicodedata
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from travel_agent.models.options import AirportInfo

# IATA metropolitan-area codes: a city code covers several airports.
METRO_CODES: dict[str, tuple[str, tuple[str, ...]]] = {
    "NYC": ("New York", ("JFK", "LGA", "EWR")),
    "LON": ("London", ("LHR", "LGW", "STN", "LCY", "LTN")),
    "PAR": ("Paris", ("CDG", "ORY")),
    "TYO": ("Tokyo", ("HND", "NRT")),
    "CHI": ("Chicago", ("ORD", "MDW")),
    "WAS": ("Washington", ("IAD", "DCA", "BWI")),
    "MIL": ("Milan", ("MXP", "LIN", "BGY")),
    "ROM": ("Rome", ("FCO", "CIA")),
    "STO": ("Stockholm", ("ARN", "BMA")),
    "OSA": ("Osaka", ("KIX", "ITM")),
    "SEL": ("Seoul", ("ICN", "GMP")),
    "BJS": ("Beijing", ("PEK", "PKX")),
    "SAO": ("Sao Paulo", ("GRU", "CGH", "VCP")),
    "BUE": ("Buenos Aires", ("EZE", "AEP")),
    "YTO": ("Toronto", ("YYZ", "YTZ")),
    "MOW": ("Moscow", ("SVO", "DME", "VKO")),
}

# Time zones of major airports, so mock flight times are local times at each airport. Other airports use an
# approximation from longitude (one hour per 15 degrees).
_ZONES = {
    "America/Los_Angeles": "SFO SJC OAK LAX BUR LGB SNA ONT SAN SMF SEA PDX LAS",
    "America/Phoenix": "PHX TUS",
    "America/Denver": "DEN SLC ABQ BOI",
    "America/Chicago": "ORD MDW DFW DAL IAH HOU AUS SAT MSP STL MCI MSY BNA MEM MKE OMA OKC",
    "America/New_York": "JFK LGA EWR BOS IAD DCA BWI PHL ATL MIA FLL MCO TPA CLT DTW PIT CLE CMH RDU BUF IND",
    "America/Anchorage": "ANC", "Pacific/Honolulu": "HNL OGG KOA LIH",
    "America/Toronto": "YYZ YTZ YOW YUL", "America/Vancouver": "YVR", "America/Edmonton": "YYC YEG",
    "America/Mexico_City": "MEX", "America/Cancun": "CUN",
    "Europe/London": "LHR LGW STN LCY LTN MAN EDI", "Europe/Dublin": "DUB", "Europe/Lisbon": "LIS",
    "Europe/Madrid": "MAD BCN", "Europe/Paris": "CDG ORY NCE", "Europe/Amsterdam": "AMS", "Europe/Brussels": "BRU",
    "Europe/Berlin": "BER FRA MUC HAM DUS", "Europe/Zurich": "ZRH GVA", "Europe/Vienna": "VIE",
    "Europe/Rome": "FCO CIA MXP LIN BGY VCE", "Europe/Copenhagen": "CPH", "Europe/Oslo": "OSL",
    "Europe/Stockholm": "ARN BMA", "Europe/Helsinki": "HEL", "Europe/Athens": "ATH", "Europe/Istanbul": "IST SAW",
    "Europe/Moscow": "SVO DME VKO", "Asia/Dubai": "DXB", "Asia/Qatar": "DOH", "Asia/Jerusalem": "TLV",
    "Asia/Kolkata": "DEL BOM BLR MAA HYD CCU", "Asia/Bangkok": "BKK", "Asia/Singapore": "SIN",
    "Asia/Hong_Kong": "HKG", "Asia/Shanghai": "PEK PKX PVG SHA CAN", "Asia/Taipei": "TPE", "Asia/Seoul": "ICN GMP",
    "Asia/Tokyo": "HND NRT KIX ITM", "Australia/Sydney": "SYD", "Australia/Melbourne": "MEL",
    "Australia/Brisbane": "BNE", "Australia/Perth": "PER", "Pacific/Auckland": "AKL",
    "America/Sao_Paulo": "GRU CGH VCP GIG", "America/Argentina/Buenos_Aires": "EZE AEP", "America/Bogota": "BOG",
    "America/Lima": "LIM", "America/Santiago": "SCL", "Africa/Johannesburg": "JNB CPT", "Africa/Cairo": "CAI",
}
AIRPORT_TIME_ZONES = {code: zone for zone, codes in _ZONES.items() for code in codes.split()}


def utc_offset(code: str, on: date) -> timedelta:
    """UTC offset of an airport (or the first airport of a metro code such as NYC) on a given day."""
    code = code.upper()
    members = METRO_CODES.get(code, ("", ()))[1]
    zone = AIRPORT_TIME_ZONES.get(code) or next((AIRPORT_TIME_ZONES[m] for m in members if m in AIRPORT_TIME_ZONES), None)
    if zone:
        try:
            offset = ZoneInfo(zone).utcoffset(datetime.combine(on, time(12)))
        except ZoneInfoNotFoundError:
            offset = None  # no time-zone data on this computer: use the approximation below
        if offset is not None:
            return offset
    info = get_airport_index().get(code)
    return timedelta(hours=round(info.longitude / 15)) if info else timedelta(0)


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.lower().replace("-", " ").split())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


class AirportIndex:
    def __init__(self, airports: list[AirportInfo], sizes: dict[str, str]) -> None:
        self._by_code = {a.iata: a for a in airports}
        self._sizes = sizes
        self._airports = airports
        for code, (city, members) in METRO_CODES.items():
            first = next((self._by_code[m] for m in members if m in self._by_code), None)
            if first is not None and code not in self._by_code:
                self._by_code[code] = AirportInfo(
                    iata=code, name=f"{city} (all airports)", city=city, country=first.country,
                    latitude=first.latitude, longitude=first.longitude, kind="metro", members=list(members),
                )

    @classmethod
    def load(cls, csv_path: Path | None = None) -> AirportIndex:
        if csv_path is not None and csv_path.is_file():
            text = csv_path.read_text(encoding="utf-8")
        else:
            text = resources.files("travel_agent.data").joinpath("airports.csv").read_text(encoding="utf-8")
        airports: list[AirportInfo] = []
        sizes: dict[str, str] = {}
        for row in csv.DictReader(text.splitlines()):
            code = row["iata"].strip().upper()
            if len(code) != 3 or code in sizes:
                continue
            airports.append(AirportInfo(
                iata=code, name=row["name"], city=row["city"] or row["name"], country=row["country"],
                latitude=float(row["latitude"]), longitude=float(row["longitude"]),
            ))
            sizes[code] = row.get("type", "")
        return cls(airports, sizes)

    def get(self, code: str) -> AirportInfo | None:
        return self._by_code.get(code.upper())

    def members(self, code: str) -> set[str]:
        code = code.upper()
        if code in METRO_CODES:
            return {code, *METRO_CODES[code][1]}
        return {code}

    def matches(self, requested: str, actual: str) -> bool:
        return actual.upper() in self.members(requested)

    def city_of(self, code: str) -> str:
        info = self.get(code)
        return info.city if info else code

    def search(self, query: str, limit: int = 5) -> list[AirportInfo]:
        raw = query.strip()
        if len(raw) == 3 and raw.isalpha() and (exact := self.get(raw)) is not None:
            return [exact]
        needle = normalize_text(raw)
        if not needle:
            return []
        scored: list[tuple[int, str, AirportInfo]] = []
        for code, (city, _members) in METRO_CODES.items():
            if normalize_text(city) == needle and (metro := self.get(code)) is not None:
                scored.append((0, code, metro))
        for airport in self._airports:
            city = normalize_text(airport.city)
            name = normalize_text(airport.name)
            size_rank = 0 if self._sizes.get(airport.iata) == "large_airport" else 1
            if city == needle or city.split(" (")[0] == needle:
                rank = 1 + size_rank
            elif city.startswith(needle):
                rank = 3 + size_rank
            elif needle in name:
                rank = 5 + size_rank
            else:
                continue
            scored.append((rank, airport.iata, airport))
        scored.sort(key=lambda item: (item[0], item[1]))
        seen: set[str] = set()
        results: list[AirportInfo] = []
        for _, code, airport in scored:
            if code not in seen:
                seen.add(code)
                results.append(airport)
            if len(results) >= limit:
                break
        return results

    def nearby(self, code: str, radius_km: float = 150, limit: int = 5) -> list[AirportInfo]:
        center = self.get(code)
        if center is None:
            return []
        exclude = self.members(code)
        candidates: list[tuple[float, AirportInfo]] = []
        for airport in self._airports:
            if airport.iata in exclude or self._sizes.get(airport.iata) != "large_airport":
                continue
            distance = haversine_km(center.latitude, center.longitude, airport.latitude, airport.longitude)
            if distance <= radius_km:
                candidates.append((distance, airport))
        candidates.sort(key=lambda item: item[0])
        return [airport for _, airport in candidates[:limit]]


@lru_cache(maxsize=4)
def get_airport_index(csv_path: str | None = None) -> AirportIndex:
    return AirportIndex.load(Path(csv_path) if csv_path else None)
