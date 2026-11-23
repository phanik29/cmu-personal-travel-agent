"""Application settings loaded from environment variables and an optional `.env` file."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from travel_agent.security import redact_secrets

ENV_PREFIX = "TRAVEL_AGENT_"
SECRET_FIELDS = frozenset({"duffel_access_token", "llm_api_key"})
SCENARIOS = frozenset({
    "happy",
    "over_budget",
    "late_arrival",
    "price_change",
    "sold_out",
    "flight_error",
    "hotel_error",
    "car_error",
    "flight_timeout",
    "no_flights",
    "no_hotels",
    "no_cars",
    "bad_weather",
})
WEATHER_SOURCES = frozenset({"sample", "open-meteo"})

# Settings whose environment names don't follow the TRAVEL_AGENT_<FIELD> pattern.
ENV_ALIASES = {
    "duffel_access_token": ("DUFFEL_ACCESS_TOKEN",),
}


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines; values in the real environment take precedence."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    host: str = "127.0.0.1"
    port: int = 8000

    scenario: str = "happy"
    simulated_delay_s: float = 0.0

    beam_width: int = 3
    max_options_per_search: int = 10
    max_tot_stages: int = 5
    max_refinement_rounds: int = 3
    max_tool_retries: int = 2
    tool_timeout_s: float = 45.0
    max_tool_calls_per_run: int = 80
    max_qa_tool_calls: int = 4
    max_llm_tokens_per_run: int = 20000

    weather_source: str = "open-meteo"

    llm_enabled: bool = False
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_allow_remote: bool = False

    duffel_access_token: str | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        dotenv_path: Path | None = Path(".env"),
    ) -> Settings:
        merged: dict[str, str] = load_dotenv(dotenv_path) if dotenv_path else {}
        merged.update(os.environ if env is None else env)
        values: dict[str, Any] = {}
        for field in fields(cls):
            names = (ENV_PREFIX + field.name.upper(), *ENV_ALIASES.get(field.name, ()))
            raw = next((merged[name] for name in names if name in merged), None)
            if raw is not None:
                values[field.name] = _coerce(field.type, raw)
        settings = cls(**values)
        settings.validate()
        return settings

    def with_overrides(self, **overrides: Any) -> Settings:
        settings = replace(self, **overrides)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not is_loopback_host(self.host):
            raise ValueError("The web app must bind to a loopback address such as 127.0.0.1")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {self.scenario!r}; expected one of {sorted(SCENARIOS)}")
        if self.weather_source not in WEATHER_SOURCES:
            raise ValueError(f"Unknown weather source {self.weather_source!r}")
        if not 2 <= self.beam_width <= 5:
            raise ValueError("beam_width must be between 2 and 5")
        if not 1 <= self.max_options_per_search <= 10:
            raise ValueError("max_options_per_search must be between 1 and 10")
        if not 1 <= self.max_tot_stages <= 5:
            raise ValueError("max_tot_stages must be between 1 and 5")
        if not 0 <= self.max_refinement_rounds <= 3:
            raise ValueError("max_refinement_rounds must be between 0 and 3")
        if not 0 <= self.max_tool_retries <= 2:
            raise ValueError("max_tool_retries must be between 0 and 2")
        if not 0 < self.tool_timeout_s <= 45:
            raise ValueError("tool_timeout_s must be greater than 0 and at most 45 seconds")
        if not 1 <= self.max_qa_tool_calls <= 4:
            raise ValueError("max_qa_tool_calls must be between 1 and 4")
        if self.llm_enabled:
            if not self.llm_model:
                raise ValueError("Set TRAVEL_AGENT_LLM_MODEL to a local model (for example gemma4:12b) to enable the LLM")
            host = urlparse(self.llm_base_url).hostname or ""
            if not self.llm_allow_remote and not is_loopback_host(host):
                raise ValueError("The LLM endpoint must run on localhost (local Ollama)")
        if self.duffel_access_token and not self.duffel_access_token.startswith("duffel_test_"):
            raise ValueError("Only Duffel test-mode tokens (duffel_test_...) are allowed")

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "travel_agent.db"

    @property
    def sandbox_db(self) -> Path:
        return self.data_dir / "sandbox_ledger.db"

    @property
    def airports_csv(self) -> Path:
        return self.data_dir / "airports.csv"

    @property
    def flight_provider(self) -> str:
        return "duffel-test" if self.duffel_access_token else "mock"

    def to_safe_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for key, value in raw.items():
            if isinstance(value, Path):
                raw[key] = str(value)
            elif isinstance(value, tuple):
                raw[key] = list(value)
        return redact_secrets(raw, extra_keys=SECRET_FIELDS)


def _coerce(annotation: Any, raw: str) -> Any:
    type_name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    if "None" in type_name and raw.strip() == "":
        return None
    if "Path" in type_name:
        return Path(raw)
    if type_name.startswith("bool"):
        return _parse_bool(raw)
    if type_name.startswith("int"):
        return int(raw)
    if type_name.startswith("float"):
        return float(raw)
    if type_name.startswith("tuple"):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return raw
