"""Agent results and trace records."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    NO_RESULTS = "no_results"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class AgentResult(BaseModel):
    """Standard contract returned by every search agent; agents never fabricate options."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    status: AgentStatus
    options: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    error: str | None = None
    calls: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)

    def summary(self) -> str:
        if self.status in {AgentStatus.SUCCESS, AgentStatus.PARTIAL}:
            return f"{len(self.options)} options"
        if self.status == AgentStatus.NO_RESULTS:
            return "no results"
        if self.status == AgentStatus.SKIPPED:
            return self.notes[0] if self.notes else "skipped"
        return self.error or self.status.value


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trip_id: str | None = None
    run_id: str | None = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event: str
    node: str | None = None
    server: str | None = None
    tool: str | None = None
    status: str | None = None
    duration_ms: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
