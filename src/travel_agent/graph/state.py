"""Short-term memory: the LangGraph state for one trip (one checkpoint thread per trip).

Values are JSON-safe dicts and lists (not Pydantic objects) so checkpoints stay portable. Parallel agents write
to `options`, `agent_status`, `agent_runs`, and `context`, whose reducers merge concurrent updates.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

REPLACE = "__replace__"
RESET = {"__reset__": True}


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Merge by key; an update carrying {REPLACE: True} replaces the whole value."""
    if right is None:
        return dict(left or {})
    if right.get(REPLACE):
        return {key: value for key, value in right.items() if key != REPLACE}
    return {**(left or {}), **right}


def append_or_reset(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Append; an update starting with RESET clears the list first."""
    if not right:
        return list(left or [])
    if right[0] == RESET:
        return list(right[1:])
    return [*(left or []), *right]


def replace(value: dict[str, Any] | None = None) -> dict[str, Any]:
    return {REPLACE: True, **(value or {})}


class TravelState(TypedDict, total=False):
    # Input of the current run
    user_message: str
    intent: str
    # Intake
    fields: dict[str, Any]
    request: dict[str, Any] | None
    missing: list[str]
    assumptions: list[str]
    # Plan-and-Execute
    task_plan: dict[str, Any] | None
    pending: list[dict[str, Any]]
    tried: list[str]
    round: int
    planner_notes: list[str]
    locked: dict[str, dict[str, Any]]
    plan_status: str | None
    exclude_seen: bool
    seen_candidates: list[str]
    understood: dict[str, Any] | None
    change_part: str | None  # the part the traveler asked to change ("hotel"), from this message
    part: str | None  # the part this plan searches again; the rest of the top pick is kept
    kept: dict[str, dict[str, Any]]
    top_pick: dict[str, Any] | None  # the best option shown for this trip, with the request it was for
    seen_options: dict[str, list[str]]  # option ids shown so far, by kind
    wishes: list[str]  # things the traveler asked for that the agent can't control ("no_rain")
    # Written by parallel agents
    options: Annotated[dict[str, dict[str, Any]], merge_dicts]
    agent_status: Annotated[dict[str, dict[str, Any]], merge_dicts]
    agent_runs: Annotated[list[dict[str, Any]], append_or_reset]
    context: Annotated[dict[str, Any], merge_dicts]
    # Tree-of-Thought evaluation and the Planner's review
    tot: dict[str, Any] | None
    review: dict[str, Any] | None
    # Proposal, approval, execution
    proposal: dict[str, Any] | None
    decision: dict[str, Any] | None
    approvals: list[dict[str, Any]]
    execution: list[dict[str, Any]]
    cancellation: dict[str, Any] | None
    # Output of the current run
    reply: str
    reply_kind: str


class AgentTask(TypedDict):
    """Payload sent to one search agent for one round (LangGraph `Send`)."""

    agent: str
    directives: list[dict[str, Any]]
    request: dict[str, Any]
    round: int
