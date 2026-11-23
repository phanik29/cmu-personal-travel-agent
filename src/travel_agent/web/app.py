"""Local web chat UI: FastAPI + one HTML page. Binds to 127.0.0.1 only (enforced by Settings)."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from travel_agent.config import Settings
from travel_agent.graph.runner import TripRunner
from travel_agent.llm import LLMClient
from travel_agent.memory.store import MemoryStore
from travel_agent.tool_gateway import ToolGateway, open_gateway

STATIC = Path(__file__).parent / "static"
GatewayFactory = Callable[[Settings, MemoryStore], AbstractAsyncContextManager[ToolGateway]]


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class DecisionIn(BaseModel):
    decision: Literal["approve", "change", "reject"]
    candidate_id: str | None = None
    feedback: str | None = Field(default=None, max_length=2000)
    kind: Literal["plan", "cancel", "start"] = "plan"


def default_gateway(settings: Settings, store: MemoryStore) -> AbstractAsyncContextManager[ToolGateway]:
    return open_gateway(settings, approval_verifier=store, trace_sink=store.add_trace)


async def ndjson(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        yield json.dumps(event, default=str) + "\n"


def create_app(settings: Settings | None = None, *, gateway_factory: GatewayFactory | None = None,
               checkpointer: Any = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = MemoryStore(settings.memory_db)
        async with AsyncExitStack() as stack:
            gateway = await stack.enter_async_context((gateway_factory or default_gateway)(settings, store))
            llm = LLMClient.from_settings(settings)
            runner = await stack.enter_async_context(TripRunner(settings, gateway, store, llm, checkpointer))
            app.state.store, app.state.gateway, app.state.runner, app.state.llm = store, gateway, runner, llm
            yield

    app = FastAPI(title="Personal Travel Agent", lifespan=lifespan)

    def services(request: Request) -> tuple[MemoryStore, TripRunner]:
        return request.app.state.store, request.app.state.runner

    def require_trip(store: MemoryStore, trip_id: str) -> dict[str, Any]:
        trip = store.get_trip(trip_id)
        if trip is None:
            raise HTTPException(404, "Trip not found")
        return trip

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def status(request: Request) -> dict[str, Any]:
        return {"flights": "Duffel test mode (search only)" if settings.duffel_access_token else "mock data",
                "hotels_cars": "mock data", "weather": "Open-Meteo.com" if settings.weather_source == "open-meteo"
                else "sample data",
                "llm": settings.llm_model if request.app.state.llm else None, "scenario": settings.scenario,
                "servers": {name: s["connected"] for name, s in request.app.state.gateway.status().items()}}

    @app.get("/api/trips")
    async def list_trips(request: Request) -> list[dict[str, Any]]:
        return services(request)[0].list_trips()

    @app.post("/api/trips")
    async def create_trip(request: Request) -> dict[str, str]:
        return {"trip_id": services(request)[0].create_trip()}

    @app.get("/api/trips/{trip_id}")
    async def get_trip(trip_id: str, request: Request) -> dict[str, Any]:
        store, runner = services(request)
        trip = require_trip(store, trip_id)
        values = await runner.values(trip_id)
        awaiting = ((await runner.pending_interrupt(trip_id)) or {}).get("type")
        bookings = []
        for booking in store.list_bookings(trip_id):
            item = store.get_item(trip_id, booking["option_id"])
            bookings.append({"booking_ref": booking["booking_ref"], "kind": booking["kind"], "status": booking["status"],
                             "total_price": booking["total_price"], "currency": booking["currency"],
                             "verified": bool(booking["verified_at"]), "label": item["label"] if item else booking["kind"]})
        return {
            "trip": trip, "messages": store.list_messages(trip_id), "awaiting": awaiting,
            "proposal": values.get("proposal") if awaiting == "approval" else None,
            "cancellation": values.get("cancellation") if awaiting == "cancel_approval" else None,
            "task_plan": values.get("task_plan"), "bookings": bookings, "plan_status": values.get("plan_status"),
        }

    @app.post("/api/trips/{trip_id}/messages")
    async def post_message(trip_id: str, body: MessageIn, request: Request) -> StreamingResponse:
        store, runner = services(request)
        require_trip(store, trip_id)
        return StreamingResponse(ndjson(runner.send_message(trip_id, body.text)), media_type="application/x-ndjson")

    @app.post("/api/trips/{trip_id}/decision")
    async def post_decision(trip_id: str, body: DecisionIn, request: Request) -> StreamingResponse:
        store, runner = services(request)
        require_trip(store, trip_id)
        decision = body.model_dump(exclude_none=True)
        if body.kind == "plan":
            decision.pop("kind")
        return StreamingResponse(ndjson(runner.decide(trip_id, decision)), media_type="application/x-ndjson")

    @app.get("/api/preferences")
    async def preferences(request: Request) -> dict[str, Any]:
        return services(request)[0].get_preferences()

    @app.delete("/api/preferences/{key}")
    async def delete_preference(key: str, request: Request) -> dict[str, bool]:
        return {"deleted": services(request)[0].delete_preference(key)}

    @app.get("/api/trips/{trip_id}/trace")
    async def trace(trip_id: str, request: Request) -> list[dict[str, Any]]:
        store = services(request)[0]
        require_trip(store, trip_id)
        return store.list_trace(trip_id, limit=300)

    return app
