"""Check the optional local LLM (Ollama) with the settings in .env (TRAVEL_AGENT_LLM_ENABLED / _LLM_MODEL).

    uv run python scripts/check_llm.py

1. Sends messages the keyword rules don't understand to the model and shows what it understood.
2. Runs one chat through the real workflow (mock travel data, sample weather) with the model on: plan, "sounds great",
   options (with the model's critic and explanation), and a free-form question (Q&A with read-only tools).
Everything runs on this computer; the model must be on localhost.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from travel_agent.config import Settings
from travel_agent.graph.nodes import classify_intent
from travel_agent.graph.runner import TripRunner
from travel_agent.inmemory import memory_gateway
from travel_agent.llm import LLMClient
from travel_agent.memory.store import MemoryStore

DEPART = date.today() + timedelta(days=40)
TRIP = {"origin": "SJC", "destination": "NYC", "departure_date": DEPART.isoformat(),
        "return_date": (DEPART + timedelta(days=5)).isoformat(), "travelers": 2, "budget_usd": 2500}
WAITING = "the traveler's go-ahead for the search plan"
# message, expected intent, expected trip details (subset)
PROBES = [
    ("sounds great, let's do it", "go_ahead", {}),
    ("nah, not yet", "decline", {}),
    ("my sister is coming with us too", "change_trip", {"travelers": 3}),
    ("can we fly back a day later?", "change_trip", {"return_date": (DEPART + timedelta(days=6)).isoformat()}),
    ("anything else you can find?", "more_options", {}),
    ("blorp", "unclear", {}),
]


def timed(start: float) -> str:
    return f"{(time.perf_counter() - start):.1f}s"


async def probe(llm: LLMClient) -> int:
    print("1. Messages the keyword rules don't understand")
    misses = 0
    for message, intent, expected in PROBES:
        if classify_intent(message, has_request=True, has_bookings=False)[1]:
            print(f"   (skipped: the rules already understand {message!r})")
            continue
        start = time.perf_counter()
        got = await llm.understand(message, TRIP, WAITING)
        fields = {k: v.isoformat() if isinstance(v, date) else v for k, v in ((got or {}).get("fields") or {}).items()}
        ok = got is not None and got["intent"] == intent and all(fields.get(k) == v for k, v in expected.items())
        misses += not ok
        print(f"   {'OK  ' if ok else 'MISS'} {message!r:36} -> {got and got['intent']} {json.dumps(fields)} "
              f"(expected {intent} {json.dumps(expected)}) {timed(start)}")
    return misses


async def chat(llm: LLMClient) -> None:
    print("\n2. One chat through the workflow with the model on")
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(data_dir=Path(tmp), weather_source="sample")
        store = MemoryStore(settings.memory_db)
        async with memory_gateway(settings, store) as gateway:
            runner = TripRunner(settings, gateway, store, llm, checkpointer=InMemorySaver())
            await runner.start()
            try:
                trip_id = store.create_trip()
                for text in (f"Plan a trip from SJC to NYC on {DEPART:%b %d} for 5 nights, 2 people, budget $2500, "
                             "balanced", "sounds great, let's do it", "How long is the flight there?"):
                    start = time.perf_counter()
                    reply = (await runner.collect(runner.send_message(trip_id, text)))[-1]
                    print(f"\n   You: {text}\n   ({reply.get('reply_kind') or reply.get('type')}, {timed(start)})")
                    print("   " + (reply.get("reply") or reply.get("message") or "").replace("\n", "\n   "))
            finally:
                await runner.close()


async def main() -> None:
    settings = Settings.from_env()
    llm = LLMClient.from_settings(settings)
    if llm is None:
        raise SystemExit("Turn the model on in .env first: TRAVEL_AGENT_LLM_ENABLED=true and TRAVEL_AGENT_LLM_MODEL=...")
    print(f"Model: {settings.llm_model} at {settings.llm_base_url}")
    start = time.perf_counter()
    if await llm._raw("Reply with the word OK.", "ping") is None:
        raise SystemExit("The model didn't answer. Is Ollama running (brew services start ollama) and was the model "
                         f"pulled (ollama pull {settings.llm_model})?")
    print(f"Model answered (first call, includes loading it into memory): {timed(start)}\n")
    misses = await probe(llm)
    await chat(llm)
    print(f"\nUnderstanding: {len(PROBES) - misses}/{len(PROBES)} as expected.")


if __name__ == "__main__":
    asyncio.run(main())
