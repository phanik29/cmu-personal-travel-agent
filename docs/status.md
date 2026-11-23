# What works and what doesn't

Checked on November 22, 2026, on macOS (Apple M4 Max) with Python 3.13, Ollama 0.40.17, and Chrome.

## Works

- A clean install and a fresh copy of the repository both install, pass the evaluation, and book a trip.
- All 17 MCP tools match their pinned definitions.
- The evaluation passes all 5 scenarios ([evaluation-report.md](evaluation-report.md)).
- Duffel test mode: a San Francisco to New York search returned 5 offers, and the price recheck worked.
- The local model (`gemma4:12b`) understood all 6 test messages. With it on, options took 5.6 seconds.
- The full demo ran in Chrome with no errors ([sample-conversation.md](sample-conversation.md)). It found nonstop flights landing at 1:21 AM with late check-in hotels and a 24-hour rental desk, kept the rest of the trip when I asked for a different hotel, booked everything in the sandbox, and asked before cancelling.

## Doesn't work yet

- Nothing is really booked or paid. That's on purpose.
- Hotels, rental cars, and airport pickups are mock data, and Duffel's test prices aren't real.
- Flights that land after 5 AM the next morning are skipped. For landings after midnight, the pickup is still dated on the departure day.
- The first three options often share the same hotel.
- Only `gemma4:12b` was tested as the model, and only on macOS with Chrome.
- The automated evaluation uses mock data. Duffel, the model, and the browser were checked by hand, and there are no unit tests.

## Check it yourself

```bash
uv run --locked python -m travel_agent.eval
uv run --locked python scripts/check_duffel.py   # needs a Duffel test token
uv run --locked python scripts/check_llm.py      # needs Ollama and the model
```
