# Evaluation report, 2026-11-22

5 scenarios run through the real LangGraph workflow with in-memory MCP servers, mock travel data, and sample weather. Each one sends a chat message, approves the plan, and approves the top option if there is one.

5 of 5 scenarios passed.

| Scenario | Expected | Result | Passed | Time after plan approval | Tool calls |
|---|---|---|---|---|---|
| Happy path | booked | booked | yes | 171 ms | 16 |
| Over budget, then a cheaper search | booked | booked | yes | 323 ms | 20 |
| Late arrival, then late check-in and pickup | booked | booked | yes | 362 ms | 22 |
| Price change after approval | stopped | stopped | yes | 146 ms | 9 |
| Hotel search outage | asks | asks | yes | 114 ms | 7 |

| Metric | Result | Target |
|---|---|---|
| Groundedness (proposed items that came from a tool result) | 48/48 | all |
| Correctness (plans within budget and party size) | 12/12 | all |
| Date accuracy (flight, hotel, car, and pickup dates that match the trip) | 84/84 | all |
| Bookings without an approval | 0 | 0 |
| Searches before the plan was approved | 0 | 0 |
| Personal data in messages and traces | 0 | 0 |
| Latency after plan approval (median / max) | 171 / 362 ms | none set |
| Tool calls / booking actions | 74 / 13 | within limits |

Mock data only (no live Duffel or Open-Meteo calls), with the model off.
