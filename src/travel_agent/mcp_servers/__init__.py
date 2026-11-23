"""Our local MCP servers. Each runs as a stdio subprocess: `python -m travel_agent.mcp_servers.<module>`."""

LOCAL_SERVERS: dict[str, str] = {
    "flights": "travel_agent.mcp_servers.flights",
    "travel-sandbox": "travel_agent.mcp_servers.travel_sandbox",
    "trip-context": "travel_agent.mcp_servers.trip_context",
}
