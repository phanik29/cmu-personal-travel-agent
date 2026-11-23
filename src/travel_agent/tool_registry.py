"""Pinned MCP tool definitions. A tool whose name, description, or input schema changes is disabled until reviewed.

Review after changing a server (developer command):
    uv run python -m travel_agent.tool_registry                 # show differences
    uv run python -m travel_agent.tool_registry --approve flights --approve trip-context
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from mcp import types

REGISTRY_PATH = Path(__file__).with_name("mcp_tool_registry.json")


def fingerprint(tool: types.Tool) -> str:
    payload = {"name": tool.name, "description": tool.description or "", "inputSchema": tool.inputSchema}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {server: dict(tools) for server, tools in data.get("servers", {}).items()}


def save_registry(registry: dict[str, dict[str, str]], path: Path = REGISTRY_PATH) -> None:
    ordered = {server: dict(sorted(tools.items())) for server, tools in sorted(registry.items())}
    path.write_text(json.dumps({"version": 1, "servers": ordered}, indent=2) + "\n", encoding="utf-8")


def diff_tools(registry: dict[str, dict[str, str]], server: str, tools: list[types.Tool]) -> list[dict[str, Any]]:
    known = registry.get(server, {})
    rows = []
    seen = set()
    for tool in tools:
        seen.add(tool.name)
        digest = fingerprint(tool)
        status = "approved" if known.get(tool.name) == digest else ("changed" if tool.name in known else "new")
        rows.append({"server": server, "tool": tool.name, "status": status, "fingerprint": digest,
                     "description": (tool.description or "").strip().splitlines()[0:1]})
    rows += [{"server": server, "tool": name, "status": "missing", "fingerprint": None, "description": []}
             for name in sorted(set(known) - seen)]
    return rows


async def review(approve: list[str], path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    from travel_agent.config import Settings
    from travel_agent.tool_gateway import ToolGateway

    settings = Settings.from_env()
    registry = load_registry(path)
    rows: list[dict[str, Any]] = []
    async with ToolGateway(settings, registry=registry) as gateway:
        for server, tools in gateway.listed_tools().items():
            rows += diff_tools(registry, server, tools)
            if server in approve:
                registry[server] = {tool.name: fingerprint(tool) for tool in tools}
    if approve:
        save_registry(registry, path)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approve", action="append", default=[], metavar="SERVER",
                        help="Pin the current tool definitions of SERVER after reviewing them")
    args = parser.parse_args(argv)
    for row in asyncio.run(review(args.approve)):
        description = row["description"][0] if row["description"] else ""
        print(f"{row['status']:9} {row['server']:15} {row['tool']:22} {description}")
    if args.approve:
        print(f"Updated {REGISTRY_PATH.name} for: {', '.join(args.approve)}")


if __name__ == "__main__":
    main()
