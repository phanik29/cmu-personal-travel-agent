"""`travel-agent` command: starts the local web chat at http://127.0.0.1:8000."""
from __future__ import annotations


def serve() -> None:
    import uvicorn

    from travel_agent.config import Settings
    from travel_agent.web.app import create_app

    settings = Settings.from_env()
    print(f"Personal Travel Agent: http://{settings.host}:{settings.port}  (sandbox mode: nothing is really booked or paid)")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    serve()
