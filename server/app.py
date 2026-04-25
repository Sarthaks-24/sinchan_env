# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
FastAPI application for the ShinChan Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    uv run --project . server
"""

import os

from fastapi import FastAPI

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .sinchan_environment import SinChanEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.sinchan_environment import SinChanEnvironment

# Enable OpenEnv web UI by default for local runs (/web route).
# Can still be overridden externally by setting ENABLE_WEB_INTERFACE.
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "32"))

app = create_app(
    SinChanEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="sinchan_env",
    max_concurrent_envs=max_concurrent,
)


def _has_route_path(fastapi_app: FastAPI, path: str) -> bool:
    """Return True when a route with exact path is already registered."""
    return any(getattr(route, "path", None) == path for route in fastapi_app.routes)


if not _has_route_path(app, "/health"):
    @app.get("/health")
    def health() -> dict[str, str]:
        """Readiness endpoint for Spaces/Colab probes."""
        return {"status": "ok"}

def main():
    """Entry point for direct execution."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
