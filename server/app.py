# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
FastAPI application for the ShinChan Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    uv run --project . server
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .sinchan_environment import SinChanEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.sinchan_environment import SinChanEnvironment

max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "32"))

app = create_app(
    SinChanEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="sinchan_env",
    max_concurrent_envs=max_concurrent,
)

def main():
    """Entry point for direct execution."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
