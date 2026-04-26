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
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from gradio import mount_gradio_app

from .gradio_ui import make_demo

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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BLOG_FILE = None
for _candidate in (
    os.path.join(_REPO_ROOT, "blog.md"),
    os.path.join(os.path.dirname(__file__), "static", "blog.md"),
):
    if os.path.isfile(_candidate):
        _BLOG_FILE = _candidate
        break
if _BLOG_FILE is not None and not _has_route_path(app, "/blog.md"):

    @app.get("/blog.md")
    def blog_markdown() -> FileResponse:
        """Hackathon mini-blog (link this URL from the README)."""
        return FileResponse(_BLOG_FILE, media_type="text/markdown; charset=utf-8")

_SINCHAN_UI_DIR = os.path.join(os.path.dirname(__file__), "static", "sinchan")
if os.path.isdir(_SINCHAN_UI_DIR):
    app.mount(
        "/sinchan-ui",
        StaticFiles(directory=_SINCHAN_UI_DIR, html=True),
        name="sinchan_ui",
    )

    if not _has_route_path(app, "/play"):
        @app.get("/play", tags=["ShinChan UI"])
        def play_lobby() -> RedirectResponse:
            """Crayon-style browser UI. The OpenEnv lab remains at /web when enabled."""
            return RedirectResponse(url="/sinchan-ui/", status_code=302)

# Gradio: simple state / action / reward UI (OpenEnv + REST remain on other routes)
if not _has_route_path(app, "/gradio"):
    mount_gradio_app(app, make_demo(), path="/gradio")

def main():
    """Entry point for direct execution."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
