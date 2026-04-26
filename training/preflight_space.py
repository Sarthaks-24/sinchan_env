# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
"""
HTTP-only preflight for a deployed OpenEnv server (e.g. Hugging Face Space).

Runs: GET /health, POST /reset, JSON-RPC on POST /mcp (session + tools/list,
tools/call new_episode, then get_scenario_info). No WebSocket.

Usage:
  py -3 training/preflight_space.py --base-url https://gladiator-codes-sinchan-env.hf.space

Exit code 0 if all required steps pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any
from urllib.parse import urljoin

import requests


def _json_rpc(
    session: requests.Session,
    mcp_url: str,
    method: str,
    params: dict[str, Any] | None = None,
    request_id: int = 1,
) -> dict[str, Any]:
    body = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": request_id,
    }
    r = session.post(mcp_url, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def run_preflight(base_url: str, *, deep: bool = True) -> int:
    """
    Return 0 on success, 1 on failure. Prints a line per step.
    """
    base = base_url.rstrip("/") + "/"
    health_url = urljoin(base, "health")
    reset_url = urljoin(base, "reset")
    mcp_url = urljoin(base, "mcp")

    session = requests.Session()
    req_id = 0

    def next_id() -> int:
        nonlocal req_id
        req_id += 1
        return req_id

    # 1) Health
    try:
        r = session.get(health_url, timeout=20)
        if r.status_code == 503:
            print(
                "[1/6] GET /health -> 503 (Space may be cold; wait 30–60s and retry) FAIL",
                file=sys.stderr,
            )
            return 1
        r.raise_for_status()
        if r.status_code != 200:
            print(f"[1/6] GET /health -> {r.status_code} FAIL", file=sys.stderr)
            return 1
        print(f"[1/6] GET /health -> 200 {r.text[:120]!r}")
    except Exception as e:
        print(f"[1/6] GET /health -> ERROR {e!r} FAIL", file=sys.stderr)
        return 1

    # 2) Reset
    try:
        r = session.post(reset_url, json={}, timeout=30)
        r.raise_for_status()
        print(f"[2/6] POST /reset -> {r.status_code}")
    except Exception as e:
        print(f"[2/6] POST /reset -> ERROR {e!r} FAIL", file=sys.stderr)
        return 1

    # 3) MCP session
    try:
        data = _json_rpc(
            session, mcp_url, "openenv/session/create", {}, request_id=next_id()
        )
        if "error" in data:
            print(
                f"[3/6] POST /mcp openenv/session/create -> {data['error']} FAIL",
                file=sys.stderr,
            )
            return 1
        session_id = (data.get("result") or {}).get("session_id")
        if not session_id:
            print(
                f"[3/6] POST /mcp openenv/session/create -> no session_id: {data!r} FAIL",
                file=sys.stderr,
            )
            return 1
        print(f"[3/6] POST /mcp session/create -> session_id={session_id!r}")
    except Exception as e:
        print(
            f"[3/6] POST /mcp openenv/session/create -> ERROR {e!r} FAIL",
            file=sys.stderr,
        )
        return 1

    # 4) tools/list
    try:
        data = _json_rpc(
            session,
            mcp_url,
            "tools/list",
            {"session_id": session_id},
            request_id=next_id(),
        )
        if "error" in data:
            print(
                f"[4/6] POST /mcp tools/list -> {data['error']} FAIL",
                file=sys.stderr,
            )
            return 1
        tools = (data.get("result") or {}).get("tools") or []
        names = [t.get("name", "") for t in tools if isinstance(t, dict)]
        print(f"[4/6] POST /mcp tools/list -> {len(names)} tools: {names[:5]}...")
    except Exception as e:
        print(f"[4/6] POST /mcp tools/list -> ERROR {e!r} FAIL", file=sys.stderr)
        return 1

    if not deep:
        print("Preflight OK (shallow: skipped tools/call).")
        return 0

    # 5) new_episode (loads a scenario for this MCP session; /reset alone does not)
    try:
        data = _json_rpc(
            session,
            mcp_url,
            "tools/call",
            {
                "session_id": session_id,
                "name": "new_episode",
                "arguments": {},
            },
            request_id=next_id(),
        )
        if "error" in data:
            print(
                f"[5/6] POST /mcp tools/call new_episode -> {data['error']} FAIL",
                file=sys.stderr,
            )
            return 1
        print(
            f"[5/6] POST /mcp tools/call new_episode -> ok keys={list((data.get('result') or {}).keys())}"
        )
    except Exception as e:
        print(
            f"[5/6] POST /mcp tools/call new_episode -> ERROR {e!r} FAIL",
            file=sys.stderr,
        )
        return 1

    # 6) get_scenario_info
    try:
        data = _json_rpc(
            session,
            mcp_url,
            "tools/call",
            {
                "session_id": session_id,
                "name": "get_scenario_info",
                "arguments": {},
            },
            request_id=next_id(),
        )
        if "error" in data:
            print(
                f"[6/6] POST /mcp tools/call get_scenario_info -> {data['error']} FAIL",
                file=sys.stderr,
            )
            return 1
        result = data.get("result") or {}
        title = (result.get("data") or {}).get("title") or (
            (result.get("structuredContent") or {}) if isinstance(result.get("structuredContent"), dict) else {}
        ).get("title")
        if not title:
            print(
                f"[6/6] POST /mcp tools/call get_scenario_info -> missing title: {data!r} FAIL",
                file=sys.stderr,
            )
            return 1
        print(
            f"[6/6] POST /mcp tools/call get_scenario_info -> title={title!r}"
        )
    except Exception as e:
        print(
            f"[6/6] POST /mcp tools/call get_scenario_info -> ERROR {e!r} FAIL",
            file=sys.stderr,
        )
        return 1

    print("Preflight OK (HTTP health + reset + MCP session + tools + new_episode + get_scenario).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP preflight: /health, /reset, /mcp (no WebSocket)."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Server base URL, e.g. https://username-spacename.hf.space",
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
        help="Stop after tools/list (skip get_scenario_info).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of full preflight attempts (use 2–4 if the Space is cold and returns 503). 15s pause between attempts.",
    )
    args = parser.parse_args()

    last = 1
    n = max(1, args.retries)
    for i in range(n):
        if i > 0:
            time.sleep(15)
            print(f"--- preflight attempt {i + 1}/{n} ---", file=sys.stderr)
        last = run_preflight(args.base_url, deep=not args.shallow)
        if last == 0:
            break
    raise SystemExit(last)


if __name__ == "__main__":
    main()
