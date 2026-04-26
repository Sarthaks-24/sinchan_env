# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
"""Gradio UI: run the Shin-chan environment in-process (no extra HTTP hop)."""

from __future__ import annotations

import json
import random

import gradio as gr
from openenv.core.env_server.mcp_types import CallToolAction

from .sinchan_environment import SinChanEnvironment

# Single in-process environment for the Space (demo-style; not multi-tenant).
_UI_ENV: SinChanEnvironment | None = None


def _get_env() -> SinChanEnvironment:
    global _UI_ENV
    if _UI_ENV is None:
        _UI_ENV = SinChanEnvironment()
    return _UI_ENV


def _obs_result(obs) -> dict:
    """
    Unwrap OpenEnv / MCP `Observation` into a dict (same as tests/test_smoke.py).
    Tool outputs are often JSON text inside result.content[0], not a bare dict.
    """
    result = getattr(obs, "result", None)
    if isinstance(result, dict):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        text = getattr(content[0], "text", "")
        if isinstance(text, str) and text.strip():
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"_parse_error": "non-json tool payload", "raw": text[:2000]}
    return {}


def _fmt_state(info: dict) -> str:
    if not info:
        return "_No scenario data — click **New episode**._"
    if info.get("error"):
        return f"**Error:** {info.get('error')}"
    title = info.get("title") or "Scenario"
    narr = (info.get("narrative") or "").strip()
    people = info.get("characters_involved") or []
    ppl = ", ".join(
        f"{c.get('name', '?')}" for c in people if isinstance(c, dict)
    )
    parts = [f"## {title}", ""]
    if ppl:
        parts.append(f"**People involved:** {ppl}")
        parts.append("")
    parts.append(narr if narr else "_(no narrative text)_")
    parts.append("")
    parts.append("### Pick an action")
    for a in info.get("available_actions") or []:
        if not isinstance(a, dict):
            continue
        n = a.get("name", "?")
        d = a.get("description", "")
        parts.append(f"- **`{n}`** — {d}")
    return "\n".join(parts)


def _action_choices(info: dict) -> tuple[list[str], str | None]:
    actions = [a for a in (info.get("available_actions") or []) if isinstance(a, dict)]
    names = [a.get("name", "") for a in actions if a.get("name")]
    if not names:
        return [], None
    return names, names[0]


def make_demo() -> gr.Blocks:
    # Note: theme/css are passed at `launch()` when running standalone; here we use `mount_gradio_app`.
    with gr.Blocks(title="Shin-chan life simulator") as demo:
        gr.Markdown(
            "# Shin-chan life simulator\n"
            "Load a new episode, **choose an action** from the list (or random), and see "
            "**reward** + what happened. Same rules as the API / OpenEnv server."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                new_ep = gr.Button("New episode", variant="primary", size="lg")
                use_random = gr.Checkbox(
                    label="Random action (ignore choice below)", value=False
                )
                action_dd = gr.Dropdown(
                    label="Action",
                    choices=[],
                    value=None,
                    allow_custom_value=True,
                    info="List refreshes every time you start an episode or finish a non-terminal step.",
                )
                reason_in = gr.Textbox(
                    label="Reasoning (why this choice?)",
                    value="I want to be funny but not hurt anyone.",
                    lines=2,
                )
                line_in = gr.Textbox(
                    label="Dialogue (in character)",
                    value="Buri buri~ leave it to Shin-chan!",
                    lines=2,
                )
                go = gr.Button("Take this action", variant="primary")
            with gr.Column(scale=1, min_width=320):
                state_md = gr.Markdown("Click **New episode** to load a story.")
                out = gr.Json(label="Last response (raw JSON)")

        def on_new():
            env = _get_env()
            env.reset(seed=None)
            info_obs = env.step(
                CallToolAction(tool_name="get_scenario_info", arguments={})
            )
            info = _obs_result(info_obs)
            choices, value = _action_choices(info)
            j = {
                "ok": True,
                "get_scenario_info": info,
            }
            if not info or (not info.get("title") and not info.get("available_actions")):
                j["warning"] = (
                    "Parsed scenario is empty; if this persists, check server logs for MCP observation shape."
                )
            return (
                _fmt_state(info),
                gr.update(choices=choices, value=value),
                j,
            )

        def on_action(
            random_pick: bool, aname: str | None, rtxt: str, dtxt: str, current_state: str
        ):
            env = _get_env()
            info_obs = env.step(
                CallToolAction(tool_name="get_scenario_info", arguments={})
            )
            info = _obs_result(info_obs)
            actions = info.get("available_actions") or []
            if not actions:
                return {
                    "error": "No scenario loaded. Click **New episode** first.",
                }, current_state, gr.update()

            names = [a.get("name") for a in actions if isinstance(a, dict) and a.get("name")]
            if random_pick and names:
                aname = str(random.choice(names))
            aname = (aname or "").strip()
            if not aname:
                return {
                    "error": "Choose an action from the list or enable **Random action**.",
                }, current_state, gr.update()

            if names and aname not in names:
                return {
                    "error": f"Invalid action {aname!r}. Use one of: {names}",
                }, current_state, gr.update()

            step_obs = env.step(
                CallToolAction(
                    tool_name="choose_action",
                    arguments={
                        "action_name": aname,
                        "reasoning": rtxt or "—",
                        "dialogue": dtxt or "—",
                    },
                )
            )
            res = _obs_result(step_obs)
            payload = {
                "action_name": aname,
                "reward": float(getattr(step_obs, "reward", 0.0) or 0.0),
                "done": bool(getattr(step_obs, "done", False)),
                "result": res,
            }

            if getattr(step_obs, "done", False):
                return (
                    payload,
                    _fmt_state(info)
                    + "\n\n---\n\n**Episode finished.** Start a **New episode** to play again.",
                    gr.update(),
                )

            next_obs = env.step(
                CallToolAction(tool_name="get_scenario_info", arguments={})
            )
            ninfo = _obs_result(next_obs)
            choices, value = _action_choices(ninfo)
            return (
                payload,
                _fmt_state(ninfo),
                gr.update(choices=choices, value=value),
            )

        new_ep.click(
            on_new,
            outputs=[state_md, action_dd, out],
        )
        go.click(
            on_action,
            inputs=[use_random, action_dd, reason_in, line_in, state_md],
            outputs=[out, state_md, action_dd],
        )

    return demo
