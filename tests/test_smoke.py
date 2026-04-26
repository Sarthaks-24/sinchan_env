import asyncio
import json

from fastapi.testclient import TestClient
from openenv.core.env_server.mcp_types import CallToolAction

from sinchan_env import CallToolEnv, SinChanEnv
from server.app import app
from server.scenario_data import ALL_SCENARIOS
from server.sinchan_environment import SinChanEnvironment


def _obs_result(obs):
    result = getattr(obs, "result", None)
    if isinstance(result, dict):
        return result

    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        text = getattr(content[0], "text", "")
        if isinstance(text, str) and text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}

    return {}


def test_scenario_count_minimum():
    assert len(ALL_SCENARIOS) >= 30


def test_health_endpoint_available():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") in {"ok", "healthy"}


def test_http_mcp_preflight_contract_matches_script():
    """Same JSON-RPC sequence as training/preflight_space.py (no WebSocket)."""
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.post("/reset", json={}).status_code == 200

    r1 = c.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "openenv/session/create", "params": {}, "id": 1},
    )
    assert r1.status_code == 200
    data1 = r1.json()
    assert "error" not in data1
    session_id = (data1.get("result") or {}).get("session_id")
    assert session_id

    r2 = c.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {"session_id": session_id},
            "id": 2,
        },
    )
    assert r2.status_code == 200
    data2 = r2.json()
    assert "error" not in data2
    tools = (data2.get("result") or {}).get("tools") or []
    names = [t.get("name") for t in tools if isinstance(t, dict)]
    assert "get_scenario_info" in names
    assert "new_episode" in names

    r_ne = c.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "session_id": session_id,
                "name": "new_episode",
                "arguments": {},
            },
            "id": 3,
        },
    )
    assert r_ne.status_code == 200
    assert "error" not in r_ne.json()

    r3 = c.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "session_id": session_id,
                "name": "get_scenario_info",
                "arguments": {},
            },
            "id": 4,
        },
    )
    assert r3.status_code == 200
    data3 = r3.json()
    assert "error" not in data3
    ginfo = (data3.get("result") or {}).get("data") or {}
    if not ginfo.get("title") and (data3.get("result") or {}).get("structuredContent"):
        ginfo = (data3.get("result") or {})["structuredContent"]
    assert ginfo.get("title"), f"get_scenario_info should return a title, got: {ginfo!r}"


def test_package_exports_calltoolenv_alias():
    assert CallToolEnv.__name__ == "CallToolEnv"
    assert issubclass(CallToolEnv, SinChanEnv)


def test_calltoolenv_http_mode_maps_step_to_call_tool():
    env = CallToolEnv(base_url="http://localhost:8000", prefer_http_mcp=True)

    async def fake_call_tool(name, **kwargs):
        return {"reward": 0.25, "done": True, "reward_components": {"total": 0.25}}

    env.call_tool = fake_call_tool  # type: ignore[method-assign]
    result = asyncio.run(
        env.step(CallToolAction(tool_name="choose_action", arguments={}))
    )

    assert result.reward == 0.25
    assert result.done is True
    assert result.observation.metadata.get("total") == 0.25


def test_reset_and_get_scenario_info():
    env = SinChanEnvironment()
    env.reset(seed=123)

    info_obs = env.step(CallToolAction(tool_name="get_scenario_info", arguments={}))
    info = _obs_result(info_obs)

    assert isinstance(info, dict)
    assert info.get("title")
    assert isinstance(info.get("available_actions"), list)
    assert len(info["available_actions"]) > 0


def test_choose_action_reward_and_episode_completion():
    env = SinChanEnvironment()
    env.reset(seed=456)

    info_obs = env.step(CallToolAction(tool_name="get_scenario_info", arguments={}))
    info = _obs_result(info_obs)
    action_name = info["available_actions"][0]["name"]

    done = False
    steps = 0

    while not done and steps < 10:
        step_obs = env.step(
            CallToolAction(
                tool_name="choose_action",
                arguments={
                    "action_name": action_name,
                    "reasoning": "I should think about tomorrow and family feelings.",
                    "dialogue": "Buri buri~ I will do my best!",
                },
            )
        )
        step_result = _obs_result(step_obs)

        assert -0.5 <= step_obs.reward <= 1.0
        assert isinstance(step_result.get("reward"), (int, float))
        assert isinstance(step_result.get("reward_components"), dict)
        assert isinstance(step_result.get("done"), bool)
        done = bool(step_obs.done)
        steps += 1

    assert done
    assert steps > 0
