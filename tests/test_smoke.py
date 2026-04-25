import json

from fastapi.testclient import TestClient
from openenv.core.env_server.mcp_types import CallToolAction

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
