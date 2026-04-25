# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
ShinChan Life Simulator — An OpenEnv MCP environment for RL training.

This environment simulates Shin-chan Nohara's daily life situations where
an LLM agent must make decisions while staying in character. Through RL
training, the agent learns to make better choices while maintaining
Shin-chan's unique personality.

Tools exposed via MCP:
- `choose_action(action_name, reasoning, dialogue)`: Make a decision as Shin-chan
- `get_scenario_info()`: Get information about the current scenario
- `get_relationships()`: View current relationship status with all characters

Example:
    >>> from sinchan_env import SinChanEnv
    >>>
    >>> with SinChanEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("get_scenario_info")
    ...     print(result)
    ...     result = env.call_tool("choose_action",
    ...         action_name="do_homework",
    ...         reasoning="Mom is angry and homework is due tomorrow",
    ...         dialogue="Buri buri~ fine, I'll do it... but only if there's Chocobi after!")
    ...     print(result)
"""

# Re-export MCP types for convenience
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import SinChanEnv

__all__ = ["SinChanEnv", "CallToolAction", "ListToolsAction"]
