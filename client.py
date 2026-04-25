# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
ShinChan Life Simulator Client.

This module provides the client for connecting to a ShinChan Environment server.
SinChanEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with SinChanEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...
    ...     # Get scenario info
    ...     info = env.call_tool("get_scenario_info")
    ...     print(info)
    ...
    ...     # Make a decision
    ...     result = env.call_tool("choose_action",
    ...         action_name="do_homework",
    ...         reasoning="Homework is due tomorrow and Mom is angry",
    ...         dialogue="Buri buri~ fine, I'll do it!")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class SinChanEnv(MCPToolClient):
    """
    Client for the ShinChan Life Simulator Environment.

    This client provides a simple interface for interacting with the ShinChan
    Environment via MCP tools. It inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)

    Available Tools:
    - `choose_action(action_name, reasoning, dialogue)`: Make a decision
    - `get_scenario_info()`: Get current scenario details
    - `get_relationships()`: View relationship meters

    Example with sync API:
        >>> with SinChanEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...     tools = env.list_tools()
        ...     result = env.call_tool("choose_action",
        ...         action_name="do_homework",
        ...         reasoning="Mom is angry",
        ...         dialogue="Fine, I'll do it!")

    Example with HuggingFace Space:
        >>> env = SinChanEnv(base_url="https://YOUR-SPACE.hf.space")
        >>> try:
        ...     env.reset()
        ...     result = env.call_tool("get_scenario_info")
        ... finally:
        ...     env.close()
    """

    pass  # MCPToolClient provides all needed functionality
