# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
ShinChan Life Simulator Environment.

This is the main environment class that connects the MCP server to the scenario
engine, character simulation, and reward calculation.
"""

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

from .scenario_data import ALL_SCENARIOS
from .scenarios import ScenarioEngine
from .characters import get_character
from .reward_engine import evaluate_action


class SinChanEnvironment(MCPEnvironment):
    """
    ShinChan Life Simulator Environment.
    
    Exposes all interaction via FastMCP tools.
    """
    
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize the environment with MCP server and tools."""
        self.scenario_engine = ScenarioEngine(ALL_SCENARIOS)
        
        # State tracking (will be reset per episode)
        self.current_scenario = None
        self.relationships = {}
        self.action_history = []
        self.episode_log = []
        self.last_reward_info = {}
        
        mcp = FastMCP("sinchan_env")

        @mcp.tool
        def new_episode(seed: Optional[int] = None) -> dict:
            """
            Start a new Shin-chan life scenario in this HTTP/WebSocket session.

            Call this after openenv/session/create. (Plain POST /reset spins up a
            throwaway environment and does not load a scenario for MCP tools.)
            """
            obs = self.reset(seed=seed)
            meta = obs.metadata or {}
            return {
                "message": meta.get("message", "New scenario!"),
                "hint": meta.get("hint", "Call get_scenario_info() to see the situation."),
            }

        @mcp.tool
        def get_scenario_info() -> dict:
            """
            Get details about the current life situation Shin-chan is facing.
            Call this first in a new episode!
            """
            if not self.current_scenario:
                return {
                    "error": "No active scenario. Call new_episode() (or the gym reset) first.",
                }
            
            return {
                "title": self.current_scenario.title,
                "narrative": self.current_scenario.narrative,
                "available_actions": [
                    {"name": a["name"], "description": a["description"]} 
                    for a in self.current_scenario.available_actions
                ],
                "characters_involved": [
                    {"id": cid, "name": get_character(cid).name}
                    for cid in self.current_scenario.characters_involved if get_character(cid)
                ]
            }

        @mcp.tool
        def get_relationships() -> dict:
            """
            Check the current relationship status with all characters.
            0.0 = Hostile, 0.5 = Neutral, 1.0 = Best Friends.
            """
            if not self.relationships:
                return {"status": "No relationship data available."}
            
            return {
                char_id: {"name": get_character(char_id).name, "value": round(val, 2)}
                for char_id, val in self.relationships.items() if get_character(char_id)
            }

        @mcp.tool
        def choose_action(action_name: str, reasoning: str, dialogue: str) -> dict:
            """
            Make a decision as Shin-chan in the current scenario.
            
            Args:
                action_name: The EXACT name of the action from available_actions (e.g. 'do_homework')
                reasoning: Internal thought process (why choose this?)
                dialogue: What Shin-chan says out loud (stay in character!)
            """
            if not self.current_scenario:
                return {"error": "No active scenario."}
                
            action_def = self.current_scenario.get_action_by_name(action_name)
            if not action_def:
                return {"error": f"Invalid action_name '{action_name}'. Check get_scenario_info()."}
                
            # Process the action
            self.action_history.append(action_name)
            step_num = self._state.step_count
            
            # 1. Calculate Rewards
            reward_info = evaluate_action(
                scenario=self.current_scenario,
                step_num=step_num,
                action_name=action_name,
                reasoning=reasoning,
                dialogue=dialogue,
                action_history=self.action_history,
                dialogue_history=[entry.get("dialogue", "") for entry in self.episode_log],
            )
            self.last_reward_info = reward_info
            
            # 2. Get character reactions
            reactions = []
            for char_id in self.current_scenario.characters_involved:
                char = get_character(char_id)
                if char:
                    reaction_text, rel_change = char.get_reaction(action_name, action_def.get("tags", []))
                    if rel_change != 0.0:
                        self.relationships[char_id] = max(0.0, min(1.0, self.relationships.get(char_id, 0.5) + rel_change))
                    reactions.append(f"[{char.name}]: {reaction_text}")
            
            # Log it
            self.episode_log.append({
                "action": action_name,
                "dialogue": dialogue,
                "reactions": reactions
            })
            
            # Compute done state from projected step after this action.
            projected_steps = len(self.action_history)
            done = bool(
                self.current_scenario
                and projected_steps >= self.current_scenario.max_steps
            )

            # Construct response with explicit reward/done for HTTP MCP clients.
            response = {
                "shinchan_said": dialogue,
                "consequences": "\n".join(reactions) if reactions else "Nothing much happened.",
                "reward": float(reward_info.get("total", 0.0)),
                "reward_components": reward_info,
                "done": done,
            }
            
            if done:
                response["status"] = "Scenario complete! Call new_episode() for another chaos round."
            else:
                response["status"] = f"Step {len(self.action_history)}/{self.current_scenario.max_steps} complete. What's next?"
                
            return response

        # Pass MCP to base class
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._total_episodes = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment to a new scenario."""
        if seed is not None:
            import random
            random.seed(seed)
            
        self._total_episodes += 1
        
        # Select scenario based on progressive difficulty
        self.current_scenario = self.scenario_engine.get_scenarios_for_phase(self._total_episodes)
        
        # Reset state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self.action_history = []
        self.episode_log = []
        self.last_reward_info = {"total": 0.0}
        
        # Initialize relationships for this episode
        self.relationships = {}
        for char_id in self.current_scenario.characters_involved:
            char = get_character(char_id)
            if char:
                self.relationships[char_id] = char.initial_relationship

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "message": f"New Scenario Loaded: {self.current_scenario.title}",
                "hint": "Use call_tool('get_scenario_info') to start!"
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (we don't support any, so return error)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": "Use ListToolsAction or CallToolAction for MCP interactions."},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step via MCP tools."""
        # Process the tool call
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        
        # Update step count if an action was actually chosen
        if hasattr(action, 'tool_name') and action.tool_name == "choose_action":
            self._state.step_count += 1
            
            # Check done condition
            done = False
            if self.current_scenario and self._state.step_count >= self.current_scenario.max_steps:
                done = True
                
            # Inject custom reward info while preserving the original observation type.
            if hasattr(obs, "done"):
                obs.done = done
            if hasattr(obs, "reward"):
                obs.reward = self.last_reward_info.get("total", 0.0)
            if hasattr(obs, "metadata"):
                obs.metadata = self.last_reward_info
            return obs
            
        return obs

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        if hasattr(action, 'tool_name') and action.tool_name == "choose_action":
            self._state.step_count += 1
            done = bool(self.current_scenario and self._state.step_count >= self.current_scenario.max_steps)
            if hasattr(obs, "done"):
                obs.done = done
            if hasattr(obs, "reward"):
                obs.reward = self.last_reward_info.get("total", 0.0)
            if hasattr(obs, "metadata"):
                obs.metadata = self.last_reward_info
            return obs
        return obs

    @property
    def state(self) -> State:
        return self._state
