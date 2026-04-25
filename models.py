# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
OpenEnv Type Models for ShinChan Life Simulator.

Defines the Action and Observation types for the environment.
These are used for type validation and client/server communication.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SinChanAction:
    """Action taken by the agent in the ShinChan environment."""

    action_name: str  # Name of the action to take (e.g., 'do_homework')
    reasoning: str  # Agent's thought process for choosing this action
    dialogue: str  # What Shin-chan says (should stay in character)


@dataclass
class SinChanObservation:
    """Observation returned by the environment after an action."""

    narrative: str  # Story continuation describing what happened
    character_reactions: str  # How NPCs reacted to the action
    available_actions: list[dict] = field(
        default_factory=list
    )  # List of actions Shin-chan can take next
    relationships: dict[str, float] = field(
        default_factory=dict
    )  # Current relationship meters with characters
    reward_breakdown: dict[str, float] = field(
        default_factory=dict
    )  # Detailed reward components
    done: bool = False  # Is the scenario complete?
    reward: float = 0.0  # Total reward for this step


@dataclass
class SinChanState:
    """Internal state of the environment."""

    episode_id: str  # Unique episode identifier
    scenario_id: str  # Current scenario ID
    step_count: int  # Number of steps taken so far
    max_steps: int  # Maximum steps allowed in this scenario
    category: str  # Scenario category (school, family, social, etc.)
    difficulty: int  # Difficulty level (1-5)
    relationships: dict[str, float] = field(
        default_factory=dict
    )  # Relationship state with all characters
