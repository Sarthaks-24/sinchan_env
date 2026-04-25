# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
Scenario definitions and engine for the ShinChan Life Simulator.
"""

from dataclasses import dataclass, field
import random


@dataclass
class Scenario:
    """A single life scenario for Shin-chan to navigate."""
    
    id: str                        # e.g., "school_homework_001"
    title: str                     # e.g., "The Homework Dilemma"
    category: str                  # school, family, social, temptation, responsibility
    difficulty: int                # 1 to 5
    narrative: str                 # The setup story
    characters_involved: list[str] # IDs of characters like "misae", "kazama"
    available_actions: list[dict]  # List of dicts: {"name": "action_id", "description": "text", "tags": [...]}
    optimal_path: list[str]        # Expected best sequence of action names
    max_steps: int                 # How many steps the scenario lasts
    personality_context: str       # How Shin-chan typically acts here
    stakeholder_impacts: dict      # How different actions generally affect characters
    
    def get_action_by_name(self, action_name: str) -> dict | None:
        """Helper to find an action definition by its name."""
        for action in self.available_actions:
            if action["name"] == action_name:
                return action
        return None


class ScenarioEngine:
    """Engine to manage and load scenarios."""
    
    def __init__(self, scenarios: list[Scenario]):
        self.scenarios = {s.id: s for s in scenarios}
        self.scenarios_by_difficulty = {}
        for s in scenarios:
            if s.difficulty not in self.scenarios_by_difficulty:
                self.scenarios_by_difficulty[s.difficulty] = []
            self.scenarios_by_difficulty[s.difficulty].append(s)

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        """Get a specific scenario by ID."""
        return self.scenarios.get(scenario_id)
        
    def get_random_scenario(self, max_difficulty: int = 5) -> Scenario:
        """Get a random scenario up to a certain difficulty."""
        valid_difficulties = [d for d in self.scenarios_by_difficulty.keys() if d <= max_difficulty]
        if not valid_difficulties:
            # Fallback if no easy scenarios exist (shouldn't happen)
            return random.choice(list(self.scenarios.values()))
            
        chosen_diff = random.choice(valid_difficulties)
        return random.choice(self.scenarios_by_difficulty[chosen_diff])

    def get_scenarios_for_phase(self, step_count: int) -> Scenario:
        """
        Graduated difficulty curve based on training steps.
        Phase 1 (Steps 0-100): Easy (1-2)
        Phase 2 (Steps 100-250): Medium (1-3)
        Phase 3 (Steps 250-400): Hard (2-4)
        Phase 4 (Steps 400+): All (1-5)
        """
        if step_count < 100:
            return self.get_random_scenario(max_difficulty=2)
        elif step_count < 250:
            return self.get_random_scenario(max_difficulty=3)
        elif step_count < 400:
            # Try to pick from 2-4
            valid_scenarios = [s for s in self.scenarios.values() if 2 <= s.difficulty <= 4]
            return random.choice(valid_scenarios) if valid_scenarios else self.get_random_scenario(4)
        else:
            return self.get_random_scenario(max_difficulty=5)
