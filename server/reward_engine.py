# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
Reward Engine for ShinChan Life Simulator.

Calculates multi-dimensional rewards based on:
1. Decision Quality (did they pick the optimal actions?)
2. Social Awareness (did they consider others' feelings?)
3. Personality Preservation (did they stay in character?)
4. Responsibility
5. Relationship Impact
6. Long-Term Thinking
7. Creativity
"""

from .scenarios import Scenario
from .characters import get_character

# Common Shin-chan catchphrases and keywords
SHINCHAN_KEYWORDS = [
    "buri buri", "action kamen", "chocobi", "ore wa", "ora",
    "pretty lady", "miss yoshinaga", "butt", "dance", "hey hey",
    "wahaha", "shinnosuke", "himawari", "shiro"
]

def evaluate_action(
    scenario: Scenario,
    step_num: int,
    action_name: str,
    reasoning: str,
    dialogue: str,
    action_history: list[str],
    dialogue_history: list[str] | None = None,
) -> dict[str, float]:
    """
    Evaluates a single action and returns reward components.
    
    Returns:
        dict containing individual scores and the 'total' reward.
    """
    scores = {
        "decision_quality": 0.0,
        "social_awareness": 0.0,
        "reasoning_quality": 0.0,
        "personality": 0.0,
        "responsibility": 0.0,
        "relationship_impact": 0.0,
        "long_term_thinking": 0.0,
        "creativity": 0.0,
        "repetition_penalty": 0.0,
        "reasoning_penalty": 0.0,
        "dialogue_repetition_penalty": 0.0,
        "total": 0.0
    }
    
    action_def = scenario.get_action_by_name(action_name)
    if not action_def:
        # Invalid action penalty
        scores["total"] = -0.5
        return scores
        
    tags = action_def.get("tags", [])
    
    # 1. Decision Quality (0.0 to 1.0)
    # Check if action aligns with the optimal path
    is_optimal = False
    if step_num < len(scenario.optimal_path):
        expected_action = scenario.optimal_path[step_num]
        if action_name == expected_action:
            scores["decision_quality"] = 1.0
            is_optimal = True
        elif action_name in scenario.optimal_path:
            scores["decision_quality"] = 0.5  # Right idea, wrong order
    
    # Penalize clearly bad tags if not optimal
    if not is_optimal:
        bad_tags = ["selfish", "mean", "bullying", "dangerous", "ignoring_duties", "stealing"]
        good_tags = ["responsible", "sharing", "honest", "brave", "caring"]
        
        if any(tag in bad_tags for tag in tags):
            scores["decision_quality"] -= 0.3
        if any(tag in good_tags for tag in tags):
            scores["decision_quality"] += 0.4
            
    # Ensure bounds
    scores["decision_quality"] = max(0.0, min(1.0, scores["decision_quality"]))

    # 2. Social Awareness (0.0 to 1.0)
    # Does the reasoning mention the characters involved?
    reasoning_lower = reasoning.lower()
    mentioned_chars = 0
    for char_id in scenario.characters_involved:
        char = get_character(char_id)
        if char:
            # Check for name or roles (mom, dad)
            if char.name.lower().split()[0] in reasoning_lower or char_id in reasoning_lower:
                mentioned_chars += 1
            elif char_id == "misae" and "mom" in reasoning_lower:
                mentioned_chars += 1
            elif char_id == "hiroshi" and "dad" in reasoning_lower:
                mentioned_chars += 1
                
    if scenario.characters_involved:
        scores["social_awareness"] = min(1.0, mentioned_chars / len(scenario.characters_involved))
        # Bonus if they mention feelings or consequences
        if any(word in reasoning_lower for word in ["sad", "angry", "happy", "cry", "feel", "because", "would"]):
            scores["social_awareness"] = min(1.0, scores["social_awareness"] + 0.3)
        if any(word in reasoning_lower for word in ["if", "then", "so that", "consequence", "later", "tomorrow"]):
            scores["social_awareness"] = min(1.0, scores["social_awareness"] + 0.1)
    else:
        scores["social_awareness"] = 0.5  # Default if no chars involved

    # 3. Reasoning Quality (0.0 to 1.0)
    # Reward non-generic, consequence-aware explanations.
    reasoning_stripped = reasoning.strip()
    generic_patterns = [
        "i should be good",
        "this is best",
        "because it is good",
        "it's the right thing",
        "just because",
    ]
    causal_markers = ["because", "if", "then", "so that", "otherwise", "consequence", "later", "tomorrow"]
    if len(reasoning_stripped) >= 20:
        scores["reasoning_quality"] += 0.4
    if len(reasoning_stripped.split()) >= 8:
        scores["reasoning_quality"] += 0.2
    if any(marker in reasoning_lower for marker in causal_markers):
        scores["reasoning_quality"] += 0.3
    if mentioned_chars > 0:
        scores["reasoning_quality"] += 0.2
    scores["reasoning_quality"] = max(0.0, min(1.0, scores["reasoning_quality"]))

    # Penalize empty/generic reasoning to reduce reward hacking.
    if len(reasoning_stripped) < 12:
        scores["reasoning_penalty"] = -0.12
    elif any(pat in reasoning_lower for pat in generic_patterns):
        scores["reasoning_penalty"] = -0.08

    # 4. Personality Preservation (0.0 to 1.0)
    # Does the dialogue sound like Shin-chan?
    dialogue_lower = dialogue.lower()
    keyword_count = sum(1 for kw in SHINCHAN_KEYWORDS if kw in dialogue_lower)
    
    if keyword_count >= 2:
        scores["personality"] = 1.0
    elif keyword_count == 1:
        scores["personality"] = 0.7
    else:
        # Check for informal/childish language style vs robotic AI
        if "I will now proceed to" in dialogue or "As an AI" in dialogue:
            scores["personality"] = 0.0
        elif len(dialogue) < 10: # Too short
            scores["personality"] = 0.3
        else:
            scores["personality"] = 0.5

    # 5. Responsibility (0.0 to 1.0)
    if any(tag in tags for tag in ["responsible", "doing_homework", "caring_for_himawari", "caring_for_shiro", "studying"]):
        scores["responsibility"] += 0.8
    if any(tag in tags for tag in ["ignoring_duties", "lazy", "disobedient", "selfish"]):
        scores["responsibility"] -= 0.4
    scores["responsibility"] = max(0.0, min(1.0, scores["responsibility"]))

    # 6. Relationship Impact (-0.5 to 0.5)
    scenario_impact = scenario.stakeholder_impacts.get(action_name, {}) if scenario.stakeholder_impacts else {}
    if scenario_impact:
        avg_impact = sum(scenario_impact.values()) / max(len(scenario_impact), 1)
        scores["relationship_impact"] = max(-0.5, min(0.5, avg_impact))
    else:
        if any(tag in tags for tag in ["sharing", "helpful", "teamwork", "caring"]):
            scores["relationship_impact"] += 0.15
        if any(tag in tags for tag in ["mean", "bullying", "lying", "stealing", "selfish"]):
            scores["relationship_impact"] -= 0.15
        scores["relationship_impact"] = max(-0.5, min(0.5, scores["relationship_impact"]))

    # 7. Long-Term Thinking (0.0 to 0.5)
    if any(word in reasoning_lower for word in ["tomorrow", "later", "future", "consequence", "if", "so that"]):
        scores["long_term_thinking"] = 0.5
    elif any(word in reasoning_lower for word in ["next", "after", "then"]):
        scores["long_term_thinking"] = 0.3

    # 8. Creativity (0.0 to 0.5)
    if any(tag in tags for tag in ["funny", "distracting", "peacemaker", "smart"]):
        scores["creativity"] = 0.4
    if action_name not in scenario.optimal_path and scores["decision_quality"] > 0.3:
        scores["creativity"] = min(0.5, scores["creativity"] + 0.1)

    # Anti-gaming: repeated identical action choice over recent history.
    if len(action_history) >= 3 and len(set(action_history[-3:])) == 1:
        scores["repetition_penalty"] = -0.15

    # Anti-gaming: repeated template dialogue over recent turns.
    if dialogue_history:
        normalized = dialogue_lower.strip()
        recent_normalized = [d.lower().strip() for d in dialogue_history[-2:] if isinstance(d, str)]
        if normalized and normalized in recent_normalized:
            scores["dialogue_repetition_penalty"] = -0.10
    if any(template in dialogue_lower for template in ["i will do my best", "okay i will do it", "fine i will do it"]):
        scores["dialogue_repetition_penalty"] = min(
            -0.05, scores["dialogue_repetition_penalty"]
        )
            
    # Combine scores with weights
    weights = {
        "decision_quality": 0.28,
        "social_awareness": 0.19,
        "reasoning_quality": 0.09,
        "personality": 0.15,
        "responsibility": 0.15,
        "creativity": 0.08,
        "long_term_thinking": 0.06,
        "relationship_impact": 0.05,
    }
    
    scores["total"] = (
        scores["decision_quality"] * weights["decision_quality"] +
        scores["social_awareness"] * weights["social_awareness"] +
        scores["reasoning_quality"] * weights["reasoning_quality"] +
        scores["personality"] * weights["personality"] +
        scores["responsibility"] * weights["responsibility"] +
        scores["creativity"] * weights["creativity"] +
        scores["long_term_thinking"] * weights["long_term_thinking"] +
        scores["relationship_impact"] * weights["relationship_impact"] +
        scores["repetition_penalty"] +
        scores["reasoning_penalty"] +
        scores["dialogue_repetition_penalty"]
    )

    scores["total"] = max(-0.5, min(1.0, scores["total"]))
    
    return scores
