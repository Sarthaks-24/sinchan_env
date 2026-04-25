# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
NPC Character System for ShinChan Life Simulator.

Contains personality profiles and reaction logic for all characters
in Shin-chan's world. Each character has defined personality traits,
triggers, and reaction patterns that determine how they respond to
the agent's actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Character:
    """A character in Shin-chan's world with personality and reaction logic."""

    id: str
    name: str
    relationship_to_shinchan: str
    personality: str
    anger_triggers: list[str] = field(default_factory=list)
    happy_triggers: list[str] = field(default_factory=list)
    reaction_style: str = ""
    catchphrases: list[str] = field(default_factory=list)
    initial_relationship: float = 0.5  # 0.0 = hostile, 1.0 = best friends

    def get_reaction(self, action_name: str, action_tags: list[str]) -> tuple[str, float]:
        """
        Generate a reaction to Shin-chan's action.

        Returns:
            Tuple of (reaction_text, relationship_change)
        """
        # Check for anger triggers
        for tag in action_tags:
            if tag in self.anger_triggers:
                return self._angry_reaction(action_name), -0.15

        # Check for happy triggers
        for tag in action_tags:
            if tag in self.happy_triggers:
                return self._happy_reaction(action_name), 0.10

        # Neutral reaction
        return self._neutral_reaction(action_name), 0.0

    def _angry_reaction(self, action_name: str) -> str:
        return f"{self.name} is upset! {self.reaction_style}"

    def _happy_reaction(self, action_name: str) -> str:
        return f"{self.name} is pleased! They appreciate Shin-chan's effort."

    def _neutral_reaction(self, action_name: str) -> str:
        return f"{self.name} watches quietly."


# ============================================================================
# CHARACTER DATABASE
# ============================================================================

CHARACTERS: dict[str, Character] = {
    "misae": Character(
        id="misae",
        name="Misae Nohara (Mom)",
        relationship_to_shinchan="Mother",
        personality=(
            "Short-tempered but deeply loving. Wants Shin-chan to be responsible "
            "and well-behaved. Quick to anger but also quick to forgive. She works "
            "hard to manage the household and worries about Shin-chan's future."
        ),
        anger_triggers=[
            "lying", "being_lazy", "embarrassing", "wasting_money",
            "skipping_homework", "being_rude", "making_mess", "disobedient",
            "sneaking", "ignoring_duties",
        ],
        happy_triggers=[
            "doing_homework", "being_honest", "helping_house", "being_responsible",
            "caring_for_himawari", "being_polite", "saving_money", "studying",
            "cleaning", "apologizing",
        ],
        reaction_style="Explosive anger → loud scolding → eventual forgiveness.",
        catchphrases=[
            "SHIN-CHAN!! 💢",
            "How many times do I have to tell you?!",
            "That's it, no dinner for you!",
            "Maybe you're growing up after all... *sniff*",
        ],
        initial_relationship=0.7,
    ),
    "hiroshi": Character(
        id="hiroshi",
        name="Hiroshi Nohara (Dad)",
        relationship_to_shinchan="Father",
        personality=(
            "Hardworking salaryman who is often tired but loves his family dearly. "
            "More lenient than Misae. Enjoys his small pleasures (beer, manga). "
            "Sometimes acts as an accomplice to Shin-chan's antics."
        ),
        anger_triggers=[
            "wasting_money", "being_rude", "embarrassing", "lying",
        ],
        happy_triggers=[
            "being_honest", "being_responsible", "caring_for_family",
            "helping_house", "being_brave", "showing_maturity",
        ],
        reaction_style="Sighs and gentle scolding, sometimes secretly amused.",
        catchphrases=[
            "Shin-chan, you rascal...",
            "Your mother is going to kill us both.",
            "That's my boy!",
            "*sigh* Where did I go wrong...",
        ],
        initial_relationship=0.75,
    ),
    "himawari": Character(
        id="himawari",
        name="Himawari Nohara (Baby Sister)",
        relationship_to_shinchan="Baby sister",
        personality=(
            "Cute baby who loves shiny things and handsome men. Often cries when "
            "neglected. Looks up to Shin-chan despite his antics. Can be surprisingly "
            "perceptive for a baby."
        ),
        anger_triggers=[
            "ignoring_duties", "being_selfish", "stealing_food", "being_mean",
        ],
        happy_triggers=[
            "caring_for_himawari", "sharing", "playing_together", "being_gentle",
        ],
        reaction_style="Cries loudly or giggles happily — no in-between.",
        catchphrases=["Waaah! 😭", "Tee hee! 😊", "*grabs shiny thing*"],
        initial_relationship=0.6,
    ),
    "kazama": Character(
        id="kazama",
        name="Kazama Toru",
        relationship_to_shinchan="Best friend",
        personality=(
            "Studious, serious, and ambitious. The most mature of Shin-chan's friend "
            "group. Gets easily annoyed by Shin-chan's antics but secretly values "
            "their friendship. Dreams of becoming an elite businessman."
        ),
        anger_triggers=[
            "being_lazy", "not_studying", "embarrassing", "being_rude",
            "cheating", "lying",
        ],
        happy_triggers=[
            "studying", "being_honest", "teamwork", "showing_effort",
            "being_brave", "helping_friends",
        ],
        reaction_style="Lectures Shin-chan but secretly cares.",
        catchphrases=[
            "Nohara, you idiot!",
            "I'm not your friend! ...Well, maybe a little.",
            "You should study more like me.",
            "I guess that was... pretty cool of you.",
        ],
        initial_relationship=0.65,
    ),
    "nene": Character(
        id="nene",
        name="Sakurada Nene",
        relationship_to_shinchan="Friend",
        personality=(
            "Appears sweet and girly but has a hidden fierce side. Loves playing "
            "'real house' (with extreme realism). Can be bossy and gets angry "
            "when things don't go her way. Punches her stuffed rabbit when upset."
        ),
        anger_triggers=[
            "being_rude", "refusing_to_play", "lying", "being_mean",
            "embarrassing",
        ],
        happy_triggers=[
            "playing_together", "being_polite", "helping_friends",
            "being_honest", "showing_empathy",
        ],
        reaction_style="Sweet on the surface, then suddenly furious. Punches Mr. Rabbit.",
        catchphrases=[
            "Let's play REAL house!",
            "Mr. Rabbit... *punch punch punch*",
            "Shin-chan, you're so mean!",
            "That's actually really nice of you.",
        ],
        initial_relationship=0.55,
    ),
    "bo": Character(
        id="bo",
        name="Tooru Bo (Bo-chan)",
        relationship_to_shinchan="Friend",
        personality=(
            "Quiet, mysterious, and zen-like. Rarely speaks but when he does, "
            "it's profound (or completely random). Has a calming presence. "
            "Often just says '...' but somehow communicates perfectly."
        ),
        anger_triggers=["being_mean", "bullying"],
        happy_triggers=[
            "being_honest", "showing_kindness", "being_brave", "teamwork",
        ],
        reaction_style="Silent nod of approval, or a quiet '...'",
        catchphrases=["...", "...!!", "Mm.", "...that was good."],
        initial_relationship=0.6,
    ),
    "masao": Character(
        id="masao",
        name="Sato Masao",
        relationship_to_shinchan="Friend",
        personality=(
            "Timid, easily scared, and often crying. A crybaby who gets picked on "
            "but is genuinely kind-hearted. Looks up to others for protection. "
            "Despite his fears, he can be surprisingly loyal."
        ),
        anger_triggers=["bullying", "being_mean", "lying"],
        happy_triggers=[
            "helping_friends", "being_brave", "showing_kindness",
            "sharing", "protecting_others",
        ],
        reaction_style="Cries easily but shows gratitude through tears.",
        catchphrases=[
            "Waaah, I'm scared! 😭",
            "Shin-chan, help me!",
            "Thank you... *sniff*",
            "You're so cool, Shin-chan!",
        ],
        initial_relationship=0.6,
    ),
    "yoshinaga": Character(
        id="yoshinaga",
        name="Yoshinaga Midori (Teacher)",
        relationship_to_shinchan="Kindergarten teacher",
        personality=(
            "Kind but strict kindergarten teacher at Futaba Kindergarten. Unmarried "
            "and a bit insecure about it (Shin-chan often brings this up). Patient "
            "but has her limits. Genuinely cares about her students' growth."
        ),
        anger_triggers=[
            "being_rude", "skipping_homework", "cheating", "embarrassing",
            "disobedient", "making_mess",
        ],
        happy_triggers=[
            "studying", "doing_homework", "being_polite", "showing_effort",
            "being_honest", "helping_friends",
        ],
        reaction_style="Patient sighing, then firm but caring discipline.",
        catchphrases=[
            "Shin-chan, please behave!",
            "Don't bring up my love life! 😤",
            "I'm proud of you today.",
            "That's what learning is all about!",
        ],
        initial_relationship=0.5,
    ),
    "shiro": Character(
        id="shiro",
        name="Shiro (Dog)",
        relationship_to_shinchan="Pet dog",
        personality=(
            "Incredibly smart and loyal dog. Often more sensible than Shin-chan. "
            "Patient, obedient, and sometimes the voice of reason (in dog form). "
            "Gets sad when neglected but always forgives."
        ),
        anger_triggers=["ignoring_duties", "being_mean"],
        happy_triggers=[
            "caring_for_shiro", "playing_together", "being_responsible",
            "sharing",
        ],
        reaction_style="Wags tail happily or whimpers sadly.",
        catchphrases=["Wan! 🐕", "*wag wag*", "*whimper*", "*happy bark!*"],
        initial_relationship=0.7,
    ),
}


def get_character(character_id: str) -> Character | None:
    """Get a character by their ID."""
    return CHARACTERS.get(character_id)


def get_all_characters() -> dict[str, Character]:
    """Get all characters."""
    return CHARACTERS.copy()


def get_characters_by_ids(character_ids: list[str]) -> list[Character]:
    """Get multiple characters by their IDs."""
    return [CHARACTERS[cid] for cid in character_ids if cid in CHARACTERS]
