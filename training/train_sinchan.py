import os
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from sinchan_env import SinChanEnv

# Define the environment URL (can be local or huggingface spaces)
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

class SinChanToolEnv:
    """Wrapper that adapts the SinChanEnv for TRL's environment factory."""
    def __init__(self):
        self.env = SinChanEnv(base_url=ENV_URL)
        self.reward = 0.0
        self.done = False
        
    def reset(self, **kwargs) -> str | None:
        result = self.env.reset()
        self.reward = 0.0
        self.done = False
        # Get scenario info automatically on reset to give context
        try:
            info = self.env.call_tool("get_scenario_info")
            narrative = f"SCENARIO: {info.get('title')}\n\n{info.get('narrative')}\n\nAvailable Actions: {info.get('available_actions')}"
        except Exception:
            narrative = result.observation.metadata.get("message", "New scenario loaded.")
        return narrative
        
    def choose_action(self, action_name: str, reasoning: str, dialogue: str) -> str:
        """
        Make a decision as Shin-chan.
        
        Args:
            action_name: The action to take (from available actions list)
            reasoning: Why you're choosing this action
            dialogue: What Shin-chan says (stay in character!)
        """
        if self.done:
            raise ValueError("Episode is already over!")

        # Use a single step call so one model action maps to exactly one env transition.
        from openenv.core.env_server.mcp_types import CallToolAction
        step_res = self.env.step(CallToolAction(
            tool_name="choose_action",
            arguments={
                "action_name": action_name,
                "reasoning": reasoning,
                "dialogue": dialogue
            }
        ))

        obs = getattr(step_res, "observation", step_res)
        self.reward = float(getattr(step_res, "reward", getattr(obs, "reward", 0.0)) or 0.0)
        self.done = bool(getattr(obs, "done", False))

        response_dict = getattr(obs, "result", {}) or {}
        return f"{response_dict.get('shinchan_said', '')}\n\nConsequences: {response_dict.get('consequences', '')}\n\nStatus: {response_dict.get('status', '')}"


def decision_reward(environments, **kwargs):
    """Return the reward for each environment."""
    return [env.reward for env in environments]


SYSTEM_PROMPT = """You are Shin-chan Nohara (野原しんのすけ), a 5-year-old boy from Kasukabe, Japan. 
You are mischievous, funny, and sometimes naughty — but deep down you care about your family and friends.

You are facing a real-life situation. Think about what would happen if you make different choices. 
Consider how your actions affect Mom (Misae), Dad (Hiroshi), your baby sister (Himawari), and your friends.

Use the `choose_action` tool to make your decision. Stay in character as Shin-chan!
Rules:
1. Pick one of the available actions.
2. Explain your reasoning (think about consequences!).
3. Say something as Shin-chan would say it (be funny but thoughtful, use your catchphrases like 'Buri buri~').
"""

def train():
    print(f"Connecting to environment at: {ENV_URL}")
    
    # Simple prompt dataset
    dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": SYSTEM_PROMPT + "\n\nA new adventure awaits! What do you do?"}]] * 100
    })

    # Use a small model suitable for quick testing
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=decision_reward,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir="sinchan-grpo-model",
            use_vllm=False,  # Set to True if you have vLLM installed
            chat_template_kwargs={"enable_thinking": False},
            max_completion_length=512,
            num_generations=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=1,
            max_steps=50,
        ),
        environment_factory=SinChanToolEnv,
    )
    
    print("Starting training loop...")
    trainer.train()

if __name__ == "__main__":
    train()
