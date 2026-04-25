# ShinChan Life Simulator

**Meta OpenEnv Hackathon Grand Finale Submission**

An OpenEnv-compliant Reinforcement Learning environment where an LLM agent plays as Shin-chan Nohara, facing hilarious life situations. The agent starts out making classic Shin-chan mistakes and through RL training (GRPO), learns to make better decisions while maintaining Shin-chan's unique personality.

## Why This Project?
- **Personalized Decision-Making:** Teaches agents to balance constraints (e.g., Mom's anger vs. personal desires).
- **Theory of Mind:** Requires the agent to understand how actions affect different characters.
- **Personality Preservation:** Evaluates if the agent stays in character (saying "Buri Buri~") instead of sounding like a robotic AI.

## Getting Started

### 1. Run the Environment Server Locally
```bash
cd sinchan_env
pip install -e .
uv run server
# or explicitly:
# uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```
*You can access the built-in web UI at `http://localhost:8000/web` to play the scenarios manually!*

### 2. Client Interaction (Python)
```python
from sinchan_env import SinChanEnv

with SinChanEnv(base_url="http://localhost:8000") as env:
    env.reset()
    info = env.call_tool("get_scenario_info")
    print(info)
    
    result = env.call_tool("choose_action",
        action_name="do_homework",
        reasoning="I don't want Mom to yell at me.",
        dialogue="Buri buri~ fine, I'll do the math!"
    )
    print(result)
```

### 3. Training the Agent
We use `trl`'s `GRPOTrainer` to train a small Qwen model to play the environment.
See the `sinchan_env/training/train_sinchan.py` script or run the Colab Notebook provided in the `training/` directory.

## Architecture

- **`scenarios.py` & `scenario_data.py`**: The engine containing branching narratives and available actions.
- **`characters.py`**: NPC system detailing triggers and relationship meters for characters like Misae, Kazama, and Himawari.
- **`reward_engine.py`**: Calculates a multi-dimensional reward (Decision Quality, Social Awareness, Personality).
- **`sinchan_environment.py`**: The FastMCP wrapper that OpenEnv exposes over HTTP.

## Deploying to Hugging Face
```bash
openenv push --repo-id your-username/sinchan-env
```
