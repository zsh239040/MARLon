from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import A2C

from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder
from marlon.baseline_models.ppo.logging_utils import (
    capture_cyberbattle_logs,
    get_existing_run_dir,
    log_run,
    suppress_noisy_gymnasium_warnings,
)
from marlon.baseline_models.a2c.train_defender import (
    DEFENDER_MODEL_FILENAME,
    EVALUATE_EPISODES,
    RUN_NAME,
    ENV_MAX_TIMESTEPS,
    ENV_MAX_NODE_COUNT,
    ENV_MAX_TOTAL_CREDENTIALS,
    USE_ACTION_MASKING,
)

RunDir = Union[Path, str]


def evaluate(run_dir: Optional[RunDir] = None, also_print: bool = True) -> Path:
    suppress_noisy_gymnasium_warnings()
    if USE_ACTION_MASKING:
        raise ValueError("Action masking is only supported for PPO via sb3-contrib MaskablePPO (see baseline_models/ppo/*).")
    base_dir = Path(__file__).resolve().parent
    resolved_run_dir = get_existing_run_dir(RUN_NAME, base_dir=base_dir, run_dir=run_dir)
    model_path = resolved_run_dir / DEFENDER_MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    hyperparams = {
        "evaluate_episodes": EVALUATE_EPISODES,
        "use_action_masking": USE_ACTION_MASKING,
        "model_path": str(model_path),
    }

    with log_run(RUN_NAME, "eval", hyperparams, run_dir=resolved_run_dir, base_dir=base_dir, also_print=also_print):
        print(f"Evaluating A2C defender from {model_path}")
        universe = MultiAgentUniverse.build(
            env_id="CyberBattleToyCtf-v0",
            max_timesteps=ENV_MAX_TIMESTEPS,
            maximum_node_count=ENV_MAX_NODE_COUNT,
            maximum_total_credentials=ENV_MAX_TOTAL_CREDENTIALS,
            attacker_action_masking=USE_ACTION_MASKING,
            attacker_builder=RandomAgentBuilder(),
            defender_builder=LoadFileBaselineAgentBuilder(
                alg_type=A2C,
                file_path=str(model_path),
            ),
            attacker_invalid_action_reward_modifier=0,
        )
        if universe.defender_agent:
            universe.defender_agent.wrapper.configure_step_logging(enabled=True, prefix="[eval]")
            universe.defender_agent.wrapper.configure_episode_end_logging(enabled=False)
        universe.attacker_agent.wrapper.configure_episode_end_logging(enabled=False)
        with capture_cyberbattle_logs():
            universe.evaluate(n_episodes=EVALUATE_EPISODES)

    return resolved_run_dir


if __name__ == "__main__":
    evaluate(also_print=False)
