from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import PPO

from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.ppo.logging_utils import (
    capture_cyberbattle_logs,
    get_existing_run_dir,
    log_run,
    resolve_latest_model,
    suppress_noisy_gymnasium_warnings,
)
from marlon.baseline_models.ppo.train import (
    ATTACKER_MODEL_FILENAME,
    EVALUATE_EPISODES,
    RUN_NAME,
    ATTACKER_CKPT_PREFIX,
    CHECKPOINTS_SUBDIR,
    ENV_MAX_NODE_COUNT,
    ENV_MAX_TOTAL_CREDENTIALS,
    USE_ACTION_MASKING,
)


RunDir = Union[Path, str]


def evaluate(run_dir: Optional[RunDir] = None, also_print: bool = True) -> Path:
    suppress_noisy_gymnasium_warnings()
    resolved_run_dir = get_existing_run_dir(RUN_NAME, run_dir=run_dir)
    model_path = resolve_latest_model(
        resolved_run_dir,
        f"{CHECKPOINTS_SUBDIR}/attacker",
        ATTACKER_CKPT_PREFIX,
        ATTACKER_MODEL_FILENAME,
    )

    hyperparams = {
        "evaluate_episodes": EVALUATE_EPISODES,
        "model_path": str(model_path),
    }

    with log_run(RUN_NAME, "eval", hyperparams, run_dir=resolved_run_dir, also_print=also_print):
        print(f"Evaluating PPO attacker from {model_path}")
        if USE_ACTION_MASKING:
            from sb3_contrib import MaskablePPO

            alg_type = MaskablePPO
        else:
            alg_type = PPO
        universe = MultiAgentUniverse.build(
            env_id="CyberBattleToyCtf-v0",
            attacker_action_masking=USE_ACTION_MASKING,
            attacker_builder=LoadFileBaselineAgentBuilder(
                alg_type=alg_type,
                file_path=str(model_path),
            ),
            attacker_invalid_action_reward_modifier=0,
            max_timesteps=1500,
            maximum_node_count=ENV_MAX_NODE_COUNT,
            maximum_total_credentials=ENV_MAX_TOTAL_CREDENTIALS,
        )

        universe.attacker_agent.wrapper.configure_step_logging(enabled=True, prefix="[eval]")
        universe.attacker_agent.wrapper.configure_episode_end_logging(enabled=False)
        with capture_cyberbattle_logs():
            universe.evaluate(n_episodes=EVALUATE_EPISODES)

    return resolved_run_dir


if __name__ == "__main__":
    evaluate(also_print=False)
