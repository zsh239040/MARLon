from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import PPO

from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.ppo.logging_utils import get_existing_run_dir, log_run, resolve_latest_model
from marlon.baseline_models.ppo.train import (
    ATTACKER_MODEL_FILENAME,
    EVALUATE_EPISODES,
    RUN_NAME,
    ATTACKER_CKPT_PREFIX,
    CHECKPOINTS_SUBDIR,
)


RunDir = Union[Path, str]


def evaluate(run_dir: Optional[RunDir] = None, also_print: bool = True) -> Path:
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
        universe = MultiAgentUniverse.build(
            env_id="CyberBattleToyCtf-v0",
            attacker_builder=LoadFileBaselineAgentBuilder(
                alg_type=PPO,
                file_path=str(model_path),
            ),
            attacker_invalid_action_reward_modifier=0,
        )

        universe.evaluate(n_episodes=EVALUATE_EPISODES)

    return resolved_run_dir


if __name__ == "__main__":
    evaluate(also_print=False)
