import shutil
from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import PPO

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.ppo.logging_utils import CheckpointManager, log_run

ENV_MAX_TIMESTEPS = 1500
LEARN_TIMESTEPS = 300_000
LEARN_EPISODES = 10000  # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
EVALUATE_EPISODES = 5
ATTACKER_MODEL_FILENAME = "ppo.zip"
RUN_NAME = "ppo_attacker"
CHECKPOINT_INTERVAL = 10
CHECKPOINT_KEEP = 5
CHECKPOINTS_SUBDIR = "checkpoints"
ATTACKER_CKPT_PREFIX = "attacker"


RunDir = Union[Path, str]


def train(evaluate_after: bool = False, run_dir: Optional[RunDir] = None, also_print: bool = True) -> Path:
    hyperparams = {
        "env_max_timesteps": ENV_MAX_TIMESTEPS,
        "learn_timesteps": LEARN_TIMESTEPS,
        "learn_episodes": LEARN_EPISODES,
        "attacker_invalid_action_reward_modifier": ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        "attacker_invalid_action_reward_multiplier": ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        "evaluate_episodes": EVALUATE_EPISODES,
    }

    with log_run(RUN_NAME, "train", hyperparams, run_dir=run_dir, also_print=also_print) as (run_path, _):
        attacker_save_path = run_path / ATTACKER_MODEL_FILENAME
        checkpoint_root = run_path / CHECKPOINTS_SUBDIR
        attacker_checkpoints = CheckpointManager(checkpoint_root / "attacker", ATTACKER_CKPT_PREFIX, keep=CHECKPOINT_KEEP)
        print(f"Training PPO attacker. Run directory: {run_path}")
        universe = MultiAgentUniverse.build(
            env_id="CyberBattleToyCtf-v0",
            attacker_builder=BaselineAgentBuilder(
                alg_type=PPO,
                policy="MultiInputPolicy",
            ),
            attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
            attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        )

        def checkpoint_saver(iteration: int):
            attacker_checkpoints.save(iteration, lambda path: universe.attacker_agent.save(str(path)))

        universe.learn(
            total_timesteps=LEARN_TIMESTEPS,
            n_eval_episodes=LEARN_EPISODES,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            checkpoint_callback=checkpoint_saver,
        )

        universe.save(attacker_filepath=str(attacker_save_path))
        latest_attacker = attacker_checkpoints.latest()
        if latest_attacker:
            shutil.copy2(latest_attacker, attacker_save_path)
        print(f"Saved attacker model to {attacker_save_path}")

        if evaluate_after:
            from marlon.baseline_models.ppo import eval as eval_script

            eval_script.evaluate(run_dir=run_path, also_print=also_print)

        return run_path


if __name__ == "__main__":
    train(evaluate_after=True, also_print=False)
