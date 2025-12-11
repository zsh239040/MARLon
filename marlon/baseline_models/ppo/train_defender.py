import shutil
from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import PPO

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder
from marlon.baseline_models.ppo.logging_utils import CheckpointManager, log_run

ENV_MAX_TIMESTEPS = 1500
LEARN_TIMESTEPS = 300_000
LEARN_EPISODES = 10000  # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ATTACKER_INVALID_ACTION_REWARD = -1
DEFENDER_INVALID_ACTION_REWARD = -1
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = False
EVALUATE_EPISODES = 5
DEFENDER_MODEL_FILENAME = "ppo_defender.zip"
RUN_NAME = "ppo_defender"
CHECKPOINT_INTERVAL = 10
CHECKPOINT_KEEP = 5
CHECKPOINTS_SUBDIR = "checkpoints"
DEFENDER_CKPT_PREFIX = "defender"


RunDir = Union[Path, str]


def train(evaluate_after: bool = False, run_dir: Optional[RunDir] = None, also_print: bool = True) -> Path:
    hyperparams = {
        "env_max_timesteps": ENV_MAX_TIMESTEPS,
        "learn_timesteps": LEARN_TIMESTEPS,
        "learn_episodes": LEARN_EPISODES,
        "attacker_invalid_action_reward": ATTACKER_INVALID_ACTION_REWARD,
        "defender_invalid_action_reward": DEFENDER_INVALID_ACTION_REWARD,
        "defender_reset_on_constraint_broken": DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        "evaluate_episodes": EVALUATE_EPISODES,
    }

    with log_run(RUN_NAME, "train", hyperparams, run_dir=run_dir, also_print=also_print) as (run_path, _):
        defender_save_path = run_path / DEFENDER_MODEL_FILENAME
        checkpoint_root = run_path / CHECKPOINTS_SUBDIR
        defender_checkpoints = CheckpointManager(checkpoint_root / "defender", DEFENDER_CKPT_PREFIX, keep=CHECKPOINT_KEEP)
        print(f"Training PPO defender. Run directory: {run_path}")
        universe = MultiAgentUniverse.build(
            env_id="CyberBattleToyCtf-v0",
            attacker_builder=RandomAgentBuilder(),
            defender_builder=BaselineAgentBuilder(
                alg_type=PPO,
                policy="MultiInputPolicy",
            ),
            attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD,
            defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
            defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        )

        def checkpoint_saver(iteration: int):
            defender_checkpoints.save(iteration, lambda path: universe.defender_agent.save(str(path)))

        universe.learn(
            total_timesteps=LEARN_TIMESTEPS,
            n_eval_episodes=LEARN_EPISODES,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            checkpoint_callback=checkpoint_saver,
        )
        universe.save(defender_filepath=str(defender_save_path))
        latest_defender = defender_checkpoints.latest()
        if latest_defender:
            shutil.copy2(latest_defender, defender_save_path)
        print(f"Saved defender model to {defender_save_path}")

        if evaluate_after:
            from marlon.baseline_models.ppo import eval_defender

            eval_defender.evaluate(run_dir=run_path, also_print=also_print)

        return run_path


if __name__ == "__main__":
    train(evaluate_after=True, also_print=False)
