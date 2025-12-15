from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import A2C

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder
from marlon.baseline_models.ppo.logging_utils import log_run, suppress_noisy_gymnasium_warnings

ENV_MAX_TIMESTEPS = 1500
ENV_MAX_NODE_COUNT = 12
ENV_MAX_TOTAL_CREDENTIALS = 10
LEARN_TIMESTEPS = 300_000
LEARN_EPISODES = 10_000  # Kept for parity; A2C stops on timesteps.
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
DEFENDER_INVALID_ACTION_REWARD = -1
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = False
EVALUATE_EPISODES = 5
DEFENDER_MODEL_FILENAME = "a2c_defender.zip"
RUN_NAME = "a2c_defender"
USE_ACTION_MASKING = False

RunDir = Union[Path, str]


def train(evaluate_after: bool = False, run_dir: Optional[RunDir] = None, also_print: bool = True) -> Path:
    suppress_noisy_gymnasium_warnings()
    if USE_ACTION_MASKING:
        raise ValueError("Action masking is only supported for PPO via sb3-contrib MaskablePPO (see baseline_models/ppo/*).")
    hyperparams = {
        "env_max_timesteps": ENV_MAX_TIMESTEPS,
        "env_max_node_count": ENV_MAX_NODE_COUNT,
        "env_max_total_credentials": ENV_MAX_TOTAL_CREDENTIALS,
        "use_action_masking": USE_ACTION_MASKING,
        "learn_timesteps": LEARN_TIMESTEPS,
        "learn_episodes": LEARN_EPISODES,
        "attacker_invalid_action_reward_modifier": ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        "attacker_invalid_action_reward_multiplier": ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        "defender_invalid_action_reward": DEFENDER_INVALID_ACTION_REWARD,
        "defender_reset_on_constraint_broken": DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        "evaluate_episodes": EVALUATE_EPISODES,
    }

    base_dir = Path(__file__).resolve().parent
    with log_run(RUN_NAME, "train", hyperparams, run_dir=run_dir, base_dir=base_dir, also_print=also_print) as (run_path, _):
        defender_save_path = run_path / DEFENDER_MODEL_FILENAME
        print(f"Training A2C defender. Run directory: {run_path}")
        universe = MultiAgentUniverse.build(
            env_id="CyberBattleToyCtf-v0",
            attacker_builder=RandomAgentBuilder(),
            defender_builder=BaselineAgentBuilder(
                alg_type=A2C,
                policy="MultiInputPolicy",
            ),
            attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
            attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
            defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
            defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
            max_timesteps=ENV_MAX_TIMESTEPS,
            maximum_node_count=ENV_MAX_NODE_COUNT,
            maximum_total_credentials=ENV_MAX_TOTAL_CREDENTIALS,
            attacker_action_masking=USE_ACTION_MASKING,
        )

        universe.learn(
            total_timesteps=LEARN_TIMESTEPS,
            n_eval_episodes=LEARN_EPISODES,
        )
        universe.save(defender_filepath=str(defender_save_path))
        print(f"Saved defender model to {defender_save_path}")

        if evaluate_after:
            from marlon.baseline_models.a2c import eval_defender

            eval_defender.evaluate(run_dir=run_path, also_print=also_print)

        return run_path


if __name__ == "__main__":
    train(evaluate_after=True, also_print=False)
