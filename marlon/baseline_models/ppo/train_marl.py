from collections import deque
import shutil
from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import PPO

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.ppo.logging_utils import CheckpointManager, log_run, suppress_noisy_gymnasium_warnings

ENV_MAX_TIMESTEPS = 2000
ENV_MAX_NODE_COUNT = 12
ENV_MAX_TOTAL_CREDENTIALS = 10
LEARN_TIMESTEPS = 300_000
LEARN_EPISODES = 10000  # Set this to a large value to stop at LEARN_TIMESTEPS instead.
STATS_WINDOW_SIZE = 10
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = -1
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 1
DEFENDER_INVALID_ACTION_REWARD = -1
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = False

USE_ACTION_MASKING = True

# PPO hyperparameters (SB3 PPO) - shared by attacker/defender
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Learning-rate schedule (progress_remaining goes 1 -> 0 during training)
LEARNING_RATE_START = 5e-4
LEARNING_RATE_MIN = 1e-4
LR_ANNEAL_ALPHA = 5.0  # larger => faster decay
EVALUATE_EPISODES = 5
ATTACKER_MODEL_FILENAME = "ppo_marl_attacker.zip"
DEFENDER_MODEL_FILENAME = "ppo_marl_defender.zip"
RUN_NAME = "ppo_marl"
CHECKPOINT_INTERVAL = 10
CHECKPOINT_KEEP = 5
CHECKPOINTS_SUBDIR = "checkpoints"
ATTACKER_CKPT_PREFIX = "attacker"
DEFENDER_CKPT_PREFIX = "defender"


RunDir = Union[Path, str]

def simulated_annealing_schedule(lr_start: float, lr_min: float, alpha: float):
    def schedule(progress_remaining: float) -> float:
        progress_remaining = float(max(0.0, min(1.0, progress_remaining)))
        from math import exp

        return float(lr_min + (lr_start - lr_min) * exp(-alpha * (1.0 - progress_remaining)))

    return schedule


def train(evaluate_after: bool = False, run_dir: Optional[RunDir] = None, also_print: bool = True) -> Path:
    suppress_noisy_gymnasium_warnings()
    ppo_kwargs = {
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "learning_rate": simulated_annealing_schedule(LEARNING_RATE_START, LEARNING_RATE_MIN, LR_ANNEAL_ALPHA),
    }
    if USE_ACTION_MASKING:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

        attacker_alg_type = MaskablePPO
        attacker_policy = MaskableMultiInputActorCriticPolicy
    else:
        attacker_alg_type = PPO
        attacker_policy = "MultiInputPolicy"
    hyperparams = {
        "env_max_timesteps": ENV_MAX_TIMESTEPS,
        "env_max_node_count": ENV_MAX_NODE_COUNT,
        "env_max_total_credentials": ENV_MAX_TOTAL_CREDENTIALS,
        "use_action_masking": USE_ACTION_MASKING,
        "learn_timesteps": LEARN_TIMESTEPS,
        "learn_episodes": LEARN_EPISODES,
        "stats_window_size": STATS_WINDOW_SIZE,
        "attacker_invalid_action_reward_modifier": ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        "attacker_invalid_action_reward_multiplier": ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        "defender_invalid_action_reward": DEFENDER_INVALID_ACTION_REWARD,
        "defender_reset_on_constraint_broken": DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        "ppo_n_steps": N_STEPS,
        "ppo_batch_size": BATCH_SIZE,
        "ppo_n_epochs": N_EPOCHS,
        "ppo_gamma": GAMMA,
        "ppo_gae_lambda": GAE_LAMBDA,
        "ppo_clip_range": CLIP_RANGE,
        "ppo_ent_coef": ENT_COEF,
        "ppo_vf_coef": VF_COEF,
        "ppo_max_grad_norm": MAX_GRAD_NORM,
        "ppo_learning_rate_start": LEARNING_RATE_START,
        "ppo_learning_rate_min": LEARNING_RATE_MIN,
        "ppo_lr_anneal_alpha": LR_ANNEAL_ALPHA,
        "evaluate_episodes": EVALUATE_EPISODES,
    }

    with log_run(RUN_NAME, "train", hyperparams, run_dir=run_dir, also_print=also_print) as (run_path, _):
        attacker_save_path = run_path / ATTACKER_MODEL_FILENAME
        defender_save_path = run_path / DEFENDER_MODEL_FILENAME
        checkpoint_root = run_path / CHECKPOINTS_SUBDIR
        attacker_checkpoints = CheckpointManager(checkpoint_root / "attacker", ATTACKER_CKPT_PREFIX, keep=CHECKPOINT_KEEP)
        defender_checkpoints = CheckpointManager(checkpoint_root / "defender", DEFENDER_CKPT_PREFIX, keep=CHECKPOINT_KEEP)
        print(f"Training PPO attacker and defender. Run directory: {run_path}")

        universe = MultiAgentUniverse.build(
            env_id="CyberBattleToyCtf-v0",
            attacker_builder=BaselineAgentBuilder(
                alg_type=attacker_alg_type,
                policy=attacker_policy,
                model_kwargs=ppo_kwargs,
            ),
            defender_builder=BaselineAgentBuilder(
                alg_type=PPO,
                policy="MultiInputPolicy",
                model_kwargs=ppo_kwargs,
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
        # Logging/statistics configuration (does not affect training gradients).
        if hasattr(universe.attacker_agent, "baseline_model"):
            universe.attacker_agent.baseline_model.ep_info_buffer = deque(maxlen=STATS_WINDOW_SIZE)
            if hasattr(universe.attacker_agent.baseline_model, "ep_success_buffer"):
                universe.attacker_agent.baseline_model.ep_success_buffer = deque(maxlen=STATS_WINDOW_SIZE)
        if universe.defender_agent and hasattr(universe.defender_agent, "baseline_model"):
            universe.defender_agent.baseline_model.ep_info_buffer = deque(maxlen=STATS_WINDOW_SIZE)
            if hasattr(universe.defender_agent.baseline_model, "ep_success_buffer"):
                universe.defender_agent.baseline_model.ep_success_buffer = deque(maxlen=STATS_WINDOW_SIZE)

        def checkpoint_saver(iteration: int):
            attacker_checkpoints.save(iteration, lambda path: universe.attacker_agent.save(str(path)))
            defender_checkpoints.save(iteration, lambda path: universe.defender_agent.save(str(path)))

        universe.learn(
            total_timesteps=LEARN_TIMESTEPS,
            n_eval_episodes=LEARN_EPISODES,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            checkpoint_callback=checkpoint_saver,
        )

        universe.save(
            attacker_filepath=str(attacker_save_path),
            defender_filepath=str(defender_save_path),
        )
        latest_attacker = attacker_checkpoints.latest()
        latest_defender = defender_checkpoints.latest()
        if latest_attacker:
            shutil.copy2(latest_attacker, attacker_save_path)
        if latest_defender:
            shutil.copy2(latest_defender, defender_save_path)
        print(f"Saved attacker model to {attacker_save_path}")
        print(f"Saved defender model to {defender_save_path}")

        if evaluate_after:
            from marlon.baseline_models.ppo import eval_marl

            eval_marl.evaluate(run_dir=run_path, also_print=also_print)

        return run_path


if __name__ == "__main__":
    train(evaluate_after=True, also_print=False)
