"""
Multi-environment multi-agent PPO training.

This file is a copy/extension of marlon/baseline_models/ppo/train_marl.py.
When N_ENVS == 1 and all strategies are 'a', it delegates to the original
single-env training to preserve identical behavior.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from cyberbattle._env.cyberbattle_env import DefenderConstraint

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.env_wrappers.environment_event_source import EnvironmentEventSource
from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineMarlonAgent
from marlon.baseline_models.multiagent.marl_algorithm_multi import RandomEnvPair, learn_multi
from marlon.baseline_models.ppo.logging_utils import CheckpointManager, log_run
from marlon.baseline_models.ppo.train_marl import train as train_single


# -------------------------
# Hyperparameters
# -------------------------

ENV_ID = "CyberBattleToyCtf-v0"
ENV_MAX_TIMESTEPS = 2000
LEARN_TIMESTEPS_PER_ENV = 30_000
LEARN_EPISODES = 10_000  # Kept for parity; PPO stops on timesteps.

# Multi-env setup
N_ENVS = 4
N_ENVS_A = 2  # a: learned attacker + learned defender
N_ENVS_B = 1  # b: learned attacker + safe-random defender
N_ENVS_C = 1  # c: random attacker + learned defender
BASE_SEED = 12345

# Constraint/reward config
DEFENDER_MAINTAIN_SLA = 0.60
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
DEFENDER_INVALID_ACTION_REWARD = 0
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = False
ATTACKER_LOSS_REWARD = -5000.0
DEFENDER_LOSS_REWARD = -5000.0

# PPO hyperparams (shared unless overridden)
ATTACKER_N_STEPS = 2048
DEFENDER_N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
LEARNING_RATE_START = 3e-4
LEARNING_RATE_MIN = 1e-5
ANNEAL_ALPHA = 5.0  # Larger => faster decay

# Logging/checkpointing
RUN_NAME = "ppo_marl_multi"
EVALUATE_EPISODES = 5
ATTACKER_MODEL_FILENAME = "ppo_marl_attacker.zip"
DEFENDER_MODEL_FILENAME = "ppo_marl_defender.zip"
CHECKPOINT_INTERVAL = 10
CHECKPOINT_KEEP = 5
CHECKPOINTS_SUBDIR = "checkpoints"
ATTACKER_CKPT_PREFIX = "attacker"
DEFENDER_CKPT_PREFIX = "defender"


RunDir = Union[Path, str]


def simulated_annealing_schedule(
    lr_start: float,
    lr_min: float = LEARNING_RATE_MIN,
    alpha: float = ANNEAL_ALPHA,
):
    """
    Simulated-annealing-like learning rate:
    lr(t) = lr_min + (lr_start - lr_min) * exp(-alpha * (1 - progress))
    where progress goes from 1 -> 0.
    """

    def schedule(progress_remaining: float) -> float:
        progress_remaining = float(np.clip(progress_remaining, 0.0, 1.0))
        return float(lr_min + (lr_start - lr_min) * np.exp(-alpha * (1.0 - progress_remaining)))

    return schedule


@dataclass
class EnvInstance:
    strategy: str
    seed: int
    cyber_env: gym.Env
    attacker_wrapper: AttackerEnvWrapper
    defender_wrapper: DefenderEnvWrapper
    rng: np.random.Generator


def _make_env_instance(strategy: str, seed: int) -> EnvInstance:
    cyber_env = gym.make(
        ENV_ID,
        defender_constraint=DefenderConstraint(maintain_sla=DEFENDER_MAINTAIN_SLA),
        losing_reward=DEFENDER_LOSS_REWARD,
    )
    cyber_env.reset(seed=seed)

    event_source = EnvironmentEventSource()
    attacker_wrapper = AttackerEnvWrapper(
        cyber_env=cyber_env,
        event_source=event_source,
        max_timesteps=ENV_MAX_TIMESTEPS,
        invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        loss_reward=ATTACKER_LOSS_REWARD,
    )
    defender_wrapper = DefenderEnvWrapper(
        cyber_env=cyber_env,
        attacker_reward_store=attacker_wrapper,
        event_source=event_source,
        max_timesteps=ENV_MAX_TIMESTEPS,
        invalid_action_reward=DEFENDER_INVALID_ACTION_REWARD,
        defender=True,
        reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        loss_reward=DEFENDER_LOSS_REWARD,
    )

    # Ensure per-env random streams differ, even with same policies.
    attacker_wrapper.action_space.seed(seed + 1)
    defender_wrapper.action_space.seed(seed + 2)

    rng = np.random.default_rng(seed + 3)

    return EnvInstance(
        strategy=strategy,
        seed=seed,
        cyber_env=cyber_env,
        attacker_wrapper=attacker_wrapper,
        defender_wrapper=defender_wrapper,
        rng=rng,
    )


def _build_instances(strategies: Sequence[str]) -> List[EnvInstance]:
    instances: List[EnvInstance] = []
    for idx, strategy in enumerate(strategies):
        instances.append(_make_env_instance(strategy=strategy, seed=BASE_SEED + idx * 997))
    return instances


def _make_vec_env_from_wrappers(wrappers: Sequence[gym.Env]) -> VecMonitor:
    env_fns = [lambda w=w: w for w in wrappers]
    return VecMonitor(DummyVecEnv(env_fns))


def train(
    evaluate_after: bool = False,
    run_dir: Optional[RunDir] = None,
    also_print: bool = True,
) -> Path:
    # Strategy list
    strategies: List[str] = (["a"] * N_ENVS_A) + (["b"] * N_ENVS_B) + (["c"] * N_ENVS_C)
    if len(strategies) != N_ENVS:
        raise ValueError("N_ENVS must equal N_ENVS_A + N_ENVS_B + N_ENVS_C")
    if ATTACKER_N_STEPS != DEFENDER_N_STEPS:
        raise ValueError("ATTACKER_N_STEPS must equal DEFENDER_N_STEPS for synchronized multi-agent rollouts.")

    # Preserve identical single-env behavior
    if N_ENVS == 1 and strategies == ["a"]:
        return train_single(evaluate_after=evaluate_after, run_dir=run_dir, also_print=also_print)

    hyperparams: Dict[str, object] = {
        "env_id": ENV_ID,
        "env_max_timesteps": ENV_MAX_TIMESTEPS,
        "learn_timesteps_per_env": LEARN_TIMESTEPS_PER_ENV,
        "learn_episodes": LEARN_EPISODES,
        "n_envs": N_ENVS,
        "n_envs_a": N_ENVS_A,
        "n_envs_b": N_ENVS_B,
        "n_envs_c": N_ENVS_C,
        "base_seed": BASE_SEED,
        "attacker_n_steps": ATTACKER_N_STEPS,
        "defender_n_steps": DEFENDER_N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "learning_rate_start": LEARNING_RATE_START,
        "learning_rate_min": LEARNING_RATE_MIN,
        "anneal_alpha": ANNEAL_ALPHA,
    }

    with log_run(RUN_NAME, "train", hyperparams, run_dir=run_dir, also_print=also_print) as (run_path, _log_path):
        attacker_save_path = run_path / ATTACKER_MODEL_FILENAME
        defender_save_path = run_path / DEFENDER_MODEL_FILENAME
        checkpoint_root = run_path / CHECKPOINTS_SUBDIR
        attacker_checkpoints = CheckpointManager(checkpoint_root / "attacker", ATTACKER_CKPT_PREFIX, keep=CHECKPOINT_KEEP)
        defender_checkpoints = CheckpointManager(checkpoint_root / "defender", DEFENDER_CKPT_PREFIX, keep=CHECKPOINT_KEEP)
        print(f"Training multi-env PPO attacker and defender. Run directory: {run_path}")

        log_level = os.environ.get("LOGLEVEL", "INFO").upper()
        logger = logging.Logger("marlon_multi", level=log_level)
        logger.addHandler(logging.StreamHandler())

        instances = _build_instances(strategies)
        envs_a = [i for i in instances if i.strategy == "a"]
        envs_b = [i for i in instances if i.strategy == "b"]
        envs_c = [i for i in instances if i.strategy == "c"]

        attacker_learn_wrappers = [i.attacker_wrapper for i in envs_a + envs_b]
        defender_learn_wrappers = [i.defender_wrapper for i in envs_a + envs_c]

        if len(attacker_learn_wrappers) == 0 or len(defender_learn_wrappers) == 0:
            raise ValueError("At least one learned env required for both attacker and defender.")

        attacker_vec_env = _make_vec_env_from_wrappers(attacker_learn_wrappers)
        defender_vec_env = _make_vec_env_from_wrappers(defender_learn_wrappers)

        attacker_lr = simulated_annealing_schedule(LEARNING_RATE_START)
        defender_lr = simulated_annealing_schedule(LEARNING_RATE_START)

        attacker_model = PPO(
            "MultiInputPolicy",
            attacker_vec_env,
            n_steps=ATTACKER_N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=attacker_lr,
            verbose=1,
        )
        defender_model = PPO(
            "MultiInputPolicy",
            defender_vec_env,
            n_steps=DEFENDER_N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=defender_lr,
            verbose=1,
        )

        attacker_agent = BaselineMarlonAgent(attacker_model, attacker_learn_wrappers[0], logger)
        defender_agent = BaselineMarlonAgent(defender_model, defender_learn_wrappers[0], logger)

        random_attack_envs = [RandomEnvPair(attacker_wrapper=i.attacker_wrapper, rng=i.rng) for i in envs_c]
        random_defend_envs = [RandomEnvPair(defender_wrapper=i.defender_wrapper, rng=i.rng) for i in envs_b]

        def checkpoint_saver(iteration: int):
            attacker_checkpoints.save(iteration, lambda path: attacker_agent.save(str(path)))
            defender_checkpoints.save(iteration, lambda path: defender_agent.save(str(path)))

        learn_multi(
            attacker_agent=attacker_agent,
            defender_agent=defender_agent,
            total_timesteps_per_env=LEARN_TIMESTEPS_PER_ENV,
            random_attack_envs=random_attack_envs,
            random_defend_envs=random_defend_envs,
            checkpoint_callback=checkpoint_saver,
            checkpoint_interval=CHECKPOINT_INTERVAL,
        )

        attacker_agent.save(str(attacker_save_path))
        defender_agent.save(str(defender_save_path))

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
