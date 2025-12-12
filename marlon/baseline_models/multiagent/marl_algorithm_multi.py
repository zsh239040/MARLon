"""
Multi-environment multi-agent training loop.

This module extends marl_algorithm.learn by supporting:
- multiple independent CyberBattle environments
- mixed strategies per env (a/b/c)
    a: attacker+defender learned
    b: attacker learned, defender safe-random
    c: attacker random, defender learned

The learned sides use SB3 PPO over a DummyVecEnv batch.
Random-only envs are stepped in python to keep cyber_env state aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from marlon.baseline_models.multiagent.marlon_agent import MarlonAgent


@dataclass
class RandomEnvPair:
    """Holds wrappers for an env where one side is random."""

    attacker_wrapper: Optional[object] = None
    defender_wrapper: Optional[object] = None
    rng: Optional[np.random.Generator] = None


def sample_safe_defender_action(defender_wrapper, rng: np.random.Generator, max_tries: int = 50):
    """
    Sample a defender action that is unlikely to reduce network availability.
    We restrict to firewall actions (block/allow traffic) which do not affect
    machine status or service running flags in our defender implementation.
    """
    for _ in range(max_tries):
        action = defender_wrapper.action_space.sample()
        action_number = int(action[0])
        if action_number not in (1, 2):
            continue
        if defender_wrapper.is_defender_action_valid(action):
            return action
    # Fall back to a likely-safe allow-traffic action on node 0.
    dim = int(defender_wrapper.action_space.nvec.shape[0])
    fallback = np.zeros((dim,), dtype=int)
    fallback[0] = 2  # allow traffic
    # indices for allow traffic: [5]=node, [6]=port, [7]=incoming/outgoing
    if dim > 7:
        fallback[5] = 0
        fallback[6] = 0
        fallback[7] = 0
    return fallback


def _step_random_attacker(attacker_wrapper, rng: np.random.Generator):
    # Random-only envs are not managed by SB3, so ensure they are initialized.
    if getattr(attacker_wrapper, "timesteps", None) is None:
        attacker_wrapper.reset()
    action = attacker_wrapper.action_space.sample()
    obs, reward, terminated, truncated, info = attacker_wrapper.step(action)
    done = bool(terminated or truncated)
    if done:
        attacker_wrapper.reset()
    return obs, reward, done, info


def _step_random_defender(defender_wrapper, rng: np.random.Generator):
    action = sample_safe_defender_action(defender_wrapper, rng)
    obs, reward, terminated, truncated, info = defender_wrapper.step(action)
    done = bool(terminated or truncated)
    if done:
        defender_wrapper.reset()
    return obs, reward, done, info


def collect_rollouts_multi(
    attacker_agent: MarlonAgent,
    defender_agent: MarlonAgent,
    random_attack_envs: Sequence[RandomEnvPair],
    random_defend_envs: Sequence[RandomEnvPair],
) -> bool:
    """
    Collect rollouts for learned agents while stepping random-only envs in sync.
    """
    attacker_agent.on_rollout_start()
    defender_agent.on_rollout_start()

    n_steps = 0
    max_steps = min(attacker_agent.n_rollout_steps, defender_agent.n_rollout_steps)

    new_obs1 = None
    new_obs2 = None
    dones1 = None
    dones2 = None

    while n_steps < max_steps:
        continue1, new_obs1, dones1 = attacker_agent.perform_step(n_steps)
        for env_pair in random_attack_envs:
            if env_pair.attacker_wrapper is not None:
                _step_random_attacker(env_pair.attacker_wrapper, env_pair.rng)

        continue2, new_obs2, dones2 = defender_agent.perform_step(n_steps)
        for env_pair in random_defend_envs:
            if env_pair.defender_wrapper is not None:
                _step_random_defender(env_pair.defender_wrapper, env_pair.rng)

        if continue1 is False or continue2 is False:
            return False

        n_steps += 1

    attacker_agent.on_rollout_end(new_obs1, dones1)
    defender_agent.on_rollout_end(new_obs2, dones2)
    return True


def learn_multi(
    attacker_agent: MarlonAgent,
    defender_agent: MarlonAgent,
    total_timesteps_per_env: int,
    random_attack_envs: Optional[Sequence[RandomEnvPair]] = None,
    random_defend_envs: Optional[Sequence[RandomEnvPair]] = None,
    checkpoint_callback: Optional[Callable[[int], None]] = None,
    checkpoint_interval: int = 0,
) -> int:
    """
    Train attacker and defender in a mixed-strategy multi-env setup.

    total_timesteps_per_env is interpreted as per-environment timesteps.
    """
    random_attack_envs = random_attack_envs or []
    random_defend_envs = random_defend_envs or []

    iteration = 0

    attacker_total_timesteps = int(total_timesteps_per_env * attacker_agent.env.num_envs)
    defender_total_timesteps = int(total_timesteps_per_env * defender_agent.env.num_envs)

    attacker_agent.setup_learn(
        attacker_total_timesteps, None, -1, 0, None, True, "OnPolicyAlgorithm"
    )
    defender_agent.setup_learn(
        defender_total_timesteps, None, -1, 0, None, True, "OnPolicyAlgorithm"
    )

    last_checkpoint_iteration: Optional[int] = None

    while attacker_agent.num_timesteps < attacker_total_timesteps and defender_agent.num_timesteps < defender_total_timesteps:
        continue_training = collect_rollouts_multi(
            attacker_agent=attacker_agent,
            defender_agent=defender_agent,
            random_attack_envs=random_attack_envs,
            random_defend_envs=random_defend_envs,
        )
        if not continue_training:
            break

        iteration += 1

        attacker_agent.update_progress(attacker_total_timesteps)
        defender_agent.update_progress(defender_total_timesteps)

        if iteration % attacker_agent.log_interval == 0:
            attacker_agent.log_training(iteration)
        if iteration % defender_agent.log_interval == 0:
            defender_agent.log_training(iteration)

        attacker_agent.train()
        defender_agent.train()

        if checkpoint_callback and checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            checkpoint_callback(iteration)
            last_checkpoint_iteration = iteration

    attacker_agent.on_training_end()
    defender_agent.on_training_end()

    if checkpoint_callback and last_checkpoint_iteration != iteration:
        checkpoint_callback(iteration)

    return iteration
