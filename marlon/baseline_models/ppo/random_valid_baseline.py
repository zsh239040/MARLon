"""
Random-but-valid attacker baseline (action masking).

This script runs episodes in an attacker environment wrapped with:
- AttackerEnvWrapper (flattens observation + tracks valid/invalid counts)
- MaskedDiscreteAttackerWrapper (exposes Discrete actions + action_masks())

Policy:
  at each step, uniformly sample an action from the current action mask.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from marlon.baseline_models.env_wrappers.action_masking import MaskedDiscreteAttackerWrapper
from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.env_wrappers.environment_event_source import EnvironmentEventSource


# Defaults (match baseline_models/ppo/train.py by default)
ENV_ID = "CyberBattleToyCtf-v0"
ENV_MAX_TIMESTEPS = 1500
ENV_MAX_NODE_COUNT = 12
ENV_MAX_TOTAL_CREDENTIALS = 10
THROWS_ON_INVALID_ACTIONS = False

ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
ATTACKER_LOSS_REWARD = -5000.0

N_EPISODES = 20
BASE_SEED = 12345


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    steps: int
    total_reward: float
    cyber_total: float
    valid: int
    invalid: int


def _make_masked_attacker_env() -> MaskedDiscreteAttackerWrapper:
    env_kwargs = dict(
        maximum_node_count=int(ENV_MAX_NODE_COUNT),
        maximum_total_credentials=int(ENV_MAX_TOTAL_CREDENTIALS),
        throws_on_invalid_actions=bool(THROWS_ON_INVALID_ACTIONS),
    )
    try:
        cyber_env = gym.make(ENV_ID, disable_env_checker=True, **env_kwargs)
    except TypeError:
        cyber_env = gym.make(ENV_ID, **env_kwargs)

    event_source = EnvironmentEventSource()
    attacker_wrapper = AttackerEnvWrapper(
        cyber_env=cyber_env,
        event_source=event_source,
        max_timesteps=int(ENV_MAX_TIMESTEPS),
        invalid_action_reward_modifier=float(ATTACKER_INVALID_ACTION_REWARD_MODIFIER),
        invalid_action_reward_multiplier=float(ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER),
        loss_reward=float(ATTACKER_LOSS_REWARD),
        log_episode_end=False,
    )
    return MaskedDiscreteAttackerWrapper(attacker_wrapper)


def _sample_valid_action(env: MaskedDiscreteAttackerWrapper, rng: np.random.Generator) -> int:
    mask = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return int(env.action_space.sample())
    return int(rng.choice(valid))


def run(*, n_episodes: int = N_EPISODES, base_seed: int = BASE_SEED) -> Tuple[EpisodeResult, ...]:
    env = _make_masked_attacker_env()
    rng = np.random.default_rng(int(base_seed))

    results = []
    for ep in range(int(n_episodes)):
        obs, _info = env.reset(seed=int(base_seed + ep * 997))
        _ = obs
        total_reward = 0.0
        steps = 0

        while True:
            action = _sample_valid_action(env, rng)
            _obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1

            if bool(terminated or truncated):
                break
            if steps >= int(ENV_MAX_TIMESTEPS):
                break

        inner = env.unwrapped  # AttackerEnvWrapper
        cyber_total = float(np.sum(getattr(inner, "cyber_rewards", [])))
        valid = int(getattr(inner, "valid_action_count", 0))
        invalid = int(getattr(inner, "invalid_action_count", 0))

        results.append(
            EpisodeResult(
                episode=ep + 1,
                steps=int(steps),
                total_reward=float(total_reward),
                cyber_total=float(cyber_total),
                valid=valid,
                invalid=invalid,
            )
        )

    env.close()
    return tuple(results)


def _print_summary(results: Tuple[EpisodeResult, ...]) -> None:
    if not results:
        print("No episodes ran.")
        return

    rewards = np.asarray([r.total_reward for r in results], dtype=float)
    steps = np.asarray([r.steps for r in results], dtype=float)
    invalid = np.asarray([r.invalid for r in results], dtype=float)

    for r in results:
        print(
            f"[ep={r.episode}] steps={r.steps} total_reward={r.total_reward:.3f} cyber_total={r.cyber_total:.3f} "
            f"valid={r.valid} invalid={r.invalid}"
        )

    print(
        "SUMMARY "
        f"episodes={len(results)} "
        f"reward_mean={float(rewards.mean()):.3f} reward_median={float(np.median(rewards)):.3f} reward_max={float(rewards.max()):.3f} "
        f"steps_mean={float(steps.mean()):.1f} invalid_mean={float(invalid.mean()):.1f}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Random-valid attacker baseline (masking).")
    parser.add_argument("--episodes", type=int, default=N_EPISODES, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base seed.")
    args = parser.parse_args(argv)

    results = run(n_episodes=int(args.episodes), base_seed=int(args.seed))
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

