from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from cyberbattle._env.cyberbattle_env import DefenderConstraint

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.env_wrappers.action_masking import MaskedDiscreteAttackerWrapper
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.env_wrappers.environment_event_source import EnvironmentEventSource
from marlon.baseline_models.multiagent.evaluation_stats import EvalutionStats
from marlon.baseline_models.ppo.logging_utils import (
    capture_cyberbattle_logs,
    get_existing_run_dir,
    get_stdout_logger,
    log_run,
    resolve_latest_model,
    suppress_noisy_gymnasium_warnings,
)
from marlon.baseline_models.ppo_multi.train_marl_multi import (
    ATTACKER_CKPT_PREFIX,
    ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
    ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
    ATTACKER_LOSS_REWARD,
    ATTACKER_MODEL_FILENAME,
    CHECKPOINTS_SUBDIR,
    DEFENDER_CKPT_PREFIX,
    DEFENDER_INVALID_ACTION_REWARD,
    DEFENDER_LOSS_REWARD,
    DEFENDER_MAINTAIN_SLA,
    DEFENDER_MODEL_FILENAME,
    DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
    ENV_ID,
    ENV_MAX_TIMESTEPS,
    EVALUATE_EPISODES,
    RUN_NAME,
    USE_ACTION_MASKING,
)

# 评估环境默认重置（与训练分开）
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = True

RunDir = Union[Path, str]

def _make_env_pair(seed: int):
    kwargs = dict(
        defender_constraint=DefenderConstraint(maintain_sla=DEFENDER_MAINTAIN_SLA),
        losing_reward=DEFENDER_LOSS_REWARD,
    )
    try:
        cyber_env = gym.make(ENV_ID, disable_env_checker=True, **kwargs)
    except TypeError:
        cyber_env = gym.make(ENV_ID, **kwargs)
    event_source = EnvironmentEventSource()
    attacker_env: gym.Env = AttackerEnvWrapper(
        cyber_env=cyber_env,
        event_source=event_source,
        max_timesteps=ENV_MAX_TIMESTEPS,
        invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        loss_reward=ATTACKER_LOSS_REWARD,
    )
    if USE_ACTION_MASKING:
        attacker_env = MaskedDiscreteAttackerWrapper(attacker_env)
    defender_wrapper = DefenderEnvWrapper(
        cyber_env=cyber_env,
        attacker_reward_store=attacker_env,
        event_source=event_source,
        max_timesteps=ENV_MAX_TIMESTEPS,
        invalid_action_reward=DEFENDER_INVALID_ACTION_REWARD,
        defender=True,
        reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        loss_reward=DEFENDER_LOSS_REWARD,
    )
    attacker_env.action_space.seed(seed + 1)
    defender_wrapper.action_space.seed(seed + 2)
    attacker_env.reset(seed=seed)
    defender_wrapper.on_reset(0)
    defender_wrapper.reset(seed=seed)
    return attacker_env, defender_wrapper


def _stack_dict_obs(obs_list: List[Dict[str, Any]], indices: List[int]) -> Dict[str, np.ndarray]:
    batch: Dict[str, np.ndarray] = {}
    keys = obs_list[indices[0]].keys()
    for key in keys:
        batch[key] = np.stack([np.asarray(obs_list[i][key]) for i in indices], axis=0)
    return batch


def evaluate(
    run_dir: Optional[RunDir] = None,
    also_print: bool = True,
    *,
    n_envs: int = 1,
    n_episodes: int = EVALUATE_EPISODES,
    max_steps: int = ENV_MAX_TIMESTEPS,
    deterministic: bool = False,
    cyberbattle_log_level: int = logging.INFO,
) -> Path:
    suppress_noisy_gymnasium_warnings()

    base_dir = Path(__file__).resolve().parent
    resolved_run_dir = get_existing_run_dir(RUN_NAME, base_dir=base_dir, run_dir=run_dir)
    attacker_model_path = resolve_latest_model(
        resolved_run_dir,
        f"{CHECKPOINTS_SUBDIR}/attacker",
        ATTACKER_CKPT_PREFIX,
        ATTACKER_MODEL_FILENAME,
    )
    defender_model_path = resolve_latest_model(
        resolved_run_dir,
        f"{CHECKPOINTS_SUBDIR}/defender",
        DEFENDER_CKPT_PREFIX,
        DEFENDER_MODEL_FILENAME,
    )

    hyperparams = {
        "env_id": ENV_ID,
        "max_steps": max_steps,
        "n_envs": n_envs,
        "n_episodes": n_episodes,
        "deterministic": deterministic,
        "cyberbattle_log_level": int(cyberbattle_log_level),
        "use_action_masking": USE_ACTION_MASKING,
        "defender_maintain_sla": DEFENDER_MAINTAIN_SLA,
        "defender_reset_on_constraint_broken": DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        "attacker_invalid_action_reward_modifier": ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        "attacker_invalid_action_reward_multiplier": ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        "defender_invalid_action_reward": DEFENDER_INVALID_ACTION_REWARD,
        "attacker_loss_reward": ATTACKER_LOSS_REWARD,
        "defender_loss_reward": DEFENDER_LOSS_REWARD,
        "attacker_model_path": str(attacker_model_path),
        "defender_model_path": str(defender_model_path),
    }

    with log_run(RUN_NAME, "eval", hyperparams, run_dir=resolved_run_dir, base_dir=base_dir, also_print=also_print):
        print("Evaluating multi-env PPO attacker/defender")
        if USE_ACTION_MASKING:
            from sb3_contrib import MaskablePPO

            attacker_model = MaskablePPO.load(str(attacker_model_path))
        else:
            attacker_model = PPO.load(str(attacker_model_path))
        defender_model = PPO.load(str(defender_model_path))

        stats_logger = get_stdout_logger("marlon_eval_multi", level=logging.INFO)

        attacker_rewards: List[float] = []
        defender_rewards: List[float] = []
        attacker_valid: List[int] = []
        attacker_invalid: List[int] = []
        defender_valid: List[int] = []
        defender_invalid: List[int] = []
        episode_steps: List[int] = []

        with capture_cyberbattle_logs(level=cyberbattle_log_level):
            for ep in range(n_episodes):
                print(f"Evaluating episode {ep + 1} of {n_episodes}")
                # Independent envs per episode; within an env, attacker/defender share the same cyber_env.
                pairs = [
                    _make_env_pair(seed=10_000 + ep * 997 + i * 101)
                    for i in range(n_envs)
                ]
                a_wrappers = [p[0] for p in pairs]
                d_wrappers = [p[1] for p in pairs]
                for i, aw in enumerate(a_wrappers):
                    aw.configure_step_logging(enabled=True, prefix=f"[ep={ep + 1}/{n_episodes} env={i}]")
                for i, dw in enumerate(d_wrappers):
                    dw.configure_step_logging(enabled=True, prefix=f"[ep={ep + 1}/{n_episodes} env={i}]")

                a_obs = [aw.reset(seed=10_000 + ep * 997 + i * 101)[0] for i, aw in enumerate(a_wrappers)]
                d_obs = []
                for i, dw in enumerate(d_wrappers):
                    dw.on_reset(0)
                    d_obs.append(dw.reset(seed=10_000 + ep * 997 + i * 101)[0])

                a_done = [False] * n_envs
                d_done = [False] * n_envs
                a_ep_rewards = [0.0] * n_envs
                d_ep_rewards = [0.0] * n_envs
                step_counts = [0] * n_envs
                a_term = [False] * n_envs
                a_trunc = [False] * n_envs
                d_term = [False] * n_envs
                d_trunc = [False] * n_envs

                while True:
                    # Batch attacker actions for unfinished envs
                    active_idx = [
                        i for i in range(n_envs) if (not a_done[i] and not d_done[i] and step_counts[i] < max_steps)
                    ]
                    if not active_idx:
                        break

                    batch_obs = _stack_dict_obs(a_obs, active_idx)
                    if USE_ACTION_MASKING:
                        masks = np.stack([a_wrappers[i].action_masks() for i in active_idx], axis=0)
                        a_actions, _ = attacker_model.predict(batch_obs, deterministic=deterministic, action_masks=masks)
                    else:
                        a_actions, _ = attacker_model.predict(batch_obs, deterministic=deterministic)

                    for j, i in enumerate(active_idx):
                        action_j = a_actions[j] if getattr(a_actions, "ndim", 1) == 1 else a_actions[j]
                        _new_obs, r, term, trunc, _info = a_wrappers[i].step(action_j)
                        a_obs[i] = _new_obs
                        a_ep_rewards[i] += float(r)
                        a_term[i] = bool(term)
                        a_trunc[i] = bool(trunc)
                        a_done[i] = bool(term or trunc)

                    # Batch defender actions for same active envs
                    active_idx_def = [i for i in active_idx if not a_done[i] and not d_done[i]]
                    if not active_idx_def:
                        continue

                    batch_obs_d = _stack_dict_obs(d_obs, active_idx_def)

                    d_actions, _ = defender_model.predict(batch_obs_d, deterministic=deterministic)

                    for j, i in enumerate(active_idx_def):
                        action_j = d_actions[j] if getattr(d_actions, "ndim", 1) == 1 else d_actions[j]
                        _new_obs, r, term, trunc, _info = d_wrappers[i].step(action_j)
                        d_obs[i] = _new_obs
                        d_ep_rewards[i] += float(r)
                        d_term[i] = bool(term)
                        d_trunc[i] = bool(trunc)
                        d_done[i] = bool(term or trunc)
                        step_counts[i] += 1

                # Episode end reasons (per env)
                for i in range(n_envs):
                    if a_term[i] or d_term[i]:
                        outcome = getattr(a_wrappers[i], "last_outcome", None) or getattr(d_wrappers[i], "last_outcome", None)
                        reason = outcome or "terminated"
                    elif step_counts[i] >= max_steps:
                        reason = "timeout"
                    elif a_trunc[i] or d_trunc[i]:
                        reason = "truncated"
                    else:
                        reason = "unknown"

                    print(
                        f"[ep={ep + 1}/{n_episodes} env={i}] EPISODE END "
                        f"steps={step_counts[i]} reason={reason} "
                        f"attacker_reward={a_ep_rewards[i]:.3f} defender_reward={d_ep_rewards[i]:.3f} "
                        f"attacker_valid={getattr(a_wrappers[i], 'valid_action_count', 0)} attacker_invalid={getattr(a_wrappers[i], 'invalid_action_count', 0)} "
                        f"defender_valid={getattr(d_wrappers[i], 'valid_action_count', 0)} defender_invalid={getattr(d_wrappers[i], 'invalid_action_count', 0)} "
                        f"availability={float(getattr(d_wrappers[i], 'last_availability', 1.0)):.3f} sla_breached={bool(getattr(d_wrappers[i], 'last_sla_breached', False))}"
                    )

                # Aggregate per-episode across envs
                for i in range(n_envs):
                    attacker_rewards.append(a_ep_rewards[i])
                    defender_rewards.append(d_ep_rewards[i])
                    attacker_valid.append(int(getattr(a_wrappers[i], "valid_action_count", 0)))
                    attacker_invalid.append(int(getattr(a_wrappers[i], "invalid_action_count", 0)))
                    defender_valid.append(int(getattr(d_wrappers[i], "valid_action_count", 0)))
                    defender_invalid.append(int(getattr(d_wrappers[i], "invalid_action_count", 0)))
                    episode_steps.append(int(step_counts[i]))

        stats = EvalutionStats(
            episode_steps=episode_steps,
            attacker_rewards=attacker_rewards,
            attacker_valid_actions=attacker_valid,
            attacker_invalid_actions=attacker_invalid,
            defender_rewards=defender_rewards,
            defender_valid_actions=defender_valid,
            defender_invalid_actions=defender_invalid,
        )

        stats.log_results(logger=stats_logger)

    return resolved_run_dir


if __name__ == "__main__":
    evaluate(also_print=False, run_dir="/home/zsh/project/MARLon/marlon/baseline_models/ppo_multi/runs/ppo_marl_multi/20251212_233516")
