from abc import ABC, abstractmethod
import logging
import os
from typing import Callable, List, Optional

import numpy as np
import gymnasium as gym

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.callbacks import BaseCallback

from cyberbattle._env.cyberbattle_env import DefenderConstraint
from marlon.baseline_models.env_wrappers.environment_event_source import EnvironmentEventSource

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.multiagent.evaluation_stats import EvalutionStats
from marlon.baseline_models.multiagent.marlon_agent import MarlonAgent
from marlon.baseline_models.multiagent import marl_algorithm


class RolloutIterationCallback(BaseCallback):
    """Runs a callback every N rollouts (iterations) and once more at the end."""

    def __init__(self, interval: int, on_checkpoint: Callable[[int], None]):
        super().__init__(verbose=0)
        self.interval = interval
        self.on_checkpoint = on_checkpoint
        self.iteration = 0
        self.last_saved: Optional[int] = None

    def _on_rollout_end(self) -> bool:
        self.iteration += 1
        if self.interval > 0 and self.iteration % self.interval == 0:
            self.on_checkpoint(self.iteration)
            self.last_saved = self.iteration
        return True

    def _on_step(self) -> bool:
        # Required by SB3 BaseCallback abstract interface.
        # We hook on rollout boundaries instead of per-step events.
        return True

    def _on_training_end(self) -> None:
        if self.on_checkpoint and self.last_saved != self.iteration:
            self.on_checkpoint(self.iteration)
        return None


class AgentBuilder(ABC):
    '''Builder capable of creating a generic MarlonAgent given a wrapper.'''

    @abstractmethod
    def build(self, wrapper: GymEnv, logger: logging.Logger) -> MarlonAgent:
        '''
        Build a generic MarlonAgent given a wrapper.

        Parameters
        ----------
            wrapper : GymEnv
                The wrapper the agent will operate on.
            logger : Logger
                Logger instance to write logs with.

        Returns
        -------
            A MarlonAgent built for the given wrapper.
        '''
        raise NotImplementedError

class MultiAgentUniverse:
    '''
    Helps build multi-agent environments by handling intricacies of various
    combinations of attacker and defender agents.
    '''

    @classmethod
    def build(cls,
        attacker_builder: AgentBuilder,
        attacker_invalid_action_reward_modifier: float = -1.0,
        attacker_invalid_action_reward_multiplier: float = 1.0,
        defender_builder: Optional[AgentBuilder] = None,
        defender_invalid_action_reward_modifier = -1,
        env_id: str = "CyberBattleToyCtf-v0",
        max_timesteps: int = 2000,
        maximum_node_count: Optional[int] = None,
        maximum_total_credentials: Optional[int] = None,
        maximum_discoverable_credentials_per_action: Optional[int] = None,
        observation_padding: Optional[bool] = None,
        throws_on_invalid_actions: Optional[bool] = False,
        attacker_action_masking: bool = False,
        attacker_loss_reward: float = -5000.0,
        defender_loss_reward: float = -5000.0,
        defender_maintain_sla: float = 0.60,
        defender_reset_on_constraint_broken: bool = True):
        '''
        Static factory method to create a MultiAgentUniverse with the given options.

        Parameters
        ----------
        attacker_builder : AgentBuilder
            A builder that will create an attacker MarlonAgent.
        attacker_invalid_action_reward_modifier : float
            A reward modifier added to all the attacker's invalid action rewards.
        attacker_invalid_action_reward_multiplier : float
            A reward multiplier applied to all the attacker's invalid action rewards.
        defender_builder : AgentBuilder
            A builder that will create a defender MarlonAgent.
        defender_invalid_action_reward_modifier : bool
            A reward modifier added to all the defenders's invalid actions rewards.
        env_id : str
            The gym environment ID to create. Should return a type that inherits CyberBattleEnv.
        max_timesteps : int
            The max timesteps per episode before the simulation is forced to end.
            Useful if training gets stuck on a single episode for too long.
        maximum_node_count : Optional[int]
            Passed through to CyberBattleSim (reduces padded observation sizes such as action masks).
        maximum_total_credentials : Optional[int]
            Passed through to CyberBattleSim (reduces padded observation sizes such as connect masks).
        maximum_discoverable_credentials_per_action : Optional[int]
            Passed through to CyberBattleSim (controls leaked_credentials tuple length).
        observation_padding : Optional[bool]
            Passed through to CyberBattleSim; when True observations are padded to the maximum sizes.
        throws_on_invalid_actions : Optional[bool]
            Passed through to CyberBattleSim; when False invalid actions return a penalty instead of raising.
        attacker_action_masking : bool
            If True, wraps the attacker env with a discrete action space and exposes action masks.
        attacker_loss_reward : float
            Reward applied to the attacker if it loses.
        defender_loss_reward : float
            Reward applied to the defender if it loses.
        defender_maintain_sla : float
            The network availability constraint for the defender.
        defender_reset_on_constraint_broken : bool
            Controls if the environment gets reset when the defender breaks its constraint.

        Returns
        -------
            A MultiAgentUniverse configured with the given options.
        '''

        log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
        logger = logging.Logger('marlon', level=log_level)
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

        env_kwargs = {}
        if maximum_node_count is not None:
            env_kwargs["maximum_node_count"] = int(maximum_node_count)
        if maximum_total_credentials is not None:
            env_kwargs["maximum_total_credentials"] = int(maximum_total_credentials)
        if maximum_discoverable_credentials_per_action is not None:
            env_kwargs["maximum_discoverable_credentials_per_action"] = int(maximum_discoverable_credentials_per_action)
        if observation_padding is not None:
            env_kwargs["observation_padding"] = bool(observation_padding)
        if throws_on_invalid_actions is not None:
            env_kwargs["throws_on_invalid_actions"] = bool(throws_on_invalid_actions)

        if defender_builder:
            cyber_env = gym.make(
                env_id,
                defender_constraint=DefenderConstraint(maintain_sla=defender_maintain_sla),
                losing_reward=defender_loss_reward,
                **env_kwargs,
            )
        else:
            cyber_env = gym.make(env_id, **env_kwargs)

        event_source = EnvironmentEventSource()

        attacker_wrapper = AttackerEnvWrapper(
            cyber_env=cyber_env,
            event_source=event_source,
            max_timesteps=max_timesteps,
            invalid_action_reward_modifier=attacker_invalid_action_reward_modifier,
            invalid_action_reward_multiplier=attacker_invalid_action_reward_multiplier,
            loss_reward=attacker_loss_reward,
            log_episode_end=True,
        )
        if attacker_action_masking:
            from marlon.baseline_models.env_wrappers.action_masking import MaskedDiscreteAttackerWrapper

            attacker_wrapper = MaskedDiscreteAttackerWrapper(attacker_wrapper)
        attacker_agent = attacker_builder.build(attacker_wrapper, logger)

        defender_agent = None
        if defender_builder:
            defender_wrapper = DefenderEnvWrapper(
                cyber_env=cyber_env,
                event_source=event_source,
                attacker_reward_store=attacker_wrapper,
                max_timesteps=max_timesteps,
                invalid_action_reward=defender_invalid_action_reward_modifier,
                defender=True,
                reset_on_constraint_broken=defender_reset_on_constraint_broken,
                loss_reward=defender_loss_reward,
                log_episode_end=True,
            )
            defender_agent = defender_builder.build(defender_wrapper, logger)

        return MultiAgentUniverse(
            attacker_agent=attacker_agent,
            defender_agent=defender_agent,
            max_timesteps=max_timesteps,
            logger=logger
        )

    def __init__(self,
        attacker_agent: MarlonAgent,
        defender_agent: Optional[MarlonAgent],
        max_timesteps: int,
        logger: logging.Logger):

        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
        self.max_timesteps = max_timesteps
        self.logger = logger

    def learn(
        self,
        total_timesteps: int,
        n_eval_episodes: int,
        checkpoint_interval: int = 0,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
    ):
        '''
        Train all agents in the universe for the specified amount of steps or episodes,
        which ever comes first.

        Parameters
        ----------
        total_timesteps : int
            The maximum number of timesteps to train for, across all episodes.
        n_eval_episodes : int
            The maximum number of episodes to train for, regardless of timesteps.
        '''

        self.logger.info('Training started')
        iterations: Optional[int] = None
        if self.defender_agent:
            iterations = marl_algorithm.learn(
                attacker_agent=self.attacker_agent,
                defender_agent=self.defender_agent,
                total_timesteps=total_timesteps,
                n_eval_episodes=n_eval_episodes,
                checkpoint_callback=checkpoint_callback,
                checkpoint_interval=checkpoint_interval,
            )
        else:
            callback: Optional[BaseCallback] = None
            if checkpoint_callback and checkpoint_interval > 0:
                callback = RolloutIterationCallback(checkpoint_interval, checkpoint_callback)
            self.attacker_agent.learn(
                total_timesteps=total_timesteps,
                n_eval_episodes=n_eval_episodes,
                callback=callback
            )
            if callback:
                iterations = callback.iteration

        self.logger.info('Training complete')
        return iterations

    def evaluate(self, n_episodes: int) -> EvalutionStats:
        '''
        Evaluate all agents in the universe for the given number of episodes.

        Parameters
        ----------
        n_episodes : int
            The number of episodes to evaluate for.
            Results will be calculated as averages per episode.
        
        Returns
        -------
            Statistics about the evaluation results.
        '''
        self.logger.info('Evaluation started')

        attacker_rewards = []
        attacker_valid_actions = []
        attacker_invalid_actions = []

        defender_rewards = []
        defender_valid_actions = []
        defender_invalid_actions = []

        episode_steps = []

        for i in range(n_episodes):
            self.logger.info(f'Evaluating episode {i+1} of {n_episodes}')
            # Unified per-step action logs for evaluation.
            if hasattr(self.attacker_agent.wrapper, "configure_step_logging"):
                self.attacker_agent.wrapper.configure_step_logging(enabled=True, prefix=f"[ep={i+1}/{n_episodes}]")
            if self.defender_agent and hasattr(self.defender_agent.wrapper, "configure_step_logging"):
                self.defender_agent.wrapper.configure_step_logging(enabled=True, prefix=f"[ep={i+1}/{n_episodes}]")
            episode_rewards1, episode_rewards2, _ = marl_algorithm.run_episode(
                attacker_agent=self.attacker_agent,
                defender_agent=self.defender_agent,
                max_steps=self.max_timesteps
            )

            # VecEnv auto-reset wipes counters; keep the last-episode snapshot if present.
            attacker_valid = getattr(self.attacker_agent.wrapper, "last_valid_action_count", self.attacker_agent.wrapper.valid_action_count)
            attacker_invalid = getattr(self.attacker_agent.wrapper, "last_invalid_action_count", self.attacker_agent.wrapper.invalid_action_count)

            attacker_rewards.append(sum(episode_rewards1))
            attacker_valid_actions.append(attacker_valid)
            attacker_invalid_actions.append(attacker_invalid)

            episode_steps.append(len(episode_rewards1))

            if self.defender_agent:
                defender_rewards.append(sum(episode_rewards2))
                defender_valid = getattr(self.defender_agent.wrapper, "last_valid_action_count", self.defender_agent.wrapper.valid_action_count)
                defender_invalid = getattr(self.defender_agent.wrapper, "last_invalid_action_count", self.defender_agent.wrapper.invalid_action_count)
                defender_valid_actions.append(defender_valid)
                defender_invalid_actions.append(defender_invalid)

                # Unified per-episode end reason (evaluation only)
                a_outcome = getattr(self.attacker_agent.wrapper, "last_outcome", None)
                d_outcome = getattr(self.defender_agent.wrapper, "last_outcome", None)
                steps = max(len(episode_rewards1), len(episode_rewards2))
                if a_outcome:
                    reason = a_outcome
                elif d_outcome:
                    reason = d_outcome
                elif steps >= self.max_timesteps:
                    reason = "timeout"
                else:
                    reason = "terminated"

                availability = float(getattr(self.defender_agent.wrapper, "last_availability", 1.0))
                sla_breached = bool(getattr(self.defender_agent.wrapper, "last_sla_breached", False))
                self.logger.warning(
                    "[ep=%d/%d] EPISODE END steps=%d reason=%s attacker_reward=%.3f defender_reward=%.3f "
                    "attacker_valid=%d attacker_invalid=%d defender_valid=%d defender_invalid=%d availability=%.3f sla_breached=%s",
                    i + 1,
                    n_episodes,
                    steps,
                    reason,
                    float(sum(episode_rewards1)),
                    float(sum(episode_rewards2)),
                    int(attacker_valid),
                    int(attacker_invalid),
                    int(defender_valid),
                    int(defender_invalid),
                    availability,
                    sla_breached,
                )
            else:
                steps = len(episode_rewards1)
                a_outcome = getattr(self.attacker_agent.wrapper, "last_outcome", None)
                reason = a_outcome or ("timeout" if steps >= self.max_timesteps else "terminated")
                self.logger.warning(
                    "[ep=%d/%d] EPISODE END steps=%d reason=%s attacker_reward=%.3f attacker_valid=%d attacker_invalid=%d",
                    i + 1,
                    n_episodes,
                    steps,
                    reason,
                    float(sum(episode_rewards1)),
                    int(attacker_valid),
                    int(attacker_invalid),
                )

        stats = EvalutionStats(
            episode_steps=episode_steps,
            attacker_rewards=attacker_rewards,
            attacker_valid_actions=attacker_valid_actions,
            attacker_invalid_actions=attacker_invalid_actions,
            defender_rewards=defender_rewards,
            defender_valid_actions=defender_valid_actions,
            defender_invalid_actions=defender_invalid_actions
        )

        stats.log_results(self.logger)

        return stats

    def save(self,
        attacker_filepath: Optional[str] = None,
        defender_filepath: Optional[str] = None):
        '''
        Save all agents in the universe at the specified file paths.

        It is safe to supply a file path for an agent that does not actually
        exist in the universe. Therefore it is safe for both file paths
        to always be supplied, regardless of universe configuration.

        Parameters
        attacker_filepath : Optional[str]
            The file path to save the attacker agent.
        '''

        if attacker_filepath is not None:
            self.logger.info('Attacker agent saving...')
            self.attacker_agent.save(attacker_filepath)

        if defender_filepath is not None and\
            self.defender_agent is not None:

            self.logger.info('Defender agent saving...')
            self.defender_agent.save(defender_filepath)
