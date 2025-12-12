import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from plotly.missing_ipywidgets import FigureWidget
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.space import Space

from cyberbattle._env.cyberbattle_env import Action, CyberBattleEnv, EnvironmentBounds, Observation
from cyberbattle.simulation import commandcontrol, model

from marlon.baseline_models.env_wrappers.environment_event_source import IEnvironmentObserver, EnvironmentEventSource
from marlon.baseline_models.env_wrappers.reward_store import IRewardStore

class AttackerEnvWrapper(gym.Env, IRewardStore, IEnvironmentObserver):
    """Wraps a CyberBattleEnv for stablebaselines-3 models to learn how to attack."""

    nested_spaces = ['credential_cache_matrix', 'leaked_credentials']
    other_removed_spaces = ['connect']
    int32_spaces = ['customer_data_found', 'escalation', 'lateral_move', 'newly_discovered_nodes_count', 'probe_result']

    def __init__(self,
        cyber_env: CyberBattleEnv,
        event_source: Optional[EnvironmentEventSource] = None,
        max_timesteps=2000,
        invalid_action_reward_modifier=-1,
        invalid_action_reward_multiplier=1,
        loss_reward=-5000):

        super().__init__()
        self.cyber_env: CyberBattleEnv = cyber_env
        self.bounds: EnvironmentBounds = self.cyber_env.bounds
        self.max_timesteps = max_timesteps
        self.invalid_action_reward_modifier = invalid_action_reward_modifier
        self.invalid_action_reward_multiplier = invalid_action_reward_multiplier
        self.loss_reward = loss_reward

        # These should be set during reset()
        self.timesteps = None
        self.cyber_rewards = [] # The rewards as given by CyberBattle, before modification.
        self.rewards = [] # The rewards returned by this wrapper, after modification.
        # Track last episode counts so they survive VecEnv auto-reset
        self.last_valid_action_count = 0
        self.last_invalid_action_count = 0

        # Use maximum node count from bounds for shaping.
        self.node_count = int(self.bounds.maximum_node_count)
        # Cache flattened sizes to avoid recomputing each step
        self._flat_local_len = self.node_count * int(self.bounds.local_attacks_count)
        self._flat_remote_len = self.node_count * self.node_count * int(self.bounds.remote_attacks_count)
        self._disc_props_len = int(self.bounds.maximum_node_count * self.bounds.property_count)
        self._nodes_priv_len = int(self.bounds.maximum_node_count)
        self._cred_cache_count = int(self.bounds.maximum_total_credentials)

        self.observation_space: Space = self.__create_observation_space(cyber_env)
        self.action_space: Space = self.__create_action_space(cyber_env)

        self.valid_action_count = 0
        self.invalid_action_count = 0

        # Add this object as an observer of the cyber env.
        if event_source is None:
            event_source = EnvironmentEventSource()

        self.event_source = event_source
        event_source.add_observer(self)

        self.reset_request = False

    def __create_observation_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        observation_space = cyber_env.observation_space.__dict__['spaces'].copy()

        # Flatten the action_mask field.
        observation_space['local_vulnerability'] = observation_space['action_mask']['local_vulnerability']
        observation_space['remote_vulnerability'] = observation_space['action_mask']['remote_vulnerability']
        observation_space['connect'] = observation_space['action_mask']['connect']
        del observation_space['action_mask']

        # Change action_mask spaces to use node count instead of maximum node count.
        observation_space['local_vulnerability'] = spaces.MultiBinary(self.node_count * int(self.bounds.local_attacks_count))
        observation_space['remote_vulnerability'] = spaces.MultiBinary(self.node_count * self.node_count * int(self.bounds.remote_attacks_count))

        # Remove 'info' fields added by cyberbattle that do not represent algorithm inputs
        for key in ['credential_cache', 'discovered_nodes', '_discovered_nodes', 'explored_network', '_explored_network']:
            if key in observation_space:
                del observation_space[key]

        # TODO: Reformat these spaces so they don't have to be removed
        # Remove nested Tuple/Dict spaces
        for space in self.nested_spaces + self.other_removed_spaces:
            del observation_space[space]

        # Replace matrix-shaped MultiDiscrete with flattened version (SB3 expects 1-D nvec)
        observation_space['discovered_nodes_properties'] = spaces.MultiDiscrete(
            np.full(self.node_count * int(self.bounds.property_count), 3, dtype=int)
        )

        # This is incorrectly set to spaces.MultiBinary(2)
        # It's a single value in the returned observations
        observation_space['customer_data_found'] = spaces.Discrete(2)

        # This is incorrectly set to spaces.MultiDiscrete(model.PrivilegeLevel.MAXIMUM + 1), when it is only one value
        observation_space['escalation'] = spaces.Discrete(model.PrivilegeLevel.MAXIMUM + 1)

        return spaces.Dict(observation_space)

    def __create_action_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        self.action_subspaces = {}
        # First action defines which action subspace to use
        # local_vulnerability, remote_vulnerability, or connect
        action_space = [3]

        # CyberBattle's action space is a dict of nested action spaces.
        # We need to flatten it into a single multidiscrete and keep
        # track of which values correspond to which nested values so
        # we can reconstruct the action later.
        subspace_index = 0
        for (key, value) in cyber_env.action_space.spaces.items():
            subspace_start = len(action_space)
            for vec in value.nvec:
                action_space.append(vec)

            # Action subspace takes the form:
            # [('subspace_name', 1, 3), ('subspace_name2', 3, 5)]
            self.action_subspaces[subspace_index] = (key, subspace_start, len(action_space))
            subspace_index += 1

        return spaces.MultiDiscrete(action_space)

    def __get_owned_nodes(self):
        return np.nonzero(self.__get_privilegelevel_array())[0]

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        # The first action value corresponds to the subspace
        action_subspace = self.action_subspaces[action[0]]

        # Translate the flattened action back into the nested
        # subspace action for CyberBattle. It takes the form:
        # {'subspace_name': [0, 1, 2]}
        translated_action = {action_subspace[0]: action[action_subspace[1]:action_subspace[2]]}

        # For reference:
        # ```python
        # action_spaces: ActionSpaceDict = {
        #     "local_vulnerability": spaces.MultiDiscrete(
        #         # source_node_id, vulnerability_id
        #         [maximum_node_count, local_vulnerabilities_count]),
        #     "remote_vulnerability": spaces.MultiDiscrete(
        #         # source_node_id, target_node_id, vulnerability_id
        #         [maximum_node_count, maximum_node_count, remote_vulnerabilities_count]),
        #     "connect": spaces.MultiDiscrete(
        #         # source_node_id, target_node_id, target_port, credential_id
        #         # (by index of discovery: 0 for initial node, 1 for first discovered node, ...)
        #         [maximum_node_count, maximum_node_count, port_count, maximum_total_credentials])
        # }
        # ```

        # Validate action; if invalid, fall back to sampling a valid one.
        reward_modifier = 0
        is_invalid = False
        if not self.cyber_env.is_action_valid(translated_action):
            translated_action = self.cyber_env.sample_valid_action(kinds=[0, 1, 2])
            self.invalid_action_count += 1
            reward_modifier += self.invalid_action_reward_modifier
            is_invalid = True
        else:
            self.valid_action_count += 1

        step_result = self.cyber_env.step(translated_action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            observation, reward, done, info = step_result
            terminated, truncated = done, False
        transformed_observation = self.transform_observation(observation)
        self.cyber_rewards.append(reward)

        if done and not truncated:
            logging.warning("Attacker Won")

        self.timesteps += 1
        if self.reset_request:
            truncated = True

        if self.timesteps > self.max_timesteps:
            truncated = True

        done = terminated or truncated

        # If action was invalid, multiplier is applied before reward modifier
        if is_invalid:
            reward = reward * self.invalid_action_reward_multiplier

        reward += reward_modifier
        self.rewards.append(reward)

        return transformed_observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> Tuple[Observation, Dict[str, Any]]:
        logging.debug('Reset Attacker')
        if not self.reset_request:
            last_reward = self.rewards[-1] if len(self.rewards) > 0 else 0
            self.event_source.notify_reset(last_reward)

        # Preserve counts from the episode that just ended (VecEnv resets automatically on done)
        self.last_valid_action_count = self.valid_action_count
        self.last_invalid_action_count = self.invalid_action_count

        reset_result = self.cyber_env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, info = reset_result
        else:
            observation = reset_result
            info = {}

        self.reset_request = False
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.timesteps = 0
        self.cyber_rewards = []
        self.rewards = []

        return self.transform_observation(observation), info

    def on_reset(self, last_rewards):
        logging.info('on_reset Attacker')
        self.reset_request = True

    def transform_observation(self, observation) -> Observation:
        # Flatten the action_mask field
        observation['local_vulnerability'] = observation['action_mask']['local_vulnerability']
        observation['remote_vulnerability'] = observation['action_mask']['remote_vulnerability']
        observation['connect'] = observation['action_mask']['connect']
        del observation['action_mask']

        # TODO: Retain real values
        observation['credential_cache_matrix'] = tuple(np.zeros((self._cred_cache_count, 2), dtype=np.float32))
        observation['discovered_nodes_properties'] = np.zeros((self._disc_props_len,), dtype=np.float32)
        observation['nodes_privilegelevel'] = np.zeros((self._nodes_priv_len,), dtype=np.float32)

        # Flatten action_mask subspaces using vectorized reshape
        # local_vulnerability comes in shape (node_count, local_attacks_count,)
        # but needs to be (node_count * local_attacks_count,)
        observation['local_vulnerability'] = np.asarray(observation['local_vulnerability'], dtype=np.float32).reshape(self._flat_local_len)

        # remote_vulnerability comes in shape (node_count, node_count, remote_attacks_count,)
        # but needs to be (node_count * node_count * remote_attacks_count,)
        observation['remote_vulnerability'] = np.asarray(observation['remote_vulnerability'], dtype=np.float32).reshape(self._flat_remote_len)

        # Remove 'info' fields added by cyberbattle that do not represent algorithm inputs
        for key in ['credential_cache', 'discovered_nodes', '_discovered_nodes', 'explored_network', '_explored_network']:
            observation.pop(key, None)

        # Stable baselines does not like numpy wrapped ints
        for space in self.int32_spaces:
            observation[space] = int(observation[space])

        # TODO: Reformat these spaces so they don't have to be removed
        # Remove nested Tuple/Dict spaces
        for space in self.nested_spaces + self.other_removed_spaces:
            del observation[space]

        return observation

    def close(self) -> None:
        return self.cyber_env.close()

    def render(self, mode: str = 'human') -> None:
        return self.cyber_env.render(mode)

    def render_as_fig(self, print_attacks=False) -> FigureWidget:
        # NOTE: This method is exactly the same as CyberBattleEnv.render_as_fig() except where noted.

        debug = commandcontrol.EnvironmentDebugging(self.cyber_env._actuator)
        # CHANGE: Parameter to decide whether to print this.
        if print_attacks:
            self.cyber_env._actuator.print_all_attacks()

        # plot the cumulative reward and network side by side using plotly
        fig = make_subplots(rows=1, cols=2)

        # CHANGE: Uses this environment's rewards instead of CyberBattle's.
        fig.add_trace(go.Scatter(y=np.array(self.cyber_rewards).cumsum(),
            name='cumulative reward'), row=1, col=1)

        traces, layout = debug.network_as_plotly_traces(xref="x2", yref="y2")
        for trace in traces:
            fig.add_trace(trace, row=1, col=2)
        fig.update_layout(layout)
        return fig

    @property
    def episode_rewards(self) -> List[float]:
        return self.cyber_rewards
