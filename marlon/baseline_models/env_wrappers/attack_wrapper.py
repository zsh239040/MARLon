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

    int32_spaces = [
        "customer_data_found",
        "escalation",
        "lateral_move",
        "newly_discovered_nodes_count",
        "probe_result",
        "credential_cache_length",
        "discovered_node_count",
    ]
    _log = logging.getLogger("cyberbattle.attacker")

    def __init__(self,
        cyber_env: CyberBattleEnv,
        event_source: Optional[EnvironmentEventSource] = None,
        max_timesteps=2000,
        invalid_action_reward_modifier=-1,
        invalid_action_reward_multiplier=1,
        loss_reward=-5000,
        log_episode_end: bool = False,
        episode_log_prefix: str = ""):

        super().__init__()
        self.cyber_env: CyberBattleEnv = cyber_env
        self._base_env = cyber_env.unwrapped
        # Use the unwrapped env to avoid gymnasium wrapper deprecation warnings (env.bounds/env.environment access).
        self.bounds: EnvironmentBounds = self._base_env.bounds
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
        # Lightweight debug fields for evaluation tracing (no printing here).
        self.last_action = None
        self.last_attempted_translated_action = None
        self.last_executed_translated_action = None
        self.last_is_invalid = False
        self.last_cyber_reward = 0.0
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_outcome: Optional[str] = None
        self._winning_reward = getattr(self._base_env, "_CyberBattleEnv__WINNING_REWARD", None)  # noqa: SLF001
        self._losing_reward = getattr(self._base_env, "_CyberBattleEnv__LOSING_REWARD", None)  # noqa: SLF001
        # Evaluation-only step logging (opt-in via configure_step_logging()).
        self._step_log_enabled = False
        self._step_log_prefix = ""
        self.last_attempted_action_valid: Optional[bool] = None
        self._last_transformed_observation: Optional[Observation] = None
        self._last_info: Dict[str, Any] = {}
        # Optional training/eval episode end logging.
        self._log_episode_end = bool(log_episode_end)
        self._episode_log_prefix = str(episode_log_prefix)
        self._episode_end_logged = False

        # Use maximum node count from bounds for shaping.
        self.node_count = int(self.bounds.maximum_node_count)
        # Cache flattened sizes to avoid recomputing each step
        self._flat_local_len = self.node_count * int(self.bounds.local_attacks_count)
        self._flat_remote_len = self.node_count * self.node_count * int(self.bounds.remote_attacks_count)
        self._disc_props_len = int(self.bounds.maximum_node_count * self.bounds.property_count)
        self._nodes_priv_len = int(self.bounds.maximum_node_count)
        self._cred_cache_count = int(self.bounds.maximum_total_credentials)

        self.observation_space: Space = self.__create_observation_space(self._base_env)
        self.action_space: Space = self.__create_action_space(self._base_env)

        self.valid_action_count = 0
        self.invalid_action_count = 0

        # Add this object as an observer of the cyber env.
        if event_source is None:
            event_source = EnvironmentEventSource()

        self.event_source = event_source
        event_source.add_observer(self)

        self.reset_request = False

    def configure_step_logging(self, *, enabled: bool, prefix: str = "") -> None:
        self._step_log_enabled = bool(enabled)
        self._step_log_prefix = str(prefix)

    def configure_episode_end_logging(self, *, enabled: bool, prefix: str = "") -> None:
        self._log_episode_end = bool(enabled)
        self._episode_log_prefix = str(prefix)

    def _disc_index_to_node_id(self, idx: int) -> str:
        try:
            discovered = self._base_env._CyberBattleEnv__discovered_nodes  # noqa: SLF001
            if 0 <= idx < len(discovered):
                return str(discovered[idx])
        except Exception:
            pass
        return f"<disc_idx:{idx}>"

    def _format_translated_action(self, translated_action: Optional[Dict[str, Any]]) -> str:
        if not translated_action:
            return "None"
        if "local_vulnerability" in translated_action:
            src_idx, vuln_idx = map(int, translated_action["local_vulnerability"])
            src = self._disc_index_to_node_id(src_idx)
            try:
                vuln_id = str(self._base_env.identifiers.local_vulnerabilities[vuln_idx])
            except Exception:
                vuln_id = f"<local_vuln_idx:{vuln_idx}>"
            return f"LOCAL_EXPLOIT src={src} vuln={vuln_id} (src_idx={src_idx} vuln_idx={vuln_idx})"
        if "remote_vulnerability" in translated_action:
            src_idx, dst_idx, vuln_idx = map(int, translated_action["remote_vulnerability"])
            src = self._disc_index_to_node_id(src_idx)
            dst = self._disc_index_to_node_id(dst_idx)
            try:
                vuln_id = str(self._base_env.identifiers.remote_vulnerabilities[vuln_idx])
            except Exception:
                vuln_id = f"<remote_vuln_idx:{vuln_idx}>"
            return f"REMOTE_EXPLOIT src={src} dst={dst} vuln={vuln_id} (src_idx={src_idx} dst_idx={dst_idx} vuln_idx={vuln_idx})"
        if "connect" in translated_action:
            src_idx, dst_idx, port_idx, cred_idx = map(int, translated_action["connect"])
            src = self._disc_index_to_node_id(src_idx)
            dst = self._disc_index_to_node_id(dst_idx)
            try:
                port = str(self._base_env.identifiers.ports[port_idx])
            except Exception:
                port = f"<port_idx:{port_idx}>"
            return f"CONNECT src={src} dst={dst} port={port} cred_idx={cred_idx} (src_idx={src_idx} dst_idx={dst_idx} port_idx={port_idx})"
        return f"UNKNOWN {translated_action}"

    def _first_owned_node_index(self) -> int:
        """Return a deterministic owned node index (external/discovered index)."""
        n_discovered = len(self._base_env._CyberBattleEnv__discovered_nodes)  # noqa: SLF001
        for idx in range(max(1, n_discovered)):
            if self._base_env.is_node_owned(idx):
                return idx
        return 0

    def __create_observation_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        observation_space = dict(cyber_env.observation_space.spaces)

        # Flatten the nested `action_mask` dict to top-level keys to keep the observation
        # as a flat gymnasium.spaces.Dict (SB3 does not support nested Dict spaces).
        action_mask_space = observation_space.pop("action_mask")
        observation_space["local_vulnerability"] = action_mask_space.spaces["local_vulnerability"]
        observation_space["remote_vulnerability"] = action_mask_space.spaces["remote_vulnerability"]
        observation_space["connect"] = action_mask_space.spaces["connect"]

        # Flatten Tuple spaces into MultiDiscrete vectors (SB3 does not support Tuple spaces).
        # leaked_credentials is a tuple of N entries of shape (4,)
        max_leaked = int(self.bounds.maximum_discoverable_credentials_per_action)
        max_creds = int(self.bounds.maximum_total_credentials)
        max_nodes = int(self.bounds.maximum_node_count)
        port_count = int(self.bounds.port_count)
        observation_space["leaked_credentials"] = spaces.MultiDiscrete(
            np.tile(np.array([2, max_creds, max_nodes, port_count], dtype=np.int32), max_leaked)
        )

        # credential_cache_matrix is a tuple of M entries of shape (2,)
        observation_space["credential_cache_matrix"] = spaces.MultiDiscrete(
            np.tile(np.array([max_nodes, port_count], dtype=np.int32), max_creds)
        )

        # Flatten matrix-shaped MultiDiscrete to a 1-D vector for consistency.
        observation_space["discovered_nodes_properties"] = spaces.MultiDiscrete(
            np.full(self.node_count * int(self.bounds.property_count), 3, dtype=np.int32)
        )

        # Remove raw/non-numeric fields (DummySpace) and legacy info-only keys if present.
        for key in [
            "_discovered_nodes",
            "_explored_network",
            "credential_cache",
            "discovered_nodes",
            "explored_network",
        ]:
            observation_space.pop(key, None)

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
        n_discovered = len(self._base_env._CyberBattleEnv__discovered_nodes)  # noqa: SLF001
        return np.array([i for i in range(n_discovered) if self.cyber_env.is_node_owned(i)], dtype=int)

    def _action_in_discovered_range(self, translated_action: Dict[str, Any]) -> bool:
        """Return True if translated_action indices cannot trigger CyberBattleSim OutOfBoundIndexError."""
        if not translated_action or len(translated_action) != 1:
            return False
        kind = next(iter(translated_action.keys()))
        coords = translated_action[kind]
        try:
            n_discovered = len(self._base_env._CyberBattleEnv__discovered_nodes)  # noqa: SLF001
        except Exception:
            return False

        if kind == "local_vulnerability":
            source_node, _vuln = map(int, coords)
            return source_node < n_discovered
        if kind == "remote_vulnerability":
            source_node, target_node, _vuln = map(int, coords)
            return source_node < n_discovered and target_node < n_discovered
        if kind == "connect":
            source_node, target_node, _port, _cred = map(int, coords)
            return source_node < n_discovered and target_node < n_discovered
        return False

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        # Some code paths (e.g., random-only env stepping) may call step() before an explicit reset().
        # Ensure internal counters are initialized to avoid NoneType arithmetic.
        if self.timesteps is None:
            self.reset()
        self.last_action = np.array(action, copy=True)
        # The first action value corresponds to the subspace
        action_subspace = self.action_subspaces[action[0]]

        # Translate the flattened action back into the nested
        # subspace action for CyberBattle. It takes the form:
        # {'subspace_name': [0, 1, 2]}
        translated_action = {action_subspace[0]: action[action_subspace[1]:action_subspace[2]]}
        self.last_attempted_translated_action = {k: np.array(v, copy=True) for k, v in translated_action.items()}

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

        # CyberBattleSim does not redirect invalid actions; it clips negative rewards to 0.
        # However, out-of-range node indices raise OutOfBoundIndexError and cause CyberBattleSim
        # to return a blank observation. To avoid injecting blank observations into learning,
        # we intercept only those out-of-range actions and treat them as a no-op at the wrapper level.
        reward_modifier = 0.0
        is_invalid = False
        attempted_action_valid = bool(self._action_in_discovered_range(translated_action))
        self.last_attempted_action_valid = attempted_action_valid
        if not attempted_action_valid:
            # translated_action = self._base_env.sample_valid_action(kinds=[0, 1, 2])
            self.invalid_action_count += 1
            reward_modifier += float(self.invalid_action_reward_modifier)
            is_invalid = True

            observation = self._last_transformed_observation
            info = dict(self._last_info) if isinstance(self._last_info, dict) else {}
            info["invalid_action"] = True
            info["cyber_step_executed"] = False
            reward = 0.0
            terminated = False
            truncated = False
            transformed_observation = observation
            self.last_executed_translated_action = None
        else:
            self.valid_action_count += 1
            self.last_executed_translated_action = {k: np.array(v, copy=True) for k, v in translated_action.items()}

            step_result = self.cyber_env.step(translated_action)
            if len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
            else:
                observation, reward, done, info = step_result
                terminated, truncated = done, False
            transformed_observation = self.transform_observation(observation)
            self._last_transformed_observation = transformed_observation
            self._last_info = dict(info) if isinstance(info, dict) else {}
            self.last_cyber_reward = float(reward)

        self.last_is_invalid = is_invalid
        self.cyber_rewards.append(float(reward))
        self.last_cyber_reward = float(reward)

        self.last_outcome = None
        if terminated or truncated:
            if not truncated and self._winning_reward is not None and float(reward) == float(self._winning_reward):
                self.last_outcome = "attacker_win"
                self._log.warning("Attacker Won")
            elif not truncated and self._losing_reward is not None and float(reward) == float(self._losing_reward):
                self.last_outcome = "attacker_loss"
                self._log.warning("Attacker Lost")
            elif truncated:
                self.last_outcome = "truncated"
            else:
                self.last_outcome = "terminated"

        self.timesteps += 1
        if self.reset_request:
            truncated = True

        if self.timesteps >= self.max_timesteps:
            truncated = True
            if self.last_outcome is None:
                self.last_outcome = "timeout"

        reward = float(reward) + float(reward_modifier)
        done = bool(terminated or truncated)
        self.rewards.append(reward)
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)

        if self._step_log_enabled and self._log.isEnabledFor(logging.INFO):
            prefix = f"{self._step_log_prefix} " if self._step_log_prefix else ""
            self._log.info(
                "%sATTACKER step=%d attempted_valid=%s fallback_sampled=%s reward=%.3f cyber_reward=%.3f attempted=%s executed_action=%s done=%s",
                prefix,
                int(self.timesteps),
                bool(self.last_attempted_action_valid),
                bool(is_invalid),
                float(reward),
                float(self.last_cyber_reward),
                self._format_translated_action(self.last_attempted_translated_action),
                self._format_translated_action(self.last_executed_translated_action),
                bool(done),
            )

        if self._log_episode_end and done and not self._episode_end_logged:
            if self.last_outcome:
                reason = self.last_outcome
            elif self.last_truncated:
                reason = "truncated"
            elif self.last_terminated:
                reason = "terminated"
            elif self.timesteps >= self.max_timesteps:
                reason = "timeout"
            else:
                reason = "unknown"
            prefix = f"{self._episode_log_prefix} " if self._episode_log_prefix else ""
            total_reward = float(np.sum(self.rewards)) if self.rewards else 0.0
            cyber_total = float(np.sum(self.cyber_rewards)) if self.cyber_rewards else 0.0
            logging.warning(
                "%sATTACKER EPISODE END steps=%d reason=%s total_reward=%.3f cyber_total=%.3f valid=%d invalid=%d",
                prefix,
                int(self.timesteps),
                reason,
                total_reward,
                cyber_total,
                int(self.valid_action_count),
                int(self.invalid_action_count),
            )
            self._episode_end_logged = True

        return transformed_observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> Tuple[Observation, Dict[str, Any]]:
        logging.debug('Reset Attacker')
        # If this reset comes after an episode ended, log the termination reason.
        if (
            self._log_episode_end
            and not self._episode_end_logged
            and self.timesteps is not None
            and (self.timesteps > 0 or len(self.rewards) > 0)
        ):
            # Determine best-effort end reason.
            if self.last_outcome:
                reason = self.last_outcome
            elif self.last_truncated:
                reason = "truncated"
            elif self.last_terminated:
                reason = "terminated"
            elif self.timesteps >= self.max_timesteps:
                reason = "timeout"
            else:
                reason = "unknown"
            prefix = f"{self._episode_log_prefix} " if self._episode_log_prefix else ""
            total_reward = float(np.sum(self.rewards)) if self.rewards else 0.0
            cyber_total = float(np.sum(self.cyber_rewards)) if self.cyber_rewards else 0.0
            logging.warning(
                "%sATTACKER EPISODE END steps=%d reason=%s total_reward=%.3f cyber_total=%.3f valid=%d invalid=%d",
                prefix,
                int(self.timesteps),
                reason,
                total_reward,
                cyber_total,
                int(self.valid_action_count),
                int(self.invalid_action_count),
            )
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
        self.last_action = None
        self.last_attempted_translated_action = None
        self.last_executed_translated_action = None
        self.last_is_invalid = False
        self.last_cyber_reward = 0.0
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_outcome = None
        self.last_attempted_action_valid = None
        self._last_transformed_observation = self.transform_observation(observation)
        self._last_info = dict(info) if isinstance(info, dict) else {}
        self._episode_end_logged = False

        return self._last_transformed_observation, info

    def on_reset(self, last_rewards):
        logging.info('on_reset Attacker')
        self.reset_request = True

    def transform_observation(self, observation) -> Observation:
        # Flatten the action_mask field to top-level keys.
        action_mask = observation.pop("action_mask")
        observation["local_vulnerability"] = np.asarray(action_mask["local_vulnerability"], dtype=np.int8)
        observation["remote_vulnerability"] = np.asarray(action_mask["remote_vulnerability"], dtype=np.int8)
        observation["connect"] = np.asarray(action_mask["connect"], dtype=np.int8)

        # Flatten tuple-encoded fields into fixed-size vectors.
        leaked = observation.get("leaked_credentials", tuple())
        if isinstance(leaked, tuple):
            if leaked:
                observation["leaked_credentials"] = np.asarray(
                    np.concatenate([np.asarray(x, dtype=np.int32) for x in leaked]),
                    dtype=np.int32,
                )
            else:
                observation["leaked_credentials"] = np.zeros((0,), dtype=np.int32)
        else:
            observation["leaked_credentials"] = np.asarray(leaked, dtype=np.int32).reshape(-1)

        cred_cache = observation.get("credential_cache_matrix", tuple())
        if isinstance(cred_cache, tuple):
            if cred_cache:
                observation["credential_cache_matrix"] = np.asarray(
                    np.concatenate([np.asarray(x, dtype=np.int32) for x in cred_cache]),
                    dtype=np.int32,
                )
            else:
                observation["credential_cache_matrix"] = np.zeros((0,), dtype=np.int32)
        else:
            observation["credential_cache_matrix"] = np.asarray(cred_cache, dtype=np.int32).reshape(-1)

        # Flatten matrix-shaped MultiDiscrete fields.
        observation["discovered_nodes_properties"] = np.asarray(observation["discovered_nodes_properties"], dtype=np.int32).reshape(self._disc_props_len)
        observation["nodes_privilegelevel"] = np.asarray(observation["nodes_privilegelevel"], dtype=np.int32).reshape(self._nodes_priv_len)

        # Remove raw/non-numeric fields that SB3 cannot consume.
        observation.pop("_discovered_nodes", None)
        observation.pop("_explored_network", None)
        observation.pop("credential_cache", None)
        observation.pop("discovered_nodes", None)
        observation.pop("explored_network", None)

        # Stable baselines does not like numpy wrapped ints.
        for space in self.int32_spaces:
            if space in observation:
                observation[space] = int(observation[space])

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
