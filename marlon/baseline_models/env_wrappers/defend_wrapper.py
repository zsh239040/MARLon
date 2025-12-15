from typing import Any, Dict, Optional, Tuple, TypedDict
import boolean
import logging

import numpy as np

from plotly.missing_ipywidgets import FigureWidget

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.space import Space

from cyberbattle.simulation import model
from cyberbattle._env.cyberbattle_env import CyberBattleEnv, EnvironmentBounds, Observation

from marlon.baseline_models.env_wrappers.environment_event_source import IEnvironmentObserver, EnvironmentEventSource
from marlon.baseline_models.env_wrappers.reward_store import IRewardStore
from marlon.defender_agents.defender import LearningDefender


Defender_Observation = TypedDict('Defender_Observation', {'infected_nodes': np.ndarray,
                                                          'incoming_firewall_status':np.ndarray,
                                                          'outgoing_firewall_status':np.ndarray,
                                                          'services_status':np.ndarray})
class DefenderEnvWrapper(gym.Env, IEnvironmentObserver):
    """Wraps a CyberBattleEnv for stablebaselines-3 models to learn how to defend."""

    nested_spaces = ['credential_cache_matrix', 'leaked_credentials']
    other_removed_spaces = ['local_vulnerability', 'remote_vulnerability', 'connect']
    int32_spaces = ['customer_data_found', 'escalation', 'lateral_move', 'newly_discovered_nodes_count', 'probe_result']
    firewall_rule_list = ["RDP", "SSH", "HTTPS", "HTTP", "su", "sudo"]
    _log = logging.getLogger("cyberbattle.defender")

    def __init__(self,
        cyber_env: CyberBattleEnv,
        attacker_reward_store: IRewardStore,
        event_source: Optional[EnvironmentEventSource] = None,
        defender: bool = False,
        max_timesteps=100,
        invalid_action_reward=0,
        reset_on_constraint_broken = True,
        loss_reward: float = -5000.0,
        sla_worsening_penalty_scale: float = 200.0,
        log_episode_end: bool = False,
        episode_log_prefix: str = ""):

        super().__init__()
        self.defender = None
        self.cyber_env: CyberBattleEnv = cyber_env
        self._base_env = cyber_env.unwrapped
        self._actuator = self._base_env._defender_actuator  # noqa: SLF001
        self._defender_constraint = self._base_env._CyberBattleEnv__defender_constraint  # noqa: SLF001
        self.bounds: EnvironmentBounds = self._base_env.bounds
        self.num_services = 0
        self.observation_space: Space = self.__create_observation_space(cyber_env)
        self.action_space: Space = self.__create_defender_action_space(cyber_env)
        self.network_availability: float = 1.0
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.max_timesteps = max_timesteps
        self.timesteps = 0
        self.rewards = []
        self.attacker_reward_store = attacker_reward_store
        self.first = True
        self.reset_request = False
        self.invalid_action_penalty = invalid_action_reward
        self.loss_reward = loss_reward
        # SLA shaping parameters. loss_reward is treated as a one-time penalty
        # when SLA is first broken; subsequent penalties are scaled by worsening only.
        self.sla_worsening_penalty_scale = sla_worsening_penalty_scale
        self._has_breached_sla = False
        self._prev_network_availability = float(self._actuator.network_availability)
        # Add this object as an observer of the cyber env.
        if event_source is None:
            event_source = EnvironmentEventSource()

        self.event_source = event_source
        event_source.add_observer(self)
        assert defender is not None, "Attempting to use the defender environment without a defender present."
        self.defender: LearningDefender = LearningDefender(cyber_env)
        self.__last_attacker_reward = None
        self.reset_on_constraint_broken = reset_on_constraint_broken
        # Track last episode counts to survive VecEnv auto-reset
        self.last_valid_action_count = 0
        self.last_invalid_action_count = 0
        # Lightweight debug fields for evaluation tracing (no printing here).
        self.last_action = None
        self.last_action_valid = None
        self.last_reward = 0.0
        self.last_availability = float(self._actuator.network_availability)
        self.last_worsening = 0.0
        self.last_sla_breached = False
        self.last_terminated = False
        self.last_truncated = False
        self.last_outcome: Optional[str] = None
        # Evaluation-only step logging (opt-in via configure_step_logging()).
        self._step_log_enabled = False
        self._step_log_prefix = ""
        # Optional training/eval episode end logging.
        self._log_episode_end = bool(log_episode_end)
        self._episode_log_prefix = str(episode_log_prefix)
        self._episode_end_logged = False

    def configure_step_logging(self, *, enabled: bool, prefix: str = "") -> None:
        self._step_log_enabled = bool(enabled)
        self._step_log_prefix = str(prefix)

    def configure_episode_end_logging(self, *, enabled: bool, prefix: str = "") -> None:
        self._log_episode_end = bool(enabled)
        self._episode_log_prefix = str(prefix)

    def _format_action(self, action) -> str:
        if action is None:
            return "None"
        try:
            arr = np.asarray(action, dtype=int)
        except Exception:
            return repr(action)
        if arr.size == 0:
            return "SKIP"
        action_number = int(arr[0])
        nodes = list(self._base_env.environment.network.nodes)

        def node_name(idx: int) -> str:
            if 0 <= idx < len(nodes):
                return str(nodes[idx])
            return f"<node_idx:{idx}>"

        def firewall_port(port_idx: int) -> str:
            if 0 <= port_idx < len(self.firewall_rule_list):
                return self.firewall_rule_list[port_idx]
            return f"<port_idx:{port_idx}>"

        if action_number == 0:
            return f"REIMAGE node={node_name(int(arr[1]))}"
        if action_number == 1:
            direction = "incoming" if bool(arr[4]) else "outgoing"
            return f"BLOCK_TRAFFIC node={node_name(int(arr[2]))} port={firewall_port(int(arr[3]))} dir={direction}"
        if action_number == 2:
            direction = "incoming" if bool(arr[7]) else "outgoing"
            return f"ALLOW_TRAFFIC node={node_name(int(arr[5]))} port={firewall_port(int(arr[6]))} dir={direction}"
        if action_number == 3:
            node_id = node_name(int(arr[8]))
            service_idx = int(arr[9])
            try:
                node_info = self._base_env.environment.get_node(nodes[int(arr[8])])
                service_name = str(node_info.services[service_idx])
            except Exception:
                service_name = f"<service_idx:{service_idx}>"
            return f"STOP_SERVICE node={node_id} service={service_name}"
        if action_number == 4:
            node_id = node_name(int(arr[10]))
            service_idx = int(arr[11])
            try:
                node_info = self._base_env.environment.get_node(nodes[int(arr[10])])
                service_name = str(node_info.services[service_idx])
            except Exception:
                service_name = f"<service_idx:{service_idx}>"
            return f"START_SERVICE node={node_id} service={service_name}"
        return f"UNKNOWN({action_number}) raw={arr.tolist()}"

    def __create_observation_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        """Creates a compatible version of the attackers observation space."""
        # Calculate how many services there are, this is used to define the maximum number of services active at once.
        for _, node in model.iterate_network_nodes(self._base_env.environment.network):
            for _ in node.services:
                self.num_services +=1
        # All spaces are MultiBinary.
        return spaces.Dict({'infected_nodes': spaces.MultiBinary(len(list(self._base_env.environment.network.nodes))),
                            'incoming_firewall_status': spaces.MultiBinary(len(self.firewall_rule_list)*len(list(self._base_env.environment.network.nodes))),
                            'outgoing_firewall_status': spaces.MultiBinary(len(self.firewall_rule_list)*len(list(self._base_env.environment.network.nodes))),
                            'services_status': spaces.MultiBinary(self.num_services)})

    def __create_defender_action_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        # 0th index of the action defines which action to use (reimage, block_traffic, allow_traffic, stop_service, start_service)
        # Index 1 is the possible nodes to reimage (all nodes) (Only used on action 0)
        # Index 2, 3, 4 are for action 1 (block traffic) 2nd = node to block on, 3rd =Port to block, 4th = incoming or outgoing
        # Index 5, 6, 7 relate to action 2 (allow traffic), 5th = node to allow on, 6th = Port to allow, 7th = incoming or outgoing
        # Index 8 and 9 are for action 3 (stop service), 8th = node to stop service on, 9th = port to stop service
        # Index 10 and 11 are for action 4 (start service), 10th = node to start service on, 11th = port to start service on.
        total_actions = 5
        reimage_node_number = len(self._base_env.environment.network.nodes)
        block_traffic_node = len(self._base_env.environment.network.nodes)
        block_traffic_port = 6
        block_traffic_incoming = 2
        allow_traffic_node = len(self._base_env.environment.network.nodes)
        allow_traffic_port = 6
        allow_traffic_incoming = 2
        stop_service_node = len(self._base_env.environment.network.nodes)
        stop_service_port = 3
        start_service_node = len(self._base_env.environment.network.nodes)
        start_service_port = 3
        action_space = [total_actions, reimage_node_number, block_traffic_node, block_traffic_port, block_traffic_incoming, allow_traffic_node, allow_traffic_port, allow_traffic_incoming, stop_service_node, stop_service_port, start_service_node, start_service_port]
        logging.info(f"Action space defender = {action_space}")
        return spaces.MultiDiscrete(action_space)

    def step(self, action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        terminated = False
        truncated = False
        reward = 0.0
        invalid_penalty = 0.0
        attacker_component = 0.0
        sla_loss_penalty = 0.0
        sla_worsening_penalty = 0.0
        used_action = action
        # Check for action validity
        if action is None or (hasattr(action, "__len__") and len(action) == 0):
            # Treat empty action as a valid no-op (useful for safe-random defender baselines).
            self.last_action = np.array([], dtype=int)
            action_valid = True
            self.last_action_valid = True
            used_action = []
        else:
            self.last_action = np.array(action, copy=True) if hasattr(action, "__len__") else action
            action_valid = bool(self.is_defender_action_valid(action))
        self.last_action_valid = action_valid
        if not action_valid:
            logging.info(f"Action chosen is outside action space. Defender will skip this turn. Action = {action}")
            self.invalid_action_count += 1
            invalid_penalty = float(self.invalid_action_penalty)
            reward += invalid_penalty
            # If the action is invalid, pass an empty list to the defender
            used_action = []
        else:
            self.valid_action_count += 1
        
        self.defender.executeAction(used_action)
        current_availability = float(self._actuator.network_availability)
        self.last_availability = current_availability
        sla_threshold = float(self._defender_constraint.maintain_sla)
        self.last_sla_breached = bool(current_availability < sla_threshold)
        worsening = float(self._prev_network_availability - current_availability)
        self.last_worsening = float(max(0.0, worsening))
        # Take the reward gained this step from the attacker's step and invert it so the defender
        # loses more reward if the attacker succeeds.
        if self.attacker_reward_store.episode_rewards:
            attacker_component = float(-1 * self.attacker_reward_store.episode_rewards[-1])
            reward += attacker_component

        if current_availability < sla_threshold:
            # One-time penalty on first breach to avoid exploding reward scales.
            if not self._has_breached_sla:
                sla_loss_penalty = float(self.loss_reward)
                reward += sla_loss_penalty
                logging.warning("Defender Lost")
                if self.reset_on_constraint_broken:
                    terminated = True
                    self.last_outcome = "sla_breached"
                self._has_breached_sla = True
            else:
                # Only keep penalizing if the constraint gets worse after already being breached.
                if worsening > 0:
                    sla_worsening_penalty = float(-self.sla_worsening_penalty_scale * worsening)
                    reward += sla_worsening_penalty
        else:
            # Reset breach state when SLA recovered so future breaches are penalized again.
            self._has_breached_sla = False

        self._prev_network_availability = current_availability
        if self._base_env._CyberBattleEnv__defender_goal_reached():  # noqa: SLF001
            reward = self._base_env._CyberBattleEnv__WINNING_REWARD  # noqa: SLF001
            terminated = True
            self.last_outcome = "defender_win"
            self._log.warning("Defender Won (eviction goal reached)")
        # Generate the new defender observation based on the defender's action
        defender_observation = self.observe()
        self.timesteps += 1

        if self.reset_request:
            truncated = True
            reward = -1*self.__last_attacker_reward
        elif self.timesteps >= self.max_timesteps:
            truncated = True

        if self.last_outcome is None and truncated and self.timesteps >= self.max_timesteps:
            self.last_outcome = "timeout"

        if self._step_log_enabled and self._log.isEnabledFor(logging.INFO):
            prefix = f"{self._step_log_prefix} " if self._step_log_prefix else ""
            act_desc = self._format_action(self.last_action)
            self._log.info(
                "%sDEFENDER step=%d action=%s valid=%s reward=%.3f (attacker=%.3f invalid=%.3f sla_loss=%.3f sla_worsen=%.3f) availability=%.3f worsening=%.3f done=%s",
                prefix,
                int(self.timesteps),
                act_desc,
                bool(action_valid),
                float(reward),
                float(attacker_component),
                float(invalid_penalty),
                float(sla_loss_penalty),
                float(sla_worsening_penalty),
                float(current_availability),
                float(max(0.0, worsening)),
                bool(terminated or truncated),
            )

        self.rewards.append(reward)
        done = terminated or truncated
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
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
            logging.warning(
                "%sDEFENDER EPISODE END steps=%d reason=%s total_reward=%.3f valid=%d invalid=%d availability=%.3f sla_breached=%s",
                prefix,
                int(self.timesteps),
                reason,
                total_reward,
                int(self.valid_action_count),
                int(self.invalid_action_count),
                float(self.last_availability),
                bool(self.last_sla_breached),
            )
            self._episode_end_logged = True
        return defender_observation, reward, terminated, truncated, {}

    def is_defender_action_valid(self, action) -> boolean:
        """Determines if a given action is valid within the environment."""
        
        def get_node_and_info(node_from_action: int):
            """Returns the node id and info for a given node"""
            node_id = get_node_from_action(node_from_action)
            node_info = get_node_info(node_id)
            return node_id, node_info

        def get_node_from_action(node_from_action: int):
            """Gets the node id from an action"""
            return list(self._base_env.environment.network.nodes)[node_from_action]

        def get_node_info(node_id: model.NodeID):
            """Given a node ID, find the corresponding node info"""
            return self._base_env.environment.get_node(node_id)

        def node_exists(node_id: model.NodeID):
            """Determines if a node exists in the network"""
            return node_id in list(self._base_env.environment.network.nodes)

        def node_running(node_info: model.NodeInfo):
            """Determines if a node is currently running"""
            return node_info.status == model.MachineStatus.Running

        def node_exists_and_running(node_from_action: int):
            """Determines if a node exists in the network, and if so if it is running."""
            node_id, node_info = get_node_and_info(node_from_action)
            return (node_exists(node_id) and node_running(node_info))

        def is_reimagable(node_info: model.NodeInfo):
            """Checks if a given node is reimagable"""
            return node_info.reimagable

        def firewall_rule_exists(node_info: model.NodeInfo, port_from_action: int, incoming :bool):
            """Checks a node to see if a given firewall rule exists on it."""
            firewall_list = []
            if incoming:
                for rule in node_info.firewall.incoming:
                    firewall_list.append(rule.port)
            else:
                for rule in node_info.firewall.outgoing:
                    firewall_list.append(rule.port)

            return self.firewall_rule_list[port_from_action] in firewall_list

        def service_exists(node_info: model.NodeInfo, service_from_action: int):
            """Checks if a service exists on a node (Only checks if the service is out of bounds for the node)"""
            return service_from_action < len(node_info.services)
        action_number = action[0]
        if action_number == 0:
            # REIMAGE
            _, node_info = get_node_and_info(action[1])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            return node_exists_and_running(action[1]) and is_reimagable(node_info)

        elif action_number == 1:
            # block traffic
            _, node_info = get_node_and_info(action[2])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # The firewall rule needs to exist as well to block the traffic.
            return node_exists_and_running(action[2]) and firewall_rule_exists(node_info, action[3], bool(action[4]))

        elif action_number == 2:
            # allow traffic
            _, node_info = get_node_and_info(action[5])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            return node_exists_and_running(action[5])

        elif action_number == 3:
            # stop service
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # Also if the service to stop does not exist, this is invalid
            _, node_info = get_node_and_info(action[8])
            return node_exists_and_running(action[8]) and service_exists(node_info, action[9])

        elif action_number == 4:
            # start service
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # Also if the service to start does not exist, this is invalid
            _, node_info = get_node_and_info(action[10])
            return node_exists_and_running(action[10]) and service_exists(node_info, action[11])
        else:
            return False

    def reset(self, *, seed=None, options=None) -> Tuple[Observation, Dict[str, Any]]:
        logging.debug('Reset Defender')
        # If this reset comes after an episode ended, log the termination reason.
        if (
            self._log_episode_end
            and not self._episode_end_logged
            and self.timesteps is not None
            and (self.timesteps > 0 or len(self.rewards) > 0)
        ):
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
            logging.warning(
                "%sDEFENDER EPISODE END steps=%d reason=%s total_reward=%.3f valid=%d invalid=%d availability=%.3f sla_breached=%s",
                prefix,
                int(self.timesteps),
                reason,
                total_reward,
                int(self.valid_action_count),
                int(self.invalid_action_count),
                float(self.last_availability),
                bool(self.last_sla_breached),
            )
        if not self.reset_request:
            self.event_source.notify_reset(last_reward=0)

        reset_result = self.cyber_env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            _, info = reset_result
        else:
            info = {}

        self.reset_request = False
        self.__last_attacker_reward = None
        self.rewards = []
        self.timesteps = 0
        # Preserve counts from the episode that just ended (VecEnv resets automatically on done)
        self.last_valid_action_count = self.valid_action_count
        self.last_invalid_action_count = self.invalid_action_count
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self._has_breached_sla = False
        self._prev_network_availability = float(self._actuator.network_availability)
        self.last_action = None
        self.last_action_valid = None
        self.last_reward = 0.0
        self.last_availability = float(self._actuator.network_availability)
        self.last_worsening = 0.0
        self.last_sla_breached = False
        self.last_terminated = False
        self.last_truncated = False
        self.last_outcome = None
        self._episode_end_logged = False

        return self.observe(), info

    def on_reset(self, last_reward):
        logging.debug('on_reset Defender')
        self.reset_request = True
        self.__last_attacker_reward = last_reward

    def get_blank_defender_observation(self):
        """ Creates a empty defender observation. """
        obs = Defender_Observation(infected_nodes = [],
                                    incoming_firewall_status=[],
                                    outgoing_firewall_status=[],
                                    services_status=[])
        return obs

    def observe(self) -> Defender_Observation:
        """Gathers information directly from the environment to generate populate an observation for the defender agent to use."""

        new_observation=self.get_blank_defender_observation()
        incoming_firewall_list = [0]*(len(self._base_env.environment.network.nodes)*len(self.firewall_rule_list))
        outgoing_firewall_list = [0]*(len(self._base_env.environment.network.nodes)*len(self.firewall_rule_list))
        all_services_list = [0]*self.num_services
        count_incoming_firewall = -1
        count_outgoing_firewall = -1
        count_services = -1

        # Iterates through all nodes in the environment.
        for _, node in model.iterate_network_nodes(self._base_env.environment.network):
            # Incoming Firewall rules section. Counts which incoming firewall rules are active.
            for rule in self.firewall_rule_list:
                count_incoming_firewall+=1
                for entry in node.firewall.incoming:
                    if rule == entry.port:
                        incoming_firewall_list[count_incoming_firewall] = 1

            # Outgoing Firewall rules section. Counts which outgoing firewall rules are active.
            for rule in self.firewall_rule_list:
                count_outgoing_firewall+=1
                for entry in node.firewall.outgoing:
                    if rule == entry.port:
                        outgoing_firewall_list[count_outgoing_firewall] = 1
                    
            # Services Section. Counts the currently running services.
            for service in node.services:
                count_services+=1
                if service.running:
                    all_services_list[count_services] = 1
                    
        # Take information from the environment and format it for defender agent observation.
        # Check all nodes and find which are infected. 1 if infected 0 if not.
        new_observation["infected_nodes"] = np.array([1 if node.agent_installed else 0 for _, node in model.iterate_network_nodes(self._base_env.environment.network)])
        # Lists all possible incoming firewall rules, 1 if active, 0 if not.
        new_observation['incoming_firewall_status'] = np.array(incoming_firewall_list)
        # Lists all possible outgoing firewall rules, 1 if active, 0 if not.
        new_observation['outgoing_firewall_status'] = np.array(outgoing_firewall_list)
        # Lists all possible services, 1 if active, 0 if not.
        new_observation['services_status'] = np.array(all_services_list)
        return new_observation

    def set_reset_request(self, reset_request):
        self.reset_request = reset_request

    def defender_constraints_broken(self):
        return self._actuator.network_availability < self._defender_constraint.maintain_sla

    def close(self) -> None:
        return self.cyber_env.close()

    def render(self, mode: str = 'human') -> None:
        return self.cyber_env.render(mode)

    def render_as_fig(self) -> FigureWidget:
        return self.cyber_env.render_as_fig()
