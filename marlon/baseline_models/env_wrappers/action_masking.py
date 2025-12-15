from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class _MaskedDiscreteLayout:
    connect_size: int
    local_size: int
    remote_size: int

    @property
    def total(self) -> int:
        return int(self.connect_size + self.local_size + self.remote_size)

    @property
    def local_offset(self) -> int:
        return int(self.connect_size)

    @property
    def remote_offset(self) -> int:
        return int(self.connect_size + self.local_size)


class MaskedDiscreteAttackerWrapper(gym.Wrapper):
    """
    Action-masking wrapper for the attacker that:
    - exposes a single Discrete action space
    - exposes `action_masks()` compatible with sb3-contrib MaskablePPO

    The discrete actions enumerate exactly the valid CyberBattleSim actions:
    - connect: (source, target, port, cred)  -> N*N*P*C
    - local exploit: (source, vuln)          -> N*L
    - remote exploit: (source, target, vuln) -> N*N*R

    This avoids the limitations of MultiDiscrete masks (which cannot encode
    cross-dimension dependencies such as "kind" selecting which dimensions matter).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("MaskedDiscreteAttackerWrapper requires Dict observation space")

        # Expect the observation to already expose the CyberBattle action masks at top-level.
        obs_spaces = env.observation_space.spaces
        if "local_vulnerability" not in obs_spaces or "remote_vulnerability" not in obs_spaces or "connect" not in obs_spaces:
            raise KeyError("Observation must include top-level 'local_vulnerability', 'remote_vulnerability', and 'connect' masks")

        # Derive sizes from observation-space shapes.
        local_shape = tuple(int(x) for x in obs_spaces["local_vulnerability"].n)  # MultiBinary
        remote_shape = tuple(int(x) for x in obs_spaces["remote_vulnerability"].n)  # MultiBinary
        connect_shape = tuple(int(x) for x in obs_spaces["connect"].n)  # MultiBinary

        if len(local_shape) != 2 or len(remote_shape) != 3 or len(connect_shape) != 4:
            raise ValueError(
                f"Unexpected mask shapes local={local_shape} remote={remote_shape} connect={connect_shape}; "
                "expected (N,L), (N,N,R), (N,N,P,C)"
            )

        self._n = int(local_shape[0])
        self._l = int(local_shape[1])
        self._r = int(remote_shape[2])
        self._p = int(connect_shape[2])
        self._c = int(connect_shape[3])

        self._layout = _MaskedDiscreteLayout(
            connect_size=self._n * self._n * self._p * self._c,
            local_size=self._n * self._l,
            remote_size=self._n * self._n * self._r,
        )

        self.action_space = spaces.Discrete(self._layout.total)
        self.observation_space = env.observation_space

        # Cache mapping from CyberBattleSim subspace name -> MultiDiscrete encoding index
        # for the wrapped attacker env (if it uses the "kind + params" MultiDiscrete format).
        self._kind_to_index: Dict[str, int] = {}
        self._kind_to_slice: Dict[str, Tuple[int, int]] = {}
        if hasattr(env, "action_subspaces"):
            for idx, (kind, start, end) in getattr(env, "action_subspaces").items():
                self._kind_to_index[str(kind)] = int(idx)
                self._kind_to_slice[str(kind)] = (int(start), int(end))

    def action_masks(self) -> np.ndarray:
        obs = getattr(self.env, "_last_transformed_observation", None)
        if obs is None:
            return np.ones((self._layout.total,), dtype=np.bool_)
        return self._mask_from_observation(obs)

    def _mask_from_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        connect = np.asarray(obs["connect"], dtype=np.int8).reshape(-1)
        local = np.asarray(obs["local_vulnerability"], dtype=np.int8).reshape(-1)
        remote = np.asarray(obs["remote_vulnerability"], dtype=np.int8).reshape(-1)

        # Keep the enumeration order consistent with decode().
        mask = np.concatenate([connect, local, remote]).astype(np.bool_, copy=False)
        if mask.shape != (self._layout.total,):
            raise ValueError(f"action mask shape mismatch: got {mask.shape}, expected {(self._layout.total,)}")
        return mask

    def _decode(self, action: int) -> Tuple[str, Tuple[int, ...]]:
        a = int(action)
        if a < 0 or a >= self._layout.total:
            raise ValueError(f"Invalid discrete action: {a}")

        if a < self._layout.connect_size:
            # connect: (((src*N + tgt)*P + port)*C + cred)
            idx = a
            cred = idx % self._c
            idx //= self._c
            port = idx % self._p
            idx //= self._p
            tgt = idx % self._n
            src = idx // self._n
            return "connect", (src, tgt, port, cred)

        if a < self._layout.remote_offset:
            # local exploit: (src*L + vuln)
            idx = a - self._layout.local_offset
            vuln = idx % self._l
            src = idx // self._l
            return "local_vulnerability", (src, vuln)

        # remote exploit: ((src*N + tgt)*R + vuln)
        idx = a - self._layout.remote_offset
        vuln = idx % self._r
        idx //= self._r
        tgt = idx % self._n
        src = idx // self._n
        return "remote_vulnerability", (src, tgt, vuln)

    def _encode_for_inner_env(self, kind: str, coords: Tuple[int, ...]) -> np.ndarray:
        """
        Encode into the wrapped attacker's MultiDiscrete format:
        - action[0] selects the subspace index
        - action[slice] contains the subspace parameters
        """
        if not hasattr(self.env, "action_space") or not isinstance(self.env.action_space, spaces.MultiDiscrete):
            raise TypeError("Inner env must use MultiDiscrete action space to be encoded")
        if kind not in self._kind_to_index or kind not in self._kind_to_slice:
            raise KeyError(f"Inner env does not expose action_subspaces for kind={kind}")

        nvec = np.asarray(self.env.action_space.nvec, dtype=np.int64)
        encoded = np.zeros((len(nvec),), dtype=np.int64)
        encoded[0] = self._kind_to_index[kind]
        start, end = self._kind_to_slice[kind]
        encoded[start:end] = np.asarray(coords, dtype=np.int64)
        return encoded.astype(np.int64)

    def step(self, action):
        # SB3 passes numpy arrays for Discrete; normalize.
        if isinstance(action, np.ndarray):
            action = int(action.item())
        kind, coords = self._decode(int(action))
        encoded = self._encode_for_inner_env(kind, coords)
        return self.env.step(encoded)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

