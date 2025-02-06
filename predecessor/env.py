# env.py
import numpy as np
import gymnasium as gym
from gym import Env, spaces
from neuronav.envs.grid_env import GridEnv
import enum
from neuronav.envs.grid_templates import (
    generate_layout,
    GridTemplate,
    GridSize,
)


class GridWorld1D(gym.Env):
    def __init__(
        self,
        size: int = 4,
        goal: int = None,
        punishment: int = None,
        noise: float = 0.0,
    ):
        self.size = size
        # set goal to the rightmost position if not specified
        self.goal = size - 1 if goal is None else goal
        # set initial position to the leftmost position if punishment is not specified
        self.init_position = 0 if punishment is None else size // 2

        self.position = self.init_position
        self.punishment = punishment
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(size,))
        self.noise = noise

    def step(self, action):
        if action == 0:  # Left
            self.position = max(self.position - 1, 0)
        elif action == 1:  # Right
            self.position = min(self.position + 1, self.size - 1)
        else:
            raise ValueError("Invalid action")

        if self.punishment is None:
            reward = 1 if self.position == self.goal else 0
            done = self.position == self.goal
        else:
            if self.position == self.goal:
                reward = 1
                done = True
            elif self.position == self.punishment:
                reward = -1
                done = True
            else:
                reward = 0
                done = False

        return self._get_observation(), reward, done, {}

    def reset(self):
        self.position = self.init_position
        return self._get_observation()

    def _get_observation(self):
        obs_noise_free = np.zeros((self.size,))
        obs_noise_free[self.position] = 1
        rng = np.random.default_rng()
        noisy_vec = rng.normal(0, self.noise, size=self.size)
        obs = obs_noise_free + np.abs(noisy_vec)
        return obs


def compute_true_q_value(env, gamma=0.95, theta=1e-5):
    Q = np.zeros((env.action_space.n, env.size))
    while True:
        delta = 0
        for s in range(env.size):
            for a in range(env.action_space.n):
                q = Q[a, s]
                if s == env.goal:
                    Q[a, s] = 0
                    continue
                next_s = min(s + 1, env.size - 1) if a == 1 else max(s - 1, 0)
                reward = 1 if next_s == env.goal else 0
                Q[a, s] = reward + gamma * np.max(Q[:, next_s])
                delta = max(delta, abs(q - Q[a, s]))
        if delta < theta:
            break
    return Q


class GridObservation(enum.Enum):
    onehot = "onehot"
    twohot = "twohot"
    geometric = "geometric"
    index = "index"
    boundary = "boundary"
    visual = "visual"
    images = "images"
    window = "window"
    symbolic = "symbolic"
    symbolic_window = "symbolic_window"
    window_tight = "window_tight"
    symbolic_window_tight = "symbolic_window_tight"
    rendered_3d = "rendered_3d"


class GridOrientation(enum.Enum):
    fixed = "fixed"
    variable = "variable"


class GridEnvNoise(GridEnv):
    def __init__(
        self,
        template: GridTemplate = GridTemplate.empty,
        size: GridSize = GridSize.small,
        obs_type: GridObservation = GridObservation.index,
        orientation_type: GridOrientation = GridOrientation.fixed,
        seed: int = None,
        use_noop: bool = False,
        torch_obs: bool = False,
        manual_collect: bool = False,
        noise: float = 0.0,
    ):
        super().__init__(
            template=template,
            size=size,
            obs_type=obs_type,
            orientation_type=orientation_type,
            seed=seed,
            use_noop=use_noop,
            torch_obs=torch_obs,
            manual_collect=manual_collect,
        )
        # print(self.orientation_type)
        self.noise = noise

    def set_action_space(self):
        if self.orientation_type == GridOrientation.variable:
            self.action_space = spaces.Discrete(3 + self.use_noop + self.manual_collect)
            self.orient_size = 4
        elif self.orientation_type == GridOrientation.fixed:
            self.orient_size = 1
            self.action_space = spaces.Discrete(4 + self.use_noop + self.manual_collect)
        else:
            raise Exception("No valid GridOrientation provided.")
        self.state_size *= self.orient_size

    def get_observation(self, perspective: list):
        rng = np.random.default_rng()
        noisy_vec = rng.normal(
            0, self.noise, size=super().get_observation(perspective).size
        )
        return super().get_observation(perspective) + noisy_vec

    @property
    def size(self):
        return self.observation.size
        # return super().get_observation(perspective).size

    # + np.random.normal(
    #        0, self.noise, size=super().get_observation(perspective).size
    #    )


if __name__ == "__main__":
    # env = GridWorld1D(size=19, punishment=0, noise=0)
    # obs = env.reset()
    # print(obs.shape)
    # for i in range(10):
    #    print(env.step(0))
    env = GridEnvNoise(size=GridSize)
