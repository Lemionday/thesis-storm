from collections import deque
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

from autoscaler import Autoscaler
from metrics_collector import MetricsCollector

register(
    id="container-autoscaling-v0",
    entry_point="container_autoscaling_env:ContainerAutoscalingEnv",
)


class Action(Enum):
    DO_NOTHING = 0
    SCALE_UP = 1
    SCALE_DOWN = 2


def discretize_state(state):
    # Discretize CPU utilization and latency into 20 bins each
    cpu_bin = int(np.clip(state[0], 0, 100) / 5)  # 0-100 to 20 bins
    latency_bin = int(np.clip(state[1], 0, 1000) / 50)  # 0-1000 to 20 bins
    containers = int(state[2])
    return np.array([cpu_bin, latency_bin, containers])


class ContainerAutoscalingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    metrics_collector: MetricsCollector
    scaler: Autoscaler
    delay_buffer: deque
    min_containers: int
    max_containers: int

    def __init__(
        self,
        scaler: Autoscaler,
        metrics_collector: MetricsCollector,
        min_containers=2,
        max_containers=5,
        render_mode=None,
    ):  # Weights for (latency, cost, stability)
        super().__init__()

        self.render_mode = render_mode

        # Define action and state space
        self.action_space = gym.spaces.Discrete(len(Action))

        # Environment parameters
        self.max_containers = max_containers
        self.min_containers = min_containers

        self.metrics_collector = metrics_collector
        self.scaler = scaler

        self.reset()

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed, options=options)

        self.scaler.set_number_of_containers(3)
        self.previous_action = Action.DO_NOTHING
        self.time_counter = 0

    def _perform_action(self, action: Action):
        current_containers = self.scaler.get_number_of_containers()
        if current_containers is None:
            print("Error getting current number of containers")

        if action == Action.SCALE_UP:
            new_numbers = min(current_containers + 1, self.max_containers)
            self.scaler.set_number_of_containers(new_numbers)
        elif action == Action.SCALE_DOWN:
            new_numbers = max(current_containers - 1, self.min_containers)
            self.scaler.set_number_of_containers(new_numbers)

    def _is_done(self):
        """Checks if the episode is done."""
        return self.time_counter > 200  # Arbitrary episode length


if __name__ == "__main__":
    env = gym.make(
        id="container-autoscaling-v0",
        render_mode="human",
        scaler=Autoscaler(url="http://localhost:8083/scale"),
        metrics_collector=MetricsCollector(url="http://localhost:9090"),
    )
    print(check_env(env.unwrapped))
