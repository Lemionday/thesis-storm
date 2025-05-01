from abc import abstractmethod
from collections import deque
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

from autoscaler import Autoscaler
from metrics_collector import PROMETHEUS_URL, MetricsCollector

register(
    id="container-autoscaling-v0",
    entry_point="container_autoscaling_env:ContainerAutoscalingEnv",
)

NUM_EPISODES = 1
STEPS_PER_EPISODE = 20

TARGET_MEMORY_PERCENT = 60.0

LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.9


class Action(Enum):
    DO_NOTHING = 0
    SCALE_UP = 1
    SCALE_DOWN = 2


class ContainerAutoscalingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    metrics_collector: MetricsCollector
    scaler: Autoscaler
    delay_buffer: deque
    min_containers: int
    max_containers: int
    return_queue: list[int]

    def __init__(
        self,
        scaler: Autoscaler,
        metrics_collector: MetricsCollector,
        target_memory_percent=TARGET_MEMORY_PERCENT,
        min_containers=2,
        max_containers=5,
        render_mode=None,
        reward_weights=(0.5, 0.7, 0.2, 0.5),  # Weights for (latency, cost, stability)
    ):
        super().__init__()

        self.render_mode = render_mode

        # Define action and state space
        self.action_space = gym.spaces.Discrete(len(Action))

        # Environment parameters
        self.max_containers = max_containers
        self.min_containers = min_containers
        self.target_memory_percent = target_memory_percent
        self.reward_weights = reward_weights

        self.metrics_collector = metrics_collector
        self.scaler = scaler

        self.return_queue = []

        self.previous_action = Action.DO_NOTHING

        self.previous_spout_messages_emitted = 0

        self.reset()

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed, options=options)

        self.scaler.set_number_of_containers(3)
        self.previous_action = Action.DO_NOTHING
        self.time_counter = 0

        return self._get_observation(), {}

    def _perform_action(self, action: Action) -> bool:
        self.time_counter += 1

        current_containers = self.scaler.get_number_of_containers()
        if current_containers is None:
            print("Error getting current number of containers")

        if action == Action.SCALE_UP and current_containers == self.max_containers:
            return True

        if action == Action.SCALE_DOWN and current_containers == self.min_containers:
            return True

        if action == Action.SCALE_UP:
            new_numbers = min(current_containers + 1, self.max_containers)
            self.scaler.set_number_of_containers(new_numbers)
        elif action == Action.SCALE_DOWN:
            new_numbers = max(current_containers - 1, self.min_containers)
            self.scaler.set_number_of_containers(new_numbers)

        return False

    def _calculate_reward(self, action: Action):
        states = self.metrics_collector.get_mem_percent()
        number_of_containers = len(states)
        if (
            number_of_containers == self.min_containers
            and np.mean(states, dtype=np.float32) < self.target_memory_percent
        ):
            memory_penalty = 0
        else:
            memory_penalty = -(
                np.sqrt(np.mean((states - self.target_memory_percent) ** 2)) / 40.0
                + np.count_nonzero(np.logical_or(states > 80, states < 20))
            )

        cost_penalty = -number_of_containers / self.max_containers

        stability_reward = 0
        if self.previous_action == Action.DO_NOTHING:
            stability_reward = 1

        spout_messages_latency_penalty = (
            -(self.metrics_collector.get_spout_messages_latency()) / 1000
        )

        self.previous_action = action

        if self.render_mode == "human":
            print(
                f"mem_pel: {self.reward_weights[0] * memory_penalty}"
                + f", cost_pel: {self.reward_weights[1] * cost_penalty}"
                + f", stab_reward: {self.reward_weights[2] * stability_reward}"
                + f", latency_pel: {self.reward_weights[3] * spout_messages_latency_penalty}"
            )

        return (
            self.reward_weights[0] * memory_penalty
            + self.reward_weights[1] * cost_penalty
            + self.reward_weights[2] * stability_reward
            + self.reward_weights[3] * spout_messages_latency_penalty
        )

    def _is_done(self):
        """Checks if the episode is done."""
        return self.time_counter > STEPS_PER_EPISODE  # Arbitrary episode length

    @abstractmethod
    def _get_observation(self):
        pass


if __name__ == "__main__":
    env = gym.make(
        id="container-autoscaling-v0",
        render_mode="human",
        scaler=Autoscaler(url="http://localhost:8083/scale"),
        metrics_collector=MetricsCollector(url=PROMETHEUS_URL),
    )
    print(check_env(env.unwrapped))
