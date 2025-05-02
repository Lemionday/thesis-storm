import gymnasium as gym
import numpy as np

from autoscaler import Autoscaler
from custom_env.container_autoscaling import Action, ContainerAutoscalingEnv
from metrics_collector import MetricsCollector
from q_learning.state import State


class QLearningEnv(ContainerAutoscalingEnv):
    def __init__(
        self,
        scaler: Autoscaler,
        metrics_collector: MetricsCollector,
        render_mode=None,
    ):
        super().__init__(
            scaler=scaler,
            metrics_collector=metrics_collector,
            render_mode=render_mode,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [0, 0, 0, self.min_containers],
                dtype=np.int64,
            ),
            high=np.array(
                [20, 40, 40, self.max_containers],
                dtype=np.int64,
            ),
            dtype=np.int64,
        )  # [memory_percent, previous_spout_messages_emitted, spout_messages_emitted, num_containers]
        self.previous_spout_messages_emitted = 0

    def _get_observation(self):
        avg_mem_percent = np.mean(
            self.metrics_collector.get_mem_percent(), dtype=np.float32
        )
        spout_messages_emitted = np.mean(
            self.metrics_collector.get_storm_spout_messages_emitted(),
            dtype=np.float32,
        )
        num_containers = self.scaler.get_number_of_containers()

        state = State(
            memory_percent=avg_mem_percent / 100.0,
            previous_spout_messages_emitted=self.previous_spout_messages_emitted,
            spout_messages_emitted=spout_messages_emitted,
            number_of_containers=num_containers,
        )
        self.previous_spout_messages_emitted = spout_messages_emitted

        return state

    def step(self, action: Action):
        """
        Executes one time step in the environment.

        Args:
            action (Action): The action to take.

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.state = self._get_observation()

        reward = self._calculate_reward(action)

        out_of_bound = self._perform_action(action)

        terminated = self._is_done()

        self.render(mode="human")

        info = {"out_of_bound": out_of_bound}

        return self.state, reward, terminated, False, info

    def render(self, mode="human"):
        """Renders the environment."""
        if mode == "human":
            print(f"Time: {self.time_counter}")
            print(self.state)
            print()
