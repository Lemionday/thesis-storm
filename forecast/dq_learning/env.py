import gymnasium as gym
import numpy as np

from autoscaler import Autoscaler
from custom_env.container_autoscaling import (
    Action,
    ContainerAutoscalingEnv,
)
from metrics_collector import MetricsCollector

SEQ_LEN = 10
BUFFER_SIZE = 10000
BATCH_SIZE = 8
target_model_update_frequency = 10  # Steps after which to update the target model
timesteps_per_sequence = 3


class DeepQLearningEnv(ContainerAutoscalingEnv):
    def __init__(
        self,
        step_interval: float,
        scaler: Autoscaler,
        metrics_collector: MetricsCollector,
        reward_weights=(0.5, 0.7, 0.2, 0.5),
        render_mode=None,
    ):
        super().__init__(
            scaler=scaler,
            metrics_collector=metrics_collector,
            render_mode=render_mode,
        )

        # [supervisor_ratio, cpu_util, latency]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.reset()

    def step(self, action: Action):
        out_of_bound = self._perform_action(action)

        # time.sleep(self.step_interval)

        next_obs = self._get_observation()

        reward = self._calculate_reward(action)

        info = {"out_of_bound": out_of_bound}

        return next_obs, reward, self._is_done(), False, info

    def _get_observation(self):
        avg_mem_usage = np.mean(
            self.metrics_collector.get_mem_percent(), dtype=np.float32
        )
        avg_mem_usage /= 100.0

        avg_spout_messages_emitted = np.mean(
            self.metrics_collector.get_storm_spout_messages_emitted(),
            dtype=np.float32,
        )
        avg_spout_messages_emitted = (
            np.clip(avg_spout_messages_emitted, 0, 100_000) / 100_000
        )

        number_of_containers = self.scaler.get_number_of_containers()
        number_of_containers /= self.max_containers

        state = (
            avg_mem_usage,
            avg_spout_messages_emitted,
            number_of_containers,
            self.previous_spout_messages_emitted,
        )

        self.previous_spout_messages_emitted = avg_spout_messages_emitted

        return np.array(state, dtype=np.float32)
