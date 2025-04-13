from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
from prometheus_api_client import PrometheusConnect

PROMETHEUS_URL = "http://localhost:9090"

try:
    prom = PrometheusConnect(
        url=PROMETHEUS_URL, disable_ssl=True
    )  # disable_ssl for local testing
except Exception as e:
    print(f"Error connecting to Prometheus: {e}")
    exit()

gym.register(id="VMAutoScale-v0", entry_point="vm_auto_scale_env:VMAutoScaleEnv")

# --- Configuration ---
INITIAL_VMS = 1
MIN_VMS = 1
MAX_VMS = 5
CPU_THRESHOLD_SCALE_UP = 0.7  # If avg CPU exceeds this, consider scaling up
CPU_THRESHOLD_SCALE_DOWN = 0.3  # If avg CPU falls below this, consider scaling down
SCALE_INCREMENT = 1
OBSERVATION_INTERVAL = 5  # Seconds to observe for CPU
NUM_EPISODES = 1000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY_RATE = 0.001
MIN_EXPLORATION_RATE = 0.01
NUM_CPU_BUCKETS = 10  # Define the number of CPU utilization buckets


class Action(Enum):
    DO_NOTHING = 0
    SCALE_UP = 1
    SCALE_DOWN = 2


class VMAutoScaleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        initial_vms=INITIAL_VMS,
        min_vms=MIN_VMS,
        max_vms=MAX_VMS,
        num_cpu_buckets=NUM_CPU_BUCKETS,
    ):
        super().__init__()
        self.min_vms = min_vms
        self.max_vms = max_vms
        self.current_vms = initial_vms
        self.num_cpu_buckets = num_cpu_buckets

        self.observation_space = gym.spaces.Discrete(
            num_cpu_buckets
        )  # Observation is now the bucket index

        self.action_space = gym.spaces.Discrete(len(Action))

        self.current_step = 0

    def _get_cpu_utilization(self):
        cpu_utilizations = np.array()
        cpu_results = prom.custom_query(query="storm_supervisor_cpu_percent")
        for cpu_utilization in cpu_results:
            cpu_utilizations.append(cpu_utilizations)
        return np.average(cpu_utilizations)

    def step(self, action: Action):
        self.current_step += 1
        reward = 0

        terminated = False
        truncated = False

        previous_vms = self.current_vms
        cpu_utilization = self._get_cpu_utilization()
        print(cpu_utilization)

        if action == Action.SCALE_UP:
            if self.current_vms < self.max_vms:
                self.current_vms += SCALE_INCREMENT
                reward = 0.1  # Small positive reward for scaling up
                print(f"Step {self.current_step}: Scaled up to {self.current_vms} VMs")
            else:
                reward = -0.05  # Small negative reward for trying to scale beyond max
        elif action == Action.SCALE_DOWN:  # Scale down
            if self.current_vms > self.min_vms:
                self.current_vms -= SCALE_INCREMENT
                reward = 0.1  # Small positive reward for scaling down
                print(
                    f"Step {self.current_step}: Scaled down to {self.current_vms} VMs"
                )
            else:
                reward = -0.05  # Small negative reward for trying to scale below min

        # Reward based on CPU utilization relative to thresholds
        if cpu_utilization > CPU_THRESHOLD_SCALE_UP:
            reward -= (
                cpu_utilization - CPU_THRESHOLD_SCALE_UP
            ) * 0.2  # Negative reward for high CPU
        elif cpu_utilization < CPU_THRESHOLD_SCALE_DOWN:
            reward += (
                CPU_THRESHOLD_SCALE_DOWN - cpu_utilization
            ) * 0.1  # Small positive for low CPU

        observation = np.array([cpu_utilization], dtype=np.float32)
        info = {"num_vms": self.current_vms}

        if self.current_step > 200:
            terminated = True

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_vms = INITIAL_VMS
        observation = np.array([self._get_cpu_utilization()], dtype=np.float32)
        self.current_step = 0
        info = {"num_vms": self.current_vms}
        return observation, info

    def render(self):
        print(
            f"Current VMs: {self.current_vms}, CPU Util: {self._get_cpu_utilization():.2f}"
        )


if __name__ == "__main__":
    env = gym.make(id="VMAutoScale-v0")
    print("Begin check env")
    check_env(env.unwrapped)
    print("Check env successfully")
