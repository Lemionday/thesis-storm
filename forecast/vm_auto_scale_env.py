from enum import Enum

import gymnasium as gym
import numpy as np
from prometheus_api_client import PrometheusConnect

from helpers import discretize_cpu_util

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
    # Parameters
    alpha = 1.0  # weight for utilization score
    beta = 0.5  # weight for worker cost
    gamma = 0.3  # penalty for scaling actions

    def __init__(
        self,
        min_vms=MIN_VMS,
        max_vms=MAX_VMS,
        num_cpu_buckets=NUM_CPU_BUCKETS,
    ):
        super().__init__()
        self.min_vms = min_vms
        self.max_vms = max_vms
        self.current_vms = min_vms
        self.num_cpu_buckets = num_cpu_buckets

        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.array([NUM_CPU_BUCKETS, MAX_VMS]),
            shape=(2,),
            dtype=np.float32,
        )  # Observation is now the bucket index
        # self.observation_space = gym.spaces.Discrete()

        self.action_space = gym.spaces.Discrete(len(Action))

        self.reset()

    def _get_cpu_utilization(self):
        """
        Queries the Prometheus server for current CPU utilization
        of Storm supervisors.

        Returns:
            np.ndarray[np.float32]: An array of CPU utilization values,
            each representing the percentage of CPU used by an individual
            Storm supervisor.
        """
        query = "storm_supervisor_cpu_percent"

        # Each record returned by Prometheus contains a 'value' field,
        # which is a tuple: (timestamp, utilization as a string).
        # We extract the utilization value (index 1), convert to float,
        # and collect all into a NumPy array.
        cpu_utilizations = np.array(
            [record["value"][1] for record in prom.custom_query(query=query)],
            dtype=np.float32,
        )

        return cpu_utilizations

    def _value_function(self, cpu_utils: np.array, action: Action) -> np.float32:
        utilization_score = -((cpu_utils - 0.7) ** 2)
        cost_penalty = np.log(len(cpu_utils) + 1)
        stability_penalty = np.where(action != Action.DO_NOTHING, 1.0, 0.0)

        penalties = (
            self.alpha * utilization_score
            - self.beta * cost_penalty
            - self.gamma * stability_penalty
        )

        return np.mean(penalties)

    def step(self, action: Action):
        self.current_step += 1

        terminated = False
        # truncated = False
        #
        # previous_vms = self.current_vms
        self.current_vms = len()
        cpu_utilizations = self._get_cpu_utilization()
        reward = self._value_function(cpu_utilizations, action)

        self.current_vms = len(cpu_utilizations)
        avg_cpu_util = discretize_cpu_util(
            np.mean(cpu_utilizations), self.num_cpu_buckets
        )

        obs = np.array([avg_cpu_util, self.current_vms], dtype=np.float32)
        info = {"num_vms": self.current_vms}

        if self.current_step > 200:
            terminated = True

        try:
            return self.previous_obs, self.previous_reward, terminated, False, info
        finally:
            self.previous_obs = obs
            self.previous_reward = reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_reward = 0
        self.current_vms = self.min_vms
        self.previous_obs = np.array([self.previous_reward, self.current_vms])

        cpu_utilizations = self._get_cpu_utilization()
        avg_cpu_util = discretize_cpu_util(
            np.mean(cpu_utilizations), self.num_cpu_buckets
        )

        obs = np.array([avg_cpu_util, self.current_vms], dtype=np.float32)
        self.current_step = 0
        info = {"num_vms": self.current_vms}
        return obs, info

    def render(self):
        print(
            f"Current VMs: {self.current_vms}, CPU Util: {self._get_cpu_utilization():.2f}"
        )


if __name__ == "__main__":
    env = gym.make(id="VMAutoScale-v0")
    # print("Begin check env")
    # check_env(env.unwrapped)
    # print("Check env successfully")
    env.reset()
    print(env.step(action=Action.SCALE_DOWN))
