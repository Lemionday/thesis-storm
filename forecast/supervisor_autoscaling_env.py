import asyncio
from collections import deque
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from autoscaler import Autoscaler
from metrics_collector import MetricsCollector

register(
    id="container-autoscaling-v0",
    entry_point="supervisor_autoscaling_env:ContainerAutoscalingEnv",
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

    def __init__(
        self,
        scaler: Autoscaler,
        metrics_collector: MetricsCollector,
        initial_containers=1,
        max_containers=5,
        min_containers=1,
        max_cpu_utilization=80,
        min_cpu_utilization=20,
        scaling_delay=1,  # Time steps before action takes effect
        reward_weights=(1.0, -1.0, -0.1),
        render_mode=None,
    ):  # Weights for (latency, cost, stability)
        super().__init__()

        self.render_mode = render_mode

        # Define action and state space
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, min_containers]),
            high=np.array([100, 1000, max_containers]),
            dtype=np.float32,
        )  # [CPU utilization, latency, num_containers]

        # Environment parameters
        self.initial_containers = initial_containers
        self.max_containers = max_containers
        self.min_containers = min_containers
        self.max_cpu_utilization = max_cpu_utilization
        self.min_cpu_utilization = min_cpu_utilization
        self.scaling_delay = scaling_delay
        self.reward_weights = reward_weights

        # Internal state
        self.num_containers = initial_containers
        self.avg_mem_percent = 50.0  # Initial CPU utilization
        self.latency = 100.0  # Initial latency
        self.time_counter = 0
        self.action_history = []
        self.state_history = []

        self.metrics_collector = metrics_collector
        self.scaler = scaler
        self.delay_buffer = deque(maxlen=scaling_delay)

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed, options=options)
        # self.num_containers = self.initial_containers
        # self.np_random.normal()
        # self.avg_mem_percent = (
        #     50.0 + self.np_random.normal() * 10
        # )  # Initialize with some noise
        # self.latency = 70.0 + self.np_random.normal() * 20
        # self.time_counter = 0
        # self.action_history = []
        # self.state_history = []
        # self.state = np.array(
        #     [self.avg_mem_percent, self.latency, self.num_containers],
        #     dtype=np.float32,
        # )

        # return self.state, {}

        return self._get_current_discretize_state()

    def _get_current_discretize_state(self):
        avg_mem_percent = np.mean(
            self.metrics_collector.get_mem_percent(), dtype=np.float32
        )
        latency = 0
        num_supervisors = self.scaler.get_number_of_supervisors()

        return discretize_state(
            np.array(
                [avg_mem_percent, latency, num_supervisors],
                dtype=np.float32,
            )
        )

    def _perform_action(self, action: Action):
        current_supervisors = self.scaler.get_number_of_supervisors()
        if current_supervisors is None:
            print("Error getting current number of supervisors")

        if action == Action.SCALE_UP:
            new_numbers = min(current_supervisors + 1, self.max_containers)
            self.scaler.set_number_of_supervisors(new_numbers)
        elif action == Action.SCALE_DOWN:
            new_numbers = max(current_supervisors - 1, self.min_containers)
            self.scaler.set_number_of_supervisors(new_numbers)

    def step(self, action: Action):
        """
        Executes one time step in the environment.

        Args:
            action (Action): The action to take.

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.state = self._get_current_discretize_state()

        reward = self._calculate_reward()

        self._perform_action(action)

        terminated = self._is_done()

        return self.state, reward, terminated, False, {}

        # self.delay_buffer.append((state, action))
        #
        # if len(self.delay_buffer) == self.scaling_delay:
        #     delayed_state, delayed_action = self.delay_buffer.popleft()
        #     reward = self._calculate_reward()
        #
        # self.action_history.append(action)
        # self.state_history.append(self.state)
        # self.time_counter += 1
        #
        # self.avg_mem_percent = np.mean(
        #     self.metrics_collector.get_mem_percent(), dtype=np.float32
        # )
        # self.latency = 0
        #
        # # Apply action
        # if self.time_counter >= self.scaling_delay:
        #     delayed_action = self.action_history[0]  # Get the delayed action
        #     if delayed_action == Action.SCALE_DOWN:
        #         self.num_containers = max(self.min_containers, self.num_containers - 1)
        #     elif delayed_action == Action.SCALE_UP:
        #         self.num_containers = min(self.max_containers, self.num_containers + 1)
        #     self.action_history = self.action_history[1:]
        #     self.state_history = self.state_history[1:]
        #     self.time_counter = self.scaling_delay - 1  # important
        #
        # self.state = np.array(
        #     [self.avg_mem_percent, self.latency, self.num_containers],
        #     dtype=np.float32,
        # )
        # reward = self._calculate_reward()
        # terminated = self._is_done()
        # info = {}
        #
        # return self.state, reward, terminated, False, info

    def _calculate_reward(self):
        """Calculates the reward based on current state."""
        latency_reward = -self.latency / 100.0  # Normalize latency
        cost_reward = -self.num_containers / self.max_containers  # Normalize cost
        stability_reward = 0
        if len(self.action_history) > 0:
            if self.action_history[-1] == 0 or self.action_history[-1] == 2:
                stability_reward = -0.1  # slight penalty for scaling actions

        return (
            self.reward_weights[0] * latency_reward
            + self.reward_weights[1] * cost_reward
            + self.reward_weights[2] * stability_reward
        )

    def _is_done(self):
        """Checks if the episode is done."""
        return self.time_counter > 200  # Arbitrary episode length

    def render(self, mode="human"):
        """Renders the environment."""
        if mode == "human":
            print(
                f"Time: {self.time_counter}, Containers: {self.num_containers}, CPU: {self.avg_mem_percent:.2f}, Latency: {self.latency:.2f}"
            )
        # Add other rendering modes if needed


async def q_learning(
    env: ContainerAutoscalingEnv,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    num_episodes=1000,
):
    """
    Q-learning algorithm implementation.

    Args:
        env (gym.Env): The Gymnasium environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        num_episodes (int): The number of episodes to train for.

    Returns:
        numpy.ndarray: The learned Q-table.
    """
    # Initialize Q-table
    num_cpu_bins = 20  # Number of CPU bins
    num_latency_bins = 20  # Number of latency bins
    max_containers = 5
    num_actions = env.action_space.n  # -> 3
    q_table = np.zeros(
        (num_cpu_bins * num_latency_bins * max_containers, num_actions)
    )  # Simplified state space discretization

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Exploration vs. Exploitation
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                discrete_state = discretize_state(state)
                action = np.argmax(q_table[discrete_state, :])  # Exploit

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Q-table update
            discrete_state = discretize_state(state)
            discrete_next_state = discretize_state(next_state)
            q_table[discrete_state, action] = q_table[
                discrete_state, action
            ] + alpha * (
                reward
                + gamma * np.max(q_table[discrete_next_state, :])
                - q_table[discrete_state, action]
            )

            state = next_state

            await asyncio.sleep(10)

        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}")

    return q_table


if __name__ == "__main__":
    env = gym.make(
        id="container-autoscaling-v0",
        render_mode=None,
        scaler=Autoscaler(url="http://localhost:8083/scale"),
        metrics_collector=MetricsCollector(url="http://localhost:9090"),
    )
    # print(check_env(env.unwrapped))
    q_table_path = "q_table.npy"  # Define the path to save/load Q-table

    # Check if Q-table exists, load if it does, else train.
    # if os.path.exists(q_table_path):
    #     q_table = np.load(q_table_path)
    #     print(f"Q-table loaded from {q_table_path}")
    # else:
    print("Start learning")
    q_table = asyncio.run(q_learning(env))  # Pass the path
    np.save(q_table_path, q_table)
    print(f"Q-table saved to {q_table_path}")

    # Example of using the trained Q-table (Greedy policy)
    # state, _ = env.reset()
    # done = False
    # print("\nTesting the learned policy:")
    # while not done:
    #     discrete_state = q_table.discretize_state(state)
    #     action = np.argmax(q_table[discrete_state, :])
    #     state, reward, done, _, _ = env.step(action)
    #     env.render(mode=None)
    #     if done:
    #         break
    env.close()
