import asyncio
import os
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from autoscaler import Autoscaler
from container_autoscaling_env import Action, ContainerAutoscalingEnv
from metrics_collector import MetricsCollector

NUM_EPISODES = 1
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.9
INITIAL_EPSILON = 1.0
EPSILON_DECAY_RATE = 0.01
MIN_EPSILON = 0.01
NUM_STATE_BINS = 10
TARGET_MEMORY_PERCENT = 70.0


class State:
    def __init__(
        self,
        memory_percent: float,
        spout_messages_emitted: float,
        number_of_containers: int,
    ):
        # Discretize CPU utilization and spout_messages_emitted into 20 bins each
        cpu_bin = int(np.clip(memory_percent, 0, 100) / 5)  # 0-100 to 20 bins
        spout_messages_emitted_bin = int(
            np.clip(spout_messages_emitted, 0, 20_000) / 500  # 40 bins
        )  # 0-1000 to 20 bins
        containers = int(number_of_containers)
        self.value = (cpu_bin, spout_messages_emitted_bin, containers)

    def __str__(self):
        return (
            f"Memory percent: {self.value[0]}"
            + f", spout_messages_emitted: {self.value[1]}"
            + f", Containers: {self.value[2]}"
        )


class QLearningContainerAutoscalingEnv(ContainerAutoscalingEnv):
    def __init__(
        self,
        scaler: Autoscaler,
        metrics_collector: MetricsCollector,
        max_cpu_utilization=80,
        min_cpu_utilization=20,
        reward_weights=(0.7, 1.0, 0.5),
        render_mode=None,
    ):
        super().__init__(
            scaler=scaler,
            metrics_collector=metrics_collector,
            render_mode=render_mode,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [0, 0, self.min_containers],
                dtype=np.int64,
            ),
            high=np.array(
                [20, 40, self.max_containers],
                dtype=np.int64,
            ),
            dtype=np.int64,
        )  # [memory_percent, spout_messages_emitted, num_containers]
        self.reward_weights = reward_weights

    def _get_current_state(self):
        avg_mem_percent = np.mean(
            self.metrics_collector.get_mem_percent(), dtype=np.float32
        )
        spout_messages_emitted = np.mean(
            self.metrics_collector.get_storm_spout_messages_emitted(),
            dtype=np.float32,
        )
        num_containers = self.scaler.get_number_of_containers()

        return State(
            memory_percent=avg_mem_percent,
            spout_messages_emitted=spout_messages_emitted,
            number_of_containers=num_containers,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        return self._get_current_state(), {}

    def step(self, action: Action):
        """
        Executes one time step in the environment.

        Args:
            action (Action): The action to take.

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.time_counter += 1

        self.state = self._get_current_state()

        reward = self._calculate_reward(action)

        self._perform_action(action)

        terminated = self._is_done()

        self.render(mode="human")

        return self.state, reward, terminated, False, {}

    def _calculate_reward(self, action: Action):
        """Calculates the reward based on current state."""
        states = self.metrics_collector.get_mem_percent()
        if len(states) == 1 and states[0] < TARGET_MEMORY_PERCENT:
            memory_penalty = 0
        else:
            memory_penalty = -np.sqrt(np.mean((states - TARGET_MEMORY_PERCENT) ** 2))
        # memory_penalty = -((state.value[0] - 14) ** 2)
        # spout_messages_emitted_reward = -state.value[1] / 100.0  # Normalize spout_messages_emitted
        number_of_containers = len(states)
        cost_penalty = -np.log(number_of_containers + 1)
        # cost_reward = -state.value[2] / self.max_containers  # Normalize cost

        stability_reward = 0
        if (
            self.previous_action == Action.SCALE_DOWN
            or self.previous_action == Action.SCALE_UP
        ):
            stability_reward = 1  # slight penalty for scaling actions

        self.previous_action = action

        if self.render_mode == "human":
            print(
                f"mem_pel: {self.reward_weights[0] * memory_penalty}"
                + f", cost_pel: {self.reward_weights[1] * cost_penalty}"
                + f", stab_reward: {self.reward_weights[2] * stability_reward}"
            )

        return (
            self.reward_weights[0] * memory_penalty
            + self.reward_weights[1] * cost_penalty
            + self.reward_weights[2] * stability_reward
        )

    def render(self, mode="human"):
        """Renders the environment."""
        if mode == "human":
            print(f"Time: {self.time_counter}")
            print(self.state)
            print(self.previous_action)
            print()


class QLearningAgent:
    def __init__(
        self,
        env: QLearningContainerAutoscalingEnv,
        learning_rate: float,
        discount_factor: float,
        initial_epsilon: float,
        epsilon_decay_rate: float,
        min_epsilon: float,
        num_memory_percent_bins: int,
        num_spout_messages_emitted_bins: int,
    ):
        self.rng = np.random.default_rng()

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # max_containers = env.max_containers

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        # self.q_table = np.zeros(
        #     (
        #         (num_memory_percent_bins, num_spout_messages_emitted_bins, max_containers),
        #         env.action_space.n,
        #     ),
        #     dtype=np.int64,
        # )
        self.training_error = []

    def select_action(self, state: State):
        if self.rng.random() < self.epsilon:
            action = self.action_space.sample()  # Explore
            print("Explore")
        else:
            action = np.argmax(self.q_values[state.value])  # Exploit
            print("Exploit")

        return Action(action)

    def update_q_table(
        self, state: State, action: Action, reward: float, next_state: State
    ):
        # best_next_action = np.argmax(self.q_table[next_state.values, :])
        # self.q_table[state.values, action] = self.q_table[
        #     state.values, action
        # ] + self.learning_rate * (
        #     reward
        #     + self.discount_factor * self.q_table[next_state.values, best_next_action]
        #     - self.q_table[state.values, action]
        # )
        future_q_value = np.max(self.q_values[next_state.value])
        temporal_difference = (
            reward
            + self.discount_factor * future_q_value
            - self.q_values[state.value][action.value]
        )

        self.q_values[state.value][action.value] = (
            self.q_values[state.value][action.value] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon, self.epsilon * (1 - self.epsilon_decay_rate)
        )

    def get_q_values(self):
        return self.q_values


async def train_agent(
    env: QLearningContainerAutoscalingEnv,
    agent: QLearningAgent,
    num_episodes=NUM_EPISODES,
):
    for episode in range(num_episodes):
        state, _ = env.reset()

        total_reward = 0
        delay_buffer = deque(maxlen=2)

        terminated = False
        while not terminated:
            action = agent.select_action(state)
            next_state, reward, terminated, _, _ = env.step(action)

            total_reward += reward

            delay_buffer.append((next_state, action))

            if len(delay_buffer) == 2:
                state, action = delay_buffer.popleft()

                agent.update_q_table(
                    state=state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                )

            agent.decay_epsilon()
            await asyncio.sleep(20)

    return agent


if __name__ == "__main__":
    env = QLearningContainerAutoscalingEnv(
        render_mode="human",
        scaler=Autoscaler(url="http://localhost:8083/scale"),
        metrics_collector=MetricsCollector(url="http://localhost:9090"),
    )
    agent = QLearningAgent(
        env=env,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        min_epsilon=MIN_EPSILON,
        num_spout_messages_emitted_bins=20,
        num_memory_percent_bins=20,
    )

    q_table_path = "q_table.npy"  # Define the path to save/load Q-table

    print("Start learning")
    # Check if Q-table exists, load if it does, else train.
    if os.path.exists(q_table_path):
        q_table = np.load(q_table_path)
        print(f"Q-table loaded from {q_table_path}")
    else:
        q_table = asyncio.run(train_agent(env, agent))
        np.save(q_table_path, q_table.get_q_values())
        print(f"Q-table saved to {q_table_path}")
        print("\nLearned Q-table:")
        print(q_table.get_q_values())

    def get_moving_avgs(arr, window, convolution_mode):
        return (
            np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode)
            / window
        )

    # Smooth over a 500 episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error, rolling_length, "same"
    )
    axs[2].plot(
        range(len(training_error_moving_average)), training_error_moving_average
    )
    plt.tight_layout()
    plt.show()
