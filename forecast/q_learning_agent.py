import asyncio
import os
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np

from autoscaler import AUTOSCALER_URL, Autoscaler
from container_autoscaling_env import NUM_EPISODES, Action, ContainerAutoscalingEnv
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from metrics_collector import PROMETHEUS_URL, MetricsCollector
from plot_result import plot_result

NUM_STATE_BINS = 10
SPOUT_MESSAGES_MAX_THRESHOLD = 25_000
SPOUT_MESSAGES_BIN_RANGE = 1250  # 25_000 / (10*60) * 30


class State:
    def __init__(
        self,
        memory_percent: float,
        spout_messages_emitted: float,
        number_of_containers: int,
    ):
        # Discretize memory usage and spout_messages_emitted into 20 bins each
        cpu_bin = int(np.clip(memory_percent, 0, 100) / 10)  # 0-100 to 20 bins
        spout_messages_emitted_bin = int(
            np.clip(spout_messages_emitted, 0, SPOUT_MESSAGES_MAX_THRESHOLD)
            / SPOUT_MESSAGES_BIN_RANGE  # 20 bins
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

    def _get_observation(self):
        avg_mem_percent = np.mean(
            self.metrics_collector.get_mem_percent(), dtype=np.float32
        )
        spout_messages_emitted = np.mean(
            self.metrics_collector.get_storm_spout_messages_emitted(),
            dtype=np.float32,
        )
        num_containers = self.scaler.get_number_of_containers()

        return State(
            memory_percent=avg_mem_percent / 100.0,
            spout_messages_emitted=spout_messages_emitted,
            number_of_containers=num_containers,
        )

    def step(self, action: Action):
        """
        Executes one time step in the environment.

        Args:
            action (Action): The action to take.

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.time_counter += 1

        self.state = self._get_observation()

        reward = self._calculate_reward(action)

        self.return_queue.append(reward)

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
            print(self.previous_action)
            print()


class QLearningAgent:
    def __init__(
        self,
        env: QLearningContainerAutoscalingEnv,
        learning_rate: float,
        discount_factor: float,
        num_memory_percent_bins: int,
        num_spout_messages_emitted_bins: int,
        q_values=None,
    ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.epsilon_strategy = EpsilonGreedyStrategy(learning=False)
        if q_values is not None:
            self.q_values.update(q_values)

            # Start exploiting but preserve some explore
            self.epsilon_strategy = EpsilonGreedyStrategy(learning=True)

        self.training_error = [1]

    def select_action(self, state: State):
        if self.epsilon_strategy.isNextActionExplore():
            action = self.action_space.sample()  # Explore
        else:
            action = np.argmax(self.q_values[state.value])  # Exploit

        self.epsilon_strategy.decay_epsilon()

        return Action(action)

    def update_q_table(
        self, state: State, action: Action, reward: float, next_state: State
    ):
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

    def get_q_values(self):
        return dict(self.q_values)


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
            next_state, reward, terminated, _, info = env.step(action)

            if info["out_of_bound"]:
                action = Action.DO_NOTHING

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

            await asyncio.sleep(30)

    return agent


if __name__ == "__main__":
    env = QLearningContainerAutoscalingEnv(
        render_mode="human",
        scaler=Autoscaler(url=AUTOSCALER_URL),
        metrics_collector=MetricsCollector(url=PROMETHEUS_URL),
    )

    q_table = None
    q_table_path = "results/q_table.npy"
    # Check if Q-table exists, load if it does.
    if os.path.exists(q_table_path):
        q_table = np.load(file=q_table_path, allow_pickle=True).item()
        print(f"Q-table loaded from {q_table_path}")
        print()

    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.9,
        num_spout_messages_emitted_bins=20,
        num_memory_percent_bins=20,
        q_values=q_table,
    )

    q_table = asyncio.run(train_agent(env, agent))

    np.save(q_table_path, q_table.get_q_values())
    print(f"Q-table saved to {q_table_path}")
    print()

    plot_result(env.return_queue, agent.training_error)
