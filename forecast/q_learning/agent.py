import asyncio
import itertools
import os
import random
from collections import defaultdict, deque

import numpy as np
import yaml

from agent import Agent
from autoscaler import Autoscaler
from custom_env.container_autoscaling import Action
from metrics_collector import MetricsCollector
from q_learning.env import QLearningEnv
from q_learning.state import State


class QLearningAgent(Agent):
    q_table_path = "results/q_table.npy"

    def __init__(
        self,
        env: QLearningEnv,
        is_training: bool = False,
    ):
        super().__init__(
            env=env,
            is_training=is_training,
            hyper_parameters_set="qscale",
        )

        self.load_hyperparameters(self.hyper_parameters_set)

        n_actions = env.action_space.n

        self.q_values = defaultdict(lambda: np.zeros(n_actions))

        if os.path.exists(self.MODEL_FILE):
            self.load_q_table()

    def select_action(self, state: State):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()  # Explore
        else:
            action = np.argmax(self.q_values[state.value])  # Exploit

        return Action(action)

    def load_hyperparameters(self, hyper_parameters_set):
        with open("hyper-parameters.yml", "r") as f:
            all_hyper_parameters_set = yaml.safe_load(f)
            hyper_parameters = all_hyper_parameters_set[hyper_parameters_set]

            self.epsilon = hyper_parameters["epsilon_init"]
            self.epsilon_decay = hyper_parameters["epsilon_decay"]
            self.epsilon_min = hyper_parameters["epsilon_min"]
            self.learning_rate = hyper_parameters["learning_rate"]
            self.discount_factor = hyper_parameters["discount_factor"]
            self.graph_update_interval = hyper_parameters["graph_update_interval"]

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
            self.q_values[state.value][action.value]
            + self.learning_rate * temporal_difference
        )
        self.losses.append(temporal_difference)

    def get_q_values(self):
        return dict(self.q_values)

    async def train(self):
        await super().train()

        for episode in itertools.count():
            state, _ = env.reset()

            episode_reward = 0
            delay_buffer = deque(maxlen=2)

            terminated = False
            while not terminated:
                action = agent.select_action(state)
                next_state, reward, terminated, _, info = env.step(action)

                if info["out_of_bound"]:
                    action = Action.DO_NOTHING

                episode_reward += reward
                self.rewards.append(reward)

                delay_buffer.append((next_state, action))

                if len(delay_buffer) == 2:
                    state, action = delay_buffer.popleft()

                    agent.update_q_table(
                        state=state,
                        action=action,
                        next_state=next_state,
                        reward=reward,
                    )

                self.decay_epsilon()
                await asyncio.sleep(30)

            self.save_q_table()
            self.update_graph()

    def save_q_table(self):
        np.save(self.q_table_path, dict(self.q_values))

    def load_q_table(self):
        q_table = np.load(file=self.q_table_path, allow_pickle=True).item()
        self.q_values.update(q_table)


if __name__ == "__main__":
    env = QLearningEnv(
        render_mode="human",
        scaler=Autoscaler(),  # is_testing=True
        metrics_collector=MetricsCollector(),
    )

    agent = QLearningAgent(env=env, is_training=True)

    asyncio.run(agent.train())
