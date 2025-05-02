import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from helpers.logs import RUNS_DIR, Log


class Agent:
    rewards = []
    losses = []

    def __init__(self, env, hyper_parameters_set: str, is_training: bool = False):
        self.env = env
        self.is_training = is_training
        self.hyper_parameters_set = hyper_parameters_set

        self.logger = Log(self.hyper_parameters_set)

        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyper_parameters_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyper_parameters_set}.png")

        if self.is_training:
            self.epsilon_hist = []

            # Track best reward
            self.best_reward = -9999999

            self.current_time = datetime.now()

    async def train(self):
        if self.is_training:
            self.last_graph_update_time = datetime.now()

            self.logger.start_training()

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )
        self.epsilon_hist.append(self.epsilon)

    def update_graph(self):
        current_time = datetime.now()

        if current_time - self.last_graph_update_time < timedelta(
            seconds=self.graph_update_interval
        ):
            return

        self.last_graph_update_time = current_time

        fig = plt.figure(1)

        # Plot rewards (Y-axis) vs episodes (X-axis)
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        plt.xlabel("Episodes")
        plt.ylabel("Mean Rewards")
        plt.plot(self.rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.xlabel("Time Steps")
        plt.ylabel("Epsilon Decay")
        plt.plot(self.epsilon_hist)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
