import argparse
import asyncio
import itertools
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn

from agent import Agent
from autoscaler import Autoscaler
from custom_env.container_autoscaling import Action
from dq_learning.dqn import DQN
from dq_learning.env import DeepQLearningEnv
from dq_learning.replay_memory import ReplayMemory
from helpers.logs import Log
from metrics_collector import MetricsCollector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"  # force cpu


class DQAgent(Agent):
    def __init__(
        self,
        env: DeepQLearningEnv,
        is_training=False,
    ):
        super().__init__(env, is_training=is_training, hyper_parameters_set="deepscale")

        self.load_hyperparameters(self.hyper_parameters_set)

        self.logger = Log(self.hyper_parameters_set)

        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Create policy and target network.
        self.policy_dqn = DQN(
            in_states=n_states,
            h1_nodes=10,
            out_actions=n_actions,
        ).to(device)

        if self.is_training:
            self.memory = ReplayMemory(capacity=self.replay_memory_size)

            self.target_dqn = DQN(
                in_states=n_states,
                h1_nodes=10,
                out_actions=n_actions,
            ).to(device)

            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(
                self.policy_dqn.parameters(), lr=self.learning_rate
            )
        else:
            # Load learned policy
            self.policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # Switch model to evaluation mode
            self.policy_dqn.eval()

        self.loss_fn = nn.MSELoss()

    async def train(self):
        await super().train()

        for episode in itertools.count():
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            terminated = False

            episode_reward = 0

            while not terminated:
                action = self.select_action(state)

                # Observe new state
                next_state, reward, terminated, _, info = self.env.step(action)

                # Accumulate reward
                episode_reward += reward

                # Convert next_state and reward to tensor
                next_state = torch.tensor(
                    next_state, dtype=torch.float32, device=device
                )
                reward = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=device,
                )
                action = torch.tensor(
                    action.value,
                    dtype=torch.long,
                    device=device,
                )
                if self.is_training:
                    self.memory.append(
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            terminated,
                        )
                    )

                state = next_state

                if len(self.memory) > self.batch_size:
                    batch = self.memory.sample(self.batch_size)

                    self.optimize(batch)

                    if self.env.time_counter % self.network_sync_rate == 0:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

                await asyncio.sleep(30)

            self.decay_epsilon()
            self.rewards.append(episode_reward)
            self.update_graph()
            self.save_model(episode=episode, episode_reward=episode_reward)

    def optimize(self, batch):
        # print("optimize neural network")

        states, actions, rewards, next_states, terminateds = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminateds = torch.tensor(terminateds).float().to(device)

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminateds)
                * self.discount_factor
                * self.target_dqn(next_states).max(dim=1)[0]
            )

        current_q = (
            self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        )

        # Compute loss for the whole batch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update network parameters

    def save_model(self, episode, episode_reward):
        # Save model when new best reward is obtained.
        if not self.is_training:
            return

        if episode_reward > self.best_reward:
            self.logger.new_best_reward(
                episode=episode,
                episode_reward=episode_reward,
                best_reward=self.best_reward,
            )

            torch.save(self.policy_dqn.state_dict(), self.MODEL_FILE)
            self.best_reward = episode_reward

    def load_hyperparameters(self, hyper_parameters_set):
        with open("hyper-parameters.yml", "r") as f:
            all_hyper_parameters_set = yaml.safe_load(f)
            hyper_parameters = all_hyper_parameters_set[hyper_parameters_set]

            self.replay_memory_size = hyper_parameters["replay_memory_size"]
            self.batch_size = hyper_parameters["batch_size"]
            self.epsilon = hyper_parameters["epsilon_init"]
            self.epsilon_decay = hyper_parameters["epsilon_decay"]
            self.epsilon_min = hyper_parameters["epsilon_min"]
            self.learning_rate = hyper_parameters["learning_rate"]
            self.discount_factor = hyper_parameters["discount_factor"]
            self.network_sync_rate = hyper_parameters["network_sync_rate"]
            self.graph_update_interval = hyper_parameters["graph_update_interval"]

    def select_action(self, state):
        if self.is_training and random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                # tensor([1, 2, 3, ...]) => tensor([[1, 2, 3, ...]])
                action = (
                    self.policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()
                )

        return Action(action)




if __name__ == "__main__":
    print(f"Use device: {device}")

    env = DeepQLearningEnv(
        render_mode="human",
        scaler=Autoscaler(),  # is_testing=True
        metrics_collector=MetricsCollector(),
        step_interval=30,
    )

    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    agent = DQAgent(env=env, is_training=args.train)
    asyncio.run(agent.train())
