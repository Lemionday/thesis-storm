import time
from argparse import Action

import numpy as np

from helpers import discretize_cpu_util
from vm_auto_scale_env import Action, VMAutoScaleEnv

NUM_EPISODES = 1000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
INITIAL_EPSILON = 1.0
EPSILON_DECAY_RATE = 0.001
MIN_EPSILON = 0.01
NUM_STATE_BINS = 10
MAX_VMS = 5


# --- Q-Learning Agent Class ---
class QLearningAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        min_epsilon=MIN_EPSILON,
        num_state_bins=NUM_STATE_BINS,
        max_vms=MAX_VMS,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.num_state_bins = num_state_bins
        self.q_table = np.zeros(((num_state_bins, max_vms), action_space.n))
        self.rng = np.random.default_rng()

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            action = self.action_space.sample()  # Explore
        else:
            action = np.argmax(self.q_table[state, :])  # Exploit
        return action

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] = self.q_table[
            state, action
        ] + self.learning_rate * (
            reward
            + self.discount_factor * self.q_table[next_state, best_next_action]
            - self.q_table[state, action]
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon, self.epsilon * (1 - self.epsilon_decay_rate)
        )

    def get_q_table(self):
        return self.q_table


# --- Training Function ---
def train_agent(env: VMAutoScaleEnv, agent: QLearningAgent, num_episodes=NUM_EPISODES):
    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = (discretize_cpu_util(observation), 1)
        pre_state = (discretize_cpu_util(observation), 1)
        pre_action = Action.DO_NOTHING
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)

            agent.update_q_table(pre_state, pre_action, reward, state)

            pre_action = action
            pre_state = state
            state = agent.discretize_state(obs)
            total_reward += reward

            # env.render() # Uncomment to see the environment evolve

            agent.decay_epsilon()
            print(
                f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.2f}, Final VMs = {info['num_vms']}"
            )

    return agent


# --- Main Execution ---
if __name__ == "__main__":
    env = VMAutoScaleEnv()
    agent = QLearningAgent(env.observation_space, env.action_space)

    trained_agent = train_agent(env, agent)

    print("\nLearned Q-table:")
    print(trained_agent.get_q_table())

    # --- Demonstrate Learned Policy ---
    print("\nDemonstrating Learned Policy:")
    observation, _ = env.reset()
    state = trained_agent.discretize_state(observation)
    terminated = False
    total_reward = 0
    for _ in range(50):
        action = np.argmax(trained_agent.get_q_table()[state, :])
        new_observation, reward, terminated, truncated, info = env.step(action)
        new_state = trained_agent.discretize_state(new_observation)
        total_reward += reward
        state = new_state
        env.render()
        if terminated:
            break
        time.sleep(0.5)

    env.close()
