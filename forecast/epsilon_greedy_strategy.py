import numpy as np

INITIAL_EPSILON = 1.0
EPSILON_DECAY_RATE = 0.01
MIN_EPSILON = 0.05


class EpsilonGreedyStrategy:
    def __init__(
        self,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        min_epsilon=MIN_EPSILON,
        render_mode=None,
        learning=False,
    ):
        self.epsilon = initial_epsilon
        if learning:
            self.epsilon = min_epsilon

        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng()
        self.render_mode = render_mode

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon, self.epsilon * (1 - self.epsilon_decay_rate)
        )

    def isNextActionExplore(self):
        if self.rng.random() < self.epsilon:
            if self.render_mode == "human":
                print("Explore")
            return True

        if self.render_mode == "human":
            print("Exploit")
        return False
