import os
from datetime import datetime

import matplotlib

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use("Agg")


class Log:
    def __init__(self, modelName: str):
        # Path to Run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{modelName}.log")

    def start_training(self):
        start_time = datetime.now()

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, "w") as file:
            file.write(log_message + "\n")

    def new_best_reward(self, episode: int, episode_reward: float, best_reward: float):
        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward - best_reward) / best_reward * 100:+.1f}%) at episode {episode}, saving model..."
        print(log_message)
        with open(self.LOG_FILE, "a") as file:
            file.write(log_message + "\n")
