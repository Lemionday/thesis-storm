import random
from collections import deque


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def append(self, transition) -> None:
        self.memory.append(transition)

    def sample(self, sample_size: int):
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        return len(self.memory)
