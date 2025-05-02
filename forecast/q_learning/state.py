import numpy as np

SPOUT_MESSAGES_MAX_THRESHOLD = 25_000
SPOUT_MESSAGES_BIN_RANGE = 1250  # 25_000 / (10*60) * 30


class State:
    def __init__(
        self,
        memory_percent: float,
        previous_spout_messages_emitted: float,
        spout_messages_emitted: float,
        number_of_containers: int,
    ):
        # Discretize memory usage and spout_messages_emitted into 20 bins each
        cpu_bin = int(np.clip(memory_percent, 0, 100) / 10)  # 0-100 to 20 bins
        previous_spout_messages_emitted = int(
            np.clip(previous_spout_messages_emitted, 0, SPOUT_MESSAGES_MAX_THRESHOLD)
            / SPOUT_MESSAGES_BIN_RANGE
        )
        spout_messages_emitted_bin = int(
            np.clip(spout_messages_emitted, 0, SPOUT_MESSAGES_MAX_THRESHOLD)
            / SPOUT_MESSAGES_BIN_RANGE  # 20 bins
        )  # 0-1000 to 20 bins
        containers = int(number_of_containers)
        self.value = (
            cpu_bin,
            previous_spout_messages_emitted,
            spout_messages_emitted_bin,
            containers,
        )

    def __str__(self):
        return (
            f"memory percent: {self.value[0]}"
            + f", previous_spout_messages_emitted: {self.value[1]}"
            + f", spout_messages_emitted: {self.value[2]}"
            + f", containers: {self.value[3]}"
        )
