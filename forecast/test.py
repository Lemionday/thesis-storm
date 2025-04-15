from collections import defaultdict

import numpy as np

from container_autoscaling_env import Action
from metrics_collector import PROMETHEUS_URL, MetricsCollector
from q_learning_agent import State

if __name__ == "__main__":
    collector = MetricsCollector(url=PROMETHEUS_URL)
    mem = collector.get_mem_percent()
    print(mem)
    print(
        State(
            memory_percent=np.mean(mem),
            latency=0,
            number_of_containers=0,
        )
    )

    q_table = defaultdict(lambda: np.zeros(3))  # Example with 3 actions
    q_table[(10, 100, 10)][Action.DO_NOTHING.value] = 1
    q_table[(10, 100, 10)][0] = 1
    print(q_table)
