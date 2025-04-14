import numpy as np


def discretize_cpu_util(cpu_util: float, num_cpu_buckets: int = 5) -> np.float32:
    """Discretizes the continuous CPU utilization state."""
    return np.float32(round(cpu_util * num_cpu_buckets / 100))
