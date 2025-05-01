import os

import numpy as np
from prometheus_api_client import PrometheusConnect

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


class MetricsCollector:
    def __init__(self, url=PROMETHEUS_URL):
        try:
            self.prom = PrometheusConnect(
                url=url, disable_ssl=True
            )  # disable_ssl for local testing
        except Exception as e:
            print(f"Error connecting to Prometheus: {e}")
            exit()

    def get_cpu_utilization(self):
        """
        Queries the Prometheus server for current CPU utilization
        of Storm supervisors.

        Returns:
            np.ndarray[np.float32]: An array of CPU utilization values,
            each representing the percentage of CPU used by an individual
            Storm supervisor.
        """
        query = "avg_over_time(storm_supervisor_cpu_percent[20s])"

        # Each record returned by Prometheus contains a 'value' field,
        # which is a tuple: (timestamp, utilization as a string).
        # We extract the utilization value (index 1), convert to float,
        # and collect all into a NumPy array.
        results = self.prom.custom_query(query=query)
        cpu_utilizations = np.array(
            [record["value"][1] for record in results],
            dtype=np.float32,
        )

        return cpu_utilizations

    def get_mem_percent(self):
        query = "avg_over_time(storm_supervisor_memory_percent[20s])"
        results = self.prom.custom_query(query=query)
        memory_percents = np.array(
            [record["value"][1] for record in results],
            dtype=np.float32,
        )

        return memory_percents[memory_percents != 0]

    def get_storm_spout_messages_emitted(self):
        query = "avg_over_time(storm_spout_messages_emitted[20s])"
        results = self.prom.custom_query(query=query)
        num_messages = np.array(
            [record["value"][1] for record in results],
            dtype=np.float32,
        )
        _ret = num_messages[num_messages != 0]
        if len(_ret) == 0:
            return [0]
        return _ret

    def get_spout_messages_latency(self):
        query = "avg_over_time(storm_spout_messages_latency[20s])"
        results = self.prom.custom_query(query=query)
        return np.array(
            [record["value"][1] for record in results],
            dtype=np.float32,
        ).sum()


# unit testing
if __name__ == "__main__":
    mc = MetricsCollector(url=PROMETHEUS_URL)
    print(mc.get_mem_percent())
