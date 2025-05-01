import os

import requests

AUTOSCALER_URL = os.getenv("AUTOSCALER_URL", "http://localhost:8083")


class Autoscaler:
    """
    A class to interact with Storm autoscaler program
    """

    """
    Args:
        url (str): The URL of the endpoint.
    """

    def __init__(self, url=AUTOSCALER_URL, is_testing=False):
        self.url = url + "/scale"
        self.is_testing = is_testing

    def _make_request(self, replicas: int = 0):
        try:
            if replicas == 0:
                resp = requests.get(self.url)
            else:
                resp = requests.post(self.url, str(replicas))

            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {self.url}: {e}")
            return None

    def get_number_of_containers(self):
        resp = self._make_request()
        if resp is None:
            return None

        body = resp.text.strip()
        # print(body)
        try:
            return int(body)
        except ValueError:
            return None

    def set_number_of_containers(self, replicas: int):
        if self.is_testing:
            return replicas

        if replicas < 2 or replicas > 5:
            return None

        resp = self._make_request(replicas=replicas)
        if resp is None:
            return None

        body = resp.text.strip()
        # print(body)
        try:
            running = int(body)
            if running != replicas:
                return None

            return running
        except ValueError:
            return None


if __name__ == "__main__":
    url = "http://localhost:8083"

    scaler = Autoscaler(url)

    running = scaler.get_number_of_containers()
    if running is not None:
        print(f"Extracted number from text: {running}")
    else:
        print("Could not extract a number from the text response.")

    running = scaler.set_number_of_containers(2)
    if running is not None:
        print(f"Extracted number from text: {running}")
    else:
        print("Could not extract a number from the text response.")
